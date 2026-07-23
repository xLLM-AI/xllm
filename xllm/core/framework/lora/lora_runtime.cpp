/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "lora_runtime.h"

#include <glog/logging.h>

#include <mutex>

#include "lora_metrics.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#endif

namespace xllm {

LoRARuntime& LoRARuntime::instance() {
  static LoRARuntime g;
  return g;
}

void LoRARuntime::init(const LoRAConfig& config) {
  {
    std::lock_guard g(materialise_mu_);
    config_ = config;
    loader_ = std::make_unique<LoRAAdapterLoader>(config_);
  }
  // P1-A.3: bind the final-removal callback so per_proj_device_pool_ frees
  // its per-adapter tensors at the exact moment the registry drops the last
  // reference (either synchronous unregister with pin=0, or the deferred
  // unpin after a drain).
  registry_.set_on_register([](uint64_t int_id, const std::string& lora_name) {
    LoRAMetrics::instance().add_adapter(int_id, lora_name);
    // Newly registered but not yet installed -> pending (0). Actual
    // active_state flip to 1 happens when install_*_on_device succeeds.
    LoRAMetrics::instance().set_state(int_id, 0);
  });
  registry_.set_on_final_removal([this](uint64_t int_id) {
    {
      std::lock_guard g(materialise_mu_);
      auto it = per_proj_device_pool_.find(int_id);
      if (it != per_proj_device_pool_.end()) {
        const size_t slot_count = it->second.size();
        per_proj_device_pool_.erase(it);
        LOG(INFO) << "[LoRARuntime] freed per_proj device pool for id="
                  << int_id << " slots=" << slot_count;
      }
      // Also drop legacy P0-A dummy pool entry if the same int_id sits there.
      auto dp = device_pool_.find(int_id);
      if (dp != device_pool_.end()) {
        device_pool_.erase(dp);
      }
      // P1 hot-swap HBM leak fix (C4 stress test 07-21): also erase the
      // MoE expert pool. Without this, each hot-swap of a MoE-experts
      // adapter (Qwen3-30B-A3B) accumulates ~1.6 GB per rank; ~2 churns
      // exhaust NPU 61 GB and crash the process with PTA memory error.
      auto mp = moe_expert_lora_pool_.find(int_id);
      if (mp != moe_expert_lora_pool_.end()) {
        const size_t layers = mp->second.size();
        moe_expert_lora_pool_.erase(mp);
        LOG(INFO) << "[LoRARuntime] freed MoE expert pool for id=" << int_id
                  << " layers=" << layers;
      }
    }
    // Drop metrics bvars AFTER releasing materialise_mu_ (LoRAMetrics
    // takes its own shared_mutex; keep locks disjoint).
    LoRAMetrics::instance().remove_adapter(int_id);
  });
  LOG(INFO) << "[LoRARuntime] initialised, enable=" << config_.enable_lora;
}

bool LoRARuntime::enabled() const { return config_.enable_lora; }

void LoRARuntime::set_model_device_dtype(torch::Device device,
                                         torch::ScalarType dtype) {
  {
    std::lock_guard g(materialise_mu_);
    model_device_ = device;
    model_dtype_ = dtype;
  }
  LOG(INFO) << "[LoRARuntime] model registered device=" << device
            << " dtype=" << dtype;

  // Path C prod v3 hot-swap: spawn the pinned executor thread here so it
  // inherits our aclrtSetDevice state (ModelContext ctor called
  // aclrtSetDevice(device_id) earlier on this same thread).
  if (!executor_started_) {
    executor_started_ = true;
    const int32_t device_index = device.index();
    executor_thread_ =
        std::thread(&LoRARuntime::executor_loop, this, device_index, dtype);
    LOG(INFO) << "[LoRARuntime] spawned hot-swap executor thread for "
              << device;
  }
}

// W4-A3: set the hot-swap install config so the pinned executor thread
// knows whether to run per-proj-only or per-proj + moe-experts (and with
// what TP + MoE args). Called from model ctor.
void LoRARuntime::set_hotswap_config(const HotswapConfig& cfg) {
  std::lock_guard<std::mutex> lk(materialise_mu_);
  hotswap_config_ = cfg;
  hotswap_config_set_ = true;
  LOG(INFO) << "[LoRARuntime] hotswap config set: tp={" << cfg.tp.tp_size << ","
            << cfg.tp.tp_rank << "} install_per_proj=" << cfg.install_per_proj
            << " install_moe=" << cfg.install_moe_experts
            << " num_experts_total=" << cfg.num_experts_total
            << " per_rank=" << cfg.num_experts_per_rank
            << " start_expert_id=" << cfg.start_expert_id
            << " moe_inter=" << cfg.moe_intermediate_size;
}

void LoRARuntime::executor_loop(int32_t device_index, torch::ScalarType dtype) {
#if defined(USE_NPU)
  aclError ret = aclrtSetDevice(device_index);
  if (ret != 0) {
    LOG(ERROR) << "[LoRARuntime] executor_loop aclrtSetDevice(" << device_index
               << ") failed, err=" << ret;
    return;
  }
#endif
  LOG(INFO) << "[LoRARuntime] executor_loop entered on device=" << device_index;

  while (!executor_stop_.load()) {
    LoadTask task;
    {
      std::unique_lock<std::mutex> lk(task_mu_);
      task_cv_.wait(
          lk, [this] { return executor_stop_.load() || !task_queue_.empty(); });
      if (executor_stop_.load() && task_queue_.empty()) break;
      task = std::move(task_queue_.front());
      task_queue_.pop();
    }

    LOG(INFO) << "[LoRARuntime] executor picked up task name='" << task.name
              << "'";
    torch::Device dev(torch::kPrivateUse1, device_index);

    // W4-A3: read hotswap config, prefer per-proj (+ optional moe experts)
    // install. Fall back to whole-block install if config not set (legacy
    // Path C prod v3 path).
    HotswapConfig cfg;
    bool have_cfg = false;
    {
      std::lock_guard<std::mutex> lk(materialise_mu_);
      cfg = hotswap_config_;
      have_cfg = hotswap_config_set_;
    }

    std::optional<uint64_t> id_opt;
    if (have_cfg && cfg.install_per_proj) {
      id_opt = install_static_adapter_on_device_per_proj(
          task.name, task.path, task.base_model_name, dev, dtype, cfg.tp);
      if (id_opt.has_value()) {
        LOG(INFO) << "[LoRARuntime] hot-swap per-proj installed '" << task.name
                  << "' id=" << *id_opt;
      } else {
        LOG(WARNING) << "[LoRARuntime] hot-swap per-proj install returned "
                     << "nullopt for '" << task.name
                     << "'; MoE-only adapters can still succeed via next step";
      }
      // MoE experts install: shares int_id with per-proj via registry
      // idempotency. Real MoE fine-tuned adapters have both attn (per-proj)
      // and experts tensors; attn-only adapters are handled by per-proj only.
      if (cfg.install_moe_experts) {
        auto moe_id =
            install_static_adapter_on_moe_experts(task.name,
                                                  task.path,
                                                  task.base_model_name,
                                                  dev,
                                                  dtype,
                                                  cfg.tp,
                                                  cfg.num_experts_total,
                                                  cfg.num_experts_per_rank,
                                                  cfg.start_expert_id,
                                                  cfg.moe_intermediate_size);
        if (moe_id.has_value()) {
          LOG(INFO) << "[LoRARuntime] hot-swap MoE-experts installed '"
                    << task.name << "' id=" << *moe_id;
          // If per-proj skipped (e.g. attn-only refused because adapter
          // contains only experts), surface moe id as the result.
          if (!id_opt.has_value()) id_opt = moe_id;
        }
      }
    } else {
      id_opt = install_static_adapter_on_device(
          task.name, task.path, task.base_model_name, dev, dtype);
    }
    task.result.set_value(id_opt);
  }

  LOG(INFO) << "[LoRARuntime] executor_loop exiting";
}

std::optional<uint64_t> LoRARuntime::load_and_activate_hotswap(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse hot-swap for '"
               << lora_name << "'";
    return std::nullopt;
  }
  if (!executor_started_) {
    LOG(ERROR) << "[LoRARuntime] executor not started; the model has not "
                  "called set_model_device_dtype yet";
    return std::nullopt;
  }

  LoadTask task;
  task.name = lora_name;
  task.path = lora_path;
  task.base_model_name = base_model_name;
  auto fut = task.result.get_future();

  {
    std::lock_guard<std::mutex> lk(task_mu_);
    task_queue_.push(std::move(task));
  }
  task_cv_.notify_one();

  LOG(INFO) << "[LoRARuntime] hot-swap enqueued '" << lora_name << "', waiting";
  auto id_opt = fut.get();
  LOG(INFO) << "[LoRARuntime] hot-swap completed '" << lora_name << "' id="
            << (id_opt.has_value() ? std::to_string(*id_opt) : "nullopt");
  return id_opt;
}

bool LoRARuntime::pick_whole_block_ab(const LoRAAdapter& adapter,
                                      torch::ScalarType dtype,
                                      torch::Tensor* A_out,
                                      torch::Tensor* B_out) const {
  // Path C is a *whole-decoder-block* delta -- one A/B for the whole
  // model. A real PEFT adapter has A/B per (layer, module). For MVP we
  // pick a first (module, layer=0) pair whose shapes match [rank, hidden]
  // / [hidden, rank] and treat that as the model-level delta. Real
  // per-proj / per-layer routing is P0-C.
  //
  // Preference order: q_proj > gate_proj > any first pair.
  static const std::vector<std::string> kPreferSubstr = {
      "layers.0.self_attn.q_proj",
      "self_attn.q_proj",
      "layers.0",
  };
  auto find_pair = [&](const std::string& subkey_hint,
                       torch::Tensor* A,
                       torch::Tensor* B) -> bool {
    for (const auto& [key, tensor] : adapter.tensors) {
      if (key.size() < 2) continue;
      const std::string subkey = key.substr(0, key.size() - 2);
      const std::string tail = key.substr(key.size() - 2);
      if (!subkey_hint.empty() && subkey.find(subkey_hint) == std::string::npos)
        continue;
      if (tail == "#A") {
        auto it = adapter.tensors.find(subkey + "#B");
        if (it == adapter.tensors.end()) continue;
        *A = tensor;
        *B = it->second;
        return true;
      }
    }
    return false;
  };
  for (const auto& hint : kPreferSubstr) {
    if (find_pair(hint, A_out, B_out)) break;
  }
  if (!A_out->defined() || !B_out->defined()) {
    if (!find_pair("", A_out, B_out)) {
      LOG(ERROR) << "[LoRARuntime] adapter '" << adapter.request.lora_name
                 << "' has no usable A/B pair";
      return false;
    }
  }

  // PICK_NO_CAST_APPLIED: skipping in-place .to(dtype) here — safetensors
  // mmap tensor + torch::to on CPU crashes torch_npu's opapi allocator on
  // ctor thread. Callers (install_static / load_and_activate) do their
  // own clone + cast now.
  return true;
}

std::optional<uint64_t> LoRARuntime::load_and_activate(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse to load '" << lora_name
               << "'";
    return std::nullopt;
  }
  torch::ScalarType dtype = torch::kFloat32;
  {
    std::lock_guard g(materialise_mu_);
    if (!model_dtype_.has_value()) {
      LOG(ERROR) << "[LoRARuntime] model has not registered dtype yet; "
                    "reject load '"
                 << lora_name << "'";
      return std::nullopt;
    }
    dtype = *model_dtype_;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  torch::Tensor A_cpu, B_cpu;
  if (!pick_whole_block_ab(*adapter_opt, dtype, &A_cpu, &B_cpu)) {
    return std::nullopt;
  }

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  {
    std::lock_guard g(materialise_mu_);
    pending_ =
        PendingDelta{A_cpu, B_cpu, adapter_opt->scaling, lora_name, *id_opt};
    // Clear the previous active adapter -- next forward will materialise
    // the new one on device.
    active_.reset();
  }
  LOG(INFO) << "[LoRARuntime] queued '" << lora_name << "' id=" << *id_opt
            << " A_cpu.shape=" << A_cpu.sizes()
            << " B_cpu.shape=" << B_cpu.sizes()
            << " scaling=" << adapter_opt->scaling
            << " (device migration deferred to first forward)";
  return id_opt;
}

std::optional<uint64_t> LoRARuntime::install_static_adapter_on_device(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse to install '" << lora_name
               << "'";
    return std::nullopt;
  }
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  // PICK_BRACKET_APPLIED: log before/after to isolate the crash location.
  LOG(INFO) << "[LoRARuntime] BEFORE pick_whole_block_ab for '" << lora_name
            << "'";
  torch::Tensor A_cpu, B_cpu;
  if (!pick_whole_block_ab(*adapter_opt, dtype, &A_cpu, &B_cpu)) {
    LOG(ERROR) << "[LoRARuntime] pick_whole_block_ab returned false";
    return std::nullopt;
  }
  LOG(INFO) << "[LoRARuntime] AFTER pick_whole_block_ab: A=" << A_cpu.sizes()
            << " B=" << B_cpu.sizes() << " A.dtype=" << A_cpu.dtype()
            << " A.device=" << A_cpu.device();

  // KEY DIFFERENCE vs load_and_activate: perform the CPU->NPU migration
  // on this thread, which is the model ctor thread and thus has a valid
  // NPU context (V60 experiment 2026-07-06 confirmed).
  // STATIC_TO_CLONE_APPLIED: safetensors-loaded tensors have mmap backing
  // that crashes torch_npu's opapi copy from ctor thread. Clone to a fresh
  // cpu allocator buffer, then .to(device).
  LOG(INFO) << "[LoRARuntime] static preload '" << lora_name
            << "' A_cpu.dtype=" << A_cpu.dtype() << " sizes=" << A_cpu.sizes()
            << " storage=" << A_cpu.storage().nbytes() << "B";
  // DIRECT_TO_APPLIED: even .clone() crashes on safetensors mmap tensors on
  // ctor thread. Try single-step .to(device, dtype).
  //
  // Strategy: build a fresh CPU tensor via torch::empty (same allocator as
  // V60 randn) and copy_ from mmap tensor - this materializes into normal
  // heap. Then .to(device) proceeds.
  torch::Tensor A_dev, B_dev;
  try {
    LOG(INFO) << "[LoRARuntime] materializing A_cpu via torch::empty+copy_";
    auto cpu_opts =
        torch::TensorOptions().dtype(A_cpu.dtype()).device(torch::kCPU);
    torch::Tensor A_cpu_owned = torch::empty(A_cpu.sizes(), cpu_opts);
    A_cpu_owned.copy_(A_cpu);
    torch::Tensor B_cpu_owned = torch::empty(B_cpu.sizes(), cpu_opts);
    B_cpu_owned.copy_(B_cpu);
    LOG(INFO) << "[LoRARuntime] materialized; casting + moving to " << device;
    A_dev = A_cpu_owned.to(device, dtype);
    LOG(INFO) << "[LoRARuntime] A_dev on " << A_dev.device()
              << " dtype=" << A_dev.dtype() << " sizes=" << A_dev.sizes();
    B_dev = B_cpu_owned.to(device, dtype);
    LOG(INFO) << "[LoRARuntime] B_dev on " << B_dev.device()
              << " dtype=" << B_dev.dtype() << " sizes=" << B_dev.sizes();
  } catch (const std::exception& e) {
    LOG(ERROR) << "[LoRARuntime] .to(device) failed for static adapter '"
               << lora_name << "': " << e.what();
    return std::nullopt;
  }

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  {
    std::lock_guard g(materialise_mu_);
    ActiveDelta ad;
    ad.A = A_dev;
    ad.B = B_dev;
    ad.scaling = adapter_opt->scaling;
    ad.name = lora_name;
    ad.int_id = *id_opt;
    device_pool_[*id_opt] = ad;
    active_ = std::move(ad);
    // Static preload wins over any pending -- unload() and future
    // load_and_activate() calls behave normally.
    pending_.reset();
  }
  LOG(INFO) << "[LoRARuntime] installed static adapter '" << lora_name
            << "' id=" << *id_opt << " A_device=" << A_dev.device()
            << " A.shape=" << A_dev.sizes() << " B.shape=" << B_dev.sizes()
            << " scaling=" << adapter_opt->scaling;
  return id_opt;
}

// TP-aware overload. For the current whole-block delta path we don't
// actually shard the A/B tensors -- every rank keeps the full copy and
// runs the delta locally on its own slice of h (which atb has already
// all_reduce'd back to full hidden by the time we get here). The
// TPInfo is stored on the resulting ActiveDelta so downstream code
// (and future M10 per-proj work) can see it, but for now the code
// path is identical to the non-TP overload.
std::optional<uint64_t> LoRARuntime::install_static_adapter_on_device(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype,
    TPInfo tp) {
  if (tp.tp_size > 1) {
    LOG(WARNING) << "[LoRARuntime] TP shard skeleton engaged (tp_size="
                 << tp.tp_size << " tp_rank=" << tp.tp_rank
                 << ") but whole-block delta ignores it -- delta stays "
                    "kReplicated. Per-proj sharding is P1 (M10).";
  }
  auto id = install_static_adapter_on_device(
      lora_name, lora_path, base_model_name, device, dtype);
  if (id.has_value()) {
    std::lock_guard g(materialise_mu_);
    auto it = device_pool_.find(*id);
    if (it != device_pool_.end()) {
      it->second.tp = tp;
      it->second.shard = LoRAShardStrategy::kReplicated;
    }
  }
  return id;
}

bool LoRARuntime::unload(const std::string& lora_name) {
  bool ok = registry_.unregister(lora_name);
  {
    std::lock_guard g(materialise_mu_);
    if (active_ && active_->name == lora_name) {
      active_.reset();
      LOG(INFO) << "[LoRARuntime] deactivated active '" << lora_name << "'";
    }
    if (pending_ && pending_->name == lora_name) {
      pending_.reset();
      LOG(INFO) << "[LoRARuntime] deactivated pending '" << lora_name << "'";
    }
  }
  return ok;
}

std::optional<LoRARuntime::ActiveDelta> LoRARuntime::active_delta() {
  std::lock_guard g(materialise_mu_);

  // Fast path: nothing to do.
  if (!pending_ && !active_) return std::nullopt;

  // Promote pending -> active. We DELIBERATELY leave the tensors on CPU
  // here even though the model wants them on device: the actual .to(npu)
  // has to happen on the worker's forward thread AND under the atb path
  // which has aclrtSetDevice set. Doing it here (from the LoRARuntime
  // mutex) even on the forward thread crashes with aclrtMemcpy 107017
  // because torch_npu's opapi copy stream is not attached.
  //
  // The caller (qwen3.h forward loop) does the .to(h.device()) inline:
  // that copy runs in the atb-managed NPU stream and works.
  if (pending_) {
    ActiveDelta ad;
    ad.A = pending_->A_cpu;  // still on CPU
    ad.B = pending_->B_cpu;  // still on CPU
    ad.scaling = pending_->scaling;
    ad.name = pending_->name;
    ad.int_id = pending_->int_id;
    active_ = std::move(ad);
    pending_.reset();
    LOG(INFO) << "[LoRARuntime] promoted '" << active_->name
              << "' id=" << active_->int_id
              << " (CPU-side; caller performs .to(device))";
  }
  return active_;
}

std::optional<LoRARuntime::ActiveDelta> LoRARuntime::get_delta_by_int_id(
    uint64_t int_id) {
  if (int_id == 0) return std::nullopt;
  std::lock_guard g(materialise_mu_);
  auto it = device_pool_.find(int_id);
  if (it == device_pool_.end()) return std::nullopt;
  return it->second;
}

// M10 per-proj installer. Loads full adapter, then walks every tensor in
// adapter.tensors (canonicalized keys like "layers.5.self_attn.q_proj#A"),
// parses (layer_index, proj_name), .to(device), and populates the pool.
std::optional<uint64_t> LoRARuntime::install_static_adapter_on_device_per_proj(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype,
    TPInfo tp) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse per-proj install '"
               << lora_name << "'";
    return std::nullopt;
  }
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;
  const uint64_t int_id = *id_opt;

  // Phase A W1 A1.3: idempotent guard (symmetric to MoE variant). If this
  // int_id already has per-proj attention slots, skip the expensive re-load.
  {
    std::lock_guard g(materialise_mu_);
    if (per_proj_device_pool_.count(int_id) > 0) {
      LOG(INFO) << "[LoRARuntime] per-proj install: idempotent skip '"
                << lora_name << "' id=" << int_id << " (already installed)";
      return int_id;
    }
  }

  // Parse canonical keys "layers.{L}.{module_path}.{proj}#A|#B" into
  // (layer_index, proj_name) and pair the A/B tensors up.
  //
  // For Qwen family, module_path values we care about are:
  //   self_attn.q_proj / self_attn.k_proj / self_attn.v_proj / self_attn.o_proj
  //   mlp.gate_proj    / mlp.up_proj      / mlp.down_proj
  // We keep only the final proj token in the ProjKey; the parent
  // (self_attn vs mlp) is implied by the proj name.
  std::unordered_map<ProjKey,
                     std::pair<torch::Tensor, torch::Tensor>,
                     ProjKeyHash>
      pairs;

  auto parse_key = [](const std::string& canon)
      -> std::optional<std::tuple<int32_t, std::string, bool /*is_a*/>> {
    // Trailing "#A" / "#B"
    if (canon.size() < 2) return std::nullopt;
    const std::string tail = canon.substr(canon.size() - 2);
    if (tail != "#A" && tail != "#B") return std::nullopt;
    const bool is_a = (tail == "#A");
    const std::string prefix = canon.substr(0, canon.size() - 2);
    // Expect "layers.{L}.<module>.<proj>" or "model.layers.{L}.<module>.<proj>"
    // (canonicalize_weight_name strips "base_model.model." but the
    // remaining "model." from PEFT paths like
    // "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight"
    // is left as-is). VL / hybrid models add "model.language_model." prefix,
    // e.g. Qwen3.5-122B PEFT adapters use
    // "base_model.model.model.language_model.layers.3.self_attn.q_proj...".
    const std::string kLayersPrefix = "layers.";
    const std::string kModelLayersPrefix = "model.layers.";
    const std::string kModelLangLayersPrefix = "model.language_model.layers.";
    const std::string kLangLayersPrefix = "language_model.layers.";
    std::string after_layers;
    if (prefix.rfind(kModelLangLayersPrefix, 0) == 0) {
      after_layers = prefix.substr(kModelLangLayersPrefix.size());
    } else if (prefix.rfind(kLangLayersPrefix, 0) == 0) {
      after_layers = prefix.substr(kLangLayersPrefix.size());
    } else if (prefix.rfind(kModelLayersPrefix, 0) == 0) {
      after_layers = prefix.substr(kModelLayersPrefix.size());
    } else if (prefix.rfind(kLayersPrefix, 0) == 0) {
      after_layers = prefix.substr(kLayersPrefix.size());
    } else {
      return std::nullopt;
    }
    const auto dot = after_layers.find('.');
    if (dot == std::string::npos) return std::nullopt;
    int32_t layer_index;
    try {
      layer_index = std::stoi(after_layers.substr(0, dot));
    } catch (...) {
      return std::nullopt;
    }
    // last-dot split: proj is the tail after the final '.'
    const auto last_dot = prefix.rfind('.');
    if (last_dot == std::string::npos) return std::nullopt;
    const std::string proj = prefix.substr(last_dot + 1);
    return std::make_tuple(layer_index, proj, is_a);
  };

  // fix #5: experts LoRA keys must NOT go through the per-proj installer.
  // xllm's per_proj_device_pool_ is keyed by (layer, proj_name) and stores
  // dense-MLP wrapper deltas. If we let an "layers.N.mlp.experts.M.gate_proj"
  // key parse to proj_name="gate_proj", 128 experts would all collide on the
  // same key and get silently overwritten, and the FusedMoE forward path
  // has no wrapper to consume them anyway --- classic silent no-op.
  // Refuse loud so callers know to either drop experts from target_modules
  // or route the adapter through install_static_adapter_on_moe_experts.
  {
    int32_t experts_keys = 0;
    std::string first_bad_key;
    for (const auto& [key, unused_tensor] : adapter_opt->tensors) {
      if (key.find(".experts.") != std::string::npos) {
        ++experts_keys;
        if (first_bad_key.empty()) first_bad_key = key;
      }
    }
    if (experts_keys > 0) {
      LOG(ERROR) << "[LoRARuntime] per-proj installer refuses adapter '"
                 << lora_name << "': contains " << experts_keys
                 << " MoE-expert key(s), e.g. '" << first_bad_key << "'. "
                 << "Per-proj installer only handles dense attention/MLP LoRA. "
                 << "Either drop experts from adapter target_modules "
                 << "(attention-only: q/k/v/o_proj) or use "
                 << "install_static_adapter_on_moe_experts (Phase D).";
      return std::nullopt;
    }
  }

  for (const auto& [key, tensor] : adapter_opt->tensors) {
    auto parsed = parse_key(key);
    if (!parsed) continue;
    auto [layer_index, proj_name, is_a] = *parsed;
    ProjKey pk{layer_index, proj_name};
    auto& slot = pairs[pk];
    if (is_a)
      slot.first = tensor;
    else
      slot.second = tensor;
  }

  // Materialize each (A, B) pair on device.
  int32_t installed = 0, skipped = 0;
  std::unordered_map<ProjKey, ProjDelta, ProjKeyHash> per_proj;
  for (auto& [pk, ab] : pairs) {
    if (!ab.first.defined() || !ab.second.defined()) {
      ++skipped;
      continue;
    }
    torch::Tensor A_cpu = ab.first;
    torch::Tensor B_cpu = ab.second;

    // TP-shard skeleton: for now we keep A/B replicated (whole hidden).
    // Per-proj Column/Row sharding is a P1 optimization; the current
    // ATB backend doesn't run per-proj anyway.
    (void)tp;

    // .to(device, dtype)
    torch::Tensor A_dev, B_dev;
    try {
      A_dev = A_cpu.to(device, dtype).contiguous();
      B_dev = B_cpu.to(device, dtype).contiguous();
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LoRARuntime] per-proj .to(device) failed for '"
                 << lora_name << "' key=layers." << pk.layer_index << "."
                 << pk.proj_name << ": " << e.what();
      ++skipped;
      continue;
    }

    ProjDelta pd;
    pd.A = A_dev;
    pd.B = B_dev;
    pd.scaling = adapter_opt->scaling;
    pd.r = adapter_opt->r;
    per_proj.emplace(pk, std::move(pd));
    ++installed;
  }

  {
    std::lock_guard g(materialise_mu_);
    per_proj_device_pool_[int_id] = std::move(per_proj);
  }
  LOG(INFO) << "[LoRARuntime] installed per-proj adapter '" << lora_name
            << "' id=" << int_id << " slots=" << installed
            << " skipped=" << skipped << " r=" << adapter_opt->r
            << " scaling=" << adapter_opt->scaling;
  // P1-D: adapter is now active on device.
  LoRAMetrics::instance().set_state(int_id, 1);
  LoRAMetrics::instance().set_device_slots(int_id, installed);
  return int_id;
}

const LoRARuntime::ProjDelta* LoRARuntime::get_per_proj_delta(
    uint64_t int_id,
    int32_t layer_index,
    const std::string& proj_name) {
  if (int_id == 0) return nullptr;
  std::lock_guard g(materialise_mu_);
  auto it = per_proj_device_pool_.find(int_id);
  if (it == per_proj_device_pool_.end()) return nullptr;
  auto pit = it->second.find(ProjKey{layer_index, proj_name});
  if (pit == it->second.end()) return nullptr;
  return &pit->second;
}

// ---------------------------------------------------------------------
// Day 1 Phase 1: MoE expert LoRA install path.
// ---------------------------------------------------------------------
//
// Parses adapter tensors keyed like
//    layers.{L}.mlp.experts.{E}.{proj}#A or #B
// (proj in {gate_proj, up_proj, down_proj}), stacks them across E into
// 3D tensors laid out to match fused_moe.cpp's group_gemm inputs after
// its lazy [E, out, in] -> [E, in, out] transpose, applies TP shard on
// out_dim for gate/up B and in_dim for down A, moves to device, and
// stores in moe_expert_lora_pool_.
//
// Only this rank's experts (start_expert_id .. start_expert_id +
// num_experts_per_rank) are kept. Other experts' A/B tensors are
// dropped after materialize.

namespace {

// Parse "layers.{L}.mlp.experts.{E}.{proj}#{A|B}" into (L, E, proj, is_a).
// Returns nullopt for non-expert keys.
struct MoeKey {
  int32_t layer_index;
  int32_t expert_index;
  std::string proj;
  bool is_a;
};

std::optional<MoeKey> parse_moe_expert_key(const std::string& canon) {
  if (canon.size() < 2) return std::nullopt;
  const std::string tail = canon.substr(canon.size() - 2);
  if (tail != "#A" && tail != "#B") return std::nullopt;
  const bool is_a = (tail == "#A");
  const std::string prefix = canon.substr(0, canon.size() - 2);

  // Strip optional "model." / "model.language_model." / "language_model."
  // leader (VL / hybrid models add these).
  std::string p = prefix;
  if (p.rfind("model.language_model.layers.", 0) == 0) {
    p = p.substr(std::string("model.language_model.").size());
  } else if (p.rfind("language_model.layers.", 0) == 0) {
    p = p.substr(std::string("language_model.").size());
  } else if (p.rfind("model.layers.", 0) == 0) {
    p = p.substr(std::string("model.").size());
  }
  if (p.rfind("layers.", 0) != 0) return std::nullopt;
  p = p.substr(std::string("layers.").size());  // "{L}.mlp.experts.{E}.{proj}"

  auto dot1 = p.find('.');
  if (dot1 == std::string::npos) return std::nullopt;
  int32_t layer;
  try {
    layer = std::stoi(p.substr(0, dot1));
  } catch (...) {
    return std::nullopt;
  }
  const std::string rest = p.substr(dot1 + 1);  // "mlp.experts.{E}.{proj}"

  // We require the exact "mlp.experts." prefix. Some adapters may use
  // "mlp.experts." vs "mlp.experts_" depending on training toolchain;
  // the mainline PEFT format uses the former.
  const std::string kExpertsPrefix = "mlp.experts.";
  if (rest.rfind(kExpertsPrefix, 0) != 0) return std::nullopt;
  const std::string rest2 = rest.substr(kExpertsPrefix.size());  // "{E}.{proj}"

  auto dot2 = rest2.find('.');
  if (dot2 == std::string::npos) return std::nullopt;
  int32_t expert;
  try {
    expert = std::stoi(rest2.substr(0, dot2));
  } catch (...) {
    return std::nullopt;
  }
  const std::string proj = rest2.substr(dot2 + 1);
  if (proj != "gate_proj" && proj != "up_proj" && proj != "down_proj") {
    return std::nullopt;
  }

  return MoeKey{layer, expert, proj, is_a};
}

}  // namespace

std::optional<uint64_t> LoRARuntime::install_static_adapter_on_moe_experts(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype,
    TPInfo tp,
    int32_t num_experts_total,
    int32_t num_experts_per_rank,
    int32_t start_expert_id,
    int32_t intermediate_size) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse MoE install '" << lora_name
               << "'";
    return std::nullopt;
  }
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  // Register (or reuse) the adapter name so per_proj + moe paths share
  // the same int_id.
  // register_adapter is idempotent when name+path match.
  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) {
    LOG(ERROR) << "[LoRARuntime] MoE install: register failed for '"
               << lora_name << "'";
    return std::nullopt;
  }
  const uint64_t int_id = *id_opt;

  // Phase A W1 A1.3: idempotent guard. If this int_id already has MoE experts
  // installed, skip the expensive re-load. Same name+path caller (via
  // register_adapter idempotency) sees the pool intact; genuine re-load
  // requires unload first.
  {
    std::lock_guard g(materialise_mu_);
    if (moe_expert_lora_pool_.count(int_id) > 0) {
      LOG(INFO) << "[LoRARuntime] MoE install: idempotent skip '" << lora_name
                << "' id=" << int_id << " (already installed)";
      return int_id;
    }
  }

  // Bucket adapter tensors by (layer, proj) -> vec<expert_idx, is_a, tensor>.
  struct Slot {
    int32_t expert_index = -1;
    bool is_a = false;
    torch::Tensor tensor;
  };
  std::unordered_map<
      int32_t /*layer*/,
      std::unordered_map<std::string /*proj*/, std::vector<Slot>>>
      by_layer_proj;

  int32_t total_expert_tensors = 0;
  for (const auto& [key, tensor] : adapter_opt->tensors) {
    auto parsed = parse_moe_expert_key(key);
    if (!parsed) continue;
    ++total_expert_tensors;
    // Keep only this rank's experts.
    if (parsed->expert_index < start_expert_id ||
        parsed->expert_index >= start_expert_id + num_experts_per_rank) {
      continue;
    }
    Slot slot{parsed->expert_index, parsed->is_a, tensor};
    by_layer_proj[parsed->layer_index][parsed->proj].push_back(std::move(slot));
  }

  if (by_layer_proj.empty()) {
    LOG(INFO) << "[LoRARuntime] MoE install '" << lora_name
              << "': no experts.* tensors matched this rank (start_expert_id="
              << start_expert_id << " num_per_rank=" << num_experts_per_rank
              << " total_expert_tensors_seen=" << total_expert_tensors << ")";
    return int_id;
  }

  const int32_t r = adapter_opt->r;
  const float scaling = adapter_opt->scaling;
  const int32_t hidden = -1;  // inferred from A tensor row count below.
  const int32_t tp_world = std::max(1, tp.tp_size);
  const int32_t tp_rank = std::max(0, tp.tp_rank);
  const int32_t local_intermediate = intermediate_size / tp_world;

  auto stack_experts = [&](const std::vector<Slot>& slots,
                           bool is_a) -> torch::Tensor {
    // Build a map expert_index -> tensor, ordered by expert_index within
    // this rank (we already filtered to this rank's expert range).
    std::vector<Slot> filtered;
    for (const auto& s : slots) {
      if (s.is_a != is_a) continue;
      filtered.push_back(s);
    }
    if (filtered.empty()) return torch::Tensor();
    std::sort(
        filtered.begin(), filtered.end(), [](const Slot& a, const Slot& b) {
          return a.expert_index < b.expert_index;
        });
    if (static_cast<int32_t>(filtered.size()) != num_experts_per_rank) {
      LOG(WARNING) << "[LoRARuntime] MoE install '" << lora_name << "': got "
                   << filtered.size()
                   << " expert tensors for a slot but expected "
                   << num_experts_per_rank << " (is_a=" << is_a
                   << "); skipping this slot";
      return torch::Tensor();
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(filtered.size());
    for (auto& s : filtered) tensors.push_back(s.tensor);
    // Stack across expert dim -> [E_per_rank, ...].
    return torch::stack(tensors, /*dim=*/0);
  };

  int32_t layers_installed = 0;
  int32_t layers_skipped = 0;
  std::unordered_map<int32_t, MoeExpertDelta> pool_layers;

  for (auto& [layer, projs] : by_layer_proj) {
    MoeExpertDelta med;
    med.r = r;
    med.scaling = scaling;

    // gate_proj.
    if (auto it = projs.find("gate_proj"); it != projs.end()) {
      auto A_stacked = stack_experts(it->second, /*is_a=*/true);
      auto B_stacked = stack_experts(it->second, /*is_a=*/false);
      if (A_stacked.defined() && B_stacked.defined()) {
        // PEFT stored A as [r, hidden]; base group_gemm wants
        // A_gate: [E_per_rank, hidden, r]. Transpose(1,2).
        // PEFT stored B as [out_full, r]; we need [E_per_rank, r, out_local].
        // Shard on dim=1 (out) then transpose to put r first.
        A_stacked = A_stacked.transpose(1, 2).contiguous();
        // B_stacked: [E, out_full, r] -> slice on out -> [E, out_local, r]
        //   -> transpose to [E, r, out_local]
        auto B_shard = B_stacked
                           .slice(1,
                                  tp_rank * local_intermediate,
                                  (tp_rank + 1) * local_intermediate)
                           .transpose(1, 2)
                           .contiguous();
        try {
          med.A_gate = A_stacked.to(device, dtype).contiguous();
          med.B_gate = B_shard.to(device, dtype).contiguous();
        } catch (const std::exception& e) {
          LOG(ERROR) << "[LoRARuntime] MoE .to(device) gate_proj L=" << layer
                     << ": " << e.what();
          med.A_gate = torch::Tensor();
          med.B_gate = torch::Tensor();
        }
      }
    }

    // up_proj.
    if (auto it = projs.find("up_proj"); it != projs.end()) {
      auto A_stacked = stack_experts(it->second, /*is_a=*/true);
      auto B_stacked = stack_experts(it->second, /*is_a=*/false);
      if (A_stacked.defined() && B_stacked.defined()) {
        A_stacked = A_stacked.transpose(1, 2).contiguous();
        auto B_shard = B_stacked
                           .slice(1,
                                  tp_rank * local_intermediate,
                                  (tp_rank + 1) * local_intermediate)
                           .transpose(1, 2)
                           .contiguous();
        try {
          med.A_up = A_stacked.to(device, dtype).contiguous();
          med.B_up = B_shard.to(device, dtype).contiguous();
        } catch (const std::exception& e) {
          LOG(ERROR) << "[LoRARuntime] MoE .to(device) up_proj L=" << layer
                     << ": " << e.what();
          med.A_up = torch::Tensor();
          med.B_up = torch::Tensor();
        }
      }
    }

    // down_proj.
    if (auto it = projs.find("down_proj"); it != projs.end()) {
      auto A_stacked = stack_experts(it->second, /*is_a=*/true);
      auto B_stacked = stack_experts(it->second, /*is_a=*/false);
      if (A_stacked.defined() && B_stacked.defined()) {
        // A: [r, in_full] -> shard on in -> [r, in_local] -> [E, in_local, r]
        auto A_shard = A_stacked
                           .slice(2,
                                  tp_rank * local_intermediate,
                                  (tp_rank + 1) * local_intermediate)
                           .transpose(1, 2)
                           .contiguous();
        // B: [hidden, r] -> [E, r, hidden]
        B_stacked = B_stacked.transpose(1, 2).contiguous();
        try {
          med.A_down = A_shard.to(device, dtype).contiguous();
          med.B_down = B_stacked.to(device, dtype).contiguous();
        } catch (const std::exception& e) {
          LOG(ERROR) << "[LoRARuntime] MoE .to(device) down_proj L=" << layer
                     << ": " << e.what();
          med.A_down = torch::Tensor();
          med.B_down = torch::Tensor();
        }
      }
    }

    // Only keep the layer if at least one proj is defined.
    if (med.A_gate.defined() || med.A_up.defined() || med.A_down.defined()) {
      pool_layers.emplace(layer, std::move(med));
      ++layers_installed;
    } else {
      ++layers_skipped;
    }
  }

  {
    std::lock_guard g(materialise_mu_);
    moe_expert_lora_pool_[int_id] = std::move(pool_layers);
  }
  LOG(INFO) << "[LoRARuntime] installed MoE expert adapter '" << lora_name
            << "' id=" << int_id << " layers=" << layers_installed
            << " skipped=" << layers_skipped << " r=" << r
            << " scaling=" << scaling << " E_per_rank=" << num_experts_per_rank
            << " start_expert=" << start_expert_id
            << " intermediate_local=" << local_intermediate
            << " tp=" << tp_world;
  return int_id;
}

const LoRARuntime::MoeExpertDelta* LoRARuntime::get_moe_expert_delta(
    uint64_t int_id,
    int32_t layer_index) {
  if (int_id == 0) return nullptr;
  std::lock_guard g(materialise_mu_);
  auto it = moe_expert_lora_pool_.find(int_id);
  if (it == moe_expert_lora_pool_.end()) return nullptr;
  auto lit = it->second.find(layer_index);
  if (lit == it->second.end()) return nullptr;
  return &lit->second;
}

}  // namespace xllm
