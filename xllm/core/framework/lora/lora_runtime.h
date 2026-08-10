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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "adapter_loader.h"
#include "lora_config.h"
#include "lora_registry.h"

namespace xllm {

// LoRARuntime is the process-global singleton that ties the LoRA modules
// together for the model forward path. It is deliberately narrow-featured
// so P0-A can end-to-end demonstrate load-via-API without waiting on M4
// (LoraCache) or M8 (Scheduler).
//
// Concretely:
//   - A single model-level dummy A/B pair (Path C semantics) can be
//     dynamically installed by loading an adapter.
//   - The A/B tensors are moved to the model's inference device once at
//     load time; there is no eviction / hot-swap yet.
//   - forward() paths call `active_delta()` and, if it returns a valid
//     value, use those tensors as the whole-block LoRA delta.
//
// Multi-tenant, per-request routing is P0-B / P0-C work. This singleton
// only holds "the currently applied adapter" -- the last one loaded wins.
// That's enough to prove the full end-to-end HTTP -> loader -> registry ->
// forward path from a single curl call.
// Path C prod v3 M10 TP shard skeleton. When TP > 1, per-proj LoRA deltas
// need to shard A/B differently depending on which xllm Linear the delta
// hooks into:
//   - ColumnParallelLinear (Q/K/V/gate/up):
//       A [rank, hidden]         replicated on every rank
//       B [hidden/tp, rank]      column-sharded (each rank owns a slice)
//   - RowParallelLinear (o_proj, down_proj):
//       A [rank, hidden/tp]      row-sharded
//       B [hidden, rank]         replicated, delta gets all_reduce'd
//
// Whole-block delta on the current NPU path (atb output is already all-
// reduced back to full hidden) uses REPLICATED for both A and B; each
// rank keeps the full tensor and computes the same delta locally. No
// cross-rank comm needed.
//
// This enum is the skeleton; the actual per-proj sharding is P1 work.
enum class LoRAShardStrategy {
  kReplicated = 0,     // whole tensor on every rank (current whole-block)
  kColumnSharded = 1,  // dim-shard on last dim (for Column-parallel proj)
  kRowSharded = 2,     // dim-shard on second-to-last (for Row-parallel proj)
};

struct TPInfo {
  int32_t tp_size = 1;
  int32_t tp_rank = 0;
};

class LoRARuntime {
 public:
  static LoRARuntime& instance();

  // One-time init; must be called before any adapter API is used. Copies
  // config by value so later registry / cache modules can inspect it
  // without another lookup.
  void init(const LoRAConfig& config);

  bool enabled() const;

  // Called once by the model (QWen3ModelImpl ctor) so subsequent HTTP
  // calls to /v1/load_lora_adapter know what device / dtype to
  // materialise weights on. Idempotent; last caller wins (single model
  // per engine today).
  void set_model_device_dtype(torch::Device device, torch::ScalarType dtype);

  // W4-A3: hot-swap install config. Passed from model ctor so that
  // load_and_activate_hotswap knows whether to run per-proj + moe-experts
  // installers (and with what TP + MoE args) or only per-proj (dense).
  // Idempotent; last caller wins (single model per engine today).
  struct HotswapConfig {
    TPInfo tp;                         // TP {world_size, rank}
    bool install_per_proj = true;      // always install per-proj attn/MLP
    bool install_moe_experts = false;  // set true for qwen3_moe
    // MoE args (ignored when install_moe_experts=false)
    int32_t num_experts_total = 0;
    int32_t num_experts_per_rank = 0;
    int32_t start_expert_id = 0;
    int32_t moe_intermediate_size = 0;
  };
  void set_hotswap_config(const HotswapConfig& cfg);

  // Load an adapter from a filesystem path, parse its PEFT files, pick a
  // whole-block A/B pair, cast to the model's dtype, and register it.
  //
  // The tensors stay on CPU here -- the actual device migration happens
  // lazily on the first active_delta() call from the model forward
  // thread, which owns the correct NPU context.
  //
  // The most recently loaded adapter becomes the pending one and will
  // be promoted to active on the next forward.
  std::optional<uint64_t> load_and_activate(const std::string& lora_name,
                                            const std::string& lora_path,
                                            const std::string& base_model_name);

  // Path C prod v3 static preload path.
  //
  // Load a PEFT adapter from disk, pick a whole-block A/B pair, cast to
  // the given dtype, .to(device) IN THIS THREAD, register it with the
  // registry, and install it as the currently active adapter with
  // *device-side* tensors. Callable ONLY from a thread with a valid NPU
  // context (in practice: QWen3ModelImpl ctor). If called after
  // model init has completed the .to(device) will crash with
  // aclrtMemcpy 107017 -- see docs/lora_investigation.
  //
  // Distinct from load_and_activate() which keeps tensors on CPU and
  // defers migration; that path is broken on CANN 8.5 + torch_npu 2.7.1
  // and only survives via ctor-time dummy fill.
  std::optional<uint64_t> install_static_adapter_on_device(
      const std::string& lora_name,
      const std::string& lora_path,
      const std::string& base_model_name,
      torch::Device device,
      torch::ScalarType dtype);

  // TP-aware overload. Same as above but the caller passes tp_size/tp_rank
  // so we can (in future M10 work) shard A/B before .to(device). For the
  // current whole-block delta path we ignore the info and store the full
  // tensor on every rank (kReplicated). This overload exists so callers
  // can start passing the info now and the internal sharding lights up
  // when M10 lands, without another API break.
  std::optional<uint64_t> install_static_adapter_on_device(
      const std::string& lora_name,
      const std::string& lora_path,
      const std::string& base_model_name,
      torch::Device device,
      torch::ScalarType dtype,
      TPInfo tp);

  // Path C prod v3 multi-adapter: look up the device-resident A/B for a
  // specific adapter by its int_id. Returns std::nullopt if int_id unknown
  // or the adapter was installed without a device pool entry.
  //
  // Populated by install_static_adapter_on_device when it succeeds.
  bool unload(const std::string& lora_name);

  // Path C prod v3 hot-swap: enqueue a load task to the pinned executor
  // thread which owns the NPU device context. Blocks until the executor
  // completes the install. Returns the assigned int_id, or nullopt on
  // failure. Callable from any thread (HTTP handler etc.).
  std::optional<uint64_t> load_and_activate_hotswap(
      const std::string& lora_name,
      const std::string& lora_path,
      const std::string& base_model_name);

  LoRARegistry& registry() { return registry_; }
  const LoRARegistry& registry() const { return registry_; }
  const LoRAConfig& config() const { return config_; }

  // The active whole-block delta tensors. std::nullopt = no adapter, the
  // forward path should just skip its delta step.
  //
  // Note: tensors are populated on CPU by the load path (which typically
  // runs on the API thread, without an NPU context) and lazily migrated
  // to device on the first active_delta() call from the model forward
  // thread (which owns the correct NPU context). This avoids the
  // aclrtMemcpy-invalid-handle failure you hit when a background thread
  // tries to allocate device memory it does not own.
  struct ActiveDelta {
    torch::Tensor A;  // [rank, hidden] (or shard thereof) on model_device
    torch::Tensor B;  // [hidden, rank] (or shard thereof) on model_device
    float scaling;
    std::string name;
    uint64_t int_id;
    // TP shard tag. kReplicated is the current whole-block default; the
    // sharded variants light up when M10 per-proj delta ships.
    LoRAShardStrategy shard = LoRAShardStrategy::kReplicated;
    TPInfo tp{};
  };
  std::optional<ActiveDelta> active_delta();

  // Path C prod v3 multi-adapter: look up the device-resident A/B for a
  // specific adapter by its int_id. Populated by
  // install_static_adapter_on_device on ctor thread. Returns nullopt if the
  // int_id is not known (base-model request or adapter installed without
  // device weights).
  std::optional<ActiveDelta> get_delta_by_int_id(uint64_t int_id);

  // ---- M10 per-proj/per-layer delta support ----
  //
  // ProjKey identifies a specific weight in a decoder layer, e.g.
  // (layer=5, proj="q_proj"). PEFT convention: proj is one of
  // {q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}.
  //
  // ProjDelta holds the device-resident A/B for that specific slot.
  // scaling is per-adapter (alpha/rank, or alpha/sqrt(rank) for rslora).
  struct ProjKey {
    int32_t layer_index;
    std::string proj_name;
    bool operator==(const ProjKey& other) const {
      return layer_index == other.layer_index && proj_name == other.proj_name;
    }
  };
  struct ProjKeyHash {
    size_t operator()(const ProjKey& k) const noexcept {
      return std::hash<int32_t>()(k.layer_index) ^
             (std::hash<std::string>()(k.proj_name) << 1);
    }
  };
  struct ProjDelta {
    torch::Tensor A;  // [rank, in_features]  or shard thereof
    torch::Tensor B;  // [out_features, rank]  or shard thereof
    float scaling = 0.0f;
    int32_t r = 0;
  };

  // Materialize every (layer, proj) tensor of the adapter onto device and
  // register in the per-proj pool. Call from the model ctor thread (V60
  // rules).
  std::optional<uint64_t> install_static_adapter_on_device_per_proj(
      const std::string& lora_name,
      const std::string& lora_path,
      const std::string& base_model_name,
      torch::Device device,
      torch::ScalarType dtype,
      TPInfo tp);

  // Look up a single per-proj delta. Called from the Linear wrapper
  // forward path once per (batch, layer, proj). Returns nullptr if the
  // adapter doesn't have anything at that slot (e.g. adapter only
  // targets q_proj/k_proj/v_proj but not gate_proj).
  const ProjDelta* get_per_proj_delta(uint64_t int_id,
                                      int32_t layer_index,
                                      const std::string& proj_name);

  // ---- MoE expert LoRA (Day 1 Phase 1) ----
  //
  // For MoE models like Qwen3-30B-A3B, adapter safetensors carry per-expert
  // LoRA weights, e.g.
  // base_model.model.model.layers.{L}.mlp.experts.{E}.gate_proj.lora_A.weight
  // These cannot use ProjDelta (single 2D pair per slot) -- we need to stack
  // over the E dim so FusedMoEImpl::forward_expert can consume them via one
  // grouped-gemm call per proj.
  //
  // MoeExpertDelta holds device-resident 3D tensors:
  //   A_gate/A_up: [E_per_rank, hidden, r]
  //   B_gate/B_up: [E_per_rank, r, out_local]     (out_local = intermediate/tp)
  //   A_down:      [E_per_rank, in_local, r]      (in_local  = intermediate/tp)
  //   B_down:      [E_per_rank, r, hidden]
  // Empty (undefined) tensor means "adapter did not target that proj".
  struct MoeExpertDelta {
    torch::Tensor A_gate;
    torch::Tensor B_gate;
    torch::Tensor A_up;
    torch::Tensor B_up;
    torch::Tensor A_down;
    torch::Tensor B_down;
    float scaling = 0.0f;
    int32_t r = 0;
  };

  // Materialize an adapter's experts.{E}.{proj} tensors onto device and
  // register in the MoE expert pool. Must be called from the model ctor
  // thread (V60 rules) -- xllm can only do CPU->NPU .to() there.
  //
  // Preconditions: (1) the same adapter has already been installed via
  // install_static_adapter_on_device_per_proj (which parses adapter_config,
  // registers name->int_id, and canonicalises tensor keys); or a fresh
  // adapter with no attention LoRA tensors is being registered by this call.
  //
  // Only the experts.{E}.{proj} keys are consumed here. Attention keys
  // (self_attn.q_proj etc.) are ignored -- the sister call handles them.
  //
  // Fails silently (returns nullopt) if the adapter has zero experts.*
  // tensors. This lets callers install attention-only adapters unchanged.
  //
  // num_experts_total: the base model's total expert count (128 for
  //   Qwen3-30B-A3B). num_experts_per_rank: how many experts this rank owns
  //   (32 when tp=4, ep=1). start_expert_id: this rank's first expert
  //   (rank * num_experts_per_rank when ep=1). intermediate_size: base
  //   model's per-expert intermediate dim (768 for Qwen3-30B-A3B, before
  //   TP shard). tp: as usual.
  std::optional<uint64_t> install_static_adapter_on_moe_experts(
      const std::string& lora_name,
      const std::string& lora_path,
      const std::string& base_model_name,
      torch::Device device,
      torch::ScalarType dtype,
      TPInfo tp,
      int32_t num_experts_total,
      int32_t num_experts_per_rank,
      int32_t start_expert_id,
      int32_t intermediate_size);

  // Look up the MoE expert delta for a decoder layer. Called from
  // FusedMoEImpl::forward_expert once per forward per layer. Returns
  // nullptr if the adapter has no MoE tensors for this layer (or the
  // adapter id is unknown).
  const MoeExpertDelta* get_moe_expert_delta(uint64_t int_id,
                                             int32_t layer_index);

 private:
  LoRARuntime() = default;

  // Called with materialise_mu_ held. Picks a plausible A/B pair from the
  // adapter's canonicalised tensor set. Result is CPU-side, dtype-cast to
  // the model dtype but kept off-device.
  bool pick_whole_block_ab(const LoRAAdapter& adapter,
                           torch::ScalarType dtype,
                           torch::Tensor* A_out,
                           torch::Tensor* B_out) const;

  // Pending / not-yet-migrated CPU tensors, seeded by load_and_activate.
  // active_delta() moves these to device on the forward thread.
  struct PendingDelta {
    torch::Tensor A_cpu;
    torch::Tensor B_cpu;
    float scaling;
    std::string name;
    uint64_t int_id;
  };
  std::optional<PendingDelta> pending_;

  mutable std::mutex materialise_mu_;
  LoRAConfig config_;
  LoRARegistry registry_;
  std::unique_ptr<LoRAAdapterLoader> loader_;

  // Recorded by the model at forward-init time so the HTTP handler knows
  // where to place freshly-loaded LoRA tensors. std::nullopt means no
  // model has registered yet.
  std::optional<torch::Device> model_device_;
  std::optional<torch::ScalarType> model_dtype_;

  // Currently-active adapter's device tensors. Guarded by materialise_mu_.
  std::optional<ActiveDelta> active_;

  // Path C prod v3 hot-swap: pinned executor thread + task queue.
  struct LoadTask {
    std::string name;
    std::string path;
    std::string base_model_name;
    std::promise<std::optional<uint64_t>> result;
  };
  std::mutex task_mu_;
  std::condition_variable task_cv_;
  std::queue<LoadTask> task_queue_;
  std::atomic<bool> executor_stop_{false};
  std::thread executor_thread_;
  bool executor_started_ = false;

  // W4-A3: hot-swap install config. Guarded by materialise_mu_ (writes
  // are rare from model ctor; reads from executor_loop task pickup).
  HotswapConfig hotswap_config_{};
  bool hotswap_config_set_ = false;

  void executor_loop(int32_t device_index, torch::ScalarType dtype);

  // Path C prod v3 multi-adapter: int_id -> device-resident A/B/scaling.
  // Written by install_static_adapter_on_device from ctor thread.
  // Read by get_delta_by_int_id from forward thread. Guarded by
  // materialise_mu_.
  std::unordered_map<uint64_t, ActiveDelta> device_pool_;

  // M10 per-proj device pool: int_id -> ((layer, proj) -> ProjDelta).
  // Populated by install_static_adapter_on_device_per_proj at ctor time
  // (V60 window). Read by get_per_proj_delta from Linear wrappers.
  // Guarded by materialise_mu_.
  std::unordered_map<uint64_t,
                     std::unordered_map<ProjKey, ProjDelta, ProjKeyHash>>
      per_proj_device_pool_;

  // Day 1 Phase 1: MoE expert LoRA pool. int_id -> (layer_index ->
  // MoeExpertDelta). Populated by install_static_adapter_on_moe_experts
  // at ctor time. Read by get_moe_expert_delta from FusedMoEImpl::forward.
  // Guarded by materialise_mu_.
  std::unordered_map<uint64_t, std::unordered_map<int32_t, MoeExpertDelta>>
      moe_expert_lora_pool_;
};

}  // namespace xllm
