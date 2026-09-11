/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <memory>
#include <vector>

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <xlite/xlite.h>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/xlite/xlite_freqs_cis.h"
#include "core/layers/xlite/xlite_init_utils.h"
#include "core/layers/xlite/xlite_model_adapter.h"
#include "runtime/options.h"

namespace xllm::xlite {

// Aggregate view over multiple StateDicts (MoE checkpoint spans many shards).
class MergedStateDict : public StateDict {
 public:
  explicit MergedStateDict(std::vector<const StateDict*> dicts)
      : StateDict({}, ""), dicts_(std::move(dicts)) {}

  torch::Tensor get_tensor(const std::string& tensor_name) const override {
    for (const StateDict* d : dicts_) {
      if (!d) {
        continue;
      }
      torch::Tensor t = d->get_tensor(tensor_name);
      if (t.defined()) {
        return t;
      }
    }
    return torch::Tensor{nullptr};
  }

 private:
  std::vector<const StateDict*> dicts_;
};

class XliteModelHolder {
 public:
  virtual ~XliteModelHolder() = default;
  virtual XRuntime& xlite_rt() = 0;
  virtual XModel& xlite_model() = 0;
  virtual const XModelConfig& xlite_config() const = 0;
  virtual const torch::Tensor& xlite_freqs_cis() const = 0;
  virtual torch::Tensor xlite_output_buf(int64_t num_tokens) = 0;
  virtual bool xlite_ready() const = 0;
};

class XliteCausalLMBase : public CausalLM, public XliteModelHolder {
 public:
  XliteCausalLMBase(const ModelContext& context,
                    std::unique_ptr<XliteModelAdapter> adapter)
      : options_(context.get_tensor_options()),
        device_(options_.device()),
        args_(context.get_model_args()),
        parallel_(context.get_parallel_args()),
        adapter_(std::move(adapter)),
        ready_(false) {
    CheckSupportedConfig();
    cfg_ = adapter_->BuildConfig(context);
    // ParallelArgs::tp_size() is not populated; derive from world_size/dp_size.
    const uint32_t tp = XliteTpSize(parallel_);
    const uint32_t tp_rank = XliteTpRank(parallel_);
    cfg_.defTpSize = tp;
    // EP=1 -> experts sharded by attention TP; EP>1 -> world_size/ep_size.
    const uint32_t ep = cfg_.moeEpSize;
    cfg_.moeTPSize = (ep > 1) ? (parallel_.world_size() / ep) : tp;
    uint32_t rank = static_cast<uint32_t>(parallel_.rank());
    rt_ =
        std::make_unique<XRuntime>(static_cast<uint32_t>(device_.index()),
                                   /*sizeMB=*/0,
                                   rank,
                                   tp,
                                   static_cast<uint32_t>(parallel_.dp_size()),
                                   cfg_.moeTPSize,
                                   static_cast<uint32_t>(parallel_.ep_size()));
    model_ = std::make_unique<XModel>(cfg_, rank);
    // MLA freqs_cis needs YaRN params from ModelArgs (XModelConfig has no such
    // fields).
    YarnConfig yarn;
    yarn.rope_factor = args_.rope_scaling_factor();
    yarn.beta_fast = static_cast<float>(args_.rope_scaling_beta_fast());
    yarn.beta_slow = static_cast<float>(args_.rope_scaling_beta_slow());
    yarn.original_seq_len =
        args_.rope_scaling_original_max_position_embeddings();
    freqs_cis_ = XliteFreqsCis::Precompute(cfg_, options_, yarn);
    output_buf_ = torch::empty(
        {static_cast<int64_t>(cfg_.maxBatchedTokens), args_.hidden_size()},
        options_);
    LOG(INFO) << "[xlite] rank=" << rank << " tp=" << tp
              << " tp_rank=" << tp_rank << " moeTPSize=" << cfg_.moeTPSize
              << " ep=" << parallel_.ep_size();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    // Reload path (WorkerImpl::update_weights re-enters load_model on the
    // same instance). XModel::Init is not idempotent (leaks its device
    // buffers on re-entry), so rebuild the model for fresh state; the runtime
    // (streams / HCCL comms / tensor pool) is weight-independent and reused
    // (InitTensorPool no-ops once inited). Old weights are released by
    // dropping their torch refs.
    if (ready_) {
      LOG(INFO) << "xlite load_model: reloading weights";
      weight_storages_.clear();
      model_ = std::make_unique<XModel>(
          cfg_, static_cast<uint32_t>(parallel_.rank()));
    }
    auto& state_dicts = loader->get_state_dicts();
    // Load once over merged shards, then Init once (not idempotent).
    std::vector<const StateDict*> dicts;
    dicts.reserve(state_dicts.size());
    for (auto& sd_ptr : state_dicts) {
      if (sd_ptr && sd_ptr->size() > 0) {
        dicts.push_back(sd_ptr.get());
      }
    }
    CHECK(!dicts.empty()) << "xlite load_model: no non-empty state dicts";
    MergedStateDict merged(std::move(dicts));
    adapter_->Load(
        merged, *model_, cfg_, args_, parallel_, device_, weight_storages_);
    model_->Init();
    // GetTensorPoolSize requires maxBatchedTokens>0 (set in BuildConfig).
    CHECK_GT(cfg_.maxBatchedTokens, 0) << "cfg_.maxBatchedTokens is 0";
    size_t pool = model_->GetTensorPoolSize(0);
    CHECK_EQ(rt_->InitTensorPool(pool), 0) << "xlite InitTensorPool failed";
    ready_ = true;
  }

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    LOG(FATAL) << "xlite forward() must not be called; use XliteExecutorImpl";
    return ModelOutput();
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    int64_t n =
        seleted_idxes.defined() ? seleted_idxes.size(0) : hidden_states.size(0);
    // head is vocab-sharded by tp; all_gather -> [tp, n, vocab/tp].
    uint32_t tp = XliteTpSize(parallel_);
    torch::Tensor out =
        torch::empty({static_cast<int64_t>(tp),
                      n,
                      args_.vocab_size() / static_cast<int64_t>(tp)},
                     options_);
    ::XTensor x_in, x_idx, x_out;
    InitXTensor(x_in, hidden_states);
    InitXTensor(x_idx,
                seleted_idxes.defined()
                    ? seleted_idxes
                    : torch::arange(n, options_.dtype(torch::kInt32)));
    InitXTensor(x_out, out);
#if defined(USE_NPU)
    aclrtStream ext = c10_npu::getCurrentNPUStream(device_.index()).stream();
    rt_->EventWaitCurrStream(ext);
    model_->ForwardGetLogits(*rt_, x_in, x_idx, x_out);
    rt_->EventRecordCurrStream(ext);
#else
    LOG(FATAL) << "xlite backend requires USE_NPU";
#endif
    // head is vocab-sharded by tp; all_gather -> [tp, n, vocab/tp].
    return out.permute({1, 0, 2}).reshape({n, args_.vocab_size()});
  }
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    out_hidden = seleted_idxes.defined()
                     ? hidden_states.index_select(0, seleted_idxes)
                     : hidden_states;
    return logits(hidden_states, seleted_idxes);
  }

  torch::Device device() const override { return device_; }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {
    LOG(FATAL) << "EPLB is not supported by xlite backend.";
  }
  void update_expert_weight(int32_t) override {
    LOG(FATAL) << "EPLB is not supported by xlite backend.";
  }
  void lazy_load_model(std::unique_ptr<ModelLoader>) override {
    LOG(FATAL) << "Rolling load / sleep mode (lazy_load_model) is not "
                  "supported by xlite backend.";
  }
  void free_model_weights() override {
    LOG(FATAL) << "Sleep mode (free_model_weights) is not supported by "
                  "xlite backend.";
  }

  xlite::XliteModelHolder* get_xlite_holder() override { return this; }

  XRuntime& xlite_rt() override { return *rt_; }
  XModel& xlite_model() override { return *model_; }
  const XModelConfig& xlite_config() const override { return cfg_; }
  const torch::Tensor& xlite_freqs_cis() const override { return freqs_cis_; }
  torch::Tensor xlite_output_buf(int64_t n) override {
    CHECK(output_buf_.defined()) << "xlite output_buf not allocated";
    CHECK_LE(n, static_cast<int64_t>(cfg_.maxBatchedTokens))
        << "xlite output_buf overflow: n=" << n
        << " > maxBatchedTokens=" << cfg_.maxBatchedTokens;
    return output_buf_.slice(0, 0, n);
  }
  bool xlite_ready() const override { return ready_; }

 private:
  // Fail fast on configurations the xlite backend does not implement. Each
  // check names the user-facing flag to disable.
  static void CheckSupportedConfig() {
    CHECK(!::xllm::EPLBConfig::get_instance().enable_eplb())
        << "EPLB is not supported by xlite backend; disable --enable_eplb"
           " or use a non-xlite backend.";
    CHECK(!::xllm::LoadConfig::get_instance().enable_rolling_load())
        << "Rolling load is not supported by xlite backend; disable "
           "--enable_rolling_load or use a non-xlite backend.";
    CHECK_LE(::xllm::SpeculativeConfig::get_instance().num_speculative_tokens(),
             0)
        << "Speculative decoding is not supported by xlite backend; set "
           "--num_speculative_tokens=0 or use a non-xlite backend.";
    CHECK_LE(::xllm::ParallelConfig::get_instance().layerwise_split_size(), 1)
        << "Layerwise split is not supported by xlite backend; set "
           "--layerwise_split_size=1 or use a non-xlite backend.";
    CHECK_NE(::xllm::KVCacheConfig::get_instance().indexer_cache_dtype(),
             "int8")
        << "INT8 indexer cache is not supported by xlite backend; set "
           "--indexer_cache_dtype=auto or use a non-xlite backend.";
  }

 protected:
  torch::TensorOptions options_;
  torch::Device device_;
  ModelArgs args_;
  ParallelArgs parallel_;
  XModelConfig cfg_{};
  std::unique_ptr<XRuntime> rt_;
  std::unique_ptr<XModel> model_;
  std::unique_ptr<XliteModelAdapter> adapter_;
  std::vector<torch::Tensor> weight_storages_;
  torch::Tensor freqs_cis_;
  torch::Tensor output_buf_;
  bool ready_;
};

}  // namespace xllm::xlite