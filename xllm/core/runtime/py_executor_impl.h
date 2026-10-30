/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <pybind11/pybind11.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "executor_impl_factory.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_params_view.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class PyCausalLM;
class SlotBuffer;
class ProcessGroup;
struct TokenResultTensors;

class __attribute__((visibility("hidden"))) PyExecutorImpl final
    : public ExecutorImpl {
 public:
  PyExecutorImpl(CausalLM* model,
                 const ModelArgs& args,
                 const torch::Device& device,
                 const runtime::Options& options);

  ~PyExecutorImpl() override;

  bool supports_adaptive_unified_graph() const override;
  void warmup_adaptive_unified_graph(ExecutorImpl& draft_executor,
                                     const AdaptiveDSparkGraphTask& task,
                                     ProcessGroup* sampling_group) override;
  torch::Tensor run_adaptive_unified_graph(
      ExecutorImpl& draft_executor,
      const AdaptiveDSparkGraphTask& task,
      ProcessGroup* sampling_group) override;

  bool supports_prepared_attention_metadata() const override {
    return supports_prepared_metadata_;
  }
  void prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                  ModelParamsView& params) override;

  void warmup_prepared_graph(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelParamsView& params) override;

  std::vector<torch::Tensor> get_prepared_aux_hidden_states(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      const ModelParamsView& params);

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override;

  // GLM sparse MTP consumes one decode row per token and no Host KV/CSR data.
  pybind11::object mtp_sparse_attention_metadata_view(
      const torch::Tensor& block_table,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& slots) const;

  pybind11::object create_mtp_graph_variant_registry(
      PyExecutorImpl& draft_executor,
      int32_t max_variants);

  torch::Tensor run_unified_block(PyExecutorImpl& draft_executor,
                                  SlotBuffer& target,
                                  SlotBuffer& draft,
                                  const TokenResultTensors& result,
                                  bool dspark,
                                  ProcessGroup* sampling_group);

 private:
  // Startup only; serving must find an already captured exact Slot binding.
  void create_adaptive_unified_registry(PyExecutorImpl& draft_executor,
                                        ProcessGroup* sampling_group);
  // Caller holds the GIL. First binding occurs on Prepare for pipeline input.
  void bind_kv_caches(std::vector<KVCache>& kv_caches);

  PyCausalLM* py_causal_lm_;
  ModelArgs args_;
  const int32_t dummy_token_count_;
  torch::Device device_;
  runtime::Options options_;
  bool enable_mla_ = false;

  pybind11::object py_executor_;
  pybind11::object block_graph_registry_;
  pybind11::object adaptive_unified_registry_;
  PyExecutorImpl* adaptive_unified_draft_ = nullptr;
  ProcessGroup* adaptive_unified_sampling_group_ = nullptr;
  bool requires_input_batch_metadata_ = false;
  bool supports_prepared_metadata_ = false;
  bool supports_layerwise_kv_load_ = false;
  bool kv_bound_ = false;
  int64_t kv_layer_count_ = 0;
  std::vector<torch::Tensor> prepared_kv_bindings_;
};

REGISTER_EXECUTOR("python", PyExecutorImpl);

}  // namespace xllm
