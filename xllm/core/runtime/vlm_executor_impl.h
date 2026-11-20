/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/types.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "executor_impl_factory.h"
#include "framework/encoder_cache/encoder_cache.h"
#include "framework/model/causal_vlm.h"
#include "framework/model/model_output.h"
#include "framework/model/model_params_view.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class VlmExecutorImpl final : public ExecutorImpl {
 public:
  VlmExecutorImpl(CausalVLM* model,
                  const ModelArgs& args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~VlmExecutorImpl() override = default;
  void clear_multimodal_cache() override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override;

  void prepare_graph_input(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelParamsView& params) override;

  void set_dp_process_group(ProcessGroup* group) override;

 private:
  CausalVLM* model_;
  DecoderModel& decoder_;
  bool collapse_mrope_;
  torch::Device device_;
  std::unique_ptr<ExecutorImpl> llm_executor_;
  std::unique_ptr<EncoderCache> encoder_cache_;

  torch::Tensor collapse_mrope_decode_positions(
      const torch::Tensor& positions) const;
};
REGISTER_EXECUTOR("vlm", VlmExecutorImpl);
}  // namespace xllm
