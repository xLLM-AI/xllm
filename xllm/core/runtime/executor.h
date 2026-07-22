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

#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class Executor final {
 public:
  Executor(CausalLM* model,
           const ModelArgs& args,
           const torch::Device& device,
           const runtime::Options& options);

  virtual ~Executor() = default;

  ForwardInput prepare_inputs(Batch& batch);

  // tokens: vector size is dp_size, each element is [num_tokens/dp_size]
  // positions: vector size is dp_size, each element is [num_tokens/dp_size]
  // token pos in the sequence returns: ModelOutput
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params);

  void prepare_graph_input(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelInputParams& params);

  bool prepare_static_graph_tasks(const torch::Tensor& tokens,
                                  const ModelInputParams& params,
                                  const Stream& signal_stream);

  bool prepare_static_mtp_graph_tasks(int64_t linear_state_id,
                                      int64_t num_accepted_tokens,
                                      int64_t spec_width,
                                      int64_t block_table_width,
                                      const Stream& signal_stream);

 private:
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace xllm
