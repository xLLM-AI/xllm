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

#include "core/framework/model_context/model_context.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_output.h"
#include "framework/model/model_params_view.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class Executor final {
 public:
  Executor(CausalLM* model,
           const ModelContext& context,
           const runtime::Options& options);

  virtual ~Executor() = default;

  void set_dp_process_group(ProcessGroup* group);

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& params);

  void prepare_graph_input(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelParamsView& params);

  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const Stream& signal_stream);

 private:
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace xllm
