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

#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "options.h"

namespace xllm::npu {

class OneRecAclGraphExecutor {
 public:
  OneRecAclGraphExecutor(CausalLM* model,
                         const ModelArgs& args,
                         const torch::Device& device,
                         const runtime::Options& options);
  ~OneRecAclGraphExecutor();

  bool is_graph_candidate(const ModelInputParams& params) const;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params);

  void reset_first_prefill_graph_state_if_needed(
      const ModelInputParams& params);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::npu
