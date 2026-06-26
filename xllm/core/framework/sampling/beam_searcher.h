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
#include <torch/types.h>

#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

namespace xllm {

struct BeamSearchOutput {
  torch::Tensor src_seq_idxes;  // [num_seq]
  torch::Tensor out_tokens;     // [num_seq]
  torch::Tensor out_logprobs;   // [num_seq]
};

class BeamSearcher {
 public:
  BeamSearcher() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logprobs: [num_seq]
  // top_tokens: [num_seq, top_k]
  // top_logprobs: [num_seq, top_k]
  // beam_width: per-group beam count, only consumed by the MLU device path.
  BeamSearchOutput forward(const torch::Tensor& logprobs,
                           const torch::Tensor& top_tokens,
                           const torch::Tensor& top_logprobs,
                           int64_t beam_width = 0) const;
};

}  // namespace xllm