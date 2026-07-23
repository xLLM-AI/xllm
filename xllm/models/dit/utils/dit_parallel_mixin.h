/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <utility>

#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace dit {

// Mixin providing classifier-free guidance (CFG) parallelism.
//
// Usage:
//   class MyPipeline : public torch::nn::Module,
//                      public dit::CFGParallelMixin { ... };
class CFGParallelMixin {
 public:
  explicit CFGParallelMixin(const DiTModelContext& context)
      : cfg_group_(context.get_parallel_args().dit_cfg_group_) {}

  // forward_fn(is_positive) -> Tensor  —  caller captures embeddings in lambda.
  // Returns {positive_noise_pred, negative_noise_pred}.
  template <typename ForwardFn>
  std::pair<torch::Tensor, torch::Tensor> exec_with_cfg(
      const ForwardFn& forward_fn) const {
    int32_t cfg_size = 1;
    if (cfg_group_ != nullptr) {
      cfg_size = cfg_group_->world_size();
    }

    CHECK(cfg_size == 1 || cfg_size == 2);

    // Serial execution: evaluate positive and negative conditionals one by one.
    if (cfg_size == 1) {
      return {forward_fn(true), forward_fn(false)};
    }

    // CFG parallel (cfg_size == 2): rank 0 → positive, rank 1 → negative,
    // gather + chunk.
    int32_t rank = cfg_group_->rank();
    torch::Tensor noise_pred = forward_fn(rank == 0);
    torch::Tensor gathered =
        parallel_state::gather(noise_pred, cfg_group_, /*dim=*/0);
    auto chunks = torch::chunk(gathered, 2, 0);
    return {chunks[0], chunks[1]};
  }

 private:
  ProcessGroup* cfg_group_ = nullptr;
};

// Mixin for VAE parallelism (to be implemented).
class VaeParallelMixin {};

// Mixin for sequence parallelism (to be implemented).
class SpParallelMixin {};

}  // namespace dit
}  // namespace xllm
