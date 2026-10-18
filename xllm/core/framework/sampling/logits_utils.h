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

#include <limits>
#include <memory>
#include <vector>

#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif
#include "core/util/tensor_helper.h"
#include "kernels/ops_api.h"

namespace xllm {

// Gathers logits at `token_ids` along the last dim. -1 sentinels clamp to
// row 0 and are the caller's to re-mask.
inline torch::Tensor gather_from_logits(const torch::Tensor& logits,
                                        const torch::Tensor& token_ids) {
  return index_select_2d(
      logits, /*dim=*/-1, token_ids.clamp_min(0).to(torch::kLong));
}

// log P(token_ids) = gathered logit − caller-computed LSE, shared between
// the acceptance test and recovery. Sentinels are not masked.
inline torch::Tensor gather_log_probs_from_logits(
    const torch::Tensor& logits,
    const torch::Tensor& lse,
    const torch::Tensor& token_ids) {
  return gather_from_logits(logits, token_ids).to(torch::kFloat32).sub(lse);
}

// P(token_ids) without materializing the dense softmax; -1 sentinels read 0.
inline torch::Tensor gather_probs_from_logits(const torch::Tensor& logits,
                                              const torch::Tensor& token_ids) {
  return gather_log_probs_from_logits(
             logits,
             torch::logsumexp(
                 logits.to(torch::kFloat32), /*dim=*/-1, /*keepdim=*/false),
             token_ids)
      .exp()
      .masked_fill_(token_ids.lt(0), 0);
}

// -inf background; a greedy one-hot is the same call with zero `values`.
inline torch::Tensor embed_sparse_logits(const torch::Tensor& token_ids,
                                         const torch::Tensor& values,
                                         int64_t width) {
  const int64_t rows = token_ids.size(0);
  torch::Tensor dense = torch::full(
      {rows, width}, -std::numeric_limits<float>::infinity(), values.options());
  dense.scatter_(/*dim=*/1, token_ids, values);
  return dense;
}

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);

void apply_repetition_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& penalties,
    const torch::Tensor& unique_token_ids_lens = {});

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures);

void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p);

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& temperatures,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);

}  // namespace xllm
