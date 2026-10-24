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

#include "core/framework/sampling/sampling_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// Transform stages shared by LogitsProcessor::process_logits and the
// rejection sampler's verify path. Penalty stages are rank-agnostic (last
// dim, per-sequence tables, stride-0 grouped broadcast); the argmax-
// preserving filters take flat 2-D [rows, vocab] logits — the contract the
// fused device kernels accept.

// [batch] -> [batch, 1, ..., 1] padded to the logits' rank (right-aligned
// broadcasting would hit the inner dims).
inline std::vector<int64_t> pad_batch_shape(int64_t batch, int64_t rank) {
  std::vector<int64_t> shape{/*batch=*/batch};
  shape.insert(shape.end(), rank - 1, 1);
  return shape;
}

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);
void apply_min_token_block(torch::Tensor& logits,
                           const torch::Tensor& min_token_ids,
                           const torch::Tensor& min_tokens_remaining);
void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties,
                                const torch::Tensor& unique_token_ids_lens);
void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p);
void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);
void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask);

// Stateless owner of the in-place pre-sampling transforms; the Sampler
// consumes already-processed logits.
class LogitsProcessor final {
 public:
  // In-place sampling transforms on [batch_size, vocab_size] logits:
  // penalties first, then the argmax-preserving transforms (temperature,
  // min-p, top-k/top-p) and the token filter mask/bitmask. A greedy batch
  // that reports no distribution values skips the argmax-preserving stage.
  static void process_logits(torch::Tensor& logits,
                             const SamplingParameters& params,
                             const torch::Tensor& filter_mask = {});

  // Bias and repetition/frequency/presence penalties; shared by paths that
  // skip the later transforms (e.g. the RecSampler fast path).
  static void apply_sampling_penalties(torch::Tensor& logits,
                                       const SamplingParameters& params);

  // Temperature only, for callers outside the full pipeline (e.g. the
  // DFlash2 selector, where target-side penalties must not apply).
  static void apply_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures);
};

}  // namespace xllm
