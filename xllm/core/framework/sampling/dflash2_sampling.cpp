/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/dflash2_sampling.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/sampling/gumbel_sampling.h"
#include "core/framework/sampling/logits_utils.h"
#include "core/util/tensor_helper.h"

namespace xllm {

DFlash2SampleOutput sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size) {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k}));
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // Target-side truncation, penalties, and grammar constraints belong to
  // verification and must not be applied to the selector's top-k
  // distribution.
  torch::Tensor edge_log_probs = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_log_probs, sampling_params.temperatures, batch_size);
  }
  edge_log_probs = torch::log_softmax(edge_log_probs, /*dim=*/-1);

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor draft_logits;
  if (sampling_params.return_logits) {
    draft_logits = torch::full({batch_size, num_steps, vocab_size},
                               -std::numeric_limits<float>::infinity(),
                               float_options);
  }
  // The [batch, num_steps] confidence signal is cheap next to the dense
  // logits; always fill it so the adaptive controller has it whenever it is
  // enabled, independent of return_logits.
  torch::Tensor confidence_probs =
      torch::empty({batch_size, num_steps}, float_options);
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  torch::Tensor greedy_row_mask;
  torch::Tensor zero_log_probs;
  if (sampling_params.return_logits &&
      sampling_params.sampling_mode == SamplingMode::MIXED) {
    greedy_row_mask =
        sampling_params.require_do_sample().view({batch_size, 1}).logical_not();
    zero_log_probs = torch::zeros({batch_size, 1}, float_options);
  }

  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor edge = edge_log_probs.select(/*dim=*/1, /*index=*/step);
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    torch::Tensor row_log_probs =
        edge.gather(/*dim=*/1, gather_indices).squeeze(/*dim=*/1);
    torch::Tensor sampled_indices = gumbel_argmax(
        row_log_probs, gumbel_noise.select(/*dim=*/1, /*index=*/step));

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        index_select_2d(step_candidates, /*dim=*/1, sampled_indices);
    if (sampling_params.return_logits) {
      // Greedy rows embed a log-space one-hot (q = 1); random rows keep
      // their sampled row log prob.
      torch::Tensor step_log_probs = row_log_probs;
      if (sampling_params.sampling_mode == SamplingMode::MIXED) {
        step_log_probs = torch::where(
            greedy_row_mask,
            embed_sparse_logits(
                sampled_indices.view({batch_size, 1}), zero_log_probs, top_k),
            step_log_probs);
      }
      draft_logits.select(/*dim=*/1, step)
          .scatter_(/*dim=*/1, step_candidates, step_log_probs);
    }
    confidence_probs.select(/*dim=*/1, step)
        .copy_(
            index_select_2d(row_log_probs, /*dim=*/1, sampled_indices).exp_());
    token_ids.select(/*dim=*/1, step).copy_(sampled_tokens);
    previous_indices = sampled_indices;
  }

  return {.token_ids = std::move(token_ids),
          .draft_logits = std::move(draft_logits),
          .confidence_probs = std::move(confidence_probs)};
}

}  // namespace xllm
