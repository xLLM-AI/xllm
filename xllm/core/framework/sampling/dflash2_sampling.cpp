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

#include "core/framework/sampling/gumbel_sampling.h"

namespace xllm {

DFlash2SampleOutput sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size,
    bool need_dense_probs) {
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

  // Pre-compute log_softmax(edge_logits / temperature) for every step at
  // once.  The previous implementation re-scaled and re-normalized the
  // gathered row inside each step through the generic sampler, which lowered
  // to an AICPU multinomial plus a per-step consensus broadcast and left the
  // compute stream idle between steps.  Gumbel-max over the same
  // log-probabilities draws from exactly the same distribution:
  //   argmax(log_softmax(logits / T) + g) ~ categorical(softmax(logits / T))
  // and greedy rows (zeroed noise) collapse to plain argmax.  Target-side
  // truncation, penalties, and grammar constraints are applied by
  // verification and must not be applied a second time to the selector's
  // top-k candidate distribution.
  torch::Tensor edge_log_probs = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_log_probs, sampling_params.temperatures, batch_size);
  }
  edge_log_probs = torch::log_softmax(edge_log_probs, /*dim=*/-1);

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor candidate_probs;
  if (need_dense_probs) {
    candidate_probs =
        torch::empty({batch_size, num_steps, top_k}, float_options);
  }
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  using ISlice = torch::indexing::Slice;
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
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    if (need_dense_probs) {
      // Random rows expose the full softmax over the top-k candidates;
      // deterministic rows expose a one-hot at the selected candidate.
      torch::Tensor step_probs;
      if (sampling_params.all_random_sample) {
        step_probs = row_log_probs.exp();
      } else {
        torch::Tensor greedy_probs =
            torch::zeros({batch_size, top_k}, float_options);
        greedy_probs.scatter_(/*dim=*/1,
                              sampled_indices.view({-1, 1}),
                              /*value=*/1.0);
        step_probs =
            torch::where(sampling_params.do_sample.view({batch_size, 1}),
                         row_log_probs.exp(),
                         greedy_probs);
      }
      candidate_probs.index_put_({ISlice(), step, ISlice()}, step_probs);
    }
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    previous_indices = sampled_indices;
  }

  if (!need_dense_probs) {
    return {.token_ids = std::move(token_ids), .dense_probs = torch::Tensor()};
  }
  torch::Tensor dense_probs =
      torch::zeros({batch_size, num_steps, vocab_size}, float_options);
  dense_probs.scatter_(
      /*dim=*/-1, candidates.candidate_ids, candidate_probs);
  return {.token_ids = std::move(token_ids),
          .dense_probs = std::move(dense_probs)};
}

}  // namespace xllm
