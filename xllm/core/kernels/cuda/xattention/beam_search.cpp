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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "cuda.h"

namespace xllm::kernel::cuda {

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t num_return_sequences,
                 uint32_t current_step) {
  (void)num_return_sequences;
  torch::Device device = acc_logprob.device();

  uint32_t beam_size = in_sequence_group.size(1);

  uint32_t top_k = top_tokens.size(1);
  uint32_t total_rounds = in_sequence_group.size(2);

  CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";

  if (current_step == 0) {
    auto tokens_view =
        top_tokens.view({batch_size, top_k}).slice(1, 0, beam_size);
    auto init_probs_view =
        top_logprobs.view({batch_size, top_k}).slice(1, 0, beam_size);

    out_token_ids.view({batch_size, beam_size}).copy_(tokens_view);
    out_acc_logprob.view({batch_size, beam_size}).copy_(init_probs_view);

    auto indices =
        torch::arange(
            beam_size,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(0)
            .expand({batch_size, -1})
            .reshape({-1, 1});
    out_token_index.copy_(indices);

    auto sequence_view =
        out_sequence_group.view({batch_size, beam_size, total_rounds});
    sequence_view.slice(2, 0, 1).squeeze(2).copy_(tokens_view);

  } else {
    auto combined_probs =
        (acc_logprob + top_logprobs).view({batch_size, beam_size * top_k});

    auto topk_result = torch::topk(combined_probs, beam_size, -1);
    auto new_probs = std::get<0>(topk_result);    // [batch_size, beam_size]
    auto new_indices = std::get<1>(topk_result);  // [batch_size, beam_size]

    auto ordered_indices = new_indices.argsort(static_cast<int64_t>(1), false);
    // Reorder new_probs (and corresponding new_indices) by ordered_indices to
    // keep alignment.
    if (current_step < total_rounds - 1) {
      new_probs = new_probs.gather(1, ordered_indices);
      new_indices = new_indices.gather(1, ordered_indices);
    }

    auto parent_beam = (new_indices / top_k).to(torch::kLong);
    auto token_in_beam = (new_indices % top_k).to(torch::kLong);

    auto top_tokens_reshaped = top_tokens.view({batch_size, beam_size, top_k});

    auto batch_idx =
        torch::arange(batch_size,
                      torch::TensorOptions().dtype(torch::kLong).device(device))
            .unsqueeze(1)
            .expand_as(parent_beam);

    using torch::indexing::TensorIndex;
    auto new_tokens = top_tokens_reshaped.index({TensorIndex(batch_idx),
                                                 TensorIndex(parent_beam),
                                                 TensorIndex(token_in_beam)});

    out_acc_logprob.view({batch_size, beam_size}).copy_(new_probs);
    out_token_index.view({batch_size, beam_size})
        .copy_(new_indices.to(torch::kInt32));
    out_token_ids.view({batch_size, beam_size}).copy_(new_tokens);

    auto batch_range =
        torch::arange(
            batch_size,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(1)
            .expand({-1, beam_size});
    auto beam_range =
        torch::arange(
            beam_size,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(0)
            .expand({batch_size, -1});

    using torch::indexing::Slice;
    using torch::indexing::TensorIndex;
    out_sequence_group.slice(2, 0, current_step) =
        in_sequence_group.index({TensorIndex(batch_range),
                                 TensorIndex(parent_beam.to(torch::kInt32)),
                                 Slice(0, current_step)});

    out_sequence_group.slice(2, current_step, current_step + 1) =
        new_tokens.unsqueeze(2);
  }
}

}  // namespace xllm::kernel::cuda
