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

#include <glog/logging.h>

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

BeamSearchOutput beam_search(const torch::Tensor& logprobs,
                             const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             int64_t beam_width) {
  const int64_t num_seq = logprobs.numel();
  const int64_t topk = top_logprobs.size(1);
  CHECK_GT(beam_width, 0) << "beam_search requires beam_width > 0: num_seq="
                          << num_seq << ", beam_width=" << beam_width
                          << ", topk=" << topk;
  CHECK_EQ(num_seq % beam_width, 0)
      << "beam_search requires num_seq divisible by beam_width: num_seq="
      << num_seq << ", beam_width=" << beam_width << ", topk=" << topk;

  const int64_t num_group = num_seq / beam_width;

  // Per-group candidate scores: combined[s, c] = logprobs[s] + top_logprobs[s,
  // c], reshaped so each group exposes beam_width * topk candidates.
  torch::Tensor combined = (logprobs.reshape({num_seq, 1}) + top_logprobs)
                               .reshape({num_group, beam_width * topk});

  auto topk_result = torch::topk(combined, beam_width, /*dim=*/-1);
  torch::Tensor vals = std::get<0>(topk_result);  // [num_group, beam_width]
  // MLU torch::topk returns int64 indices.
  torch::Tensor idx = std::get<1>(topk_result);  // [num_group, beam_width]

  torch::Tensor src_local = idx / topk;  // source beam within the group
  torch::Tensor cand_col = idx % topk;   // candidate column within that beam

  // Global source sequence index = group_id * beam_width + src_local.
  torch::Tensor group_id =
      torch::arange(num_group, idx.options()).view({num_group, 1});
  torch::Tensor src_seq = group_id * beam_width + src_local;

  // out_tokens[g, b] = top_tokens[g, src_local[g, b], cand_col[g, b]].
  torch::Tensor top_tokens_g =
      top_tokens.reshape({num_group, beam_width, topk});
  torch::Tensor row_index =
      src_local.unsqueeze(-1).expand({num_group, beam_width, topk});
  torch::Tensor gathered_rows = top_tokens_g.gather(/*dim=*/1, row_index);
  torch::Tensor out_tokens =
      gathered_rows.gather(/*dim=*/2, cand_col.unsqueeze(-1)).squeeze(-1);

  BeamSearchOutput output;
  output.src_seq_idxes = src_seq.to(torch::kInt32).reshape({-1});
  output.out_tokens = out_tokens.to(torch::kInt32).reshape({-1});
  output.out_logprobs = vals.to(torch::kFloat32).reshape({-1});
  return output;
}

}  // namespace xllm::kernel::mlu
