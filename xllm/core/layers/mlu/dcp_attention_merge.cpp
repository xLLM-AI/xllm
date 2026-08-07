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

#include "layers/mlu/dcp_attention_merge.h"

#include <glog/logging.h>

#include <limits>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {

DcpAttentionResult merge_dcp_attention_shards(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse) {
  CHECK_EQ(partial_outputs.dim(), 5)
      << "partial outputs must have shape [dcp, batch, query, heads, value]";
  CHECK_EQ(partial_lse.dim(), 4)
      << "partial LSE must have shape [dcp, batch, heads, query]";
  CHECK_EQ(partial_outputs.size(0), partial_lse.size(0));
  CHECK_EQ(partial_outputs.size(1), partial_lse.size(1));
  CHECK_EQ(partial_outputs.size(2), partial_lse.size(3));
  CHECK_EQ(partial_outputs.size(3), partial_lse.size(2));

  torch::Tensor lse = partial_lse.to(torch::kFloat32);
  torch::Tensor max_lse = std::get<0>(lse.max(/*dim=*/0, /*keepdim=*/true));
  torch::Tensor has_finite_shard = torch::isfinite(max_lse);
  torch::Tensor safe_max_lse =
      torch::where(has_finite_shard, max_lse, torch::zeros_like(max_lse));
  torch::Tensor weights = torch::where(torch::isfinite(lse),
                                       torch::exp(lse - safe_max_lse),
                                       torch::zeros_like(lse));
  torch::Tensor weight_sum = weights.sum(/*dim=*/0);
  torch::Tensor safe_weight_sum =
      torch::where(weight_sum > 0, weight_sum, torch::ones_like(weight_sum));

  torch::Tensor weighted_output =
      (partial_outputs.to(torch::kFloat32) * weights.unsqueeze(/*dim=*/2))
          .sum(/*dim=*/0);
  torch::Tensor has_output = (weight_sum > 0).unsqueeze(/*dim=*/1);
  torch::Tensor output =
      torch::where(has_output,
                   weighted_output / safe_weight_sum.unsqueeze(/*dim=*/1),
                   torch::zeros_like(weighted_output));

  const float negative_infinity = -std::numeric_limits<float>::infinity();
  torch::Tensor merged_lse = torch::where(
      weight_sum > 0,
      safe_max_lse.squeeze(/*dim=*/0) + torch::log(safe_weight_sum),
      torch::full_like(weight_sum, negative_infinity));
  return {output.to(partial_outputs.scalar_type()), merged_lse};
}

DcpAttentionResult all_gather_and_merge_dcp_attention(
    const torch::Tensor& local_output,
    const torch::Tensor& local_lse,
    ProcessGroup& dcp_group) {
  torch::Tensor partial_outputs =
      dcp_group.allgather_base_sync(local_output.contiguous());
  torch::Tensor partial_lse =
      dcp_group.allgather_base_sync(local_lse.contiguous());
  return merge_dcp_attention_shards(partial_outputs, partial_lse);
}

}  // namespace xllm::layer
