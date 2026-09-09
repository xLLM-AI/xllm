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

#include <glog/logging.h>

#include "ilu_ops_api.h"

namespace xllm::kernel::ilu {

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    const torch::Tensor& input,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool normalize,
    const std::optional<torch::Tensor>& mask,
    const std::string& normed_by,
    const std::string& scoring_func,
    double route_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias) {
  torch::Tensor input_ = input.to(torch::kFloat32);
  auto reduce_weight =
      torch::empty({input.size(0), topk},
                   torch::dtype(torch::kFloat).device(input.device()));
  auto topk_indices =
      torch::empty({input.size(0), topk},
                   torch::dtype(torch::kInt32).device(input.device()));
  auto token_expert_indices =
      torch::empty({input.size(0), topk},
                   torch::dtype(torch::kInt32).device(input.device()));

  infer::topk_softmax(
      reduce_weight, topk_indices, token_expert_indices, input_, false);

  auto tt = reduce_weight.sum(-1);
  if (normalize) {
    reduce_weight = reduce_weight / reduce_weight.sum(-1).unsqueeze(-1);
  }
  return std::make_tuple(reduce_weight, topk_indices);
}

std::vector<torch::Tensor> moe_gen_idx(torch::Tensor& expert_id,
                                       int64_t expert_num) {
  auto src_dst = expert_id.new_empty({expert_id.numel()});
  auto dst_src = torch::empty_like(src_dst);
  auto expert_sizes_gpu = expert_id.new_empty({expert_num});
  auto expert_sizes_gpu_cumsum = expert_id.new_zeros({expert_id.numel() + 1});
  infer::moe_compute_token_index_api(expert_id,
                                     src_dst,
                                     dst_src,
                                     expert_sizes_gpu,
                                     /*expert_mask=*/std::nullopt,
                                     /*expert_sizes_cpu*/ std::nullopt,
                                     /*expert_sizes_gpu*/ std::nullopt,
                                     0,
                                     expert_num,
                                     expert_num);

  expert_sizes_gpu_cumsum = expert_sizes_gpu.cumsum(-1);
  return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_gpu_cumsum};
}

torch::Tensor moe_expand_input(const torch::Tensor& input,
                               const torch::Tensor& gather_index,
                               const torch::Tensor& combine_idx,
                               int64_t topk) {
  int64_t dst_tokens = input.size(0) * topk;
  auto output = input.new_empty({dst_tokens, input.size(1)});
  infer::moe_expand_input(
      output, input, combine_idx, gather_index, dst_tokens, topk);

  return output;
}

torch::Tensor moe_combine_result(torch::Tensor& input, torch::Tensor& weight) {
  input = input.view({-1, weight.size(1), input.size(1)});
  auto output = input.new_empty({input.size(0), input.size(2)});
  infer::moe_output_reduce_sum(output,
                               input,
                               weight,
                               /*mask=*/std::nullopt,
                               /*extra_residual*/ std::nullopt,
                               /*scaling_factor=*/1.0);
  return output;
}

}  // namespace xllm::kernel::ilu
