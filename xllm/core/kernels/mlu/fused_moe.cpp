/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
  auto reduce_weight =
      torch::empty({input.size(0), topk},
                   torch::dtype(torch::kFloat).device(input.device()));
  auto expert_id =
      torch::empty({input.size(0), topk},
                   torch::dtype(torch::kInt32).device(input.device()));
  tmo::torch_api::moe_active_topk(input,
                                  topk,
                                  num_expert_group,
                                  topk_group,
                                  normalize,
                                  mask,
                                  normed_by,
                                  scoring_func,
                                  route_scale,
                                  e_score_correction_bias,
                                  reduce_weight,
                                  expert_id);
  return std::make_tuple(reduce_weight, expert_id);
}

std::tuple<torch::Tensor, torch::Tensor> moe_softplus_topk(
    const torch::Tensor& input,
    int64_t topk,
    const std::optional<torch::Tensor>& input_ids,
    const std::optional<torch::Tensor>& tid2eid,
    const std::optional<torch::Tensor>& bias,
    float route_scale) {
  std::vector<int64_t> out_shape = input.sizes().vec();
  out_shape.pop_back();
  out_shape.push_back(topk);
  auto reduce_weight = torch::empty(
      out_shape, torch::dtype(torch::kFloat).device(input.device()));
  auto expert_id = torch::empty(
      out_shape, torch::dtype(torch::kInt32).device(input.device()));
  tmo::torch_api::moe_softplus_topk(input,
                                    input_ids,
                                    tid2eid,
                                    bias,
                                    topk,
                                    route_scale,
                                    reduce_weight,
                                    expert_id);
  return std::make_tuple(reduce_weight, expert_id);
}

std::vector<torch::Tensor> moe_gen_idx(const torch::Tensor& expert_id,
                                       int64_t expert_num) {
  return tmo::torch_api::moe_gen_idx(expert_id, expert_num);
}

torch::Tensor moe_expand_input(
    const torch::Tensor& input,
    const torch::Tensor& gather_index,
    const std::optional<torch::Tensor>& cusum_token_count,
    int64_t start_expert_id,
    int64_t expert_size) {
  return tmo::torch_api::moe_expand_input(
      input, gather_index, cusum_token_count, start_expert_id, expert_size);
}

torch::Tensor moe_combine_result(
    const torch::Tensor& input,
    const torch::Tensor& reduce_weight,
    const torch::Tensor& gather_ids,
    const std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& cusum_token_count,
    const int64_t start_expert_id,
    const int64_t expert_size,
    const std::optional<torch::Tensor>& bias) {
  auto output =
      torch::empty({reduce_weight.size(0), input.size(1)}, input.options());
  tmo::torch_api::moe_combine_result(input,
                                     output,
                                     reduce_weight,
                                     gather_ids,
                                     residual,
                                     cusum_token_count,
                                     start_expert_id,
                                     expert_size,
                                     bias);
  return output;
}

}  // namespace xllm::kernel::mlu
