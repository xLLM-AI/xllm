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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor moe_all2all_gen_send_layout(const torch::Tensor& token_count,
                                          int64_t nrank) {
  return tmo::torch_api::moe_all2all_gen_send_layout(token_count, nrank);
}

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    const torch::Tensor& token_num,
    int64_t pad_num,
    bool return_cusum_token_count) {
  // get dimension information
  int32_t rank_num = token_num.size(0);
  int32_t expert_num = token_num.size(1);

  // prepare tensor options (keep same device as input, enforce int32)
  auto options =
      torch::TensorOptions().dtype(torch::kInt).device(token_num.device());

  // output tensors
  torch::Tensor gather_by_expert_index =
      torch::empty({rank_num * pad_num}, options);
  torch::Tensor gather_by_rank_index =
      torch::empty({rank_num * pad_num}, options);
  torch::Tensor token_count = torch::empty({expert_num}, options);
  torch::Tensor token_sum = torch::empty({1}, options);

  // handle optional tensor allocation
  torch::Tensor cusum_token_count;
  if (return_cusum_token_count) {
    cusum_token_count = torch::empty({expert_num + 1}, options);
  }

  tmo::torch_api::moe_all2all_gen_gather_index(gather_by_expert_index,
                                               gather_by_rank_index,
                                               token_count,
                                               cusum_token_count,
                                               token_sum,
                                               token_num,
                                               pad_num);

  // pack and return results using std::vector
  std::vector<torch::Tensor> results;
  results.reserve(return_cusum_token_count ? 5 : 4);

  results.push_back(gather_by_expert_index);
  results.push_back(gather_by_rank_index);
  results.push_back(token_count);
  results.push_back(token_sum);

  if (return_cusum_token_count) {
    results.push_back(cusum_token_count);
  }

  return results;
}

std::vector<torch::Tensor> moe_all2all_create(int64_t dispatch_token_byte,
                                              int64_t combine_token_byte,
                                              int64_t max_expert_num,
                                              int64_t max_token_num,
                                              int64_t rank,
                                              int64_t nrank,
                                              const torch::Device& device) {
  // Create placeholder tensor on the specified device
  auto options = torch::TensorOptions().device(device);
  torch::Tensor place_holder = torch::empty({0}, options);

  // Call the underlying operator
  // Since the return type is explicitly std::vector<torch::Tensor>, we capture
  // it directly.
  std::vector<torch::Tensor> outputs =
      tmo::torch_api::moe_all2all_create(dispatch_token_byte,
                                         combine_token_byte,
                                         max_expert_num,
                                         max_token_num,
                                         rank,
                                         nrank,
                                         place_holder);
  // Return all 6 tensors
  // Construct a new vector from the iterator range
  return std::vector<torch::Tensor>(outputs.begin(), outputs.end());
}

void moe_all2all_init(int64_t handle,
                      const torch::Tensor& all_exchange_info,
                      const torch::Device& device) {
  auto options = torch::TensorOptions().device(device);
  torch::Tensor place_holder = torch::empty({0}, options);
  tmo::torch_api::moe_all2all_init(handle, all_exchange_info, place_holder);
}

void moe_all2all_dispatch(int64_t handle,
                          int64_t token_byte,
                          int64_t token_num,
                          const torch::Tensor& send_layout,
                          const torch::Tensor& send_token_num,
                          const torch::Tensor& recv_layout,
                          const torch::Tensor& recv_token_num,
                          const std::optional<torch::Tensor>& send_token,
                          const std::optional<torch::Tensor>& recv_token) {
  tmo::torch_api::moe_all2all_dispatch(handle,
                                       token_byte,
                                       token_num,
                                       send_layout,
                                       send_token_num,
                                       recv_layout,
                                       recv_token_num,
                                       send_token,
                                       recv_token);
}

void moe_all2all_combine(int64_t handle,
                         int64_t token_byte,
                         int64_t token_num,
                         const torch::Tensor& send_src_layout,
                         const torch::Tensor& send_dst_layout,
                         const std::optional<torch::Tensor>& send_token,
                         const std::optional<torch::Tensor>& recv_token) {
  tmo::torch_api::moe_all2all_combine(handle,
                                      token_byte,
                                      token_num,
                                      send_src_layout,
                                      send_dst_layout,
                                      send_token,
                                      recv_token);
}

void moe_all2all_destroy(int64_t handle, const torch::Device& device) {
  auto options = torch::TensorOptions().device(device);
  torch::Tensor place_holder = torch::empty({0}, options);
  tmo::torch_api::moe_all2all_destroy(handle, place_holder);
}

}  // namespace xllm::kernel::mlu
