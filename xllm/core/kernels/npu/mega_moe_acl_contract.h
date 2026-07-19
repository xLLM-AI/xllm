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
#pragma once

#include <torch/torch.h>

#include <optional>
#include <utility>

namespace xllm::kernel::npu {

// Deployed custom-vendor ABI from ops-transformer commit 09f2ed7.
template <typename Execute>
void execute_mega_moe_acl_contract(
    Execute&& execute,
    torch::Tensor& context,
    torch::Tensor& x,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weights,
    torch::TensorList& weight1,
    torch::TensorList& weight2,
    std::optional<torch::TensorList>& weight_scales1,
    std::optional<torch::TensorList>& weight_scales2,
    std::optional<torch::TensorList>& bias1,
    std::optional<torch::TensorList>& bias2,
    std::optional<torch::Tensor>& x_active_mask,
    int64_t& moe_expert_num,
    int64_t& ep_world_size,
    int64_t& ccl_buffer_size,
    int64_t& max_recv_token_num,
    int64_t& dispatch_quant_mode,
    int64_t& dispatch_quant_out_dtype,
    int64_t& combine_quant_mode,
    char*& comm_alg,
    int64_t& num_max_tokens_per_rank,
    char*& activation,
    float& activation_clamp,
    int64_t& topo_type,
    int64_t& rank_num_per_server,
    torch::Tensor& output,
    torch::Tensor& expert_token_nums) {
  std::forward<Execute>(execute)(context,
                                 x,
                                 topk_ids,
                                 topk_weights,
                                 weight1,
                                 weight2,
                                 weight_scales1,
                                 weight_scales2,
                                 bias1,
                                 bias2,
                                 x_active_mask,
                                 moe_expert_num,
                                 ep_world_size,
                                 ccl_buffer_size,
                                 max_recv_token_num,
                                 dispatch_quant_mode,
                                 dispatch_quant_out_dtype,
                                 combine_quant_mode,
                                 comm_alg,
                                 num_max_tokens_per_rank,
                                 activation,
                                 activation_clamp,
                                 topo_type,
                                 rank_num_per_server,
                                 output,
                                 expert_token_nums);
}

}  // namespace xllm::kernel::npu
