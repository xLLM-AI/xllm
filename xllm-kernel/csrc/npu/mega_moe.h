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

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace xllm::kernel::npu {

// MegaMoe dispatch-quant modes and the op's own dtype codes shared between the
// kernel implementation and its op registration. These dtype codes are the
// MegaMoe operator's own encoding, not aclDataType values (e.g. the bf16 code
// 28 differs from ACL_BF16 == 27), so they must not be replaced with SDK enums.
inline constexpr int64_t kMegaMoeDispatchQuantModeNone = 0;
inline constexpr int64_t kMegaMoeDispatchQuantModeDynamic = 2;
inline constexpr int64_t kMegaMoeCombineQuantModeNone = 0;
inline constexpr int64_t kMegaMoeDtypeInt8 = 2;
inline constexpr int64_t kMegaMoeDtypeBFloat16 = 28;

bool has_mega_moe();

std::tuple<torch::Tensor, torch::Tensor> apply_npu_mega_moe(
    const torch::Tensor& context,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const std::optional<torch::TensorList>& weight_scales1 = std::nullopt,
    const std::optional<torch::TensorList>& weight_scales2 = std::nullopt,
    const std::optional<torch::TensorList>& bias1 = std::nullopt,
    const std::optional<torch::TensorList>& bias2 = std::nullopt,
    const std::optional<torch::Tensor>& x_active_mask = std::nullopt,
    int64_t max_recv_token_num = 0,
    int64_t dispatch_quant_mode = 0,
    int64_t combine_quant_mode = 0,
    const std::string& comm_alg = "",
    int64_t num_max_tokens_per_rank = 0,
    const std::string& activation = "swiglu",
    float activation_clamp = std::numeric_limits<float>::max(),
    int64_t dispatch_quant_out_dtype = 0,
    int64_t topo_type = 0,
    int64_t rank_num_per_server = 2);

}  // namespace xllm::kernel::npu
