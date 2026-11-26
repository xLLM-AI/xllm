// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#pragma once

#include <glog/logging.h>

#include "csrc/npu/mega_moe.h"

namespace xllm_kernel {

inline std::tuple<torch::Tensor, torch::Tensor> mega_moe(
    const torch::Tensor& context,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    const torch::TensorList weight_scales1,
    const torch::TensorList weight_scales2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    int64_t num_max_tokens_per_rank,
    const std::optional<torch::Tensor>& x_active_mask) {
  CHECK_EQ(weight_scales1.empty(), weight_scales2.empty())
      << "MegaMoe weight scales must be provided for both projections or "
         "neither.";
  const bool use_w8a8 = !weight_scales1.empty();
  return xllm::kernel::npu::apply_npu_mega_moe(
      context,
      x,
      topk_ids,
      topk_weights,
      weight1,
      weight2,
      moe_expert_num,
      ep_world_size,
      ccl_buffer_size,
      use_w8a8 ? std::optional<torch::TensorList>(weight_scales1)
               : std::nullopt,
      use_w8a8 ? std::optional<torch::TensorList>(weight_scales2)
               : std::nullopt,
      /*bias1=*/std::nullopt,
      /*bias2=*/std::nullopt,
      x_active_mask,
      /*max_recv_token_num=*/0,
      use_w8a8 ? xllm::kernel::npu::kMegaMoeDispatchQuantModeDynamic
               : xllm::kernel::npu::kMegaMoeDispatchQuantModeNone,
      xllm::kernel::npu::kMegaMoeCombineQuantModeNone,
      /*comm_alg=*/"",
      num_max_tokens_per_rank,
      /*activation=*/"swiglu",
      /*activation_clamp=*/std::numeric_limits<float>::max(),
      use_w8a8 ? xllm::kernel::npu::kMegaMoeDtypeInt8
               : xllm::kernel::npu::kMegaMoeDtypeBFloat16);
}

}  // namespace xllm_kernel
