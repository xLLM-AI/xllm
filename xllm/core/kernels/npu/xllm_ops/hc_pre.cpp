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

#include <torch/library.h>

#include <cmath>
#include <optional>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_pre(
    const torch::Tensor& x,
    const torch::Tensor& hc_fn,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x.dim(),
              ".");

  c10::ScalarType original_type = x.scalar_type();
  torch::Tensor x_bf16 = x;
  if (x_bf16.scalar_type() != torch::kBFloat16) {
    x_bf16 = x_bf16.to(torch::kBFloat16);
  }

  static const bool has_fused_hc_pre =
      aclnn::detail::get_op_api_func_addr("aclnnHcPreGetWorkspaceSize") !=
          nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnHcPre") != nullptr;
  if (has_fused_hc_pre && x_bf16.device().type() == torch::kPrivateUse1 &&
      hc_mult == 4 && x_bf16.size(-2) == hc_mult && x_bf16.numel() > 0 &&
      hc_fn.dim() == 2 && hc_fn.size(0) == 24 &&
      hc_fn.size(1) == hc_mult * x_bf16.size(-1) && hc_scale.dim() == 1 &&
      hc_scale.numel() == 3 && hc_base.dim() == 1 && hc_base.numel() == 24 &&
      hc_fn.scalar_type() == torch::kFloat32 &&
      hc_scale.scalar_type() == torch::kFloat32 &&
      hc_base.scalar_type() == torch::kFloat32 &&
      hc_fn.device() == x_bf16.device() &&
      hc_scale.device() == x_bf16.device() &&
      hc_base.device() == x_bf16.device() && x_bf16.is_contiguous() &&
      hc_fn.is_contiguous() && hc_scale.is_contiguous() &&
      hc_base.is_contiguous() && hc_sinkhorn_iters > 0 &&
      std::isfinite(norm_eps) && norm_eps > 0 && std::isfinite(hc_eps) &&
      hc_eps > 0) {
    auto output_shape = x_bf16.sizes().vec();
    output_shape.erase(output_shape.end() - 2);
    auto post_shape = x_bf16.sizes().vec();
    post_shape.pop_back();
    auto combination_shape = post_shape;
    combination_shape.emplace_back(hc_mult);
    auto output = torch::empty(output_shape, x_bf16.options());
    auto post =
        torch::empty(post_shape, x_bf16.options().dtype(torch::kFloat32));
    auto combination = torch::empty(combination_shape,
                                    x_bf16.options().dtype(torch::kFloat32));
    auto pre = torch::empty_like(post);
    std::optional<torch::Tensor> pre_mix;
    EXEC_NPU_CMD(aclnnHcPre,
                 x_bf16,
                 hc_fn,
                 hc_scale,
                 hc_base,
                 pre_mix,
                 hc_mult,
                 hc_sinkhorn_iters,
                 hc_eps,
                 norm_eps,
                 output,
                 post,
                 combination,
                 pre);
    return {output.to(original_type), post, combination};
  }

  torch::Tensor rsqrt = hc_pre_inv_rms(x_bf16, norm_eps);
  // Run the gating projection on the BF16 cube (the cube accumulates in FP32
  // internally), then cast the result back to FP32 which hc_pre_sinkhorn
  // requires. The original FP32 cube path was the single largest decode
  // hotspot (~8% of device time); the K=16384 reduction is safe in BF16
  // because the accumulator stays FP32 and the downstream sinkhorn
  // renormalizes the logits.
  torch::Tensor hc_fn_bf16 = hc_fn.scalar_type() == torch::kBFloat16
                                 ? hc_fn
                                 : hc_fn.to(torch::kBFloat16);
  torch::Tensor x_flattened =
      x.dim() == 4 ? x_bf16.flatten(2, -1) : x_bf16.flatten(1, -1);
  torch::Tensor mixes =
      torch::linear(x_flattened, hc_fn_bf16).to(torch::kFloat);

  auto output = hc_pre_sinkhorn(mixes,
                                rsqrt,
                                hc_scale,
                                hc_base,
                                x_bf16,
                                hc_mult,
                                hc_sinkhorn_iters,
                                hc_eps);

  torch::Tensor y = std::get<0>(output).to(original_type);
  return std::make_tuple(y, std::get<1>(output), std::get<2>(output));
}

}  // namespace xllm::kernel::npu
