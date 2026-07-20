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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

bool is_supported_layer_norm_dtype(at::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

torch::Tensor make_contiguous_2d(const torch::Tensor& tensor) {
  return tensor.reshape({-1, tensor.size(-1)}).contiguous();
}

}  // namespace

torch::Tensor layer_norm_fwd_aclnn(const torch::Tensor& x,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& bias,
                                   double eps,
                                   const std::optional<torch::Tensor>& z,
                                   int64_t group_size,
                                   bool norm_before_gate,
                                   bool is_rms_norm) {
  TORCH_CHECK(x.defined(), "layer_norm_fwd_aclnn: x must be defined");
  TORCH_CHECK(weight.defined(), "layer_norm_fwd_aclnn: weight must be defined");
  TORCH_CHECK(x.dim() >= 1, "layer_norm_fwd_aclnn: x must have at least 1 dim");
  TORCH_CHECK(is_supported_layer_norm_dtype(x.scalar_type()),
              "layer_norm_fwd_aclnn: x dtype must be fp16, bf16 or fp32, got ",
              x.scalar_type());
  TORCH_CHECK(weight.dim() == 1, "layer_norm_fwd_aclnn: weight must be 1D");

  const int64_t full_n = x.size(-1);
  if (group_size < 0) {
    group_size = full_n;
  }
  TORCH_CHECK(group_size > 0,
              "layer_norm_fwd_aclnn: group_size must be positive");
  TORCH_CHECK(full_n % group_size == 0,
              "layer_norm_fwd_aclnn: last dim ",
              full_n,
              " must be divisible by group_size ",
              group_size);
  TORCH_CHECK(weight.numel() == full_n,
              "layer_norm_fwd_aclnn: weight numel must equal x last dim");

  if (bias.defined()) {
    TORCH_CHECK(bias.dim() == 1, "layer_norm_fwd_aclnn: bias must be 1D");
    TORCH_CHECK(bias.numel() == full_n,
                "layer_norm_fwd_aclnn: bias numel must equal x last dim");
  }
  if (z.has_value() && z->defined()) {
    TORCH_CHECK(z->sizes() == x.sizes(),
                "layer_norm_fwd_aclnn: z shape must match x");
  }

  torch::Tensor x_2d = make_contiguous_2d(x);
  torch::Tensor weight_contiguous = weight.contiguous();

  c10::optional<torch::Tensor> bias_contiguous = c10::nullopt;
  if (bias.defined()) {
    bias_contiguous = bias.contiguous();
  }

  c10::optional<torch::Tensor> z_2d = c10::nullopt;
  if (z.has_value() && z->defined()) {
    z_2d = make_contiguous_2d(*z);
  }

  torch::Tensor y_2d = torch::empty_like(x_2d);
  const int64_t m = x_2d.size(0);
  const int64_t group_count = full_n / group_size;
  torch::Tensor mean =
      is_rms_norm
          ? torch::empty({0}, x.options().dtype(at::kFloat))
          : torch::empty({group_count * m}, x.options().dtype(at::kFloat));
  torch::Tensor rstd =
      torch::empty({group_count * m}, x.options().dtype(at::kFloat));

  const float eps_f = static_cast<float>(eps);
  EXEC_NPU_CMD(aclnnLayerNormFwd,
               x_2d,
               weight_contiguous,
               bias_contiguous,
               z_2d,
               eps_f,
               group_size,
               norm_before_gate,
               is_rms_norm,
               y_2d,
               mean,
               rstd);

  return y_2d.reshape(x.sizes());
}

}  // namespace xllm::kernel::npu
