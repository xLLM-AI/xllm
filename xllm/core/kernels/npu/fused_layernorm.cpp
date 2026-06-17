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
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <array>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps,
                       const std::string& mode) {
  if (mode != "rmsnorm") {
    LOG(FATAL) << "Only rmsnorm mode is supported in NPU rms_norm";
  }
  std::tuple<at::Tensor, at::Tensor> result =
      at_npu::native::custom_ops::npu_rms_norm(input, weight, eps);
  auto normalized_input = std::get<0>(result);
  return normalized_input;
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_dynamic_quant(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  CHECK(input.numel() > 0) << "Input tensor should not be empty.";
  CHECK(weight.numel() > 0) << "Weight tensor should not be empty.";
  CHECK(input.dim() >= 1) << "Input tensor dim should be >= 1.";
  CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1))
      << "Weight dim must equal input last dim.";

  at::SmallVector<int64_t, 8> scale_shape;
  const int64_t scale_dim = input.dim() - 1;
  for (int64_t index = 0; index < scale_dim; ++index) {
    scale_shape.push_back(input.size(index));
  }

  torch::Tensor output =
      torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
  torch::Tensor unused_output =
      torch::empty({1}, input.options().dtype(torch::kInt8));
  torch::Tensor scale =
      torch::empty(scale_shape, input.options().dtype(torch::kFloat32));
  torch::Tensor unused_scale = torch::empty_like(scale);
  std::optional<torch::Tensor> smooth_scale = std::nullopt;
  std::optional<torch::Tensor> smooth_scale2 = std::nullopt;
  std::optional<torch::Tensor> beta = std::nullopt;
  std::array<bool, 2>* output_mask = nullptr;
  int64_t* dst_type = nullptr;

  EXEC_NPU_CMD(aclnnRmsNormDynamicQuant,
               input,
               weight,
               smooth_scale,
               smooth_scale2,
               beta,
               eps,
               output_mask,
               dst_type,
               output,
               unused_output,
               scale,
               unused_scale);
  return {output, scale};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon) {
  return at_npu::native::custom_ops::npu_add_rms_norm(x1, x2, gamma, epsilon);
}

}  // namespace xllm::kernel::npu
