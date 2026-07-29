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
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gamma_add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon,
    bool add_gamma_offset) {
  CHECK_GT(x1.numel(), 0) << "x1 must not be empty.";
  CHECK_GT(x2.numel(), 0) << "x2 must not be empty.";
  CHECK_GT(gamma.numel(), 0) << "gamma must not be empty.";
  CHECK_EQ(x1.sizes(), x2.sizes()) << "x1 and x2 must have the same shape.";
  CHECK_GE(x1.dim(), gamma.dim())
      << "x1 must have at least as many dimensions as gamma.";

  const torch::IntArrayRef x1_shape = x1.sizes();
  const int64_t keep_dims = x1.dim() - gamma.dim();
  std::vector<int64_t> rstd_shape;
  rstd_shape.reserve(static_cast<size_t>(x1.dim()));
  for (int64_t index = 0; index < x1.dim(); ++index) {
    rstd_shape.emplace_back(index < keep_dims ? x1_shape[index] : 1);
  }

  torch::Tensor y_out = torch::empty_like(x1);
  torch::Tensor rstd_out =
      torch::empty(rstd_shape, x1.options().dtype(torch::kFloat32));
  torch::Tensor x_out = torch::empty_like(x1);
  EXEC_NPU_CMD(aclnnGammaAddRmsNorm,
               x1,
               x2,
               gamma,
               epsilon,
               add_gamma_offset,
               y_out,
               rstd_out,
               x_out);
  return {y_out, rstd_out, x_out};
}

}  // namespace xllm::kernel::npu
