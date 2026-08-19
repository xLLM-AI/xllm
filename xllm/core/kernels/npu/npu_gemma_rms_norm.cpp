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

#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {

void npu_gemma_rms_norm(const torch::Tensor& x,
                        const torch::Tensor& gamma,
                        double epsilon,
                        torch::Tensor& rstd_out,
                        torch::Tensor& y_out) {
  int64_t dim_x = x.dim();
  int64_t dim_gamma = gamma.dim();
  int64_t diff = dim_x - dim_gamma;
  std::vector<int64_t> rstd_shape;
  if (diff > 0) {
    rstd_shape.reserve(dim_x);
    auto x_sizes = x.sizes();
    for (int64_t index = 0; index < diff; ++index) {
      rstd_shape.push_back(x_sizes[index]);
    }
    for (int64_t index = 0; index < dim_gamma; ++index) {
      rstd_shape.push_back(1);
    }
  } else {
    rstd_shape.assign(dim_x, 1);
  }

  rstd_out = torch::empty(rstd_shape, x.options().dtype(torch::kFloat));
  y_out = torch::empty(x.sizes(), x.options());
  EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y_out, rstd_out);
}

}  // namespace xllm::kernel::npu
