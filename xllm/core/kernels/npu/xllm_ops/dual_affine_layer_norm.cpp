/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>

#include <array>
#include <cmath>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_affine_param(const torch::Tensor& param,
                        const torch::Tensor& x,
                        const char* name) {
  CHECK(param.defined()) << name << " must be defined";
  CHECK(param.device().is_privateuseone()) << name << " must be on NPU";
  CHECK(param.device() == x.device()) << name << " must be on the input NPU";
  CHECK(param.dim() == 1) << name << " must be 1D";
  CHECK(param.scalar_type() == x.scalar_type())
      << name << " dtype must match x";
  CHECK(param.is_contiguous()) << name << " must be contiguous";
  CHECK(param.numel() == x.size(1)) << name << " length must equal H";
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> dual_affine_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma1,
    const torch::Tensor& beta1,
    const torch::Tensor& gamma2,
    const torch::Tensor& beta2,
    double epsilon) {
  CHECK(x.defined()) << "x must be defined";
  CHECK(x.device().is_privateuseone()) << "x must be on NPU";
  CHECK(x.dim() == 2) << "x must be a 2D [N,H] tensor";
  CHECK(x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16)
      << "x dtype must be FP16 or BF16";
  CHECK(x.is_contiguous()) << "x must be contiguous";

  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  CHECK(rows >= 1 && rows <= 4096) << "N must be in [1,4096]";
  CHECK(hidden >= 1 && hidden <= 8192) << "H must be in [1,8192]";
  CHECK(std::isfinite(epsilon) && epsilon >= 1e-7 && epsilon <= 1e-2)
      << "epsilon must be finite and in [1e-7,1e-2]";

  const std::array<const torch::Tensor*, 4> params = {
      &gamma1, &beta1, &gamma2, &beta2};
  const std::array<const char*, 4> names = {
      "gamma1", "beta1", "gamma2", "beta2"};
  for (size_t index = 0; index < params.size(); ++index) {
    check_affine_param(*params[index], x, names[index]);
  }

  torch::Tensor y1 = torch::empty_like(x);
  torch::Tensor y2 = torch::empty_like(x);
  const float epsilon_float = static_cast<float>(epsilon);
  EXEC_NPU_CMD(aclnnDualAffineLayerNorm,
               x,
               gamma1,
               beta1,
               gamma2,
               beta2,
               epsilon_float,
               y1,
               y2);
  return std::make_tuple(y1, y2);
}

}  // namespace xllm::kernel::npu
