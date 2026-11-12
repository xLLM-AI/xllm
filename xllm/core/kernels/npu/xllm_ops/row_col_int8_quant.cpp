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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kMaxM = 4096;
constexpr int64_t kMaxN = 4096;

}  // namespace

void row_col_int8_quant(const torch::Tensor& x,
                        torch::Tensor& q_row,
                        torch::Tensor& q_col,
                        torch::Tensor& row_absmax,
                        torch::Tensor& col_absmax) {
  TORCH_CHECK(x.defined(), "row_col_int8_quant: x must be defined");
  TORCH_CHECK(x.dim() == 2, "row_col_int8_quant: x must be 2D, got ", x.dim(),
              "D");
  TORCH_CHECK(x.scalar_type() == at::kHalf,
              "row_col_int8_quant: x dtype must be fp16, got ",
              x.scalar_type());
  TORCH_CHECK(x.is_contiguous(),
              "row_col_int8_quant: x must be contiguous");
  TORCH_CHECK(x.is_cuda() || x.device().type() == c10::DeviceType::PrivateUse1,
              "row_col_int8_quant: x must be on NPU");

  const int64_t m = x.size(0);
  const int64_t n = x.size(1);
  TORCH_CHECK(m >= 1 && m <= kMaxM,
              "row_col_int8_quant: M must be in [1,", kMaxM, "], got ", m);
  TORCH_CHECK(n >= 2 && n <= kMaxN,
              "row_col_int8_quant: N must be in [2,", kMaxN, "], got ", n);
  TORCH_CHECK(n % 2 == 0, "row_col_int8_quant: N must be even, got ", n);

  q_row = torch::empty({m, n}, x.options().dtype(at::kChar));
  q_col = torch::empty({m, n}, x.options().dtype(at::kChar));
  row_absmax = torch::empty({m}, x.options().dtype(at::kFloat));
  col_absmax = torch::empty({n}, x.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnRowColInt8Quant, x, q_row, q_col, row_absmax, col_absmax);
}

}  // namespace xllm::kernel::npu
