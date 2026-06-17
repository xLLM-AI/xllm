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

at::Tensor construct_quant_matmul_output_tensor(const at::Tensor& x1,
                                                const at::Tensor& x2,
                                                at::ScalarType output_dtype,
                                                bool transpose2) {
  TORCH_CHECK(x1.dim() >= 2, "x1 dim must be >= 2 for quant matmul");
  TORCH_CHECK(x2.dim() >= 2, "x2 dim must be >= 2 for quant matmul");
  if (transpose2) {
    TORCH_CHECK(x1.size(-1) == x2.size(-1),
                "while transpose2 is true",
                "x1 last dim must match x2 last dim, got ",
                x1.size(-1),
                " vs ",
                x2.size(-1));
  } else {
    TORCH_CHECK(x1.size(-1) == x2.size(-2),
                "while transpose2 is false",
                "x1 dim[-1] must match x2 dim[-2], got ",
                x1.size(-1),
                " vs ",
                x2.size(-2));
  }

  auto out_shape = x1.sizes().vec();
  out_shape.back() = transpose2 ? x2.size(0) : x2.size(1);
  return at::empty(out_shape, x1.options().dtype(output_dtype));
}

}  // namespace

at::Tensor quant_matmul(const at::Tensor& x1,
                        const at::Tensor& x2,
                        const bool transpose2,
                        const at::Tensor& scale,
                        const c10::optional<at::Tensor>& offset,
                        const c10::optional<at::Tensor>& pertoken_scale,
                        const c10::optional<at::Tensor>& bias,
                        c10::optional<at::ScalarType> output_dtype) {
  const at::Tensor& offset_real = offset.value_or(at::Tensor());
  const at::Tensor& pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
  const at::Tensor& bias_real = bias.value_or(at::Tensor());
  const bool transpose1 = false;
  const at::ScalarType out_dtype = output_dtype.value_or(at::kChar);

  at::Tensor result =
      construct_quant_matmul_output_tensor(x1, x2, out_dtype, transpose2);
  EXEC_NPU_CMD(aclnnQuantMatmulV4,
               x1,
               x2,
               scale,
               offset_real,
               pertoken_scale_real,
               bias_real,
               transpose1,
               transpose2,
               result);
  return result;
}

}  // namespace xllm::kernel::npu
