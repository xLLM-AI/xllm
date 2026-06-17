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

constexpr int64_t kInt4NumsInInt32Space = 8;

int64_t get_tensor_npu_format(const at::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

at::Tensor apply_int4_quantize(const at::Tensor& self,
                               const at::Tensor& scales,
                               const at::Tensor& zero_points,
                               int64_t axis) {
  TORCH_CHECK(self.dim() >= 1, "input dim must be >= 1 for int4 quantize.");

  auto output_shape = self.sizes().vec();
  const int64_t last_dim = self.dim() - 1;
  TORCH_CHECK(output_shape[last_dim] % kInt4NumsInInt32Space == 0,
              "input shape last dim must be divisible by 8 for int4 "
              "quantize, got ",
              output_shape[last_dim]);
  output_shape[last_dim] /= kInt4NumsInInt32Space;

  const auto output_options = self.options().dtype(at::kInt);
  at::Tensor result;
  const int64_t npu_format = get_tensor_npu_format(self);
  if (npu_format == ACL_FORMAT_FRACTAL_NZ) {
    result = at_npu::native::OpPreparation::apply_tensor_with_format(
        output_shape, output_options, ACL_FORMAT_FRACTAL_NZ, true);
  } else {
    result = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_shape, output_options);
  }

  const bool sqrt_mode = false;
  const c10::optional<at::Tensor> zero_points_opt =
      zero_points.defined() ? c10::optional<at::Tensor>(zero_points)
                            : c10::nullopt;
  // Keep the same behavior as torch_npu.npu_quantize(..., div_mode=false):
  // if AscendQuantV3 exists, force axis to -1 unless the caller passed a value
  // lower than -1.
  const int64_t ascend_quant_axis = axis < -1 ? axis : -1;
  static const bool has_ascend_quant_v3 =
      aclnn::detail::get_op_api_func_addr("aclnnAscendQuantV3") != nullptr &&
      aclnn::detail::get_op_api_func_addr(
          "aclnnAscendQuantV3GetWorkspaceSize") != nullptr;
  std::string quant_model = "round";
  if (has_ascend_quant_v3) {
    EXEC_NPU_CMD(aclnnAscendQuantV3,
                 self,
                 scales,
                 zero_points_opt,
                 sqrt_mode,
                 quant_model,
                 at::kInt,
                 ascend_quant_axis,
                 result);
  } else {
    EXEC_NPU_CMD(aclnnAscendQuant,
                 self,
                 scales,
                 zero_points_opt,
                 sqrt_mode,
                 quant_model,
                 at::kInt,
                 result);
  }
  return result;
}

}  // namespace

at::Tensor quantize_per_tensor(const at::Tensor& self,
                               const at::Tensor& scales,
                               const at::Tensor& zero_points,
                               at::ScalarType dtype,
                               int64_t axis) {
  if (dtype == at::ScalarType::QUInt4x2) {
    return apply_int4_quantize(self, scales, zero_points, axis);
  }

  at::ScalarType output_dtype = at::kInt;
  if (dtype == at::ScalarType::QInt8) {
    output_dtype = at::kChar;
  } else if (dtype == at::ScalarType::QUInt8) {
    output_dtype = at::kByte;
  } else if (dtype == at::ScalarType::QInt32) {
    output_dtype = at::kInt;
  }

  at::Tensor result =
      at::empty(self.sizes().vec(), self.options().dtype(output_dtype));
  EXEC_NPU_CMD(
      aclnnQuantize, self, scales, zero_points, output_dtype, axis, result);
  return result;
}

}  // namespace xllm::kernel::npu
