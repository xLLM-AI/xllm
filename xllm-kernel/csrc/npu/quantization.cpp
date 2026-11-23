// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch/library.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <cstdint>
#include <optional>
#include <tuple>

#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

constexpr int64_t kInt4ValuesPerWord = 8;

torch::Tensor quantize_int4(const torch::Tensor& value,
                            const torch::Tensor& scales,
                            const torch::Tensor& zero_points,
                            int64_t axis) {
  CHECK_GE(value.dim(), 1);
  auto shape = value.sizes().vec();
  CHECK_EQ(shape.back() % kInt4ValuesPerWord, 0);
  shape.back() /= kInt4ValuesPerWord;

  const auto options = value.options().dtype(torch::kInt32);
  auto output = at_npu::native::get_npu_format(value) == ACL_FORMAT_FRACTAL_NZ
                    ? at_npu::native::empty_with_format(
                          shape, options, ACL_FORMAT_FRACTAL_NZ, true)
                    : torch::empty(shape, options);
  constexpr const char* kRoundMode = "round";
  constexpr bool kSqrtMode = false;
  static const bool has_v3 =
      GetOpApiFuncAddr("aclnnAscendQuantV3") &&
      GetOpApiFuncAddr("aclnnAscendQuantV3GetWorkspaceSize");
  if (has_v3) {
    // Preserve the last-axis default of the legacy packed INT4 path.
    const int32_t quant_axis = axis < -1 ? static_cast<int32_t>(axis) : -1;
    EXEC_NPU_CMD_EXT(aclnnAscendQuantV3,
                     value,
                     scales,
                     zero_points,
                     kSqrtMode,
                     kRoundMode,
                     torch::kInt32,
                     quant_axis,
                     output);
  } else {
    EXEC_NPU_CMD_EXT(aclnnAscendQuant,
                     value,
                     scales,
                     zero_points,
                     kSqrtMode,
                     kRoundMode,
                     torch::kInt32,
                     output);
  }
  return output;
}

torch::Tensor quantize_per_tensor(const torch::Tensor& value,
                                  const torch::Tensor& scales,
                                  const torch::Tensor& zero_points,
                                  torch::ScalarType dtype,
                                  int64_t axis) {
  const torch::DeviceGuard guard(value.device());
  if (dtype == torch::ScalarType::QUInt4x2) {
    return quantize_int4(value, scales, zero_points, axis);
  }

  torch::ScalarType output_dtype = torch::kInt32;
  if (dtype == torch::ScalarType::QInt8) {
    output_dtype = torch::kInt8;
  } else if (dtype == torch::ScalarType::QUInt8) {
    output_dtype = torch::kUInt8;
  }
  auto output =
      torch::empty(value.sizes(), value.options().dtype(output_dtype));
  const int32_t quant_axis = static_cast<int32_t>(axis);
  EXEC_NPU_CMD_EXT(aclnnQuantize,
                   value,
                   scales,
                   zero_points,
                   output_dtype,
                   quant_axis,
                   output);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant(
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& smooth_scales,
    const std::optional<torch::Tensor>& group_index,
    std::optional<torch::ScalarType> dst_type) {
  const torch::ScalarType dtype = dst_type.value_or(torch::kInt8);
  const bool packed_int4 = dtype == torch::ScalarType::QUInt4x2;
  CHECK(dtype == torch::kInt8 || packed_int4);
  CHECK_GE(value.dim(), 1);

  const torch::DeviceGuard guard(value.device());
  auto shape = value.sizes().vec();
  auto scale = torch::empty(value.sizes().slice(0, value.dim() - 1),
                            value.options().dtype(torch::kFloat32));
  if (packed_int4) {
    CHECK_EQ(shape.back() % kInt4ValuesPerWord, 0);
    shape.back() /= kInt4ValuesPerWord;
  }
  const torch::ScalarType output_dtype =
      packed_int4 ? torch::kInt32 : torch::kInt8;
  auto output = torch::empty(shape, value.options().dtype(output_dtype));
  const std::optional<torch::Tensor> offset;
  // DynamicQuantV2 takes a GE dtype ID. Packed INT4 uses DT_INT32 storage.
  constexpr int64_t kGeInt32 = 3;
  constexpr int64_t kGeInt8 = 2;
  const int64_t ge_dtype = packed_int4 ? kGeInt32 : kGeInt8;
  EXEC_NPU_CMD_EXT(aclnnDynamicQuantV2,
                   value,
                   smooth_scales,
                   group_index,
                   ge_dtype,
                   output,
                   scale,
                   offset);
  return {output, scale};
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "quantize_per_tensor(Tensor value, Tensor scales, Tensor zero_points, "
      "ScalarType dtype, int axis) -> Tensor");
  m.def(
      "dynamic_quant(Tensor value, Tensor? smooth_scales, Tensor? group_index, "
      "ScalarType? dst_type) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("quantize_per_tensor", TORCH_FN(quantize_per_tensor));
  m.impl("dynamic_quant", TORCH_FN(dynamic_quant));
}

}  // namespace
}  // namespace xllm_kernel
