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

#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kStaticQuantMode = 0;
constexpr int64_t kDynamicQuantMode = 1;
constexpr int64_t kDstTypeInt8 = 2;
constexpr int64_t kActivateDimDefault = -1;
constexpr int64_t kSwigluModeDefault = 0;
constexpr double kClampLimitDefault = 7.0;
constexpr double kGluAlphaDefault = 1.702;
constexpr double kGluBiasDefault = 1.0;

std::tuple<at::Tensor, at::Tensor> construct_dequant_swiglu_quant_output_tensor(
    const at::Tensor& x) {
  TORCH_CHECK(x.dim() > 1, "x dim should be larger than 1.");
  TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even.");

  at::SmallVector<int64_t, op_infer::SIZE> y_size;
  at::SmallVector<int64_t, op_infer::SIZE> scale_size;
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    y_size.push_back(x.size(i));
    scale_size.push_back(x.size(i));
  }
  y_size.push_back(x.size(x.dim() - 1) / 2);

  at::Tensor y = at::empty(y_size, x.options().dtype(at::kChar));
  at::Tensor scale = at::empty(scale_size, x.options().dtype(at::kFloat));
  return std::make_tuple(y, scale);
}

bool is_dequant_swiglu_quant_v2_available() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnDequantSwigluQuantV2") !=
          nullptr;
  return is_available;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> dequant_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode) {
  TORCH_CHECK(quant_mode == kStaticQuantMode || quant_mode == kDynamicQuantMode,
              "quant_mode only supports 0(static) or 1(dynamic), but got ",
              quant_mode,
              ".");

  auto [y, scale] = construct_dequant_swiglu_quant_output_tensor(x);
  std::string quant_mode_str =
      (quant_mode == kDynamicQuantMode) ? "dynamic" : "static";
  char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

  if (is_dequant_swiglu_quant_v2_available()) {
    std::string round_mode_str = "rint";
    char* round_mode_ptr = round_mode_str.data();

    EXEC_NPU_CMD(aclnnDequantSwigluQuantV2,
                 x,
                 weight_scale,
                 activation_scale,
                 bias,
                 quant_scale,
                 quant_offset,
                 group_index,
                 activate_left,
                 quant_mode_ptr,
                 kDstTypeInt8,
                 round_mode_ptr,
                 kActivateDimDefault,
                 kSwigluModeDefault,
                 kClampLimitDefault,
                 kGluAlphaDefault,
                 kGluBiasDefault,
                 y,
                 scale);
  } else {
    EXEC_NPU_CMD(aclnnDequantSwigluQuant,
                 x,
                 weight_scale,
                 activation_scale,
                 bias,
                 quant_scale,
                 quant_offset,
                 group_index,
                 activate_left,
                 quant_mode_ptr,
                 y,
                 scale);
  }

  return std::make_tuple(y, scale);
}

}  // namespace xllm::kernel::npu
