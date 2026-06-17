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

const int64_t INT4_IN_INT32_NUM = 8;
// Substitute INT32_OUTPUT_TYPE for ge::DataType::DT_INT32, INT8_OUTPUT_TYPE
// for ge::DataType::DT_INT8.
constexpr int64_t INT32_OUTPUT_TYPE = 3;
constexpr int64_t INT8_OUTPUT_TYPE = 2;

std::tuple<at::Tensor, c10::optional<at::Tensor>> dynamic_quant(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& group_index,
    c10::optional<at::ScalarType> dst_type) {
  const auto quant_dtype = dst_type.value_or(at::kChar);
  TORCH_CHECK(
      quant_dtype == at::kChar || quant_dtype == at::ScalarType::QUInt4x2,
      "dst_type must be torch.int8 or torch.quint4x2.");
  TORCH_CHECK(input.dim() >= 1, "input dim must be >= 1.");

  at::SmallVector<int64_t, op_infer::SIZE> scale_size;
  const auto scale_dim = input.dim() - 1;
  for (int64_t index = 0; index < scale_dim; ++index) {
    scale_size.push_back(input.size(index));
  }

  at::Tensor scale = at::empty(scale_size, input.options().dtype(at::kFloat));
  c10::optional<at::Tensor> offset;
  // default quant type: Int8
  int64_t output_type = INT8_OUTPUT_TYPE;
  at::Tensor output;
  if (quant_dtype == at::ScalarType::QUInt4x2) {
    const auto last_dim = input.size(scale_dim);
    TORCH_CHECK(last_dim % INT4_IN_INT32_NUM == 0,
                "input last dim must be divisible by 8 for int4 quantization.");
    auto output_size = scale_size;
    output_size.push_back(last_dim / INT4_IN_INT32_NUM);
    output_type = INT32_OUTPUT_TYPE;
    output = at::empty(output_size, input.options().dtype(at::kInt));
  } else {
    output = at::empty(input.sizes().vec(), input.options().dtype(at::kChar));
  }

  EXEC_NPU_CMD(aclnnDynamicQuantV2,
               input,
               smooth_scales,
               group_index,
               output_type,
               output,
               scale,
               offset);
  return std::make_tuple(output, c10::optional<at::Tensor>(scale));
}

}  // namespace xllm::kernel::npu
