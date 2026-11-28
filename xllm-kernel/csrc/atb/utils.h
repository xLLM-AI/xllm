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

#pragma once

#include <atb/types.h>
#include <c10/util/Logging.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <cstdint>

namespace xllm_kernel {

inline bool is_base_format(int64_t format) {
  return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
         format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW;
}

inline void check_base_format(const torch::Tensor& tensor) {
  CHECK(is_base_format(at_npu::native::get_npu_format(tensor)))
      << "MLA projection requires ND/base format";
}

inline void check_operands(const torch::Tensor& input,
                           const torch::Tensor& weight) {
  CHECK(input.device().type() == torch::kPrivateUse1)
      << "MLA projection requires NPU input";
  CHECK(input.device() == weight.device());
  CHECK(input.scalar_type() == torch::kBFloat16);
  CHECK(weight.scalar_type() == input.scalar_type());
  CHECK(input.dim() == 3) << "input must be [T,H,D]";
  CHECK(weight.dim() == 3) << "weight must be [H,D,O]";
  CHECK(input.size(0) > 0);
  CHECK(input.size(1) > 0);
  CHECK(input.size(2) > 0);
  CHECK(weight.size(2) > 0);
  CHECK(input.size(1) == weight.size(0));
  CHECK(input.size(2) == weight.size(1));
  CHECK(weight.is_contiguous());
  check_base_format(input);
  check_base_format(weight);
}

inline atb::Tensor to_atb_tensor(const torch::Tensor& tensor) {
  atb::Tensor result;
  result.desc.format = ACL_FORMAT_ND;
  result.desc.dtype = ACL_BF16;
  result.desc.shape.dimNum = 3;
  for (int32_t dimension = 0; dimension < 3; ++dimension) {
    result.desc.shape.dims[dimension] = tensor.size(dimension);
  }
  result.deviceData = tensor.data_ptr();
  result.dataSize = tensor.numel() * tensor.element_size();
  return result;
}

}  // namespace xllm_kernel
