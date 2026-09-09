/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_helper.hpp

#include <torch/all.h>

#include "cutlass_extensions/common.hpp"
#include "kernels/cuda/utils.h"

namespace xllm {
namespace kernel {
namespace cuda {

template <typename Fp8Func, typename Int8Func, typename BlockwiseFunc>
void dispatch_scaled_mm(torch::Tensor& c,
                        torch::Tensor const& a,
                        torch::Tensor const& b,
                        torch::Tensor const& a_scales,
                        torch::Tensor const& b_scales,
                        std::optional<torch::Tensor> const& bias,
                        Fp8Func fp8_func,
                        Int8Func int8_func,
                        BlockwiseFunc blockwise_func) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  int M = a.size(0), N = b.size(1), K = a.size(1);

  if ((a_scales.numel() == 1 || a_scales.numel() == a.size(0)) &&
      (b_scales.numel() == 1 || b_scales.numel() == b.size(1))) {
    // Standard per-tensor/per-token/per-channel scaling
    TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
    if (a.dtype() == torch::kFloat8_e4m3fn) {
      fp8_func(c, a, b, a_scales, b_scales, bias);
    } else {
      TORCH_CHECK(a.dtype() == torch::kInt8);
      if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
        int8_func(c, a, b, a_scales, b_scales, bias);
      } else {
        int32_t version_num = get_sm_version_num();
        TORCH_CHECK(
            false,
            "Int8 not supported on SM",
            version_num,
            ". Use FP8 quantization instead, or run on older arch (SM < 100).");
      }
    }
  } else {
    TORCH_CHECK(a_scales.dim() == 2, "a scale must be 2d tensor.");
    TORCH_CHECK(b_scales.dim() == 2, "b scale must be 2d tensor.");
    if constexpr (!std::is_same_v<BlockwiseFunc, std::nullptr_t>) {
      int32_t version_num = get_sm_version_num();
      if (version_num >= 90) {
        TORCH_CHECK(a.size(0) == a_scales.size(0) &&
                        ceil_div(a.size(1), int64_t(128)) == a_scales.size(1),
                    "a_scale_group_shape must be [1, 128].");
        TORCH_CHECK(ceil_div(b.size(0), int64_t(128)) == b_scales.size(0) &&
                        ceil_div(b.size(1), int64_t(128)) == b_scales.size(1),
                    "b_scale_group_shape must be [128, 128].");
      }

      TORCH_CHECK(!bias, "Bias not yet supported blockwise scaled_mm");
      blockwise_func(c, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(
          false,
          "Blockwise scaling is not supported. "
          "Only per-tensor, per-token, or per-channel scaling is supported.");
    }
  }
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
