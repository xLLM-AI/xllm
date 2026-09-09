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
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8.cu

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_fp8_dispatch.cuh"

namespace xllm {
namespace kernel {
namespace cuda {

void cutlass_scaled_mm_sm90_fp8(torch::Tensor& out,
                                torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ",
                out.dtype());
    return cutlass_scaled_mm_sm90_fp8_epilogue<true>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm90_fp8_epilogue<false>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
