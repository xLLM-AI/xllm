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
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm90.cu

#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

// Quantized GEMM operations using CUTLASS 3.x API for sm90a (Hopper).

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90

namespace xllm::kernel::cuda {

void cutlass_scaled_mm_sm90(torch::Tensor& c,
                            torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias) {
  dispatch_scaled_mm(c,
                     a,
                     b,
                     a_scales,
                     b_scales,
                     bias,
                     cutlass_scaled_mm_sm90_fp8,
                     nullptr,   // int8 not supported
                     nullptr);  // blockwise not supported
}

}  // namespace xllm::kernel::cuda

#endif
