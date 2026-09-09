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
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_kernels.hpp

#pragma once

#include <torch/all.h>

namespace xllm {
namespace kernel {
namespace cuda {

void cutlass_scaled_mm_sm90_fp8(torch::Tensor& out,
                                torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm100_fp8(torch::Tensor& out,
                                 torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm120_fp8(torch::Tensor& out,
                                 torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 std::optional<torch::Tensor> const& bias);

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
