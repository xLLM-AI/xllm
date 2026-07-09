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

#pragma once

#include <torch/torch.h>

#include <optional>

namespace xllm {
namespace kernel {
namespace dcu {
namespace hipblaslt_fp8 {

// Dense FP8 NT GEMM through hipBLASLt outer-vector channelwise scaling:
//   output = (activation * activation_scale) @
//            (weight * weight_scale).transpose(0, 1)
//
// activation:       [M, K] FP8 contiguous
// weight:           [N, K] FP8 contiguous
// activation_scale: scalar, [1], [M], or [M,1] float32
// weight_scale:     scalar, [1], [N], or [N,1] float32
// output:           optional [M, N] fp16/bf16 contiguous
torch::Tensor fp8_gemm_nt(const torch::Tensor& activation,
                          const torch::Tensor& weight,
                          const torch::Tensor& activation_scale,
                          const torch::Tensor& weight_scale,
                          torch::ScalarType output_dtype,
                          const std::optional<torch::Tensor>& bias,
                          const std::optional<torch::Tensor>& output);

}  // namespace hipblaslt_fp8
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
