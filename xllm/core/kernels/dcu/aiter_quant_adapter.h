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

// Thin adapter around AITER's xLLM-facing quantization symbols.

#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>

namespace xllm {
namespace kernel {
namespace dcu {
namespace aiter {

// Dynamic per-token FP8 activation quantization.
//
// input:  [M, K] fp32/fp16/bf16 contiguous
// output: optional [M, K] fp8 contiguous
// scale:  returns [M, 1] fp32 dequant scale
std::tuple<torch::Tensor, torch::Tensor> per_token_quant_fp8(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output);

}  // namespace aiter
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
