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

#include <torch/types.h>

namespace xllm::kernel::cuda {

// In-place temperature, min-p, top-k and top-p transforms. Row parameters
// may be undefined; otherwise they are contiguous device vectors with one
// entry per row. Logits have contiguous columns and nonoverlapping rows.
// Top-k is int64; temperature/min-p/top-p are FP32, FP16 or BF16,
// independently of logits.
void fused_logits_transform(torch::Tensor& logits,
                            const torch::Tensor& temperatures,
                            const torch::Tensor& min_p,
                            const torch::Tensor& top_k,
                            const torch::Tensor& top_p);

}  // namespace xllm::kernel::cuda
