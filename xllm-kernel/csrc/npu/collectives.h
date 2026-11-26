/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>

namespace xllm::kernel::npu {

// Submit on the current NPU stream. Buffers must be dense, contiguous, nonempty
// and in ND storage format. The caller owns the communicator (encoded as an
// integer for Torch) and buffers, and must retain them through graph replay.
// ACLGraph capture requires HCCL_OP_EXPANSION_MODE=AIV.
// In-place SUM, preserving dtype.
void all_reduce_on_current_stream(torch::Tensor& input, int64_t comm);

// Out-of-place, rank-ordered concatenation: output.numel = W * input.numel.
void all_gather_on_current_stream(const torch::Tensor& input,
                                  torch::Tensor& output,
                                  int64_t comm);

// Out-of-place SUM of rank-ordered blocks: input.numel = W * output.numel.
void reduce_scatter_on_current_stream(const torch::Tensor& input,
                                      torch::Tensor& output,
                                      int64_t comm);

}  // namespace xllm::kernel::npu
