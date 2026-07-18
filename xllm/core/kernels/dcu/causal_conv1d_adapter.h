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

#include <vector>

namespace xllm {
namespace kernel {
namespace dcu {

// Varlen (packed) causal conv1d forward for prefill.
//
// x_tc         : [T_total, C]              bf16/fp16/fp32, contiguous
// weight_ck    : [C, K]                    same dtype
// cu_seqlens   : N+1 host vec, prefix sums (only >0 seqs, no padding rows)
// state_indices: N host vec of ssm-cache slot ids for each real sequence
// conv_cache   : [num_slots, K-1, C]       (xllm layout) — updated in place
// activation   : true = silu
//
// Returns out_tc: [T_total, C] same dtype as x_tc.
torch::Tensor causal_conv1d_varlen_fwd(
    const torch::Tensor& x_tc,
    const torch::Tensor& weight_ck,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& state_indices,
    torch::Tensor& conv_cache,
    bool activation);

// Decode single-step update.
//
// x_bc            : [B, C]           bf16/fp16/fp32
// weight_ck       : [C, K]           same dtype
// state_indices   : [B]              int32 device tensor of ssm-cache slot ids
// conv_cache      : [num_slots, K-1, C]  updated in place
// activation      : true = silu
//
// Returns out_bc: [B, C].
torch::Tensor causal_conv1d_update(const torch::Tensor& x_bc,
                                   const torch::Tensor& weight_ck,
                                   const torch::Tensor& state_indices,
                                   torch::Tensor& conv_cache,
                                   bool activation);

}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
