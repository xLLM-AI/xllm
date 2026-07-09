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

// Thin adapter around AITER's xLLM-facing MoE FP8 symbols.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace xllm {
namespace kernel {
namespace dcu {
namespace aiter {

struct MoeSortedTokens {
  torch::Tensor sorted_token_ids;
  torch::Tensor sorted_weights;
  torch::Tensor expert_ids;
  torch::Tensor tokens_positions_per_expert;
  torch::Tensor num_tokens_post_pad;
};

torch::Tensor moe_layout_shuffle_gemm1(const torch::Tensor& weight);
torch::Tensor moe_layout_shuffle_gemm2(const torch::Tensor& weight);

int64_t select_moe_gemm1_mode(int64_t num_tokens);
int64_t select_moe_gemm2_mode(int64_t num_tokens);
int64_t select_moe_m_key(int64_t num_tokens);

MoeSortedTokens moe_align_block_size(const torch::Tensor& topk_ids,
                                     const torch::Tensor& topk_weights,
                                     int64_t num_experts,
                                     int64_t block_size);

torch::Tensor moe_gemm_fp8_channelwise(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& topk_weights,
    const MoeSortedTokens& sorted_tokens,
    int64_t topk,
    int64_t mode,
    int64_t delta,
    int64_t selected_m);

}  // namespace aiter
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
