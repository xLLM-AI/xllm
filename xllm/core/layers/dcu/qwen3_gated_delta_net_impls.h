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

// Common backend interface for Qwen3 gated-delta-net operators on DCU.
//
// Two implementations live behind identical function signatures:
//   * torch_impl::  — pure-torch fallback (portable, no external kernels).
//   * kernel_impl:: — optimized HIP kernels (aiter fla + causal-conv1d).
//
// The layer's forward() chooses between them via runtime env flags
// (XLLM_DCU_USE_TORCH_CONV1D=1, XLLM_DCU_USE_TORCH_RECURRENT=1). All shared
// tensor-shape conventions are documented once, here.

#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

namespace xllm {
namespace layer {
namespace qwen3_gdn {

// ---------------------------------------------------------------------------
// causal_conv1d
// ---------------------------------------------------------------------------
// Prefill (varlen, packed):
//   flat_input     : [T_total, C] contiguous, same dtype as conv_cache/weight.
//   conv_weight_2d : [C, K] contiguous.
//   conv_cache     : [num_slots, K-1, C] in place — one row per real seq
//                    (matching state_indices below).
//   cu_seqlens     : N+1 int64 prefix sums (only real seqs; caller filters
//                    zero-length padding rows).
//   state_indices  : N int64 slot ids in conv_cache for the same N real seqs.
//   kernel_size    : K
//   activation     : true => silu.
// Returns [T_total, C] output with silu applied.
using CausalConv1dPrefillFn =
    torch::Tensor (*)(const torch::Tensor& flat_input,
                      const torch::Tensor& conv_weight_2d,
                      torch::Tensor& conv_cache,
                      const std::vector<int64_t>& cu_seqlens,
                      const std::vector<int64_t>& state_indices,
                      int32_t kernel_size,
                      bool activation);

// Decode (per-token):
//   flat_input    : [B, C] contiguous.
//   conv_weight_2d: [C, K].
//   conv_cache    : [num_slots, K-1, C] updated in place.
//   state_indices : [B] int32/int64 device tensor of slot ids.
// Returns [B, C] output with silu applied.
using CausalConv1dDecodeFn =
    torch::Tensor (*)(const torch::Tensor& flat_input,
                      const torch::Tensor& conv_weight_2d,
                      torch::Tensor& conv_cache,
                      const torch::Tensor& state_indices,
                      int32_t kernel_size,
                      bool activation);

// ---------------------------------------------------------------------------
// gated_delta_rule
// ---------------------------------------------------------------------------
// Shared for both prefill (long T) and decode (T=1):
//   q, k     : [1, T, Hk, K] bf16, before optional l2norm.
//   v        : [1, T, Hv, V] bf16.
//   g        : [1, T, Hv]    log-gate (bf16 or fp32 accepted).
//   beta     : [1, T, Hv].
//   initial_state (optional): [N, Hv, K, V] fp32 (fla layout).
// Returns:
//   core_attn_out : [1, T, Hv, V] in q's dtype.
//   last_state    : [N, Hv, K, V] fp32 (fla layout) — final state per seq.

namespace torch_impl {
torch::Tensor causal_conv1d_prefill(const torch::Tensor& flat_input,
                                    const torch::Tensor& conv_weight_2d,
                                    torch::Tensor& conv_cache,
                                    const std::vector<int64_t>& cu_seqlens,
                                    const std::vector<int64_t>& state_indices,
                                    int32_t kernel_size,
                                    bool activation);

torch::Tensor causal_conv1d_decode(const torch::Tensor& flat_input,
                                   const torch::Tensor& conv_weight_2d,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& state_indices,
                                   int32_t kernel_size,
                                   bool activation);

// Naive per-token recurrent; used for both prefill and decode.
std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true);
}  // namespace torch_impl

namespace kernel_impl {
torch::Tensor causal_conv1d_prefill(const torch::Tensor& flat_input,
                                    const torch::Tensor& conv_weight_2d,
                                    torch::Tensor& conv_cache,
                                    const std::vector<int64_t>& cu_seqlens,
                                    const std::vector<int64_t>& state_indices,
                                    int32_t kernel_size,
                                    bool activation);

torch::Tensor causal_conv1d_decode(const torch::Tensor& flat_input,
                                   const torch::Tensor& conv_weight_2d,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& state_indices,
                                   int32_t kernel_size,
                                   bool activation);

// Chunked WY decomposition + aiter h-recurrence + chunk_fwd_o. Prefill only —
// decode always uses torch_impl::gated_delta_rule since its per-token cost is
// already the bottleneck-free case.
std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule_prefill(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    const std::vector<int64_t>& cu_seqlens,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true);
}  // namespace kernel_impl

}  // namespace qwen3_gdn
}  // namespace layer
}  // namespace xllm
