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
#include <tuple>
#include <utility>
#include <vector>

namespace xllm::kernel::npu::tilelang {

// Public TileLang kernel APIs exported to the xLLM NPU runtime.
//
// Return whether the packed allowed-token mask can use the TileLang fast path.
bool can_apply_token_bitmask_inplace(const torch::Tensor& logits,
                                     const torch::Tensor& bitmask);

// Apply [rows, ceil(vocab / 32)] packed allowed-token words to logits in place.
// Invalid inputs trigger CHECK failures; callers should use the predicate when
// an ATen fallback is available.
void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask);

// Apply TileLang RoPE kernel in-place on a single input tensor.
// Invalid inputs trigger CHECK failures.
// Supports input not contiguous, with stride.
void rope_in_place(torch::Tensor& input,
                   const torch::Tensor& sin_cache,
                   const torch::Tensor& cos_cache);

// Compute fused GDN gating outputs on NPU.
// Invalid inputs trigger CHECK failures.
std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float softplus_beta,
    float softplus_threshold);

// Build merged mRoPE gather offsets for split_qkv_rmsnorm_mrope.
torch::Tensor build_split_qkv_rmsnorm_mrope_gather_pattern(
    int64_t rope_dim,
    const std::vector<int64_t>& mrope_section,
    bool is_interleaved,
    const torch::Device& device);

// Split fused [Q|K|V|G], apply q/k RMSNorm + mRoPE, and return
// (q, k, v, gate) on NPU.
//
// qkvg: [T, q_size + kv_size + kv_size + q_size] in Q | K | V | G layout.
// cos_sin: [T, 3 * rope_dim] with row layout
// [t_cos|t_sin|h_cos|h_sin|w_cos|w_sin].
// gather_pattern: merged gather offsets built by
// build_split_qkv_rmsnorm_mrope_gather_pattern(...).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
split_qkv_rmsnorm_mrope(const torch::Tensor& qkvg,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin,
                        const torch::Tensor& gather_pattern,
                        float eps,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_size);

bool has_split_qkv_rmsnorm_mrope_specialization(int64_t num_q_heads,
                                                int64_t num_kv_heads,
                                                int64_t head_size);

// Compute chunk_gated_delta_rule forward pass for hidden state on NPU.
// Returns (h, v_new, final_state).
//   h: [N, NT_max, H, K, V] (flattened per-sequence chunks)
//   v_new: [T_total, H, V] (updated values)
//   final_state: [N, H, K, V] (optional, empty tensor if !output_final_state)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
chunk_gated_delta_rule_fwd_h(const torch::Tensor& k,
                             const torch::Tensor& w,
                             const torch::Tensor& u,
                             const std::optional<torch::Tensor>& g,
                             const std::optional<torch::Tensor>& initial_state,
                             bool output_final_state,
                             int64_t chunk_size,
                             bool save_new_value,
                             const std::optional<torch::Tensor>& cu_seqlens,
                             const std::optional<torch::Tensor>& chunk_offsets);

}  // namespace xllm::kernel::npu::tilelang
