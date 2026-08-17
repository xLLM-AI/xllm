/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kernels/npu/paged_attention_tiling_layout.h"

namespace xllm::kernel::npu::tilelang {

// Public TileLang kernel APIs exported to the xLLM NPU runtime.

// Take the first token from each row of an existing row-major int32 verify
// buffer and pack it with `spec_width - 1` proposer columns into graph-owned
// row-major int32 storage on the current NPU stream. `spec_width` equals
// draft_tokens.size() + 1; compiled variants currently cover widths 4, 5 and 6
// (MTP depths 3, 4 and 5). Each draft source is contiguous int64 [batch], and
// persistent_tokens is updated in place. The batch-1 variant performs an
// eight-int32 vector store and clears its unused tail, so the destination must
// provide at least eight elements.
void spec_verify_token_update(const torch::Tensor& base_tokens,
                              const std::vector<torch::Tensor>& draft_tokens,
                              torch::Tensor& persistent_tokens,
                              int64_t spec_width);

bool has_spec_verify_token_update_specialization(int64_t spec_width);

bool has_spec_verify_attention_tiling_update_specialization(int64_t spec_width,
                                                            int64_t block_size);

bool has_spec_verify_graph_update_specialization(int64_t spec_width,
                                                 int64_t block_size);

// Update only replay-varying fields in graph-owned CustomPagedAttention tiling
// storage on the current NPU stream: max KV length, KV split length, and
// per-row KV lengths. `layout` must describe a recognized ATB tiling schema;
// `spec_width` equals the MTP depth plus one.
void spec_verify_attention_tiling_update(
    const torch::Tensor& src_kv_seq_lens,
    torch::Tensor& tiling_data,
    const PagedAttentionTilingLayout& layout,
    int64_t spec_width,
    int64_t block_size,
    int64_t max_kv_seq_len,
    int64_t kv_split_core_count);

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

// CausalConv1D decode kernel for batched single-token decode on NPU.
//
// Expects tensors in kernel-native layout:
//   x:            [batch, dim]                     contiguous, float16
//   conv_state:   [cache_lines, state_len, dim]    contiguous, float16
//                (in/out: last state_len positions written back)
//   weight:       [width, dim]                     contiguous, float16
//   bias:         [dim]                            contiguous, float16
//   init_indices: [batch]                          contiguous, int32
//   current_indices: [batch]                       contiguous, int32
//   initial_state_mode: [batch]                    contiguous, int32
//
// All batches must have seqlen=1 (one token per batch).
// Returns y: [batch, dim] float16.
torch::Tensor causal_conv1d_decode(torch::Tensor& conv_state,
                                   const torch::Tensor& x,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& bias,
                                   const torch::Tensor& init_indices,
                                   const torch::Tensor& current_indices,
                                   const torch::Tensor& initial_state_mode,
                                   bool has_silu);

// Check whether a decode kernel specialization exists for the given
// batch_size, dim, and has_silu. Returns false if no compiled variant matches,
// in which case callers should fall back to the per-batch prefill kernel.
bool has_causal_conv1d_decode_specialization(int64_t batch_size,
                                             int64_t dim,
                                             bool has_silu);

// CausalConv1D general kernel for variable-length batches on NPU.
//
// Expects tensors in kernel-native layout:
//   x:            [total_tokens, dim]               contiguous, fp16
//   conv_state:   [cache_lines, state_len, dim]     contiguous, fp16
//   weight:       [width, dim]                      contiguous, fp16
//   bias:         [dim]                             contiguous, fp16
//   cu_seqlens:   [batch+1]                         contiguous, int32
//   init_indices: [batch]                           contiguous, int32
//   current_indices: [batch]                        contiguous, int32
//   initial_state_mode: [batch]                     contiguous, int32
//
// Returns y: [total_tokens, dim] fp16.
torch::Tensor causal_conv1d(torch::Tensor& conv_state,
                            const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& bias,
                            const torch::Tensor& cu_seqlens,
                            const torch::Tensor& init_indices,
                            const torch::Tensor& current_indices,
                            const torch::Tensor& initial_state_mode,
                            bool has_silu);

// Run fused sigmoid-gating delta-rule SSM scan on NPU.
// Returns (out, final_state).
//   out: [T_padded, nv, dv] (padded token dim; caller strips padding)
//   final_state: [num_seqs, nv, dk, dv], float32 accumulator state
// Invalid inputs trigger CHECK failures.
std::tuple<torch::Tensor, torch::Tensor> fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    float softplus_beta,
    float softplus_threshold);

}  // namespace xllm::kernel::npu::tilelang
