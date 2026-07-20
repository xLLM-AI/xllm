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

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

// #include "ATen/Tensor.h"

namespace xllm::kernel::dcu {

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

// MoE group GEMM via CK Tile grouped_gemm_tileloop.
//
// Computes a batched per-expert GEMM:
//   for each expert e:
//     C[e] = A[t_start:t_end, :] * B[e, :, :]^T
//
// Parameters:
//   input       — [total_tokens, K]  contiguous, fp16 or bf16
//   weight      — [num_experts, N, K] contiguous, same dtype as input
//   token_count — [num_experts] int32, number of tokens per expert
//   output      — optional [total_tokens, N] contiguous; allocated if omitted
//
// Returns output tensor.
torch::Tensor group_gemm(const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& token_count,
                         std::optional<torch::Tensor> output = std::nullopt);

// Build a 2D block_table [B, max_pages] int32 from CSR-format paged KV
// metadata.
//
// paged_kv_indptr:  [B + 1]       int32  — cumulative page offsets per sequence
// paged_kv_indices: [total_pages]  int32  — flattened page (block) IDs
//
// Returns block_table of shape [B, total_pages] filled with page IDs and -1 for
// padding. total_pages (= indices.size(0)) is the worst-case row width, read
// from tensor metadata with zero GPU sync. The flash attention kernel only
// accesses up to the per-sequence page count, so extra -1 columns are never
// touched.
//
// Implementation:
//   - Pre-allocates [B, total_pages] and initializes to -1 via cudaMemsetAsync
//     (hardware fill engine, no kernel launch)
//   - Single kernel (one block per sequence) copies valid page IDs
torch::Tensor build_block_table_from_paged_kv_cuda(
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices);

// Gather DCU DeepSeek MLA latent cache from paged layout to a contiguous
// [total_kv_len, latent_dim] tensor for full-context prefill.
torch::Tensor gather_mla_latent_cache(const torch::Tensor& k_cache,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& kv_cu_seq_lens,
                                      int64_t total_kv_len,
                                      int64_t max_seq_len);

torch::Tensor random_sample(const torch::Tensor& probs);
torch::Tensor rejection_sample(const torch::Tensor& draft_token_ids,
                               const torch::Tensor& num_draft_tokens,
                               const torch::Tensor& cu_num_draft_tokens,
                               const std::optional<torch::Tensor>& draft_probs,
                               const torch::Tensor& target_probs,
                               const torch::Tensor& bonus_token_ids,
                               const torch::Tensor& uniform_rand,
                               const torch::Tensor& uniform_probs,
                               int64_t max_spec_len);

std::tuple<torch::Tensor, torch::Tensor> moe_grouped_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func,
    double routed_scaling_factor);

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func,
    double routed_scaling_factor);

// DCU W8A8 dynamic activation quantization.
// Current DCU implementation supports only no-smooth per-token INT8
// quantization and returns one fp32 scale per token row.
std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero,
    const std::optional<torch::Tensor>& token_count,
    const std::optional<torch::Tensor>& gather_index,
    const std::optional<torch::Tensor>& gather_index_start_position,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& output_scale,
    const std::string& act_mode,
    double active_coef,
    bool is_gated,
    torch::ScalarType quant_type);

// Torch fallback for gated RMSNorm / LayerNorm (with optional SiLU-gated
// residual). DCU does not have a fused kernel yet.
torch::Tensor gated_layer_norm(torch::Tensor& x,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               double eps,
                               const std::optional<torch::Tensor>& gate,
                               int64_t group_size,
                               bool norm_before_gate);

// Torch fallback for Gemma-style RMSNorm (weight is applied as `1 + gamma`).
// DCU does not have a fused kernel yet.
torch::Tensor gemma_rms_norm(const torch::Tensor& x,
                             const torch::Tensor& gamma,
                             double eps,
                             torch::Tensor& norm_out);

// W8A8: INT8 x INT8 scaled matmul via hipBLASLt.
// Equivalent to lmslim's hipblaslt_w8a8_gemm.
torch::Tensor scaled_matmul(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& a_scale,
                            const torch::Tensor& b_scale,
                            torch::ScalarType output_dtype,
                            const std::optional<torch::Tensor>& bias,
                            const std::optional<torch::Tensor>& c,
                            const std::string& act_mode,
                            int64_t quant_bit_size,
                            double alpha,
                            double beta,
                            bool use_hp_active,
                            int64_t a_quant_bit_size,
                            const std::optional<torch::Tensor>& a_calib,
                            const std::optional<torch::Tensor>& b_calib,
                            const std::optional<torch::Tensor>& output);

}  // namespace xllm::kernel::dcu
