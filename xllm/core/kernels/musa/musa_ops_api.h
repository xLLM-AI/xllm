/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

// Replaces negative dst entries with indexed values from src.
void replace_token(torch::Tensor& dst,
                   torch::Tensor& src,
                   bool synchronize_stream = true);

torch::Tensor moe_combine_result_indexed(const torch::Tensor& gemm2_sorted,
                                         const torch::Tensor& sorted_positions,
                                         const torch::Tensor& reduce_weight,
                                         int64_t num_tokens,
                                         int32_t top_k);

}  // namespace xllm::kernel::musa

namespace xllm::kernel::musa {

void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      bool is_neox);

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate);

void fused_shared_expert_gate_inplace(torch::Tensor& shared_output,
                                      const torch::Tensor& hidden_states,
                                      const torch::Tensor& gate_weight);

}  // namespace xllm::kernel::musa

namespace xllm::kernel::cuda {

void reshape_paged_cache(torch::Tensor slot_ids,
                         torch::Tensor keys,
                         torch::Tensor values,
                         torch::Tensor key_cache,
                         torch::Tensor value_cache);

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype);

}  // namespace xllm::kernel::cuda

namespace xllm::kernel::musa {

void batch_prefill(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& mask = std::nullopt);

void batch_prefill_non_causal(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& mask = std::nullopt);

void batch_chunked_prefill(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    bool causal = true,
    const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
    const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
    const torch::Tensor& paged_kv_last_page_len_host = torch::Tensor());

void batch_decode(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    bool use_tensor_core,
    std::optional<torch::Tensor> qo_indptr = std::nullopt,
    const torch::Tensor& paged_kv_indptr_host = torch::Tensor(),
    const torch::Tensor& paged_kv_indices_host = torch::Tensor(),
    const torch::Tensor& paged_kv_last_page_len_host = torch::Tensor());
void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse);

void fa3_prefill(const torch::Tensor& query,
                 const torch::Tensor& key,
                 const torch::Tensor& value,
                 const torch::Tensor& cu_seqlens_q,
                 const torch::Tensor& cu_seqlens_k,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 int64_t window_left,
                 int64_t window_right,
                 double sm_scale,
                 torch::Tensor& output,
                 torch::Tensor& output_lse);

// Builds metadata for paged-KV FA3 prefill.
torch::Tensor fa3_prefill_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k_new,
    const torch::Tensor& seqused_k);

void fa3_prefill_paged(const torch::Tensor& query,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& cu_seqlens_k_new,
                       const torch::Tensor& seqused_k,
                       const torch::Tensor& page_table,
                       const torch::Tensor& scheduler_metadata,
                       int64_t max_seqlen_q,
                       int64_t window_left,
                       int64_t window_right,
                       double sm_scale,
                       torch::Tensor& output,
                       torch::Tensor& output_lse);

torch::Tensor fa3_decode_scheduler_metadata(const torch::Device& device,
                                            int32_t batch_size,
                                            int32_t num_heads_q,
                                            int32_t num_heads_kv,
                                            int32_t head_dim_qk,
                                            int32_t head_dim_vo,
                                            int32_t max_seqlen_q,
                                            int32_t max_seqlen_k,
                                            int32_t window_size_left,
                                            int32_t window_size_right,
                                            const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_k);

}  // namespace xllm::kernel::musa

namespace xllm::kernel::cuda {

void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,
                        torch::Tensor& residual,
                        torch::Tensor& weight,
                        double epsilon);

}  // namespace xllm::kernel::cuda

namespace xllm::kernel::musa {

void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps);

void fused_add_gemma_rms_norm(torch::Tensor& input,
                              torch::Tensor& residual,
                              torch::Tensor& weight,
                              double epsilon);

void fused_qk_norm_rope(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t num_heads_v,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        bool interleaved,
                        const torch::Tensor& position_ids,
                        int64_t k_head_offset = 0);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias,
                     std::optional<torch::Tensor> output_buf = std::nullopt);

void gdn_fused_qkvzba_split_contiguous(const torch::Tensor& mixed_qkvz,
                                       const torch::Tensor& mixed_ba,
                                       torch::Tensor& mixed_qkv,
                                       torch::Tensor& z,
                                       torch::Tensor& b,
                                       torch::Tensor& a,
                                       int64_t num_heads_qk,
                                       int64_t num_heads_v,
                                       int64_t head_qk,
                                       int64_t head_v);

void partial_rotary_embedding_inplace(torch::Tensor& positions,
                                      torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      int64_t head_size,
                                      int64_t rotary_dim,
                                      bool is_neox);

torch::Tensor gemm_fp8_nt_groupwise(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Returns {quantized_input, per_group_scale}.
std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size);

// Returns {fp8_rows, scales, src_to_dst, expert_counts}.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_fp8(const torch::Tensor& input,
                         const torch::Tensor& topk_ids,
                         int64_t num_experts,
                         int64_t group_size);

// Returns {padded_hidden, row_expert_ids, original_to_padded, group_m_counts}.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_bf16(const torch::Tensor& input,
                          const torch::Tensor& topk_ids,
                          int64_t num_experts,
                          int64_t alignment);

// Decode-only ragged MoE helpers use aligned 128-row blocks.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_ragged_preprocess_fp8(const torch::Tensor& input,
                                const torch::Tensor& topk_ids,
                                int64_t group_size,
                                int64_t alignment);

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_preprocess_bf16(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    int64_t alignment);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_decode_preprocess_bf16(const torch::Tensor& input,
                                 const torch::Tensor& topk_ids,
                                 int64_t num_experts,
                                 int64_t alignment);

torch::Tensor fused_moe_ragged_swiglu_bf16(const torch::Tensor& input,
                                           int64_t alignment);

torch::Tensor fused_moe_indexed_swiglu_bf16(const torch::Tensor& input,
                                            const torch::Tensor& valid_rows);

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_swiglu_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size,
    int64_t alignment);

torch::Tensor fused_moe_ragged_combine(const torch::Tensor& down,
                                       const torch::Tensor& topk_weights,
                                       int64_t num_tokens,
                                       int64_t alignment);

// AOT kernels support decode batch sizes 1 through 8.
bool musa_fused_moe_aot_available(int64_t num_tokens);

bool musa_fused_moe_bf16_aot_available(int64_t num_tokens);

void prepare_musa_fused_moe_aot(const torch::Device& device);

void prepare_musa_fused_moe_bf16_aot(const torch::Device& device);

torch::Tensor musa_fused_moe_aot_fp8(const torch::Tensor& hidden_states,
                                     const torch::Tensor& w13,
                                     const torch::Tensor& w13_scale,
                                     const torch::Tensor& w2,
                                     const torch::Tensor& w2_scale,
                                     const torch::Tensor& topk_weights,
                                     const torch::Tensor& topk_ids);

torch::Tensor musa_fused_moe_aot_bf16(const torch::Tensor& hidden_states,
                                      const torch::Tensor& w13,
                                      const torch::Tensor& w2,
                                      const torch::Tensor& topk_weights,
                                      const torch::Tensor& topk_ids);

}  // namespace xllm::kernel::musa

namespace xllm::kernel::cuda {

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func);

}  // namespace xllm::kernel::cuda

namespace xllm::kernel::musa {

torch::Tensor random_sample(const torch::Tensor& probs);

// Returns [B, 2] with rejected suffix positions masked to -1.
torch::Tensor rejection_sample_target_only_k1(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& recovery_exponential,
    const torch::Tensor& bonus_token_ids);

torch::Tensor masked_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& token_counts,
                                   torch::ScalarType output_dtype,
                                   int64_t expected_tokens);

torch::Tensor masked_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& token_counts,
                                  torch::ScalarType output_dtype,
                                  int64_t expected_tokens);

// Compact BF16 grouped GEMM. Input rows must be sorted by expert and
// token_counts must contain the number of consecutive rows for each expert.
torch::Tensor contiguous_moe_gemm_bf16(const torch::Tensor& input,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& token_counts,
                                       torch::ScalarType output_dtype);

// Fixed-block BF16 Ragged GEMM. row_expert_ids has one entry per input row;
// valid expert ids occur in aligned blocks and padding rows contain -1.
torch::Tensor ragged_moe_gemm_bf16(const torch::Tensor& input,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& row_expert_ids,
                                   torch::ScalarType output_dtype,
                                   int64_t alignment);

// Input rows must be sorted by expert and grouped by token_counts.
torch::Tensor contiguous_moe_gemm_fp8(const torch::Tensor& input,
                                      const torch::Tensor& input_scale,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& token_counts,
                                      torch::ScalarType output_dtype);

// Fixed-block FP8 Ragged GEMM. row_expert_ids has one entry per input row;
// valid expert ids occur at aligned block starts and padding rows contain -1.
torch::Tensor ragged_moe_gemm_fp8(const torch::Tensor& input,
                                  const torch::Tensor& input_scale,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& weight_scale,
                                  const torch::Tensor& row_expert_ids,
                                  torch::ScalarType output_dtype,
                                  int64_t alignment);

std::tuple<torch::Tensor, torch::Tensor> musa_moe_topk_softmax(
    const torch::Tensor& router_logits,
    int64_t topk);
bool musa_moe_topk_softmax_available();

}  // namespace xllm::kernel::musa

namespace xllm::kernel::cuda {

// Returns {src_dst, dst_src, expert_sizes}.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts);

torch::Tensor moe_combine_result(const torch::Tensor& gemm2,
                                 const torch::Tensor& reduce_weight,
                                 int64_t N,
                                 int32_t topk);

}  // namespace xllm::kernel::cuda

#include "core/kernels/musa/gdn_ops.h"
