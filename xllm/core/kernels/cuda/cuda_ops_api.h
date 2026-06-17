/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <optional>
#include <tuple>
#include <vector>

#include "utils.h"

namespace xllm::kernel::cuda {

// TODO: add head_size parameter
void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      // int64_t head_size,
                      bool is_neox);

// act_mode only support silu, gelu, gelu_tanh
void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache);

void block_copy(torch::Tensor key_cache_ptrs,
                torch::Tensor value_cache_ptrs,
                torch::Tensor src_block_indices,
                torch::Tensor dst_block_indices,
                torch::Tensor cum_sum,
                int64_t numel_per_block,
                torch::ScalarType cache_dtype);
#if !defined(USE_DCU)
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

// Wrapper function for batch_prefill that conditionally uses AttentionRunner
// for piecewise CUDA Graph capture
void batch_prefill_with_optional_piecewise_capture(
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
    std::optional<torch::Tensor>& output_lse);

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
    bool causal = true);

void batch_decode(const std::string& uri,
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
                  std::optional<torch::Tensor> qo_indptr = std::nullopt);
#endif  // !defined(USE_DCU)
void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

void cutlass_scaled_mm(torch::Tensor& c,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

// Static scaled FP8 quantization
// Quantizes input tensor to FP8 using a pre-computed scale factor
void static_scaled_fp8_quant(torch::Tensor& out,           // [..., d]
                             torch::Tensor const& input,   // [..., d]
                             torch::Tensor const& scale);  // [1]

// FP8 scaled quantize: quantizes input tensor to FP8 e4m3 format
// Returns: (quantized_output, scale)
std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& scale = std::nullopt);

// ============================================================================
// Fused RMSNorm + Static FP8 Quantization
// ============================================================================
// These functions combine RMSNorm and FP8 quantization to reduce memory
// bandwidth by avoiding the intermediate write-back to global memory.

// Fused RMSNorm + Static FP8 Quantization (without residual)
// Combines RMSNorm normalization and FP8 quantization in a single kernel.
// This is optimal for the first layer where no residual connection exists.
void rms_norm_static_fp8_quant(
    torch::Tensor& out,     // [..., hidden_size], FP8 output
    torch::Tensor& input,   // [..., hidden_size], input tensor
    torch::Tensor& weight,  // [hidden_size], RMSNorm weight
    torch::Tensor& scale,   // [1], FP8 quantization scale
    double epsilon);        // RMSNorm epsilon

// Fused Add + RMSNorm + Static FP8 Quantization (with residual)
// Combines residual addition, RMSNorm, and FP8 quantization in a single kernel.
// The residual tensor is updated in-place with the sum of input and residual.
void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size], FP8 output
    torch::Tensor& input,     // [..., hidden_size], input tensor
    torch::Tensor& residual,  // [..., hidden_size], residual (updated in-place)
    torch::Tensor& weight,    // [hidden_size], RMSNorm weight
    torch::Tensor& scale,     // [1], FP8 quantization scale
    double epsilon);          // RMSNorm epsilon

// FP8 scaled matmul for W8A8 quantization using CUTLASS kernels
// Performs: c = (a @ b.T) with scales applied
torch::Tensor fp8_scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device);

std::pair<torch::Tensor, torch::Tensor> compute_topk_general(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device);

torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures);

void fused_qk_norm_rope(
    torch::Tensor& qkv,   // Combined QKV tensor [num_tokens,
                          // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,  // Number of query heads
    int64_t num_heads_k,  // Number of key heads
    int64_t num_heads_v,  // Number of value heads
    int64_t head_dim,     // Dimension per head
    double eps,           // Epsilon for RMS normalization
    const torch::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    const torch::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    const torch::Tensor&
        cos_sin_cache,  // Cos/sin cache [max_position, rotary_dim]
    bool interleaved,   // Whether RoPE is applied in interleaved style
    const torch::Tensor& position_ids  // Position IDs for RoPE [num_tokens]
);

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::optional<torch::Tensor>& correction_bias,
    const std::string& scoring_func);

torch::Tensor random_sample(const torch::Tensor& probs);

torch::Tensor cutlass_fused_moe(
    const torch::Tensor& input,                   // [num_tokens, hidden]
    const torch::Tensor& token_selected_experts,  // [num_tokens, top_k]
    const torch::Tensor& token_final_scales,      // [num_tokens, top_k]
    const torch::Tensor&
        fc1_expert_weights,  // [num_experts, inter_dim, hidden]
    const torch::Tensor&
        fc2_expert_weights,  // [num_experts, hidden, inter_dim]
    torch::ScalarType output_dtype,
    const std::vector<torch::Tensor>& quant_scales,
    int32_t tp_size,
    int32_t tp_rank,
    int32_t ep_size,
    int32_t ep_rank,
    int32_t cluster_size,
    int32_t cluster_rank,
    const std::optional<torch::Tensor>& fc1_expert_biases = std::nullopt,
    const std::optional<torch::Tensor>& fc2_expert_biases = std::nullopt,
    const std::optional<torch::Tensor>& input_sf = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_alpha = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_beta = std::nullopt,
    const std::optional<torch::Tensor>& swiglu_limit = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    bool enable_alltoall = false,
    bool use_deepseek_fp8_block_scale = false,
    bool use_w4_group_scaling = false,
    bool use_mxfp8_act_scaling = false,
    bool min_latency_mode = false,
    bool use_packed_weights = false,
    int32_t tune_max_num_tokens = 8192,
    ActivationType activation_type = ActivationType::SWIGLU);
}  // namespace xllm::kernel::cuda
