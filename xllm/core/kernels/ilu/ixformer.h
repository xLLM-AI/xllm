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
#include <torch/all.h>

#include "ATen/Tensor.h"
#include "utils.h"

namespace ixformer::infer {
torch::Tensor ixinfer_flash_attn_unpad_with_block_tables(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& out,
    torch::Tensor& block_tables,
    torch::Tensor& cu_seq_q,
    torch::Tensor& cu_seq_k,
    int64_t max_seq_q,
    int64_t max_seq_k,
    bool is_causal,
    int64_t window_left,
    int64_t window_right,
    double scale,
    double softcap,
    bool sqrt_alibi,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::optional<torch::Tensor>& sinks,
    std::optional<torch::Tensor>& lse);

void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

torch::Tensor xllm_paged_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    bool causal,
    int32_t window_left,
    int32_t window_right,
    double softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    const std::optional<torch::Tensor>& sinks);

torch::Tensor ixformer_linear(torch::Tensor& input,
                              torch::Tensor& weight,
                              int64_t act_type,
                              const std::optional<torch::Tensor>& bias,
                              const std::optional<torch::Tensor>& out,
                              const std::optional<bool> persistent);

torch::Tensor ixformer_linear_ex(torch::Tensor& input,
                                 torch::Tensor& weight,
                                 const c10::optional<torch::Tensor>& bias,
                                 const c10::optional<torch::Tensor>& out);

void xllm_reshape_and_cache(torch::Tensor& key,
                            torch::Tensor& value,
                            torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping,
                            int64_t key_token_stride,
                            int64_t value_token_stride);

void xllm_rotary_embedding(torch::Tensor& positions,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           int64_t head_size,
                           torch::Tensor& cos_sin_cache,
                           bool is_neox);

void residual_rms_norm(torch::Tensor& input,
                       torch::Tensor& residual,
                       torch::Tensor& weight,
                       torch::Tensor& output,
                       torch::Tensor& residual_output,
                       const std::optional<torch::Tensor>& fused_bias,
                       double alpha,
                       double eps,
                       bool is_post);

void rms_norm(torch::Tensor& input,
              torch::Tensor& weight,
              torch::Tensor& output,
              const std::optional<torch::Tensor>& fused_bias,
              double eps);

void topk_softmax(torch::Tensor& topk_weights,
                  torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output,
                  bool renormalize);

void moe_compute_token_index_api(
    torch::Tensor& topk_ids,
    torch::Tensor& src_dst,
    torch::Tensor& dst_src,
    torch::Tensor& expert_sizes_gpu,
    const c10::optional<torch::Tensor>& expert_mask,
    const c10::optional<torch::Tensor>& expert_sizes_cpu,
    const c10::optional<torch::Tensor>& expand_tokens_gpu,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t num_experts);

void moe_expand_input(torch::Tensor outputs,
                      torch::Tensor inputs,
                      torch::Tensor dst_to_src,
                      const c10::optional<torch::Tensor>& src_to_dst,
                      int64_t dst_tokens,
                      int64_t expand_factor);

void moe_w16a16_group_gemm(torch::Tensor output,
                           torch::Tensor inputs,
                           torch::Tensor weights,
                           torch::Tensor tokens_per_experts,
                           const c10::optional<torch::Tensor>& dst_to_src,
                           const c10::optional<torch::Tensor>& bias,
                           std::string format,
                           int64_t persistent,
                           int64_t output_n);

void moe_output_reduce_sum(torch::Tensor outputs,
                           torch::Tensor inputs,
                           const c10::optional<torch::Tensor>& mul_weight,
                           const c10::optional<torch::Tensor>& mask,
                           const c10::optional<torch::Tensor>& extra_residual,
                           double scaling_factor);
}  // namespace ixformer::infer
