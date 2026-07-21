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
#include <string>
#include <tuple>
#include <vector>

#include "ATen/Tensor.h"
#include "torch_mlu_ops.h"

namespace xllm::kernel::mlu {

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const std::optional<torch::Tensor>& position_ids,
                  const std::optional<torch::Tensor>& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk,
                  int64_t max_query_len);

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int64_t start_expert_id,
            int64_t expert_size);

void reshape_paged_cache(torch::Tensor& key,
                         const std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         const std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction);

void reshape_from_cache(torch::Tensor& key,
                        const std::optional<torch::Tensor>& value,
                        const torch::Tensor& key_cache,
                        const std::optional<torch::Tensor>& value_cache,
                        const torch::Tensor& context_lengths,
                        const int64_t max_context_len,
                        const std::optional<torch::Tensor>& context_seq_offset,
                        const std::optional<torch::Tensor>& block_tables,
                        const std::optional<torch::Tensor>& cache_seq_offset);

// Quantize and store KV cache to paged cache (INT8 quantization)
// k/v: [token_nums, head_num_kv, head_size] (FP16/BF16)
// k_cache/v_cache: [block_nums, head_num_kv, block_size, head_size] (INT8)
// k_cache_scale/v_cache_scale: [block_nums, head_num_kv, block_size] (FP32)
// slot_mapping: [token_nums] (INT32)
void quant_to_paged_cache(const torch::Tensor& k,
                          const std::optional<torch::Tensor>& v,
                          torch::Tensor& k_cache,
                          const std::optional<torch::Tensor>& v_cache,
                          torch::Tensor& k_cache_scale,
                          const std::optional<torch::Tensor>& v_cache_scale,
                          const torch::Tensor& slot_mapping);

// Dequantize KV cache from paged cache (INT8 to FP16/BF16)
// key/value: [total_seqlens, head_num, head_size] (FP16/BF16) - output
// key_cache/value_cache: [block_nums, head_num, block_size, head_size] (INT8)
// key_cache_scale/value_cache_scale: [block_nums, head_num, block_size] or
// [head_num, head_size] (FP32) context_lengths: [batch] (INT32)
// max_context_len: maximum context length
// context_seq_offset: [batch] (INT32) - optional sequence offset
// block_tables: [batch, max_block_num] (INT32)
// quant_mode: 0 for per-channel, 1 for per-token
// quant_bit: quantization bit (default 8)
void dequant_from_paged_cache(
    torch::Tensor& key,
    const std::optional<torch::Tensor>& value,
    const torch::Tensor& key_cache,
    const std::optional<torch::Tensor>& value_cache,
    const torch::Tensor& key_cache_quant_scale,
    const std::optional<torch::Tensor>& value_cache_quant_scale,
    const torch::Tensor& context_lengths,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& context_seq_offset,
    const torch::Tensor& block_tables,
    int64_t quant_mode,
    int64_t quant_bit);

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_tables,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  int64_t kv_cache_quant_bit_size,
                  const std::optional<torch::Tensor>& cu_seq_q = std::nullopt,
                  int64_t max_seq_q = -1,
                  const std::optional<torch::Tensor>& sink = std::nullopt);

void update_out_and_lse(
    torch::Tensor& out,
    torch::Tensor& lse,
    const torch::Tensor& block_out,
    const torch::Tensor& block_lse,
    const std::optional<torch::Tensor>& seq_offsets = std::nullopt,
    const std::optional<torch::Tensor>& cu_seqs = std::nullopt,
    const std::optional<torch::Tensor>& block_cu_seqs = std::nullopt);

void masked_indexer_select_paged_kv(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& weights,
    const torch::Tensor& kv_cache_block_table,
    const std::optional<torch::Tensor>& cu_seq_q_lens,
    const std::optional<torch::Tensor>& cu_seq_k_lens,
    const std::optional<torch::Tensor>& k_context_lens,
    const std::optional<torch::Tensor>& k_cache_block_table,
    const bool is_prefill,
    const int64_t index_topk,
    const int64_t kv_cache_block_size,
    const double softmax_scale,
    const std::optional<torch::Tensor>& q_scale,
    const std::optional<torch::Tensor>& k_scale_cache,
    const torch::Tensor& sparse_block_table,
    const torch::Tensor& sparse_context_lens,
    bool is_score_float = false,
    int64_t compress_ratio = 1,
    const std::optional<torch::Tensor>& kv_cache_block_table_offset =
        std::nullopt);

void fused_layernorm(const torch::Tensor& input,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& residual,
                     const std::optional<torch::Tensor>& weight,
                     const std::optional<torch::Tensor>& beta,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& quant_scale,
                     const std::optional<torch::Tensor>& residual_out,
                     const std::optional<torch::Tensor>& smooth_quant_scale,
                     const std::optional<torch::Tensor>& normed_out,
                     const std::string& mode,
                     double eps,
                     bool store_output_before_norm,
                     bool store_output_after_norm,
                     bool dynamic_quant);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& c,
                     double alpha,
                     double beta);

torch::Tensor group_gemm(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& token_count,
                         torch::Tensor& output,
                         const std::optional<torch::Tensor>& a_scale,
                         const std::optional<torch::Tensor>& b_scale,
                         const std::optional<torch::List<int64_t>>& quant_flag,
                         const int64_t max_dim,
                         const bool trans_a,
                         const bool trans_b,
                         const int64_t a_quant_bit);

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    const torch::Tensor& input,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool normalize,
    const std::optional<torch::Tensor>& mask,
    const std::string& normed_by,
    const std::string& scoring_func,
    double route_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias);

std::vector<torch::Tensor> moe_gen_idx(const torch::Tensor& expert_id,
                                       int64_t expert_num);

torch::Tensor moe_expand_input(
    const torch::Tensor& input,
    const torch::Tensor& gather_index,
    const std::optional<torch::Tensor>& cusum_token_count,
    int64_t start_expert_id,
    int64_t expert_size);

torch::Tensor moe_combine_result(
    const torch::Tensor& input,
    const torch::Tensor& reduce_weight,
    const torch::Tensor& gather_ids,
    const std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& cusum_token_count,
    const int64_t start_expert_id,
    const int64_t expert_size,
    const std::optional<torch::Tensor>& bias);

torch::Tensor moe_all2all_gen_send_layout(const torch::Tensor& token_count,
                                          int64_t nrank);

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    const torch::Tensor& token_num,
    int64_t pad_num,
    bool return_cusum_token_count);

std::vector<torch::Tensor> moe_all2all_create(int64_t dispatch_token_byte,
                                              int64_t combine_token_byte,
                                              int64_t max_expert_num,
                                              int64_t max_token_num,
                                              int64_t rank,
                                              int64_t nrank,
                                              const torch::Device& device);

void moe_all2all_init(int64_t handle,
                      const torch::Tensor& all_exchange_info,
                      const torch::Device& device);

void moe_all2all_dispatch(int64_t handle,
                          int64_t token_byte,
                          int64_t token_num,
                          const torch::Tensor& send_layout,
                          const torch::Tensor& send_token_num,
                          const torch::Tensor& recv_layout,
                          const torch::Tensor& recv_token_num,
                          const std::optional<torch::Tensor>& send_token,
                          const std::optional<torch::Tensor>& recv_token);

void moe_all2all_combine(int64_t handle,
                         int64_t token_byte,
                         int64_t token_num,
                         const torch::Tensor& send_src_layout,
                         const torch::Tensor& send_dst_layout,
                         const std::optional<torch::Tensor>& send_token,
                         const std::optional<torch::Tensor>& recv_token);

void moe_all2all_destroy(int64_t handle, const torch::Device& device);

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero = std::nullopt,
    const std::optional<torch::Tensor>& token_count = std::nullopt,
    const std::optional<torch::Tensor>& gather_index = std::nullopt,
    const std::optional<torch::Tensor>& gather_index_start_position =
        std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& output_scale = std::nullopt,
    const std::string& act_mode = "none",
    double active_coef = 1.0,
    bool is_gated = false,
    torch::ScalarType quant_type = torch::kChar);

torch::Tensor scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& c = std::nullopt,
    const std::string& act_mode = "none",
    int64_t quant_bit_size = 8,
    double alpha = 1.0,
    double beta = 1.0,
    bool use_hp_active = false,
    int64_t a_quant_bit_size = -1,
    const std::optional<torch::Tensor>& a_calib = std::nullopt,
    const std::optional<torch::Tensor>& b_calib = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor apply_top_k_top_p(const torch::Tensor& logits,
                                const torch::Tensor& temperature_list,
                                const torch::Tensor& topk_list,
                                const torch::Tensor& topp_list);

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

void gather_split(const torch::Tensor& input,
                  const torch::Tensor& gather_index,
                  const torch::Tensor& valid_token_num,
                  const torch::Tensor& output_head,
                  const torch::Tensor& output_tail);

torch::Tensor fused_mul_reduce_sum(const torch::Tensor& x,
                                   const torch::Tensor& w);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_split_sinkhorn(
    const torch::Tensor& mixes,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    const std::optional<torch::Tensor>& pre_scale,
    int64_t hc_mult,
    int64_t sinkhorn_iter,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> fused_mhc_post(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    bool compute_rms,
    double eps);

void fused_mla_q(const torch::Tensor& input,
                 torch::Tensor& output,
                 torch::Tensor& output_scale,
                 const std::optional<torch::Tensor>& output_norm,
                 const torch::Tensor& gamma,
                 const std::optional<torch::Tensor>& smooth_quant_scale,
                 const torch::Tensor& weight_b,
                 const torch::Tensor& weight_b_scale,
                 const torch::Tensor& weight_c,
                 const torch::Tensor& sin,
                 const torch::Tensor& cos,
                 const torch::Tensor& position_id,
                 const std::string& quant_mode,
                 double eps,
                 bool interleaved);

void fused_mla_kv(const torch::Tensor& input_kv,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const torch::Tensor& position_id,
                  const torch::Tensor& gamma,
                  const torch::Tensor& kv_cache,
                  const std::optional<torch::Tensor>& kv_cache_scale,
                  const std::optional<torch::Tensor>& slot_mapping,
                  const std::optional<torch::Tensor>& cache_bs_id,
                  const std::optional<torch::Tensor>& cache_seq_offset,
                  const std::string& quant_mode,
                  bool is_paged_cache,
                  double eps,
                  bool interleaved);

void fused_indexer_q(const torch::Tensor& input_q,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& output_scale,
                     const torch::Tensor& w_q,
                     const std::optional<torch::Tensor>& w_q_scale,
                     const std::optional<torch::Tensor>& hadamard_matrix,
                     const torch::Tensor& sin,
                     const torch::Tensor& cos,
                     const torch::Tensor& position_id,
                     const std::string& quant_mode,
                     bool interleaved,
                     bool rope_at_front);

void fused_indexer_k(const torch::Tensor& x,
                     const torch::Tensor& wk,
                     const torch::Tensor& wproj,
                     const torch::Tensor& sin_table,
                     const torch::Tensor& cos_table,
                     const torch::Tensor& position_id,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& head_weights,
                     const torch::Tensor& k_cache,
                     const std::optional<torch::Tensor>& k_cache_scale,
                     const std::optional<torch::Tensor>& hadamard_matrix,
                     bool interleaved,
                     const std::optional<torch::Tensor>& gamma,
                     const std::optional<torch::Tensor>& beta,
                     double eps);

torch::Tensor gated_layer_norm(torch::Tensor& x,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               double eps,
                               const std::optional<torch::Tensor>& gate,
                               int64_t group_size,
                               bool norm_before_gate);

torch::Tensor gemma_rms_norm(const torch::Tensor& x,
                             const torch::Tensor& gamma,
                             double eps,
                             torch::Tensor& norm_out,
                             const std::optional<torch::Tensor>& residual,
                             std::optional<torch::Tensor>& residual_out);

std::tuple<torch::Tensor, torch::Tensor> moe_softplus_topk(
    const torch::Tensor& input,
    int64_t topk,
    const std::optional<torch::Tensor>& input_ids = std::nullopt,
    const std::optional<torch::Tensor>& tid2eid = std::nullopt,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    float route_scale = 1.0);

void fused_compress_single_kv(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& position,
    const torch::Tensor& ape,
    const torch::Tensor& gamma,
    const torch::Tensor& sin,
    const torch::Tensor& cos,
    const std::optional<torch::Tensor>& hadamard_matrix,
    const torch::Tensor& slot_mapping,
    torch::Tensor& kv_cache,
    const std::optional<torch::Tensor>& kv_cache_scale,
    double eps,
    bool overlap,
    torch::Tensor& state_cache,
    const torch::Tensor& state_bt,
    int64_t state_width,
    int64_t state_block_size,
    const torch::Tensor& cu_query_len,
    int64_t K = 0);

void fused_compress_multi_kv(const torch::Tensor& kv,
                             const torch::Tensor& score,
                             torch::Tensor& state_cache,
                             const torch::Tensor& state_block_table,
                             const torch::Tensor& cu_seqlens,
                             const torch::Tensor& positions,
                             const torch::Tensor& ape,
                             int64_t max_seqlen,
                             bool overlap,
                             torch::Tensor& compressed_kv);

torch::Tensor causal_conv1d_fn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& batch,
    const torch::Tensor& token_block_offset,
    int32_t nt,
    const std::optional<torch::Tensor>& bias_opt = std::nullopt,
    const std::optional<torch::Tensor>& cache_indices_opt = std::nullopt,
    const std::optional<torch::Tensor>& has_initial_state_opt = std::nullopt,
    const std::optional<torch::Tensor>& initial_state_idx_opt = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt = std::nullopt,
    bool inplace_final_state = true);

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const std::optional<torch::Tensor>& beta_opt = std::nullopt,
    const std::optional<torch::Tensor>& initial_state_opt = std::nullopt,
    bool inplace_final_state = true,
    const std::optional<torch::Tensor>& cu_seqlens_opt = std::nullopt,
    const std::optional<torch::Tensor>& ssm_state_indices_opt = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt = std::nullopt,
    bool use_qk_l2norm_in_kernel = true);

std::pair<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    double scale,
    torch::Tensor& ssm_cache,
    const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel = true);

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
fused_post_conv_prep(const torch::Tensor& conv_output,
                     const torch::Tensor& a,
                     const torch::Tensor& b,
                     const torch::Tensor& A_log,
                     const torch::Tensor& dt_bias,
                     int64_t num_k_heads,
                     int64_t head_k_dim,
                     int64_t head_v_dim,
                     bool apply_l2norm = true,
                     bool output_g_exp = false);

std::pair<torch::Tensor, torch::Tensor> fused_sigmoid_gating_delta_rule_update(
    const torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& b,
    const torch::Tensor& dt_bias,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& initial_state,
    torch::Tensor& ssm_state_indices,
    torch::Tensor& cu_seqlens,
    double scale,
    bool use_qk_l2norm_in_kernel = true,
    float softplus_beta = 1.0f,
    float softplus_threshold = 20.0f,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt = std::nullopt,
    bool inplace_final_state = true,
    bool is_kda = false);

torch::Tensor causal_conv1d_update_decode(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt,
    const std::optional<torch::Tensor>& conv_state_indices_opt,
    bool activation = true,
    int32_t pad_slot_id = -1,
    const std::optional<torch::Tensor>& query_start_loc_opt = std::nullopt,
    int32_t max_query_len = -1,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt = std::nullopt,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token_opt =
        std::nullopt,
    const std::optional<torch::Tensor>& initial_state_idx_opt = std::nullopt);

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f);
}  // namespace xllm::kernel::mlu
