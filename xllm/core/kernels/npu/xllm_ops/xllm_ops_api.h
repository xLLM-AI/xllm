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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace xllm::kernel::npu {
inline constexpr int64_t kDsaMetadataBufferElements = 1024;

namespace op_infer {
constexpr int32_t N = 32;
// npu tensor max size
constexpr int32_t SIZE = 8;
constexpr int32_t INT4_NUMS_IN_INT32_SPACE = 8;
constexpr int32_t NPU_NSA_COMPRESS_INPUT_DIM_SECOND = 1;
constexpr int32_t NPU_NSA_COMPRESS_INPUT_DIM_THIRD = 2;
constexpr int32_t DIM_0 = 0;
constexpr int32_t DIM_1 = 1;
constexpr int32_t DIM_2 = 2;
constexpr int32_t DIM_3 = 3;
}  // namespace op_infer

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_token_ids);

void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);

void replace_token(torch::Tensor& dst,
                   torch::Tensor& src,
                   bool synchronize_stream = true);

// Laser attention (MindIE-SD kernel tuned for Wan2.2). q/k/v in BNSD layout;
// returns attention output in BNSD layout, cast back to the input dtype.
torch::Tensor laser_attention(const torch::Tensor& q_bnsd,
                              const torch::Tensor& k_bnsd,
                              const torch::Tensor& v_bnsd,
                              double scale_value,
                              int64_t head_num);

void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence);

void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     int64_t result_width,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence);

void select_unshared_kv(const torch::Tensor& beam_index,
                        const std::vector<torch::Tensor>& x_key_block,
                        const std::vector<torch::Tensor>& x_value_block,
                        const torch::Tensor& block_table,
                        const torch::Tensor& group_offset,
                        int64_t decode_step,
                        int64_t beam_size,
                        int64_t layer_num);

std::optional<std::tuple<torch::Tensor, torch::Tensor>>
rec_constrained_topk_fused(const torch::Tensor& logits,
                           const torch::Tensor& sequence_group,
                           const torch::Tensor& first_token_ids,
                           const torch::Tensor& prefix1_offsets,
                           const torch::Tensor& prefix1_values,
                           const torch::Tensor& prefix1_pair_keys,
                           const torch::Tensor& prefix2_value_offsets,
                           const torch::Tensor& prefix2_values,
                           const torch::Tensor& temperatures,
                           int64_t current_step,
                           int64_t top_k,
                           int64_t max_prefix1_degree,
                           int64_t max_prefix2_degree);

at::Tensor quant_matmul(const at::Tensor& x1,
                        const at::Tensor& x2,
                        const bool transpose2,
                        const at::Tensor& scale,
                        const c10::optional<at::Tensor>& offset,
                        const c10::optional<at::Tensor>& pertoken_scale,
                        const c10::optional<at::Tensor>& bias,
                        c10::optional<at::ScalarType> output_dtype);

at::Tensor quantize_per_tensor(const at::Tensor& self,
                               const at::Tensor& scales,
                               const at::Tensor& zero_points,
                               at::ScalarType dtype,
                               int64_t axis);

std::tuple<at::Tensor, c10::optional<at::Tensor>> dynamic_quant(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& group_index,
    c10::optional<at::ScalarType> dst_type);

std::tuple<at::Tensor, at::Tensor> dequant_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias);

at::Tensor hc_post(const at::Tensor& x,
                   const at::Tensor& residual,
                   const at::Tensor& post,
                   const at::Tensor& comb);

std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& metadata,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    bool return_value);
at::Tensor hc_pre_inv_rms(const at::Tensor& x, double epsilon);

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_sinkhorn(
    const at::Tensor& mixes,
    const at::Tensor& rsqrt,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& x,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double hc_eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double norm_eps,
    double hc_eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag);

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse);

at::Tensor sparse_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_preprocess(
    const at::Tensor& input,
    const at::Tensor& gamma0,
    const at::Tensor& beta0,
    const at::Tensor& quant_scale0,
    const at::Tensor& quant_offset0,
    const at::Tensor& wdqkv,
    const at::Tensor& descale0,
    const at::Tensor& bias0,
    const at::Tensor& gamma1,
    const at::Tensor& beta1,
    const at::Tensor& quant_scale1,
    const at::Tensor& quant_offset1,
    const at::Tensor& wuq,
    const at::Tensor& descale1,
    const at::Tensor& bias1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slot_mapping,
    const at::Tensor& ctkv_scale,
    const at::Tensor& q_nope_scale,
    int64_t wdq_dim,
    int64_t q_rope_dim,
    int64_t k_rope_dim,
    double epsilon,
    int64_t q_rotary_coeff,
    int64_t k_rotary_coeff,
    bool transepose_wdq,
    bool transepose_wuq,
    bool transepose_wuk,
    int64_t cache_mode,
    int64_t quant_mode,
    bool do_rms_norm,
    int64_t wdkv_split_count);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compressor(const at::Tensor& x,
           const at::Tensor& wkv,
           const at::Tensor& wgate,
           at::Tensor& kv_state,
           at::Tensor& score_state,
           const at::Tensor& ape,
           const at::Tensor& norm_weight,
           const at::Tensor& rope_sin,
           const at::Tensor& rope_cos,
           const c10::optional<at::Tensor>& kv_block_table,
           const c10::optional<at::Tensor>& score_block_table,
           const c10::optional<at::Tensor>& cu_seqlens,
           const c10::optional<at::Tensor>& seqused,
           const c10::optional<at::Tensor>& start_pos,
           int64_t rope_head_dim,
           int64_t cmp_ratio,
           int64_t coff,
           double norm_eps,
           int64_t rotary_mode,
           bool enable_grad);

at::Tensor quant_lightning_indexer_metadata(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    const c10::string_view device);

at::Tensor sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv);

void npu_inplace_partial_rotary_mul(torch::Tensor& x,
                                    const torch::Tensor& r1,
                                    const torch::Tensor& r2,
                                    c10::string_view rotary_mode,
                                    at::IntArrayRef partial_slice);

void scatter_nd_update(torch::Tensor& var,
                       const torch::Tensor& indices,
                       const torch::Tensor& updates);

std::pair<torch::Tensor, torch::Tensor> npu_mega_chunk_gdn(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    const std::optional<float>& scale = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    c10::ArrayRef<int32_t> q_seq_lens = {},
    bool use_qk_l2norm_in_kernel = false);

torch::Tensor layer_norm_fwd_aclnn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps,
    const std::optional<torch::Tensor>& z = std::nullopt,
    int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false);
}  // namespace xllm::kernel::npu
