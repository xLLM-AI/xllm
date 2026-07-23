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

#include "custom_functions_npu/atb_common.h"

namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping);

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   torch::Tensor& output);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output);

std::tuple<torch::Tensor, torch::Tensor> npu_fused_infer_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag = false);

void batch_chunked_paged_prefill(const torch::Tensor& query,
                                 const torch::Tensor& k_cache,
                                 const torch::Tensor& v_cache,
                                 float scale,
                                 const torch::Tensor& block_table,
                                 const torch::Tensor& seq_lens,
                                 const torch::Tensor& attn_mask,
                                 const torch::Tensor& q_seq_lens,
                                 torch::Tensor& output);

// Custom batch decode for ACL graph execution
// This variant uses CustomPagedAttention to avoid .to(kCPU) operations
// that break ACL graph capture
void batch_decode_acl_graph(const torch::Tensor& query,
                            const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache,
                            float scale,
                            const torch::Tensor& block_table,
                            const torch::Tensor& seq_lens,
                            const torch::Tensor& tiling_data,
                            torch::Tensor& output);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias);

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode);

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps,
                       const std::string& mode);

std::tuple<torch::Tensor, torch::Tensor> rms_norm_dynamic_quant(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps);

void npu_gemma_rms_norm(const torch::Tensor& x,
                        const torch::Tensor& gamma,
                        double epsilon,
                        torch::Tensor& rstd_out,
                        torch::Tensor& y_out);

std::tuple<torch::Tensor, torch::Tensor> npu_block_sparse_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_sparse_mask,
    torch::IntArrayRef block_shape,
    const std::string& q_input_layout,
    const std::string& kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t inner_precise,
    int64_t softmax_lse_flag = 0,
    std::optional<torch::IntArrayRef> actual_seq_lengths = std::nullopt,
    std::optional<torch::IntArrayRef> actual_seq_lengths_kv = std::nullopt);

std::tuple<torch::Tensor, torch::Tensor> npu_rain_fusion_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& select_idx,
    const torch::Tensor& select_num_idx,
    torch::IntArrayRef blockshape,
    const std::string& q_input_layout,
    const std::string& kv_input_layout,
    int64_t head_num,
    double scale,
    int64_t inner_precise,
    std::optional<torch::IntArrayRef> actual_seq_lengths = std::nullopt,
    std::optional<torch::IntArrayRef> actual_seq_lengths_kv = std::nullopt);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon);

/// @brief Fused adaptive LayerNorm: LayerNorm(x, weight, bias) * (1 + scale) +
///        shift. Calls aclnnAdaLayerNorm under the hood.
///        The (1 + scale) is applied inside the kernel, so pass the raw scale.
/// @param input  Input tensor [B, L, C]
/// @param scale  Raw scale modulation factor [B, C] (kernel applies +1)
/// @param shift  Shift modulation factor [B, C]
/// @param weight Optional affine weight [C]
/// @param bias   Optional affine bias [C]
/// @param eps    Epsilon for numerical stability
/// @return Normalized and modulated output [B, L, C]
torch::Tensor fused_adalayer_norm(const torch::Tensor& input,
                                  const torch::Tensor& scale,
                                  const torch::Tensor& shift,
                                  std::optional<torch::Tensor> weight,
                                  std::optional<torch::Tensor> bias,
                                  double eps);

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& positions);

torch::Tensor apply_npu_moe_token_unpermute(
    const torch::Tensor& permuted_tokens,
    const torch::Tensor& sorted_indices,
    const std::optional<torch::Tensor>& probes,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
apply_moe_gating_topk_softmax(const torch::Tensor& x,
                              const std::optional<torch::Tensor>& finished,
                              int k);

std::vector<torch::Tensor> apply_npu_grouped_matmul(
    const torch::TensorList x,
    const torch::TensorList weight,
    const std::optional<torch::TensorList> bias,
    const std::optional<torch::TensorList> scale,
    const std::optional<torch::TensorList> offset,
    const std::optional<torch::TensorList> antiquant_scale,
    const std::optional<torch::TensorList> antiquant_offset,
    const std::optional<torch::TensorList> per_token_scale,
    const std::optional<torch::Tensor>& group_list,
    const std::optional<torch::TensorList> activation_input,
    const std::optional<torch::TensorList> activation_quant_scale,
    const std::optional<torch::TensorList> activation_quant_offset,
    std::optional<int64_t> split_item,
    std::optional<int64_t> group_type,
    std::optional<int64_t> group_list_type,
    std::optional<int64_t> act_type,
    const c10::OptionalIntArrayRef tuning_config,
    std::optional<torch::ScalarType> output_dtype);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
apply_npu_moe_init_routing_v2(const torch::Tensor& x,
                              const torch::Tensor& expert_idx,
                              const std::optional<torch::Tensor>& scale,
                              const std::optional<torch::Tensor>& offset,
                              int active_num,
                              int expert_capacity,
                              int expert_num,
                              int drop_pad_mode,
                              int expert_tokens_num_type,
                              bool expert_tokens_num_flag,
                              int quant_mode,
                              torch::IntArrayRef active_expert_range,
                              int row_idx_type);

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
apply_npu_moe_distribute_dispatch_v2(
    const torch::Tensor& x,
    const torch::Tensor& expert_ids,
    const std::optional<torch::Tensor>& expert_scales,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::optional<torch::Tensor>& scales,
    const std::string& group_ep,
    int64_t ep_world_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    const std::string& group_tp,
    int64_t tp_world_size,
    int64_t tp_rank_id,
    int64_t expert_shard_type,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs,
    int64_t expert_token_nums_type,
    const std::string& comm_alg);

torch::Tensor apply_npu_moe_distribute_combine_v2(
    const torch::Tensor& expand_x,
    const torch::Tensor& expert_ids,
    const torch::Tensor& assist_info_for_combine,
    const torch::Tensor& ep_send_counts,
    const torch::Tensor& expert_scales,
    const std::optional<torch::Tensor>& tp_send_counts,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::optional<torch::Tensor>& expand_scales,
    const std::optional<torch::Tensor>& shared_expert_x,
    const std::string& group_ep,
    int64_t ep_world_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    const std::string& group_tp,
    int64_t tp_world_size,
    int64_t tp_rank_id,
    int64_t expert_shard_type,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t global_bs,
    int64_t comm_quant_mode,
    const std::string& comm_alg);

bool has_moe_distribute_dispatch_combine_v2();

std::tuple<torch::Tensor, torch::Tensor> apply_npu_dispatch_ffn_combine(
    const torch::Tensor& x,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    const torch::Tensor& expert_ids,
    const torch::TensorList scale1,
    const torch::TensorList scale2,
    const torch::Tensor& probs,
    const std::string& group,
    int64_t max_output_size,
    double swiglu_limit,
    const std::optional<torch::Tensor>& output,
    const std::optional<torch::Tensor>& expert_token_nums);

bool has_dispatch_ffn_combine();

std::tuple<torch::Tensor, torch::Tensor> apply_npu_dispatch_gmm_combine_decode(
    const torch::Tensor& x,
    const torch::Tensor& expert_ids,
    const torch::TensorList gmm1_permuted_weight,
    const torch::TensorList gmm1_permuted_weight_scale,
    const torch::TensorList gmm2_weight,
    const torch::TensorList gmm2_weight_scale,
    const torch::Tensor& expert_scales,
    const std::optional<torch::Tensor>& expert_smooth_scales,
    const std::optional<torch::Tensor>& x_active_mask,
    const std::string& group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs);

bool has_dispatch_gmm_combine_decode();

std::pair<torch::Tensor, torch::Tensor> apply_npu_partial_rotary_embedding(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    int64_t rotary_dim,
    const torch::Tensor& cos_sin_cache,
    bool is_neox_style);

torch::Tensor npu_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk);

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>>
w4a8_dynamic_moe_preprocess(
    const torch::Tensor& w13_weight,
    const torch::Tensor& w2_weight,
    const torch::Tensor& w13_weight_scale,
    const torch::Tensor& w2_weight_scale,
    const std::optional<torch::Tensor>& w13_weight_scale_second,
    const std::optional<torch::Tensor>& w2_weight_scale_second,
    const std::optional<torch::Tensor>& w13_scale_bias,
    const std::optional<torch::Tensor>& w2_scale_bias,
    int64_t group_size,
    bool pack_weight_to_int32);

std::tuple<torch::Tensor, torch::Tensor> rec_constrained_topk(
    const torch::Tensor& logits,
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

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d_qkv(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    int64_t num_qk_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim);

void causal_conv1d_out(const torch::Tensor& output,
                       const torch::Tensor& x,
                       const torch::Tensor& weight,
                       const torch::Tensor& conv_state,
                       const std::optional<torch::Tensor>& bias_opt,
                       const torch::IntArrayRef query_start_loc_opt,
                       const torch::IntArrayRef cache_indices_opt,
                       const torch::IntArrayRef initial_state_mode_opt,
                       const torch::IntArrayRef num_accepted_tokens_opt,
                       int64_t activation_mode,
                       int64_t pad_slot_id,
                       int64_t run_mode);
}  // namespace xllm::kernel::npu
