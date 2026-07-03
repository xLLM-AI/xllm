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
#include <string>
#include <tuple>
#include <utility>

namespace xllm {
namespace kernel {

struct CausalConv1dUpdateParams;
struct ChunkGatedDeltaRuleParams;
struct FusedGdnGatingParams;
struct FusedQkvzbaSplitReshapeParams;
struct FusedRecurrentGatedDeltaRuleParams;
struct GatedLayerNormParams;
struct MateGatedDeltaRuleDecodeParams;
struct MateGatedDeltaRulePrefillParams;
struct PartialRotaryEmbeddingParams;
struct FusedSigmoidGatingDeltaRuleUpdateParams;

namespace cuda {

torch::Tensor l2_norm(torch::Tensor& x, double eps);

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params);

std::pair<torch::Tensor, torch::Tensor> gdn_gating(const torch::Tensor& a,
                                                   const torch::Tensor& b,
                                                   const torch::Tensor& A_log,
                                                   const torch::Tensor& dt_bias,
                                                   double sp_beta,
                                                   double threshold);

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params);

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params);

torch::Tensor gated_layer_norm(GatedLayerNormParams& params);

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params);

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params);

torch::Tensor recurrent_gated_delta_rule(
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

std::string get_mate_gdn_prefill_uri(int64_t num_q_heads,
                                     int64_t num_v_heads,
                                     torch::ScalarType dtype);

std::string get_mate_gdn_decode_uri(int64_t num_q_heads,
                                    int64_t num_v_heads,
                                    torch::ScalarType dtype);

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params);

torch::Tensor mate_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

torch::Tensor fused_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params);

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

torch::Tensor fused_sigmoid_gating_delta_rule_update(
    FusedSigmoidGatingDeltaRuleUpdateParams& params);

void causal_conv1d_decode_fused(const torch::Tensor& x,
                                const torch::Tensor& weight,
                                const std::optional<torch::Tensor>& bias,
                                torch::Tensor conv_state,
                                const torch::Tensor& cache_indices,
                                torch::Tensor output_buf,
                                int pad_slot_id,
                                bool silu_activation);

void gated_rms_norm_fused(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor& z,
                          torch::Tensor output,
                          double eps);

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
