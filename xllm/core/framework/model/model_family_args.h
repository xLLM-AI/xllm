/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_family.h"
#include "core/framework/model/model_params.h"

namespace xllm {

namespace model_family_args_detail {

inline const ModelArgs& default_model_args() {
  static const ModelArgs args{};
  return args;
}

}  // namespace model_family_args_detail

// Preserve the property interface while each field lives in its capability.
#define MODEL_PARAM(T, property, params)                               \
 public:                                                               \
  [[nodiscard]] const T& property() const& noexcept {                  \
    return params.property();                                          \
  }                                                                    \
  [[nodiscard]] T& property() & noexcept { return params.property(); } \
  [[nodiscard]] T&& property() && noexcept {                           \
    return std::move(params).property();                               \
  }                                                                    \
  auto property(const T& value) & -> decltype(*this) {                 \
    params.property(value);                                            \
    return *this;                                                      \
  }                                                                    \
  auto property(T&& value) & -> decltype(*this) {                      \
    params.property(std::move(value));                                 \
    return *this;                                                      \
  }                                                                    \
  void property(const T& value) && = delete;                           \
  void property(T&& value) && = delete

// Model-family arguments compose capability parameter objects directly.
#define COMMON_MODEL_PARAMS                                                    \
  MODEL_PARAM(std::string, model_type, execution_params_);                     \
  MODEL_PARAM(std::string, dtype, execution_params_);                          \
  MODEL_PARAM(int64_t, hidden_size, execution_params_);                        \
  MODEL_PARAM(std::string, hidden_act, execution_params_);                     \
  MODEL_PARAM(int64_t, intermediate_size, execution_params_);                  \
  MODEL_PARAM(int64_t, n_layers, execution_params_);                           \
  MODEL_PARAM(int64_t, n_encoder_layers, execution_params_);                   \
  MODEL_PARAM(int64_t, head_dim, attention_params_);                           \
  MODEL_PARAM(int64_t, kv_cache_storage_head_dim, kv_cache_params_);           \
  MODEL_PARAM(int64_t, decoder_head_dim, attention_params_);                   \
  MODEL_PARAM(int64_t, n_heads, attention_params_);                            \
  MODEL_PARAM(int64_t, decoder_n_heads, attention_params_);                    \
  MODEL_PARAM(int64_t, actual_n_heads, attention_params_);                     \
  MODEL_PARAM(std::optional<int64_t>, n_kv_heads, attention_params_);          \
  MODEL_PARAM(std::optional<int64_t>, decoder_n_kv_heads, attention_params_);  \
  MODEL_PARAM(int64_t, vocab_size, execution_params_);                         \
  MODEL_PARAM(int32_t, qk_nope_head_dim, attention_params_);                   \
  MODEL_PARAM(int32_t, qk_rope_head_dim, attention_params_);                   \
  MODEL_PARAM(int32_t, v_head_dim, attention_params_);                         \
  MODEL_PARAM(int32_t, q_lora_rank, attention_params_);                        \
  MODEL_PARAM(int32_t, kv_lora_rank, attention_params_);                       \
  MODEL_PARAM(int32_t, num_nextn_predict_layers, speculative_params_);         \
  MODEL_PARAM(bool, enable_mla, attention_params_);                            \
  MODEL_PARAM(int32_t, num_experts_per_tok, moe_params_);                      \
  MODEL_PARAM(bool, use_qk_norm, attention_params_);                           \
  MODEL_PARAM(float, rms_norm_eps, attention_params_);                         \
  MODEL_PARAM(float, layer_norm_eps, attention_params_);                       \
  MODEL_PARAM(int64_t, rotary_dim, attention_params_);                         \
  MODEL_PARAM(float, rope_theta, attention_params_);                           \
  MODEL_PARAM(std::string, rope_scaling_rope_type, attention_params_);         \
  MODEL_PARAM(float, rope_scaling_factor, attention_params_);                  \
  MODEL_PARAM(float, rope_scaling_low_freq_factor, attention_params_);         \
  MODEL_PARAM(float, rope_scaling_high_freq_factor, attention_params_);        \
  MODEL_PARAM(float, rope_extrapolation_factor, attention_params_);            \
  MODEL_PARAM(int64_t,                                                         \
              rope_scaling_original_max_position_embeddings,                   \
              attention_params_);                                              \
  MODEL_PARAM(int64_t, rope_scaling_beta_fast, attention_params_);             \
  MODEL_PARAM(int64_t, rope_scaling_beta_slow, attention_params_);             \
  MODEL_PARAM(float, rope_scaling_attn_factor, attention_params_);             \
  MODEL_PARAM(float, rope_scaling_mscale, attention_params_);                  \
  MODEL_PARAM(float, rope_scaling_mscale_all_dim, attention_params_);          \
  MODEL_PARAM(                                                                 \
      std::vector<int64_t>, rope_scaling_mrope_section, attention_params_);    \
  MODEL_PARAM(bool, rope_scaling_mrope_interleaved, attention_params_);        \
  MODEL_PARAM(int64_t, max_position_embeddings, execution_params_);            \
  MODEL_PARAM(bool, use_sliding_window, attention_params_);                    \
  MODEL_PARAM(int32_t, sliding_window, attention_params_);                     \
  MODEL_PARAM(int32_t, index_head_dim, kv_cache_params_);                      \
  MODEL_PARAM(int32_t, index_n_heads, kv_cache_params_);                       \
  MODEL_PARAM(int32_t, index_topk, kv_cache_params_);                          \
  MODEL_PARAM(int32_t, index_kpool, kv_cache_params_);                         \
  MODEL_PARAM(bool, index_kpool_compress, kv_cache_params_);                   \
  MODEL_PARAM(bool, index_kpool_always_select_tail, kv_cache_params_);         \
  MODEL_PARAM(bool, indexer_rope_interleave, kv_cache_params_);                \
  MODEL_PARAM(int32_t, index_topk_freq, kv_cache_params_);                     \
  MODEL_PARAM(std::string, index_topk_pattern, kv_cache_params_);              \
  MODEL_PARAM(int32_t, index_skip_topk_offset, kv_cache_params_);              \
  MODEL_PARAM(bool, index_share_for_mtp_iteration, kv_cache_params_);          \
  MODEL_PARAM(std::vector<std::string>, indexer_types, kv_cache_params_);      \
  MODEL_PARAM(std::vector<std::string>, mlp_layer_types, moe_params_);         \
  MODEL_PARAM(int32_t, rope_head_dim, kv_cache_params_);                       \
  MODEL_PARAM(int32_t, o_lora_rank, kv_cache_params_);                         \
  MODEL_PARAM(int32_t, o_groups, kv_cache_params_);                            \
  MODEL_PARAM(std::vector<int32_t>, compress_ratios, kv_cache_params_);        \
  MODEL_PARAM(float, compress_rope_theta, kv_cache_params_);                   \
  MODEL_PARAM(int32_t, window_size, kv_cache_params_);                         \
  MODEL_PARAM(int32_t, n_activated_experts, moe_params_);                      \
  MODEL_PARAM(int32_t, n_hash_layers, moe_params_);                            \
  MODEL_PARAM(float, factor, kv_cache_params_);                                \
  MODEL_PARAM(float, beta_fast, kv_cache_params_);                             \
  MODEL_PARAM(float, beta_slow, kv_cache_params_);                             \
  MODEL_PARAM(std::string, scale_fmt, kv_cache_params_);                       \
  MODEL_PARAM(int32_t, hc_mult, kv_cache_params_);                             \
  MODEL_PARAM(int32_t, hc_sinkhorn_iters, kv_cache_params_);                   \
  MODEL_PARAM(float, hc_eps, kv_cache_params_);                                \
  MODEL_PARAM(int64_t, max_batch_size, kv_cache_params_);                      \
  MODEL_PARAM(int64_t, max_seq_len, kv_cache_params_);                         \
  MODEL_PARAM(bool, use_absolute_position_embedding, attention_params_);       \
  MODEL_PARAM(int32_t, bos_token_id, execution_params_);                       \
  MODEL_PARAM(int32_t, eos_token_id, execution_params_);                       \
  MODEL_PARAM(std::vector<int32_t>, eos_token_id_vec, execution_params_);      \
  MODEL_PARAM(int32_t, pad_token_id, execution_params_);                       \
  MODEL_PARAM(std::optional<float>, attn_scalar, attention_params_);           \
  MODEL_PARAM(bool, no_bias, execution_params_);                               \
  MODEL_PARAM(bool, qkv_bias, attention_params_);                              \
  MODEL_PARAM(std::unordered_set<int32_t>, stop_token_ids, execution_params_); \
  MODEL_PARAM(bool, tie_word_embeddings, execution_params_);                   \
  MODEL_PARAM(bool, attention_bias, attention_params_);                        \
  MODEL_PARAM(float, attention_dropout, attention_params_);                    \
  MODEL_PARAM(int32_t, decoder_sparse_step, moe_params_);                      \
  MODEL_PARAM(float, initializer_range, execution_params_);                    \
  MODEL_PARAM(std::vector<int32_t>, mlp_only_layers, moe_params_);             \
  MODEL_PARAM(int64_t, num_attention_heads, attention_params_);                \
  MODEL_PARAM(int32_t, num_experts, moe_params_);                              \
  MODEL_PARAM(bool, output_router_logits, moe_params_);                        \
  MODEL_PARAM(int32_t, rope_scaling, execution_params_);                       \
  MODEL_PARAM(float, router_aux_loss_coef, moe_params_);                       \
  MODEL_PARAM(bool, attn_output_gate, attention_params_);                      \
  MODEL_PARAM(int32_t, full_attention_interval, linear_attention_params_);     \
  MODEL_PARAM(                                                                 \
      std::vector<int32_t>, full_attn_layers, linear_attention_params_);       \
  MODEL_PARAM(int32_t, linear_conv_kernel_dim, linear_attention_params_);      \
  MODEL_PARAM(int32_t, linear_key_head_dim, linear_attention_params_);         \
  MODEL_PARAM(int32_t, linear_value_head_dim, linear_attention_params_);       \
  MODEL_PARAM(int64_t, linear_num_key_heads, linear_attention_params_);        \
  MODEL_PARAM(int32_t, linear_num_value_heads, linear_attention_params_);      \
  MODEL_PARAM(float, linear_lower_bound, linear_attention_params_);            \
  MODEL_PARAM(std::string, mamba_ssm_dtype, linear_attention_params_);         \
  MODEL_PARAM(int32_t, shared_expert_intermediate_size, moe_params_);          \
  MODEL_PARAM(float, partial_rotary_factor, attention_params_);                \
  MODEL_PARAM(std::vector<std::string>, layer_types, linear_attention_params_);

class LlmModelArgs final {
 private:
  AttentionModelParams attention_params_;
  LinearAttentionModelParams linear_attention_params_;
  KVCacheModelParams kv_cache_params_;
  MoEModelParams moe_params_;
  SpeculativeModelParams speculative_params_;
  EmbeddingModelParams embedding_params_;
  ExecutionModelParams execution_params_;

 public:
  LlmModelArgs()
      : LlmModelArgs(model_family_args_detail::default_model_args()) {}
  explicit LlmModelArgs(const ModelArgs& args)
      : attention_params_(args),
        linear_attention_params_(args),
        kv_cache_params_(args),
        moe_params_(args),
        speculative_params_(args),
        embedding_params_(args),
        execution_params_(args) {}

  explicit LlmModelArgs(ModelArgs&&) = delete;
  explicit LlmModelArgs(const ModelArgs&&) = delete;

  const AttentionModelParams& attention_params() const noexcept {
    return attention_params_;
  }
  AttentionModelParams& attention_params() noexcept {
    return attention_params_;
  }
  const LinearAttentionModelParams& linear_attention_params() const noexcept {
    return linear_attention_params_;
  }
  LinearAttentionModelParams& linear_attention_params() noexcept {
    return linear_attention_params_;
  }
  const KVCacheModelParams& kv_cache_params() const noexcept {
    return kv_cache_params_;
  }
  KVCacheModelParams& kv_cache_params() noexcept { return kv_cache_params_; }
  const MoEModelParams& moe_params() const noexcept { return moe_params_; }
  MoEModelParams& moe_params() noexcept { return moe_params_; }
  const SpeculativeModelParams& speculative_params() const noexcept {
    return speculative_params_;
  }
  SpeculativeModelParams& speculative_params() noexcept {
    return speculative_params_;
  }
  const EmbeddingModelParams& embedding_params() const noexcept {
    return embedding_params_;
  }
  EmbeddingModelParams& embedding_params() noexcept {
    return embedding_params_;
  }
  const ExecutionModelParams& execution_params() const noexcept {
    return execution_params_;
  }
  ExecutionModelParams& execution_params() noexcept {
    return execution_params_;
  }
  EmbeddingModelParams& mutable_embedding_params() noexcept {
    return embedding_params();
  }

  COMMON_MODEL_PARAMS

  MODEL_PARAM(int32_t, first_k_dense_replace, moe_params_);
  MODEL_PARAM(int32_t, moe_layer_freq, moe_params_);
  MODEL_PARAM(int64_t, draft_vocab_size, speculative_params_);
  MODEL_PARAM(int64_t, markov_rank, speculative_params_);
  MODEL_PARAM(int32_t, dspark_num_layers, speculative_params_);
  MODEL_PARAM(int32_t, dspark_block_size, speculative_params_);
  MODEL_PARAM(bool, dspark_use_native_sas, speculative_params_);
  MODEL_PARAM(bool, enable_confidence_head, speculative_params_);
  MODEL_PARAM(bool, confidence_head_with_markov, speculative_params_);
  MODEL_PARAM(int32_t, dflash2_block_size, speculative_params_);
  MODEL_PARAM(int32_t, dflash2_conv_group_size, speculative_params_);
  MODEL_PARAM(int32_t, dflash2_conv_kernel_size, speculative_params_);
  MODEL_PARAM(int32_t, dflash2_selector_rank, speculative_params_);
  MODEL_PARAM(int32_t, dflash2_selector_top_k, speculative_params_);
  MODEL_PARAM(bool, use_moe, moe_params_);
  MODEL_PARAM(std::string, moe_score_func, moe_params_);
  MODEL_PARAM(float, moe_route_scale, moe_params_);
  MODEL_PARAM(bool, moe_use_shared_experts, moe_params_);
  MODEL_PARAM(std::string, topk_method, moe_params_);
  MODEL_PARAM(int32_t, n_routed_experts, moe_params_);
  MODEL_PARAM(int32_t, n_shared_experts, moe_params_);

  MODEL_PARAM(int32_t, moe_intermediate_size, moe_params_);
  MODEL_PARAM(float, routed_scaling_factor, moe_params_);
  MODEL_PARAM(bool, norm_topk_prob, moe_params_);
  MODEL_PARAM(int32_t, n_group, moe_params_);
  MODEL_PARAM(int32_t, topk_group, moe_params_);
  MODEL_PARAM(std::string, scoring_func, moe_params_);
  MODEL_PARAM(float, swiglu_limit, moe_params_);

  MODEL_PARAM(int64_t, mtp_start_layer_idx, speculative_params_);
  MODEL_PARAM(std::string, mtp_mlp_type, speculative_params_);

  MODEL_PARAM(int32_t, max_window_layers, attention_params_);
  MODEL_PARAM(int64_t, num_speculative_tokens, speculative_params_);
  MODEL_PARAM(bool, enable_embedded_eagle3_draft, speculative_params_);
  MODEL_PARAM(int32_t, max_concurrent_prefills_per_dp, execution_params_);
  MODEL_PARAM(std::vector<int32_t>, layers_to_capture, speculative_params_);

  MODEL_PARAM(bool, norm_before_fc, speculative_params_);
  MODEL_PARAM(bool, norm_before_residual, speculative_params_);
  MODEL_PARAM(bool, norm_output, speculative_params_);
  MODEL_PARAM(bool, fc_norm, speculative_params_);
  MODEL_PARAM(bool, embedding_mode, embedding_params_);
};

class VlmVisionModelArgs final {
 public:
  VlmVisionModelArgs()
      : VlmVisionModelArgs(model_family_args_detail::default_model_args()) {}
  explicit VlmVisionModelArgs(const ModelArgs& args)
      : dtype_(args.dtype()),
        rms_norm_eps_(args.rms_norm_eps()),
        vision_start_token_id_(args.vision_start_token_id()),
        vision_end_token_id_(args.vision_end_token_id()),
        vision_token_id_(args.vision_token_id()),
        image_token_id_(args.image_token_id()),
        video_token_id_(args.video_token_id()),
        audio_token_id_(args.audio_token_id()),
        image_start_token_id_(args.image_start_token_id()),
        image_end_token_id_(args.image_end_token_id()),
        video_start_token_id_(args.video_start_token_id()),
        video_end_token_id_(args.video_end_token_id()),
        vision_custom_adapter_(args.vision_custom_adapter()),
        vision_max_slice_nums_(args.vision_max_slice_nums()),
        mm_dropout_(args.mm_dropout()),
        mm_hidden_act_(args.mm_hidden_act()),
        mm_hidden_size_(args.mm_hidden_size()),
        mm_image_size_(args.mm_image_size()),
        mm_intermediate_size_(args.mm_intermediate_size()),
        mm_num_channels_(args.mm_num_channels()),
        mm_initializer_range_(args.mm_initializer_range()),
        mm_layer_norm_eps_(args.mm_layer_norm_eps()),
        mm_num_attention_heads_(args.mm_num_attention_heads()),
        mm_num_beam_groups_(args.mm_num_beam_groups()),
        mm_num_beams_(args.mm_num_beams()),
        mm_num_hidden_layers_(args.mm_num_hidden_layers()),
        mm_num_return_sequences_(args.mm_num_return_sequences()),
        mm_output_attentions_(args.mm_output_attentions()),
        mm_output_hidden_states_(args.mm_output_hidden_states()),
        mm_output_scores_(args.mm_output_scores()),
        mm_patch_size_(args.mm_patch_size()),
        mm_projection_dim_(args.mm_projection_dim()),
        mm_projector_hidden_size_(args.mm_projector_hidden_size()),
        mm_spatial_merge_size_(args.mm_spatial_merge_size()),
        mm_spatial_patch_size_(args.mm_spatial_patch_size()),
        mm_remove_invalid_values_(args.mm_remove_invalid_values()),
        mm_repetition_penalty_(args.mm_repetition_penalty()),
        mm_return_dict_(args.mm_return_dict()),
        mm_return_dict_in_generate_(args.mm_return_dict_in_generate()),
        mm_temperature_(args.mm_temperature()),
        mm_tie_encoder_decoder_(args.mm_tie_encoder_decoder()),
        mm_tie_word_embeddings_(args.mm_tie_word_embeddings()),
        mm_top_k_(args.mm_top_k()),
        mm_top_p_(args.mm_top_p()),
        mm_torchscript_(args.mm_torchscript()),
        mm_use_bfloat16_(args.mm_use_bfloat16()),
        mm_head_dim_(args.mm_head_dim()),
        mm_vocab_size_(args.mm_vocab_size()),
        mm_window_size_(args.mm_window_size()),
        mm_fullatt_block_indexes_(args.mm_fullatt_block_indexes()),
        mm_deepstack_visual_indexes_(args.mm_deepstack_visual_indexes()),
        mm_tokens_per_second_(args.mm_tokens_per_second()),
        mm_temporal_patch_size_(args.mm_temporal_patch_size()),
        mm_projector_type_(args.mm_projector_type()),
        mm_num_position_embeddings_(args.mm_num_position_embeddings()),
        mm_projector_hidden_act_(args.mm_projector_hidden_act()),
        mm_projector_n_layers_(args.mm_projector_n_layers()),
        mm_vision_feature_layer_(args.mm_vision_feature_layer()),
        mm_vision_feature_select_strategy_(
            args.mm_vision_feature_select_strategy()),
        mm_image_do_center_crop_(args.mm_image_do_center_crop()),
        mm_image_crop_height_size_(args.mm_image_crop_height_size()),
        mm_image_crop_width_size_(args.mm_image_crop_width_size()),
        mm_image_do_resize_(args.mm_image_do_resize()),
        mm_image_resize_shortest_edge_(args.mm_image_resize_shortest_edge()),
        mm_image_resample_(args.mm_image_resample()),
        mm_image_do_rescale_(args.mm_image_do_rescale()),
        mm_video_do_rescale_(args.mm_video_do_rescale()),
        mm_image_rescale_factor_(args.mm_image_rescale_factor()),
        mm_image_do_normalize_(args.mm_image_do_normalize()),
        mm_image_normalize_mean_(args.mm_image_normalize_mean()),
        mm_image_normalize_std_(args.mm_image_normalize_std()),
        mm_init_pos_emb_width_(args.mm_init_pos_emb_width()),
        mm_init_pos_emb_height_(args.mm_init_pos_emb_height()),
        mm_init_pos_emb_time_(args.mm_init_pos_emb_time()),
        mm_km_in_patch_limit_(args.mm_km_in_patch_limit()),
        mm_km_patch_size_(args.mm_km_patch_size()),
        mm_km_image_mean_(args.mm_km_image_mean()),
        mm_km_image_std_(args.mm_km_image_std()),
        mm_km_merge_kernel_size_(args.mm_km_merge_kernel_size()),
        mm_km_fixed_output_tokens_(args.mm_km_fixed_output_tokens()),
        mm_km_patch_limit_on_one_side_(args.mm_km_patch_limit_on_one_side()),
        mm_km_in_patch_limit_each_frame_(
            args.mm_km_in_patch_limit_each_frame()),
        mm_km_in_patch_limit_video_(args.mm_km_in_patch_limit_video()),
        mm_km_sample_fps_(args.mm_km_sample_fps()),
        mm_km_max_num_frames_each_video_(
            args.mm_km_max_num_frames_each_video()),
        mm_km_temporal_merge_kernel_size_(
            args.mm_km_temporal_merge_kernel_size()),
        mm_km_timestamp_mode_(args.mm_km_timestamp_mode()),
        mm_video_normalize_mean_(args.mm_video_normalize_mean()),
        mm_video_normalize_std_(args.mm_video_normalize_std()),
        mm_image_min_pixels_(args.mm_image_min_pixels()),
        mm_image_max_pixels_(args.mm_image_max_pixels()),
        mm_image_shortest_edge_(args.mm_image_shortest_edge()),
        mm_image_longest_edge_(args.mm_image_longest_edge()),
        mm_video_shortest_edge_(args.mm_video_shortest_edge()),
        mm_video_longest_edge_(args.mm_video_longest_edge()),
        mm_image_patch_size_(args.mm_image_patch_size()),
        mm_image_temporal_patch_size_(args.mm_image_temporal_patch_size()),
        mm_image_merge_size_(args.mm_image_merge_size()),
        mm_image_min_tokens_(args.mm_image_min_tokens()),
        mm_image_max_tokens_(args.mm_image_max_tokens()),
        mm_video_patch_size_(args.mm_video_patch_size()),
        mm_video_temporal_patch_size_(args.mm_video_temporal_patch_size()),
        mm_video_merge_size_(args.mm_video_merge_size()),
        mm_video_min_tokens_(args.mm_video_min_tokens()),
        mm_video_max_tokens_(args.mm_video_max_tokens()),
        mm_image_feature_size_(args.mm_image_feature_size()),
        mm_scale_resolution_(args.mm_scale_resolution()),
        mm_slice_mode_(args.mm_slice_mode()),
        mm_use_image_id_(args.mm_use_image_id()),
        mm_image_token_index_(args.mm_image_token_index()),
        mm_pad_token_id_(args.mm_pad_token_id()),
        mm_audio_sampling_rate_(args.mm_audio_sampling_rate()),
        mm_audio_num_mel_bins_(args.mm_audio_num_mel_bins()),
        mm_audio_frame_length_(args.mm_audio_frame_length()),
        mm_audio_frame_shift_(args.mm_audio_frame_shift()),
        mm_audio_max_frames_(args.mm_audio_max_frames()),
        mm_audio_downsample_rate_(args.mm_audio_downsample_rate()),
        mm_audio_cmvn_means_(args.mm_audio_cmvn_means()),
        mm_audio_cmvn_inverse_std_(args.mm_audio_cmvn_inverse_std()),
        mm_audio_idim_(args.mm_audio_idim()),
        mm_audio_d_model_(args.mm_audio_d_model()),
        mm_audio_n_layers_(args.mm_audio_n_layers()),
        mm_audio_n_head_(args.mm_audio_n_head()),
        mm_audio_kernel_size_(args.mm_audio_kernel_size()),
        mm_audio_pe_maxlen_(args.mm_audio_pe_maxlen()),
        mm_audio_ctc_pad_token_id_(args.mm_audio_ctc_pad_token_id()) {}

  explicit VlmVisionModelArgs(ModelArgs&&) = delete;
  explicit VlmVisionModelArgs(const ModelArgs&&) = delete;

  PROPERTY(std::string, dtype) = {};
  PROPERTY(float, rms_norm_eps) = 0.0f;
  PROPERTY(int32_t, vision_start_token_id) = 0;
  PROPERTY(int32_t, vision_end_token_id) = 0;
  PROPERTY(int32_t, vision_token_id) = 0;
  PROPERTY(int32_t, image_token_id) = 0;
  PROPERTY(int32_t, video_token_id) = 0;
  PROPERTY(int32_t, audio_token_id) = 0;
  PROPERTY(int32_t, image_start_token_id) = 0;
  PROPERTY(int32_t, image_end_token_id) = 0;
  PROPERTY(int32_t, video_start_token_id) = 0;
  PROPERTY(int32_t, video_end_token_id) = 0;
  PROPERTY(std::string, vision_custom_adapter) = {};
  PROPERTY(int32_t, vision_max_slice_nums) = 0;
  PROPERTY(float, mm_dropout) = 0.0f;
  PROPERTY(std::string, mm_hidden_act) = {};
  PROPERTY(int64_t, mm_hidden_size) = 0;
  PROPERTY(int64_t, mm_image_size) = 0;
  PROPERTY(int64_t, mm_intermediate_size) = 0;
  PROPERTY(int64_t, mm_num_channels) = 0;
  PROPERTY(float, mm_initializer_range) = 0.0f;
  PROPERTY(float, mm_layer_norm_eps) = 0;
  PROPERTY(int64_t, mm_num_attention_heads) = 0;
  PROPERTY(int64_t, mm_num_beam_groups) = 0;
  PROPERTY(int64_t, mm_num_beams) = 0;
  PROPERTY(int64_t, mm_num_hidden_layers) = 0;
  PROPERTY(int64_t, mm_num_return_sequences) = 0;
  PROPERTY(bool, mm_output_attentions) = false;
  PROPERTY(bool, mm_output_hidden_states) = false;
  PROPERTY(bool, mm_output_scores) = false;
  PROPERTY(int64_t, mm_patch_size) = 0;
  PROPERTY(int64_t, mm_projection_dim) = 0;
  PROPERTY(int64_t, mm_projector_hidden_size) = 0;
  PROPERTY(int64_t, mm_spatial_merge_size) = 0;
  PROPERTY(int64_t, mm_spatial_patch_size) = 0;
  PROPERTY(bool, mm_remove_invalid_values) = false;
  PROPERTY(float, mm_repetition_penalty) = 0.0f;
  PROPERTY(bool, mm_return_dict) = false;
  PROPERTY(bool, mm_return_dict_in_generate) = false;
  PROPERTY(float, mm_temperature) = 0.0f;
  PROPERTY(bool, mm_tie_encoder_decoder) = false;
  PROPERTY(bool, mm_tie_word_embeddings) = false;
  PROPERTY(int64_t, mm_top_k) = 0;
  PROPERTY(float, mm_top_p) = 0.0f;
  PROPERTY(bool, mm_torchscript) = false;
  PROPERTY(bool, mm_use_bfloat16) = false;
  PROPERTY(int64_t, mm_head_dim) = 0;
  PROPERTY(int64_t, mm_vocab_size) = 0;
  PROPERTY(int32_t, mm_window_size) = 0;
  PROPERTY(std::vector<int64_t>, mm_fullatt_block_indexes) = {};
  PROPERTY(std::vector<int64_t>, mm_deepstack_visual_indexes) = {};
  PROPERTY(int32_t, mm_tokens_per_second) = 0;
  PROPERTY(int32_t, mm_temporal_patch_size) = 0;
  PROPERTY(std::string, mm_projector_type) = {};
  PROPERTY(int64_t, mm_num_position_embeddings) = {};
  PROPERTY(std::string, mm_projector_hidden_act) = {};
  PROPERTY(int64_t, mm_projector_n_layers) = 0;
  PROPERTY(int64_t, mm_vision_feature_layer) = 0;
  PROPERTY(std::string, mm_vision_feature_select_strategy) = {};
  PROPERTY(bool, mm_image_do_center_crop) = false;
  PROPERTY(int32_t, mm_image_crop_height_size) = 336;
  PROPERTY(int32_t, mm_image_crop_width_size) = 336;
  PROPERTY(bool, mm_image_do_resize) = false;
  PROPERTY(int32_t, mm_image_resize_shortest_edge) = 336;
  PROPERTY(int32_t, mm_image_resample) = 0;
  PROPERTY(bool, mm_image_do_rescale) = false;
  PROPERTY(bool, mm_video_do_rescale) = false;
  PROPERTY(double, mm_image_rescale_factor) = 0;
  PROPERTY(bool, mm_image_do_normalize) = false;
  PROPERTY(std::vector<double>, mm_image_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_image_normalize_std) = {};
  PROPERTY(int64_t, mm_init_pos_emb_width) = 64;
  PROPERTY(int64_t, mm_init_pos_emb_height) = 64;
  PROPERTY(int64_t, mm_init_pos_emb_time) = 4;
  PROPERTY(int64_t, mm_km_in_patch_limit) = 16384;
  PROPERTY(int64_t, mm_km_patch_size) = 14;
  PROPERTY(std::vector<int64_t>, mm_km_image_mean) = {};
  PROPERTY(std::vector<int64_t>, mm_km_image_std) = {};
  PROPERTY(int64_t, mm_km_merge_kernel_size) = 2;
  PROPERTY(int64_t, mm_km_fixed_output_tokens) = -1;
  PROPERTY(int64_t, mm_km_patch_limit_on_one_side) = 512;
  PROPERTY(int64_t, mm_km_in_patch_limit_each_frame) = 4096;
  PROPERTY(int64_t, mm_km_in_patch_limit_video) = 200;
  PROPERTY(float, mm_km_sample_fps) = 2.0;
  PROPERTY(int64_t, mm_km_max_num_frames_each_video) = 2;
  PROPERTY(int64_t, mm_km_temporal_merge_kernel_size) = 4;
  PROPERTY(std::string, mm_km_timestamp_mode) = "hh:mm:ss.fff";
  PROPERTY(std::vector<double>, mm_video_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_video_normalize_std) = {};
  PROPERTY(int32_t, mm_image_min_pixels) = 0;
  PROPERTY(int32_t, mm_image_max_pixels) = 0;
  PROPERTY(int64_t, mm_image_shortest_edge) = 0;
  PROPERTY(int64_t, mm_image_longest_edge) = 0;
  PROPERTY(int64_t, mm_video_shortest_edge) = 0;
  PROPERTY(int64_t, mm_video_longest_edge) = 0;
  PROPERTY(int32_t, mm_image_patch_size) = 0;
  PROPERTY(int32_t, mm_image_temporal_patch_size) = 0;
  PROPERTY(int32_t, mm_image_merge_size) = 0;
  PROPERTY(int32_t, mm_image_min_tokens) = 0;
  PROPERTY(int32_t, mm_image_max_tokens) = 0;
  PROPERTY(int32_t, mm_video_patch_size) = 0;
  PROPERTY(int32_t, mm_video_temporal_patch_size) = 0;
  PROPERTY(int32_t, mm_video_merge_size) = 0;
  PROPERTY(int32_t, mm_video_min_tokens) = 0;
  PROPERTY(int32_t, mm_video_max_tokens) = 0;
  PROPERTY(int32_t, mm_image_feature_size) = 0;
  PROPERTY(int32_t, mm_scale_resolution) = 0;
  PROPERTY(bool, mm_slice_mode) = false;
  PROPERTY(bool, mm_use_image_id) = false;
  PROPERTY(int64_t, mm_image_token_index) = 0;
  PROPERTY(int64_t, mm_pad_token_id) = 0;
  PROPERTY(int64_t, mm_audio_sampling_rate) = 16000;
  PROPERTY(int64_t, mm_audio_num_mel_bins) = 80;
  PROPERTY(int64_t, mm_audio_frame_length) = 400;
  PROPERTY(int64_t, mm_audio_frame_shift) = 160;
  PROPERTY(int64_t, mm_audio_max_frames) = 3000;
  PROPERTY(int64_t, mm_audio_downsample_rate) = 2;
  PROPERTY(std::vector<double>, mm_audio_cmvn_means) = {};
  PROPERTY(std::vector<double>, mm_audio_cmvn_inverse_std) = {};
  PROPERTY(int64_t, mm_audio_idim) = 80;
  PROPERTY(int64_t, mm_audio_d_model) = 1280;
  PROPERTY(int64_t, mm_audio_n_layers) = 16;
  PROPERTY(int64_t, mm_audio_n_head) = 20;
  PROPERTY(int64_t, mm_audio_kernel_size) = 33;
  PROPERTY(int64_t, mm_audio_pe_maxlen) = 5000;
  PROPERTY(int64_t, mm_audio_ctc_pad_token_id) = 151643;
};

class RecModelArgs final {
 private:
  AttentionModelParams attention_params_;
  LinearAttentionModelParams linear_attention_params_;
  KVCacheModelParams kv_cache_params_;
  MoEModelParams moe_params_;
  SpeculativeModelParams speculative_params_;
  EmbeddingModelParams embedding_params_;
  ExecutionModelParams execution_params_;

 public:
  RecModelArgs()
      : RecModelArgs(model_family_args_detail::default_model_args()) {}
  explicit RecModelArgs(const ModelArgs& args)
      : attention_params_(args),
        linear_attention_params_(args),
        kv_cache_params_(args),
        moe_params_(args),
        speculative_params_(args),
        embedding_params_(args),
        execution_params_(args) {}

  explicit RecModelArgs(ModelArgs&&) = delete;
  explicit RecModelArgs(const ModelArgs&&) = delete;

  const AttentionModelParams& attention_params() const noexcept {
    return attention_params_;
  }
  AttentionModelParams& attention_params() noexcept {
    return attention_params_;
  }
  const LinearAttentionModelParams& linear_attention_params() const noexcept {
    return linear_attention_params_;
  }
  LinearAttentionModelParams& linear_attention_params() noexcept {
    return linear_attention_params_;
  }
  const KVCacheModelParams& kv_cache_params() const noexcept {
    return kv_cache_params_;
  }
  KVCacheModelParams& kv_cache_params() noexcept { return kv_cache_params_; }
  const MoEModelParams& moe_params() const noexcept { return moe_params_; }
  MoEModelParams& moe_params() noexcept { return moe_params_; }
  const SpeculativeModelParams& speculative_params() const noexcept {
    return speculative_params_;
  }
  SpeculativeModelParams& speculative_params() noexcept {
    return speculative_params_;
  }
  const EmbeddingModelParams& embedding_params() const noexcept {
    return embedding_params_;
  }
  EmbeddingModelParams& embedding_params() noexcept {
    return embedding_params_;
  }
  const ExecutionModelParams& execution_params() const noexcept {
    return execution_params_;
  }
  ExecutionModelParams& execution_params() noexcept {
    return execution_params_;
  }
  EmbeddingModelParams& mutable_embedding_params() noexcept {
    return embedding_params();
  }

  COMMON_MODEL_PARAMS

  MODEL_PARAM(int32_t, query_num, embedding_params_);
  MODEL_PARAM(bool, encoder_embedding_mode, embedding_params_);
  MODEL_PARAM(bool, embedding_mode, embedding_params_);
  MODEL_PARAM(bool, use_moe, moe_params_);
  MODEL_PARAM(float, moe_route_scale, moe_params_);
  MODEL_PARAM(bool, moe_use_shared_experts, moe_params_);
  MODEL_PARAM(int32_t, n_routed_experts, moe_params_);
  MODEL_PARAM(int32_t, n_shared_experts, moe_params_);
  MODEL_PARAM(int32_t, moe_intermediate_size, moe_params_);

  MODEL_PARAM(int64_t, relative_attention_num_buckets, attention_params_);
  MODEL_PARAM(int64_t, relative_attention_max_distance, attention_params_);
};

class VlmModelArgs final {
 private:
  AttentionModelParams attention_params_;
  LinearAttentionModelParams linear_attention_params_;
  KVCacheModelParams kv_cache_params_;
  MoEModelParams moe_params_;
  SpeculativeModelParams speculative_params_;
  EmbeddingModelParams embedding_params_;
  ExecutionModelParams execution_params_;
  VlmVisionModelArgs vision_;

 public:
  VlmModelArgs()
      : VlmModelArgs(model_family_args_detail::default_model_args()) {}
  explicit VlmModelArgs(const ModelArgs& args)
      : attention_params_(args),
        linear_attention_params_(args),
        kv_cache_params_(args),
        moe_params_(args),
        speculative_params_(args),
        embedding_params_(args),
        execution_params_(args),
        vision_(args) {}

  explicit VlmModelArgs(ModelArgs&&) = delete;
  explicit VlmModelArgs(const ModelArgs&&) = delete;

  const AttentionModelParams& attention_params() const noexcept {
    return attention_params_;
  }
  AttentionModelParams& attention_params() noexcept {
    return attention_params_;
  }
  const LinearAttentionModelParams& linear_attention_params() const noexcept {
    return linear_attention_params_;
  }
  LinearAttentionModelParams& linear_attention_params() noexcept {
    return linear_attention_params_;
  }
  const KVCacheModelParams& kv_cache_params() const noexcept {
    return kv_cache_params_;
  }
  KVCacheModelParams& kv_cache_params() noexcept { return kv_cache_params_; }
  const MoEModelParams& moe_params() const noexcept { return moe_params_; }
  MoEModelParams& moe_params() noexcept { return moe_params_; }
  const SpeculativeModelParams& speculative_params() const noexcept {
    return speculative_params_;
  }
  SpeculativeModelParams& speculative_params() noexcept {
    return speculative_params_;
  }
  const EmbeddingModelParams& embedding_params() const noexcept {
    return embedding_params_;
  }
  EmbeddingModelParams& embedding_params() noexcept {
    return embedding_params_;
  }
  const ExecutionModelParams& execution_params() const noexcept {
    return execution_params_;
  }
  ExecutionModelParams& execution_params() noexcept {
    return execution_params_;
  }
  EmbeddingModelParams& mutable_embedding_params() noexcept {
    return embedding_params();
  }

  const VlmVisionModelArgs& vision() const noexcept { return vision_; }
  VlmVisionModelArgs& mutable_vision() noexcept { return vision_; }

  COMMON_MODEL_PARAMS

  MODEL_PARAM(int32_t, query_num, embedding_params_);
  MODEL_PARAM(bool, encoder_embedding_mode, embedding_params_);
  MODEL_PARAM(bool, embedding_mode, embedding_params_);
  MODEL_PARAM(int64_t, num_speculative_tokens, speculative_params_);
};

class DiTTransformerModelArgs final {
 private:
  AttentionModelParams attention_params_;
  LinearAttentionModelParams linear_attention_params_;
  KVCacheModelParams kv_cache_params_;
  MoEModelParams moe_params_;
  SpeculativeModelParams speculative_params_;
  EmbeddingModelParams embedding_params_;
  ExecutionModelParams execution_params_;

 public:
  DiTTransformerModelArgs()
      : DiTTransformerModelArgs(
            model_family_args_detail::default_model_args()) {}
  explicit DiTTransformerModelArgs(const ModelArgs& args)
      : attention_params_(args),
        linear_attention_params_(args),
        kv_cache_params_(args),
        moe_params_(args),
        speculative_params_(args),
        embedding_params_(args),
        execution_params_(args),
        in_channels_(args.in_channels()),
        out_channels_(args.out_channels()),
        latent_channels_(args.latent_channels()),
        joint_attention_dim_(args.joint_attention_dim()),
        pooled_projection_dim_(args.pooled_projection_dim()),
        guidance_embeds_(args.guidance_embeds()),
        axes_dims_rope_(args.axes_dims_rope()),
        num_layers_(args.num_layers()),
        num_single_layers_(args.num_single_layers()),
        mlp_ratio_(args.mlp_ratio()),
        timestep_guidance_channels_(args.timestep_guidance_channels()),
        mm_patch_size_(args.mm_patch_size()),
        patch_size_(args.patch_size()),
        wan_patch_size_(args.wan_patch_size()),
        cross_attn_norm_(args.cross_attn_norm()),
        ffn_dim_(args.ffn_dim()),
        time_freq_dim_(args.time_freq_dim()),
        dit_in_channels_(args.dit_in_channels()),
        dit_out_channels_(args.dit_out_channels()),
        qk_norm_(args.qk_norm()),
        rope_max_seq_len_(args.rope_max_seq_len()),
        text_embed_dim_(args.text_embed_dim()),
        image_embed_dim_(args.image_embed_dim()),
        added_kv_proj_dim_(args.added_kv_proj_dim()),
        pos_embed_seq_len_(args.pos_embed_seq_len()),
        num_refiner_layers_(args.num_refiner_layers()),
        audio_in_channels_(args.audio_in_channels()),
        time_embed_hidden_dim_(args.time_embed_hidden_dim()),
        time_embed_dim_(args.time_embed_dim()),
        rope_freq_dim_(args.rope_freq_dim()),
        qk_norm_eps_(args.qk_norm_eps()),
        final_norm_eps_(args.final_norm_eps()),
        mlp_width_ratio_(args.mlp_width_ratio()),
        text_dim_(args.text_dim()),
        rope_dim_list_(args.rope_dim_list()),
        rope_theta_dit_(args.rope_theta_dit()),
        eps_(args.eps()),
        txt_dim_(args.txt_dim()),
        txt_in_channels_(args.txt_in_channels()),
        txt_out_channels_(args.txt_out_channels()),
        emb_dim_(args.emb_dim()),
        heads_(args.heads()),
        rope_dim_(args.rope_dim()),
        expand_ratio_(args.expand_ratio()),
        block_size_(args.block_size()),
        latent_dim_(args.latent_dim()),
        qk_bias_(args.qk_bias()),
        norm_eps_(args.norm_eps()),
        zero_cond_t_(args.zero_cond_t()),
        use_additional_t_cond_(args.use_additional_t_cond()),
        use_layer3d_rope_(args.use_layer3d_rope()) {}

  explicit DiTTransformerModelArgs(ModelArgs&&) = delete;
  explicit DiTTransformerModelArgs(const ModelArgs&&) = delete;

  const AttentionModelParams& attention_params() const noexcept {
    return attention_params_;
  }
  AttentionModelParams& attention_params() noexcept {
    return attention_params_;
  }
  const LinearAttentionModelParams& linear_attention_params() const noexcept {
    return linear_attention_params_;
  }
  LinearAttentionModelParams& linear_attention_params() noexcept {
    return linear_attention_params_;
  }
  const KVCacheModelParams& kv_cache_params() const noexcept {
    return kv_cache_params_;
  }
  KVCacheModelParams& kv_cache_params() noexcept { return kv_cache_params_; }
  const MoEModelParams& moe_params() const noexcept { return moe_params_; }
  MoEModelParams& moe_params() noexcept { return moe_params_; }
  const SpeculativeModelParams& speculative_params() const noexcept {
    return speculative_params_;
  }
  SpeculativeModelParams& speculative_params() noexcept {
    return speculative_params_;
  }
  const EmbeddingModelParams& embedding_params() const noexcept {
    return embedding_params_;
  }
  EmbeddingModelParams& embedding_params() noexcept {
    return embedding_params_;
  }
  const ExecutionModelParams& execution_params() const noexcept {
    return execution_params_;
  }
  ExecutionModelParams& execution_params() noexcept {
    return execution_params_;
  }
  EmbeddingModelParams& mutable_embedding_params() noexcept {
    return embedding_params();
  }

  COMMON_MODEL_PARAMS

  PROPERTY(int64_t, in_channels) = -1;
  PROPERTY(int64_t, out_channels) = -1;
  PROPERTY(int64_t, latent_channels) = -1;
  PROPERTY(int64_t, joint_attention_dim) = 0;
  PROPERTY(int64_t, pooled_projection_dim) = 0;
  PROPERTY(bool, guidance_embeds) = true;
  PROPERTY(std::vector<int64_t>, axes_dims_rope) = {};
  PROPERTY(int64_t, num_layers) = 0;
  PROPERTY(int64_t, num_single_layers) = 0;
  PROPERTY(float, mlp_ratio) = 3.0f;
  PROPERTY(int32_t, timestep_guidance_channels) = 256;
  PROPERTY(int64_t, mm_patch_size) = 0;
  PROPERTY(int64_t, patch_size) = 1;
  PROPERTY(std::vector<int64_t>, wan_patch_size) = { 1, 2, 2 };
  PROPERTY(bool, cross_attn_norm) = true;
  PROPERTY(int64_t, ffn_dim) = 13824;
  PROPERTY(int64_t, time_freq_dim) = 256;
  PROPERTY(int64_t, dit_in_channels) = 36;
  PROPERTY(int64_t, dit_out_channels) = 16;
  PROPERTY(std::string, qk_norm) = "rms_norm_across_heads";
  PROPERTY(int64_t, rope_max_seq_len) = 1024;
  PROPERTY(int64_t, text_embed_dim) = 4096;
  PROPERTY(int64_t, image_embed_dim) = -1;
  PROPERTY(int64_t, added_kv_proj_dim) = -1;
  PROPERTY(int64_t, pos_embed_seq_len) = -1;
  PROPERTY(int64_t, num_refiner_layers) = 0;
  PROPERTY(int64_t, audio_in_channels) = 0;
  PROPERTY(int64_t, time_embed_hidden_dim) = 0;
  PROPERTY(int64_t, time_embed_dim) = 0;
  PROPERTY(int64_t, rope_freq_dim) = 0;
  PROPERTY(float, qk_norm_eps) = 1e-5f;
  PROPERTY(float, final_norm_eps) = 1e-5f;
  PROPERTY(double, mlp_width_ratio) = 4.0;
  PROPERTY(int64_t, text_dim) = 4096;
  PROPERTY(std::vector<int64_t>, rope_dim_list) = { 16, 56, 56 };
  PROPERTY(int64_t, rope_theta_dit) = 10000;
  PROPERTY(double, eps) = 1e-6;
  PROPERTY(int64_t, txt_dim) = 0;
  PROPERTY(int64_t, txt_in_channels) = 0;
  PROPERTY(int64_t, txt_out_channels) = 0;
  PROPERTY(int64_t, emb_dim) = 0;
  PROPERTY(int64_t, heads) = 0;
  PROPERTY(int64_t, rope_dim) = 0;
  PROPERTY(int64_t, expand_ratio) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, latent_dim) = 0;
  PROPERTY(bool, qk_bias) = false;
  PROPERTY(float, norm_eps) = 1e-5f;
  PROPERTY(bool, zero_cond_t) = false;
  PROPERTY(bool, use_additional_t_cond) = false;
  PROPERTY(bool, use_layer3d_rope) = false;
};

class DiTVaeModelArgs final {
 public:
  DiTVaeModelArgs()
      : DiTVaeModelArgs(model_family_args_detail::default_model_args()) {}
  explicit DiTVaeModelArgs(const ModelArgs& args)
      : in_channels_(args.in_channels()),
        out_channels_(args.out_channels()),
        down_block_types_(args.down_block_types()),
        up_block_types_(args.up_block_types()),
        block_out_channels_(args.block_out_channels()),
        layers_per_block_(args.layers_per_block()),
        latent_channels_(args.latent_channels()),
        norm_num_groups_(args.norm_num_groups()),
        sample_size_(args.sample_size()),
        scale_factor_(args.scale_factor()),
        shift_factor_(args.shift_factor()),
        mid_block_add_attention_(args.mid_block_add_attention()),
        force_upcast_(args.force_upcast()),
        use_quant_conv_(args.use_quant_conv()),
        use_post_quant_conv_(args.use_post_quant_conv()),
        vae_scale_factor_temporal_(args.vae_scale_factor_temporal()),
        vae_scale_factor_spatial_(args.vae_scale_factor_spatial()),
        vae_is_residual_(args.vae_is_residual()),
        batch_norm_eps_(args.batch_norm_eps()),
        batch_norm_momentum_(args.batch_norm_momentum()),
        ae_patch_size_(args.ae_patch_size()),
        base_dim_(args.base_dim()),
        z_dim_(args.z_dim()),
        dim_mult_(args.dim_mult()),
        latents_mean_(args.latents_mean()),
        latents_std_(args.latents_std()),
        attn_scales_(args.attn_scales()),
        temperal_downsample_(args.temperal_downsample()),
        num_res_blocks_(args.num_res_blocks()),
        dropout_(args.dropout()),
        spatial_downsample_factors_(args.spatial_downsample_factors()),
        temporal_downsample_factors_(args.temporal_downsample_factors()),
        spatial_padding_mode_(args.spatial_padding_mode()),
        decoder_num_register_tokens_(args.decoder_num_register_tokens()),
        decoder_ffn_mult_(args.decoder_ffn_mult()),
        decoder_head_dim_(args.decoder_head_dim()),
        decoder_n_heads_(args.decoder_n_heads()),
        decoder_rope_theta_(args.decoder_rope_theta()),
        decoder_rope_dim_ratio_(args.decoder_rope_dim_ratio()),
        decoder_norm_eps_(args.decoder_norm_eps()),
        encoder_dim_(args.encoder_dim()),
        encoder_rates_(args.encoder_rates()),
        decoder_dim_(args.decoder_dim()),
        decoder_rates_(args.decoder_rates()),
        decoder_kernel_sizes_(args.decoder_kernel_sizes()),
        resblock_kernel_sizes_(args.resblock_kernel_sizes()),
        sampling_rate_(args.sampling_rate()),
        clip_length_(args.clip_length()),
        token_drop_(args.token_drop()),
        vae_dim_(args.vae_dim()),
        vae_num_heads_(args.vae_num_heads()),
        encoder_num_blocks_(args.encoder_num_blocks()),
        decoder_num_blocks_(args.decoder_num_blocks()),
        shared_heads_kv_(args.shared_heads_kv()),
        vae_rope_theta_(args.vae_rope_theta()),
        vae_block_size_(args.vae_block_size()),
        vae_patch_size_(args.vae_patch_size()),
        encoder_last_ln_(args.encoder_last_ln()),
        shifting_factor_(args.shifting_factor()),
        scaling_factor_(args.scaling_factor()),
        use_variation_(args.use_variation()),
        num_attention_heads_(args.num_attention_heads()),
        ffn_dim_(args.ffn_dim()),
        vocab_size_(args.vocab_size()),
        qk_bias_(args.qk_bias()),
        norm_eps_(args.norm_eps()) {}

  explicit DiTVaeModelArgs(ModelArgs&&) = delete;
  explicit DiTVaeModelArgs(const ModelArgs&&) = delete;

  PROPERTY(int64_t, in_channels) = -1;
  PROPERTY(int64_t, out_channels) = -1;
  PROPERTY(std::vector<std::string>, down_block_types) = {};
  PROPERTY(std::vector<std::string>, up_block_types) = {};
  PROPERTY(std::vector<int64_t>, block_out_channels) = {};
  PROPERTY(int64_t, layers_per_block) = 1;
  PROPERTY(int64_t, latent_channels) = -1;
  PROPERTY(int64_t, norm_num_groups) = -1;
  PROPERTY(int64_t, sample_size) = -1;
  PROPERTY(float, scale_factor) = 0.0f;
  PROPERTY(float, shift_factor) = 0.0f;
  PROPERTY(bool, mid_block_add_attention) = true;
  PROPERTY(bool, force_upcast) = true;
  PROPERTY(bool, use_quant_conv) = false;
  PROPERTY(bool, use_post_quant_conv) = false;
  PROPERTY(int64_t, vae_scale_factor_temporal) = 0;
  PROPERTY(int64_t, vae_scale_factor_spatial) = 0;
  PROPERTY(bool, vae_is_residual) = false;
  PROPERTY(float, batch_norm_eps) = 1e-04f;
  PROPERTY(float, batch_norm_momentum) = 0.1f;
  PROPERTY(std::vector<int64_t>, ae_patch_size) = {};
  PROPERTY(int64_t, base_dim) = 0;
  PROPERTY(int64_t, z_dim) = 0;
  PROPERTY(std::vector<int64_t>, dim_mult) = {};
  PROPERTY(std::vector<double>, latents_mean) = {};
  PROPERTY(std::vector<double>, latents_std) = {};
  PROPERTY(std::vector<double>, attn_scales) = {};
  PROPERTY(std::vector<bool>, temperal_downsample) = {};
  PROPERTY(int64_t, num_res_blocks) = 0;
  PROPERTY(double, dropout) = 0;
  PROPERTY(std::vector<int64_t>, spatial_downsample_factors) = {};
  PROPERTY(std::vector<int64_t>, temporal_downsample_factors) = {};
  PROPERTY(std::string, spatial_padding_mode) = "reflect";
  PROPERTY(int64_t, decoder_num_register_tokens) = 0;
  PROPERTY(int64_t, decoder_ffn_mult) = 0;
  PROPERTY(int64_t, decoder_head_dim) = 0;
  PROPERTY(int64_t, decoder_n_heads) = 0;
  PROPERTY(float, decoder_rope_theta) = 0.0f;
  PROPERTY(float, decoder_rope_dim_ratio) = 0.0f;
  PROPERTY(float, decoder_norm_eps) = 1e-5f;
  PROPERTY(int64_t, encoder_dim) = 0;
  PROPERTY(std::vector<int64_t>, encoder_rates) = {};
  PROPERTY(int64_t, decoder_dim) = 0;
  PROPERTY(std::vector<int64_t>, decoder_rates) = {};
  PROPERTY(std::vector<int64_t>, decoder_kernel_sizes) = {};
  PROPERTY(std::vector<int64_t>, resblock_kernel_sizes) = {};
  PROPERTY(int64_t, sampling_rate) = 32000;
  PROPERTY(int64_t, clip_length) = 0;
  PROPERTY(int64_t, token_drop) = 0;
  PROPERTY(int64_t, vae_dim) = 0;
  PROPERTY(int64_t, vae_num_heads) = 0;
  PROPERTY(int64_t, encoder_num_blocks) = 0;
  PROPERTY(int64_t, decoder_num_blocks) = 0;
  PROPERTY(int64_t, shared_heads_kv) = 0;
  PROPERTY(int64_t, vae_rope_theta) = 0;
  PROPERTY(int64_t, vae_block_size) = 0;
  PROPERTY(int64_t, vae_patch_size) = 0;
  PROPERTY(bool, encoder_last_ln) = true;
  PROPERTY(float, shifting_factor) = 0.0f;
  PROPERTY(float, scaling_factor) = 0.0f;
  PROPERTY(bool, use_variation) = true;
  PROPERTY(int64_t, num_attention_heads) = 32;
  PROPERTY(int64_t, ffn_dim) = 13824;
  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(bool, qk_bias) = false;
  PROPERTY(float, norm_eps) = 1e-5f;
};

class DiTTextEncoderModelArgs final {
 public:
  DiTTextEncoderModelArgs()
      : DiTTextEncoderModelArgs(
            model_family_args_detail::default_model_args()) {}
  explicit DiTTextEncoderModelArgs(const ModelArgs& args)
      : n_heads_(args.n_heads()),
        vocab_size_(args.vocab_size()),
        layer_norm_eps_(args.layer_norm_eps()),
        eos_token_id_(args.eos_token_id()),
        mm_head_dim_(args.mm_head_dim()),
        mm_hidden_size_(args.mm_hidden_size()),
        mm_intermediate_size_(args.mm_intermediate_size()),
        mm_layer_norm_eps_(args.mm_layer_norm_eps()),
        mm_num_attention_heads_(args.mm_num_attention_heads()),
        mm_num_hidden_layers_(args.mm_num_hidden_layers()),
        d_model_(args.d_model()),
        max_position_embeddings_(args.max_position_embeddings()),
        num_layers_(args.num_layers()),
        d_kv_(args.d_kv()),
        d_ff_(args.d_ff()),
        act_fn_(args.act_fn()),
        is_gated_act_(args.is_gated_act()),
        relative_attention_num_buckets_(args.relative_attention_num_buckets()),
        relative_attention_max_distance_(
            args.relative_attention_max_distance()) {}

  explicit DiTTextEncoderModelArgs(ModelArgs&&) = delete;
  explicit DiTTextEncoderModelArgs(const ModelArgs&&) = delete;

  PROPERTY(int64_t, n_heads) = 0;
  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(float, layer_norm_eps) = 0.0f;
  PROPERTY(int32_t, eos_token_id) = -1;
  PROPERTY(int64_t, mm_head_dim) = 0;
  PROPERTY(int64_t, mm_hidden_size) = 0;
  PROPERTY(int64_t, mm_intermediate_size) = 0;
  PROPERTY(float, mm_layer_norm_eps) = 0.0f;
  PROPERTY(int64_t, mm_num_attention_heads) = 0;
  PROPERTY(int64_t, mm_num_hidden_layers) = 0;
  PROPERTY(int64_t, d_model) = 0;
  PROPERTY(int64_t, max_position_embeddings) = 0;
  PROPERTY(int64_t, num_layers) = 0;
  PROPERTY(int64_t, d_kv) = 0;
  PROPERTY(int64_t, d_ff) = 0;
  PROPERTY(std::string, act_fn) = "";
  PROPERTY(bool, is_gated_act) = true;
  PROPERTY(int64_t, relative_attention_num_buckets) = 0;
  PROPERTY(int64_t, relative_attention_max_distance) = 0;
};

class DiTSchedulerModelArgs final {
 public:
  DiTSchedulerModelArgs()
      : DiTSchedulerModelArgs(model_family_args_detail::default_model_args()) {}
  explicit DiTSchedulerModelArgs(const ModelArgs& args)
      : num_train_timesteps_(args.num_train_timesteps()),
        shift_(args.shift()),
        use_dynamic_shifting_(args.use_dynamic_shifting()),
        base_shift_(args.base_shift()),
        max_shift_(args.max_shift()),
        base_image_seq_len_(args.base_image_seq_len()),
        max_image_seq_len_(args.max_image_seq_len()),
        shift_terminal_(args.shift_terminal()),
        beta_start_(args.beta_start()),
        beta_end_(args.beta_end()),
        beta_schedule_(args.beta_schedule()),
        trained_betas_(args.trained_betas()),
        solver_order_(args.solver_order()),
        prediction_type_(args.prediction_type()),
        thresholding_(args.thresholding()),
        dynamic_thresholding_ratio_(args.dynamic_thresholding_ratio()),
        sample_max_value_(args.sample_max_value()),
        predict_x0_(args.predict_x0()),
        solver_type_(args.solver_type()),
        lower_order_final_(args.lower_order_final()),
        disable_corrector_(args.disable_corrector()),
        use_karras_sigmas_(args.use_karras_sigmas()),
        use_exponential_sigmas_(args.use_exponential_sigmas()),
        use_beta_sigmas_(args.use_beta_sigmas()),
        use_flow_sigmas_(args.use_flow_sigmas()),
        flow_shift_(args.flow_shift()),
        timestep_spacing_(args.timestep_spacing()),
        steps_offset_(args.steps_offset()),
        final_sigmas_type_(args.final_sigmas_type()),
        rescale_betas_zero_snr_(args.rescale_betas_zero_snr()),
        time_shift_type_(args.time_shift_type()) {}

  explicit DiTSchedulerModelArgs(ModelArgs&&) = delete;
  explicit DiTSchedulerModelArgs(const ModelArgs&&) = delete;

  PROPERTY(int64_t, num_train_timesteps) = 0;
  PROPERTY(int64_t, shift) = 0;
  PROPERTY(bool, use_dynamic_shifting) = false;
  PROPERTY(float, base_shift) = 0;
  PROPERTY(float, max_shift) = 0;
  PROPERTY(int64_t, base_image_seq_len) = 0;
  PROPERTY(int64_t, max_image_seq_len) = 0;
  PROPERTY(float, shift_terminal) = 0;
  PROPERTY(float, beta_start) = 0.0001f;
  PROPERTY(float, beta_end) = 0.02f;
  PROPERTY(std::string, beta_schedule) = "linear";
  PROPERTY(std::vector<float>, trained_betas) = {};
  PROPERTY(int64_t, solver_order) = 2;
  PROPERTY(std::string, prediction_type) = "flow_prediction";
  PROPERTY(bool, thresholding) = false;
  PROPERTY(float, dynamic_thresholding_ratio) = 0.995f;
  PROPERTY(float, sample_max_value) = 1.0f;
  PROPERTY(bool, predict_x0) = true;
  PROPERTY(std::string, solver_type) = "bh2";
  PROPERTY(bool, lower_order_final) = true;
  PROPERTY(std::vector<int64_t>, disable_corrector) = {};
  PROPERTY(bool, use_karras_sigmas) = false;
  PROPERTY(bool, use_exponential_sigmas) = false;
  PROPERTY(bool, use_beta_sigmas) = false;
  PROPERTY(bool, use_flow_sigmas) = true;
  PROPERTY(float, flow_shift) = 3.0f;
  PROPERTY(std::string, timestep_spacing) = "linspace";
  PROPERTY(int64_t, steps_offset) = 0;
  PROPERTY(std::string, final_sigmas_type) = "zero";
  PROPERTY(bool, rescale_betas_zero_snr) = false;
  PROPERTY(std::string, time_shift_type) = "exponential";
};

#undef COMMON_MODEL_PARAMS
#undef MODEL_PARAM

enum class DiTComponentType : uint8_t {
  UNKNOWN = 0,
  TRANSFORMER = 1,
  VAE = 2,
  TEXT_ENCODER = 3,
  SCHEDULER = 4,
};

using DiTComponentModelArgsVariant = std::variant<DiTTransformerModelArgs,
                                                  DiTVaeModelArgs,
                                                  DiTTextEncoderModelArgs,
                                                  DiTSchedulerModelArgs,
                                                  VlmModelArgs>;

class DiTComponentModelArgs final {
 public:
  explicit DiTComponentModelArgs(
      const ModelArgs& args,
      DiTComponentType component_type = DiTComponentType::TRANSFORMER,
      ModelFamily family = ModelFamily::DIT)
      : component_type_(component_type),
        family_(family),
        args_(make_args(args, component_type, family)) {
    CHECK(family_ == ModelFamily::DIT || family_ == ModelFamily::VLM)
        << "Unsupported DiT component model family: "
        << static_cast<int>(family_);
    CHECK(component_type_ != DiTComponentType::UNKNOWN)
        << "Cannot construct arguments for an unknown DiT component type";
  }

  DiTComponentType component_type() const noexcept { return component_type_; }
  ModelFamily family() const noexcept { return family_; }

  const DiTTransformerModelArgs& transformer() const {
    CHECK(component_type_ == DiTComponentType::TRANSFORMER)
        << "DiT component does not contain transformer arguments";
    return std::get<DiTTransformerModelArgs>(args_);
  }

  const DiTVaeModelArgs& vae() const {
    CHECK(component_type_ == DiTComponentType::VAE)
        << "DiT component does not contain VAE arguments";
    return std::get<DiTVaeModelArgs>(args_);
  }

  const DiTTextEncoderModelArgs& text_encoder() const {
    CHECK(component_type_ == DiTComponentType::TEXT_ENCODER)
        << "DiT component does not contain text encoder arguments";
    return std::get<DiTTextEncoderModelArgs>(args_);
  }

  const DiTSchedulerModelArgs& scheduler() const {
    CHECK(component_type_ == DiTComponentType::SCHEDULER)
        << "DiT component does not contain scheduler arguments";
    return std::get<DiTSchedulerModelArgs>(args_);
  }

  VlmModelArgs& vlm() {
    CHECK(family_ == ModelFamily::VLM)
        << "DiT component does not contain VLM arguments";
    return std::get<VlmModelArgs>(args_);
  }

  const VlmModelArgs& vlm() const {
    CHECK(family_ == ModelFamily::VLM)
        << "DiT component does not contain VLM arguments";
    return std::get<VlmModelArgs>(args_);
  }

 private:
  static DiTComponentModelArgsVariant make_args(const ModelArgs& args,
                                                DiTComponentType component_type,
                                                ModelFamily family) {
    if (family == ModelFamily::VLM) {
      return VlmModelArgs(args);
    }
    switch (component_type) {
      case DiTComponentType::TRANSFORMER:
        return DiTTransformerModelArgs(args);
      case DiTComponentType::VAE:
        return DiTVaeModelArgs(args);
      case DiTComponentType::TEXT_ENCODER:
        return DiTTextEncoderModelArgs(args);
      case DiTComponentType::SCHEDULER:
        return DiTSchedulerModelArgs(args);
      case DiTComponentType::UNKNOWN:
        break;
    }
    LOG(FATAL) << "Unknown DiT model component kind";
    return DiTTransformerModelArgs();
  }

  DiTComponentType component_type_;
  ModelFamily family_;
  DiTComponentModelArgsVariant args_;
};

// ModelArgsBundle owns the active family or DiT component. ModelContext shares
// the aggregate so capability access returns references to existing objects.
class ModelArgsBundle final {
 public:
  ModelArgsBundle() = default;

  // Construct a bundle for a regular model context. DiT components use the
  // component-specific overload below so a VLM text encoder remains wrapped
  // in a DiTComponentModelArgs while retaining its VlmModelArgs payload.
  ModelArgsBundle(const ModelArgs& model_args, ModelFamily family)
      : family_(family) {
    initialize(model_args, family, DiTComponentType::UNKNOWN);
  }

  ModelArgsBundle(const ModelArgs& model_args,
                  ModelFamily family,
                  DiTComponentType dit_component_type)
      : family_(family) {
    initialize(model_args, family, dit_component_type);
  }

  ModelFamily family() const noexcept { return family_; }

  const LlmModelArgs& llm_model_args() const {
    const auto* args = std::get_if<LlmModelArgs>(&args_);
    CHECK(args != nullptr) << "Model arguments do not contain LLM arguments";
    return *args;
  }

  const VlmModelArgs& vlm_model_args() const {
    if (const auto* args = std::get_if<VlmModelArgs>(&args_)) {
      return *args;
    }
    const auto* args = std::get_if<DiTComponentModelArgs>(&args_);
    CHECK(args != nullptr) << "Model arguments do not contain VLM arguments";
    return args->vlm();
  }

  const RecModelArgs& rec_model_args() const {
    const auto* args = std::get_if<RecModelArgs>(&args_);
    CHECK(args != nullptr) << "Model arguments do not contain Rec arguments";
    return *args;
  }

  const DiTComponentModelArgs& dit_model_args() const {
    const auto* args = std::get_if<DiTComponentModelArgs>(&args_);
    CHECK(args != nullptr) << "Model arguments do not contain DiT arguments";
    return *args;
  }

  const AttentionModelParams& attention_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().attention_params();
      case ModelFamily::VLM:
        return vlm_model_args().attention_params();
      case ModelFamily::REC:
        return rec_model_args().attention_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().attention_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "Attention parameters require a registered model family";
    return llm_model_args().attention_params();
  }

  const LinearAttentionModelParams& linear_attention_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().linear_attention_params();
      case ModelFamily::VLM:
        return vlm_model_args().linear_attention_params();
      case ModelFamily::REC:
        return rec_model_args().linear_attention_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().linear_attention_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL)
        << "Linear attention parameters require a registered model family";
    return llm_model_args().linear_attention_params();
  }

  const KVCacheModelParams& kv_cache_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().kv_cache_params();
      case ModelFamily::VLM:
        return vlm_model_args().kv_cache_params();
      case ModelFamily::REC:
        return rec_model_args().kv_cache_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().kv_cache_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "KV cache parameters require a registered model family";
    return llm_model_args().kv_cache_params();
  }

  const MoEModelParams& moe_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().moe_params();
      case ModelFamily::VLM:
        return vlm_model_args().moe_params();
      case ModelFamily::REC:
        return rec_model_args().moe_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().moe_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "MoE parameters require a registered model family";
    return llm_model_args().moe_params();
  }

  const SpeculativeModelParams& speculative_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().speculative_params();
      case ModelFamily::VLM:
        return vlm_model_args().speculative_params();
      case ModelFamily::REC:
        return rec_model_args().speculative_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().speculative_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "Speculative parameters require a registered model family";
    return llm_model_args().speculative_params();
  }

  const EmbeddingModelParams& embedding_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().embedding_params();
      case ModelFamily::VLM:
        return vlm_model_args().embedding_params();
      case ModelFamily::REC:
        return rec_model_args().embedding_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().embedding_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "Embedding parameters require a registered model family";
    return llm_model_args().embedding_params();
  }

  const ExecutionModelParams& execution_params() const {
    switch (family_) {
      case ModelFamily::LLM:
        return llm_model_args().execution_params();
      case ModelFamily::VLM:
        return vlm_model_args().execution_params();
      case ModelFamily::REC:
        return rec_model_args().execution_params();
      case ModelFamily::DIT:
        return dit_transformer_model_args().execution_params();
      case ModelFamily::UNKNOWN:
        break;
    }
    LOG(FATAL) << "Execution parameters require a registered model family";
    return llm_model_args().execution_params();
  }

  const std::string& model_type() const {
    return execution_params().model_type();
  }
  const std::string& dtype() const { return execution_params().dtype(); }
  int64_t n_layers() const { return execution_params().n_layers(); }
  int64_t n_encoder_layers() const {
    return execution_params().n_encoder_layers();
  }
  int64_t hidden_size() const { return execution_params().hidden_size(); }
  const std::string& hidden_act() const {
    return execution_params().hidden_act();
  }
  int64_t intermediate_size() const {
    return execution_params().intermediate_size();
  }
  int64_t max_position_embeddings() const {
    return execution_params().max_position_embeddings();
  }
  int64_t vocab_size() const { return execution_params().vocab_size(); }
  int32_t bos_token_id() const { return execution_params().bos_token_id(); }
  int32_t eos_token_id() const { return execution_params().eos_token_id(); }
  const std::vector<int32_t>& eos_token_id_vec() const {
    return execution_params().eos_token_id_vec();
  }
  int32_t pad_token_id() const { return execution_params().pad_token_id(); }
  const std::unordered_set<int32_t>& stop_token_ids() const {
    return execution_params().stop_token_ids();
  }
  bool tie_word_embeddings() const {
    return execution_params().tie_word_embeddings();
  }
  bool no_bias() const { return execution_params().no_bias(); }
  float initializer_range() const {
    return execution_params().initializer_range();
  }
  int32_t rope_scaling() const { return execution_params().rope_scaling(); }
  int32_t max_concurrent_prefills_per_dp() const {
    return execution_params().max_concurrent_prefills_per_dp();
  }
  int64_t head_dim() const { return attention_params().head_dim(); }
  int64_t n_heads() const { return attention_params().n_heads(); }
  std::optional<int64_t> n_kv_heads() const {
    return attention_params().n_kv_heads();
  }
  bool enable_mla() const { return attention_params().enable_mla(); }
  int64_t decoder_head_dim() const {
    return attention_params().decoder_head_dim();
  }
  int64_t decoder_n_heads() const {
    return attention_params().decoder_n_heads();
  }
  std::optional<int64_t> decoder_n_kv_heads() const {
    return attention_params().decoder_n_kv_heads();
  }
  int64_t actual_n_heads() const { return attention_params().actual_n_heads(); }
  int64_t rotary_dim() const { return attention_params().rotary_dim(); }
  float partial_rotary_factor() const {
    return attention_params().partial_rotary_factor();
  }
  bool use_qk_norm() const { return attention_params().use_qk_norm(); }
  float rms_norm_eps() const { return attention_params().rms_norm_eps(); }
  float layer_norm_eps() const { return attention_params().layer_norm_eps(); }
  bool attention_bias() const { return attention_params().attention_bias(); }
  bool qkv_bias() const { return attention_params().qkv_bias(); }
  std::optional<float> attn_scalar() const {
    return attention_params().attn_scalar();
  }
  bool use_sliding_window() const {
    return attention_params().use_sliding_window();
  }
  int32_t sliding_window() const { return attention_params().sliding_window(); }
  int32_t max_window_layers() const {
    return attention_params().max_window_layers();
  }
  int32_t kv_lora_rank() const { return attention_params().kv_lora_rank(); }
  int32_t q_lora_rank() const { return attention_params().q_lora_rank(); }
  int32_t qk_nope_head_dim() const {
    return attention_params().qk_nope_head_dim();
  }
  int32_t qk_rope_head_dim() const {
    return attention_params().qk_rope_head_dim();
  }
  int32_t v_head_dim() const { return attention_params().v_head_dim(); }
  int64_t num_attention_heads() const {
    return attention_params().num_attention_heads();
  }
  float rope_theta() const { return attention_params().rope_theta(); }
  const std::string& rope_scaling_rope_type() const {
    return attention_params().rope_scaling_rope_type();
  }
  float rope_scaling_factor() const {
    return attention_params().rope_scaling_factor();
  }
  float rope_scaling_low_freq_factor() const {
    return attention_params().rope_scaling_low_freq_factor();
  }
  float rope_scaling_high_freq_factor() const {
    return attention_params().rope_scaling_high_freq_factor();
  }
  float rope_extrapolation_factor() const {
    return attention_params().rope_extrapolation_factor();
  }
  int64_t rope_scaling_original_max_position_embeddings() const {
    return attention_params().rope_scaling_original_max_position_embeddings();
  }
  int64_t rope_scaling_beta_fast() const {
    return attention_params().rope_scaling_beta_fast();
  }
  int64_t rope_scaling_beta_slow() const {
    return attention_params().rope_scaling_beta_slow();
  }
  float rope_scaling_attn_factor() const {
    return attention_params().rope_scaling_attn_factor();
  }
  float rope_scaling_mscale() const {
    return attention_params().rope_scaling_mscale();
  }
  float rope_scaling_mscale_all_dim() const {
    return attention_params().rope_scaling_mscale_all_dim();
  }
  const std::vector<int64_t>& rope_scaling_mrope_section() const {
    return attention_params().rope_scaling_mrope_section();
  }
  bool rope_scaling_mrope_interleaved() const {
    return attention_params().rope_scaling_mrope_interleaved();
  }
  float attention_dropout() const {
    return attention_params().attention_dropout();
  }
  bool attn_output_gate() const {
    return attention_params().attn_output_gate();
  }
  bool use_absolute_position_embedding() const {
    return attention_params().use_absolute_position_embedding();
  }
  int64_t relative_attention_num_buckets() const {
    return attention_params().relative_attention_num_buckets();
  }
  int64_t relative_attention_max_distance() const {
    return attention_params().relative_attention_max_distance();
  }
  int64_t kv_cache_storage_head_dim() const {
    return kv_cache_params().kv_cache_storage_head_dim();
  }
  int32_t index_head_dim() const { return kv_cache_params().index_head_dim(); }
  int32_t index_n_heads() const { return kv_cache_params().index_n_heads(); }
  int32_t index_topk() const { return kv_cache_params().index_topk(); }
  int32_t index_kpool() const { return kv_cache_params().index_kpool(); }
  bool index_kpool_compress() const {
    return kv_cache_params().index_kpool_compress();
  }
  bool index_kpool_always_select_tail() const {
    return kv_cache_params().index_kpool_always_select_tail();
  }
  bool indexer_rope_interleave() const {
    return kv_cache_params().indexer_rope_interleave();
  }
  int32_t index_topk_freq() const {
    return kv_cache_params().index_topk_freq();
  }
  const std::string& index_topk_pattern() const {
    return kv_cache_params().index_topk_pattern();
  }
  int32_t index_skip_topk_offset() const {
    return kv_cache_params().index_skip_topk_offset();
  }
  bool index_share_for_mtp_iteration() const {
    return kv_cache_params().index_share_for_mtp_iteration();
  }
  const std::vector<std::string>& indexer_types() const {
    return kv_cache_params().indexer_types();
  }
  int32_t full_attention_interval() const {
    return linear_attention_params().full_attention_interval();
  }
  const std::vector<std::string>& layer_types() const {
    return linear_attention_params().layer_types();
  }
  int32_t linear_conv_kernel_dim() const {
    return linear_attention_params().linear_conv_kernel_dim();
  }
  int32_t linear_key_head_dim() const {
    return linear_attention_params().linear_key_head_dim();
  }
  int32_t linear_value_head_dim() const {
    return linear_attention_params().linear_value_head_dim();
  }
  int64_t linear_num_key_heads() const {
    return linear_attention_params().linear_num_key_heads();
  }
  int32_t linear_num_value_heads() const {
    return linear_attention_params().linear_num_value_heads();
  }
  float linear_lower_bound() const {
    return linear_attention_params().linear_lower_bound();
  }
  const std::string& mamba_ssm_dtype() const {
    return linear_attention_params().mamba_ssm_dtype();
  }
  float factor() const { return kv_cache_params().factor(); }
  float beta_fast() const { return kv_cache_params().beta_fast(); }
  float beta_slow() const { return kv_cache_params().beta_slow(); }
  const std::vector<int32_t>& compress_ratios() const {
    return kv_cache_params().compress_ratios();
  }
  int32_t window_size() const { return kv_cache_params().window_size(); }
  int64_t max_batch_size() const { return kv_cache_params().max_batch_size(); }
  int64_t max_seq_len() const { return kv_cache_params().max_seq_len(); }
  const std::string& scale_fmt() const { return kv_cache_params().scale_fmt(); }
  int32_t hc_mult() const { return kv_cache_params().hc_mult(); }
  int32_t hc_sinkhorn_iters() const {
    return kv_cache_params().hc_sinkhorn_iters();
  }
  float hc_eps() const { return kv_cache_params().hc_eps(); }
  int32_t rope_head_dim() const { return kv_cache_params().rope_head_dim(); }
  int32_t o_lora_rank() const { return kv_cache_params().o_lora_rank(); }
  int32_t o_groups() const { return kv_cache_params().o_groups(); }
  float compress_rope_theta() const {
    return kv_cache_params().compress_rope_theta();
  }
  int32_t num_nextn_predict_layers() const {
    return speculative_params().num_nextn_predict_layers();
  }
  int64_t draft_vocab_size() const {
    return speculative_params().draft_vocab_size();
  }
  int64_t num_speculative_tokens() const {
    return speculative_params().num_speculative_tokens();
  }
  bool enable_embedded_eagle3_draft() const {
    return speculative_params().enable_embedded_eagle3_draft();
  }
  const std::vector<int32_t>& layers_to_capture() const {
    return speculative_params().layers_to_capture();
  }
  int64_t mtp_start_layer_idx() const {
    return speculative_params().mtp_start_layer_idx();
  }
  const std::string& mtp_mlp_type() const {
    return speculative_params().mtp_mlp_type();
  }
  int64_t markov_rank() const { return speculative_params().markov_rank(); }
  int32_t dspark_num_layers() const {
    return speculative_params().dspark_num_layers();
  }
  int32_t dspark_block_size() const {
    return speculative_params().dspark_block_size();
  }
  bool dspark_use_native_sas() const {
    return speculative_params().dspark_use_native_sas();
  }
  bool enable_confidence_head() const {
    return speculative_params().enable_confidence_head();
  }
  bool confidence_head_with_markov() const {
    return speculative_params().confidence_head_with_markov();
  }
  int32_t dflash2_block_size() const {
    return speculative_params().dflash2_block_size();
  }
  int32_t dflash2_conv_group_size() const {
    return speculative_params().dflash2_conv_group_size();
  }
  int32_t dflash2_conv_kernel_size() const {
    return speculative_params().dflash2_conv_kernel_size();
  }
  int32_t dflash2_selector_rank() const {
    return speculative_params().dflash2_selector_rank();
  }
  int32_t dflash2_selector_top_k() const {
    return speculative_params().dflash2_selector_top_k();
  }
  bool norm_before_fc() const { return speculative_params().norm_before_fc(); }
  bool norm_before_residual() const {
    return speculative_params().norm_before_residual();
  }
  bool norm_output() const { return speculative_params().norm_output(); }
  bool fc_norm() const { return speculative_params().fc_norm(); }

  bool use_moe() const { return moe_params().use_moe(); }
  int32_t first_k_dense_replace() const {
    return moe_params().first_k_dense_replace();
  }
  int32_t moe_layer_freq() const { return moe_params().moe_layer_freq(); }
  const std::string& moe_score_func() const {
    return moe_params().moe_score_func();
  }
  float moe_route_scale() const { return moe_params().moe_route_scale(); }
  bool moe_use_shared_experts() const {
    return moe_params().moe_use_shared_experts();
  }
  const std::string& topk_method() const { return moe_params().topk_method(); }
  int32_t n_routed_experts() const { return moe_params().n_routed_experts(); }
  int32_t n_shared_experts() const { return moe_params().n_shared_experts(); }
  int32_t num_experts_per_tok() const {
    return moe_params().num_experts_per_tok();
  }
  int32_t moe_intermediate_size() const {
    return moe_params().moe_intermediate_size();
  }
  float routed_scaling_factor() const {
    return moe_params().routed_scaling_factor();
  }
  bool norm_topk_prob() const { return moe_params().norm_topk_prob(); }
  int32_t n_group() const { return moe_params().n_group(); }
  int32_t topk_group() const { return moe_params().topk_group(); }
  const std::string& scoring_func() const {
    return moe_params().scoring_func();
  }
  float swiglu_limit() const { return moe_params().swiglu_limit(); }
  int32_t num_experts() const { return moe_params().num_experts(); }
  int32_t n_activated_experts() const {
    return moe_params().n_activated_experts();
  }
  int32_t shared_expert_intermediate_size() const {
    return moe_params().shared_expert_intermediate_size();
  }
  bool output_router_logits() const {
    return moe_params().output_router_logits();
  }
  float router_aux_loss_coef() const {
    return moe_params().router_aux_loss_coef();
  }
  const std::vector<int32_t>& mlp_only_layers() const {
    return moe_params().mlp_only_layers();
  }
  int32_t n_hash_layers() const { return moe_params().n_hash_layers(); }
  const std::vector<std::string>& mlp_layer_types() const {
    return moe_params().mlp_layer_types();
  }
  int32_t decoder_sparse_step() const {
    return moe_params().decoder_sparse_step();
  }
  int32_t query_num() const { return embedding_params().query_num(); }
  bool encoder_embedding_mode() const {
    return embedding_params().encoder_embedding_mode();
  }
  bool embedding_mode() const { return embedding_params().embedding_mode(); }

  void set_encoder_embedding_mode(bool encoder_embedding_mode) {
    if (auto* args = std::get_if<VlmModelArgs>(&args_)) {
      args->encoder_embedding_mode(encoder_embedding_mode);
      return;
    }
    if (auto* args = std::get_if<RecModelArgs>(&args_)) {
      args->encoder_embedding_mode(encoder_embedding_mode);
      return;
    }
    if (auto* args = std::get_if<DiTComponentModelArgs>(&args_);
        args != nullptr && args->family() == ModelFamily::VLM) {
      args->vlm().encoder_embedding_mode(encoder_embedding_mode);
    }
  }

 private:
  const DiTTransformerModelArgs& dit_transformer_model_args() const {
    const auto& component_args = dit_model_args();
    CHECK(component_args.component_type() == DiTComponentType::TRANSFORMER)
        << "Capability parameters require a DiT transformer component";
    return component_args.transformer();
  }

  void initialize(const ModelArgs& model_args,
                  ModelFamily family,
                  DiTComponentType dit_component_type) {
    switch (family) {
      case ModelFamily::LLM:
        args_.emplace<LlmModelArgs>(model_args);
        return;
      case ModelFamily::VLM:
        if (dit_component_type != DiTComponentType::UNKNOWN) {
          args_.emplace<DiTComponentModelArgs>(
              model_args, dit_component_type, ModelFamily::VLM);
          return;
        }
        args_.emplace<VlmModelArgs>(model_args);
        return;
      case ModelFamily::REC:
        args_.emplace<RecModelArgs>(model_args);
        return;
      case ModelFamily::DIT:
        args_.emplace<DiTComponentModelArgs>(
            model_args,
            dit_component_type == DiTComponentType::UNKNOWN
                ? DiTComponentType::TRANSFORMER
                : dit_component_type);
        return;
      case ModelFamily::UNKNOWN:
        return;
    }
  }

  ModelFamily family_ = ModelFamily::UNKNOWN;
  std::variant<std::monostate,
               LlmModelArgs,
               VlmModelArgs,
               RecModelArgs,
               DiTComponentModelArgs>
      args_;
};

inline bool has_linear_attention_layers(const ModelArgsBundle& args) {
  return args.linear_attention_params().has_linear_attention_layers();
}

inline bool is_full_attention_layer(const ModelArgsBundle& args,
                                    int64_t layer_id) {
  return args.linear_attention_params().is_full_attention_layer(layer_id);
}

}  // namespace xllm
