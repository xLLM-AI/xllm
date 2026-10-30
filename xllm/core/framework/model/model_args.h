/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"

namespace xllm {

inline constexpr std::string_view kDFlash2DraftModelType = "DFlash2DraftModel";
inline constexpr std::string_view kDSparkGLM5DraftModelType = "glm5_dspark";
inline constexpr std::string_view kDeepseekV4DSparkModelType =
    "deepseek_v4_dspark";

inline constexpr bool is_dflash2_draft_model_type(std::string_view model_type) {
  return model_type == kDFlash2DraftModelType;
}

// DSpark draft model types (Qwen/DSparkDraftModel, GLM5, DeepSeek-V4).
inline constexpr bool is_dspark_draft_model_type(std::string_view model_type) {
  return model_type == "DSparkDraftModel" ||
         model_type == kDSparkGLM5DraftModelType ||
         model_type == kDeepseekV4DSparkModelType;
}

// All block-diffusion draft model families (DFlash/DFlash2/DSpark).
inline constexpr bool is_block_draft_model_type(std::string_view model_type) {
  return model_type == "DFlashDraftModel" ||
         is_dflash2_draft_model_type(model_type) ||
         is_dspark_draft_model_type(model_type);
}

class TokenEmbeddingArgs final {
 public:
  REFLECT_PROPERTIES(TokenEmbeddingArgs);

  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(bool, tie_word_embeddings) = false;
  PROPERTY(bool, encoder_embedding_mode) = false;
  PROPERTY(bool, embedding_mode) = false;
};

// Model-defined sequence boundary, padding, and default decoding stop tokens.
class SpecialTokenArgs final {
 public:
  REFLECT_PROPERTIES(SpecialTokenArgs);

  PROPERTY(int32_t, bos_token_id) = 0;
  PROPERTY(int32_t, eos_token_id) = -1;
  PROPERTY(std::vector<int32_t>, eos_token_id_vec);
  PROPERTY(int32_t, pad_token_id) = 0;
  PROPERTY(std::unordered_set<int32_t>, stop_token_ids);
};

// Shared expert geometry and token-routing configuration.
class MoEArgs final {
 public:
  REFLECT_PROPERTIES(MoEArgs);

  PROPERTY(std::string, topk_method);
  PROPERTY(int32_t, n_routed_experts) = 0;
  PROPERTY(int32_t, n_shared_experts) = 0;
  PROPERTY(int32_t, num_experts_per_tok) = 0;
  PROPERTY(int32_t, moe_intermediate_size) = 0;
  PROPERTY(float, routed_scaling_factor) = 0.0f;
  PROPERTY(bool, norm_topk_prob) = false;
  PROPERTY(int32_t, n_group) = 0;
  PROPERTY(int32_t, topk_group) = 0;
  PROPERTY(std::string, scoring_func);
  PROPERTY(int32_t, first_k_dense_replace) = 0;
  PROPERTY(int32_t, moe_layer_freq) = 0;
  PROPERTY(bool, use_moe) = false;
  PROPERTY(std::string, moe_score_func);
  PROPERTY(float, moe_route_scale) = 1.0f;
  PROPERTY(bool, moe_use_shared_experts) = false;
  PROPERTY(float, swiglu_limit) = 0.0f;
  PROPERTY(int32_t, decoder_sparse_step) = 1;
  PROPERTY(std::vector<int32_t>, mlp_only_layers) = {};
  PROPERTY(int32_t, num_experts) = 128;
  PROPERTY(bool, output_router_logits) = false;
  PROPERTY(float, router_aux_loss_coef) = 0.001f;
  PROPERTY(int32_t, shared_expert_intermediate_size) = 0;
};

// Multi-token prediction layer structure and checkpoint addressing.
class MtpArgs final {
 public:
  REFLECT_PROPERTIES(MtpArgs);

  PROPERTY(int32_t, num_nextn_predict_layers) = 0;
  PROPERTY(int64_t, mtp_start_layer_idx) = -1;
  PROPERTY(std::string, mtp_mlp_type) = "moe";
};

// Attention head geometry, excluding backend-specific cache storage padding.
class AttentionHeadArgs final {
 public:
  REFLECT_PROPERTIES(AttentionHeadArgs);

  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, decoder_head_dim) = 0;
  PROPERTY(int64_t, n_heads) = 0;
  PROPERTY(int64_t, decoder_n_heads) = 0;
  PROPERTY(int64_t, actual_n_heads) = 0;
  PROPERTY(int64_t, num_attention_heads) = 32;
  PROPERTY(std::optional<int64_t>, n_kv_heads);
  PROPERTY(std::optional<int64_t>, decoder_n_kv_heads);
};

// Low-rank attention projection ranks and query/key/value head geometry.
class MlaArgs final {
 public:
  REFLECT_PROPERTIES(MlaArgs);

  PROPERTY(int32_t, qk_nope_head_dim) = 0;
  PROPERTY(int32_t, qk_rope_head_dim) = 0;
  PROPERTY(int32_t, v_head_dim) = 0;
  PROPERTY(int32_t, q_lora_rank) = 0;
  PROPERTY(int32_t, kv_lora_rank) = 0;
};

// Attention implementation selection shared by model attention variants.
// Runtime execution options remain outside ModelArgs; this flag only records
// the checkpoint-defined MLA attention type.
class AttentionArgs final {
 public:
  REFLECT_PROPERTIES(AttentionArgs);

  PROPERTY(bool, enable_mla) = false;
};

// Linear and recurrent attention geometry and checkpoint-defined state dtype.
class LinearAttentionArgs final {
 public:
  REFLECT_PROPERTIES(LinearAttentionArgs);

  PROPERTY(int32_t, linear_conv_kernel_dim) = 0;
  PROPERTY(int32_t, linear_key_head_dim) = 0;
  PROPERTY(int32_t, linear_value_head_dim) = 0;
  PROPERTY(int64_t, linear_num_key_heads) = 0;
  PROPERTY(int32_t, linear_num_value_heads) = 0;
  PROPERTY(float, linear_lower_bound) = -5.0f;
  PROPERTY(std::string, mamba_ssm_dtype);
};

// Checkpoint-defined placement of full and linear attention layers.
class HybridAttentionArgs final {
 public:
  REFLECT_PROPERTIES(HybridAttentionArgs);

  PROPERTY(int32_t, full_attention_interval) = 0;
  PROPERTY(std::vector<int32_t>, full_attn_layers) = {};
  PROPERTY(std::vector<std::string>, layer_types) = {};
};

// Rotary position embedding geometry and checkpoint-defined scaling.
class RopeArgs final {
 public:
  REFLECT_PROPERTIES(RopeArgs);

  PROPERTY(int64_t, rotary_dim) = 0;
  // The base period of the rotary position embeddings.
  PROPERTY(float, rope_theta) = 10000.0f;
  PROPERTY(std::string, rope_scaling_rope_type);
  PROPERTY(float, rope_scaling_factor) = 0.0f;
  PROPERTY(float, rope_scaling_low_freq_factor) = 0.0f;
  PROPERTY(float, rope_scaling_high_freq_factor) = 0.0f;
  PROPERTY(float, rope_extrapolation_factor) = 0.0f;
  PROPERTY(int64_t, rope_scaling_original_max_position_embeddings) = 0;
  PROPERTY(int64_t, rope_scaling_beta_fast) = 0;
  PROPERTY(int64_t, rope_scaling_beta_slow) = 0;
  PROPERTY(float, rope_scaling_attn_factor) = 0.0f;
  PROPERTY(float, rope_scaling_mscale) = 0.0f;
  PROPERTY(float, rope_scaling_mscale_all_dim) = 0.0f;
  PROPERTY(std::vector<int64_t>, rope_scaling_mrope_section);
  PROPERTY(bool, rope_scaling_mrope_interleaved) = false;
  PROPERTY(int32_t, rope_scaling) = -1;
  PROPERTY(float, partial_rotary_factor) = 0.0f;
};

// Token IDs used to delimit multimodal content in model inputs.
class MultimodalTokenArgs final {
 public:
  REFLECT_PROPERTIES(MultimodalTokenArgs);

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
};

// Image processor settings for cropping, resizing, rescaling, and
// normalization.
class ImagePreprocessArgs final {
 public:
  REFLECT_PROPERTIES(ImagePreprocessArgs);

  PROPERTY(bool, mm_image_do_center_crop) = false;
  PROPERTY(int32_t, mm_image_crop_height_size) = 336;
  PROPERTY(int32_t, mm_image_crop_width_size) = 336;
  PROPERTY(bool, mm_image_do_resize) = false;
  PROPERTY(int32_t, mm_image_resize_shortest_edge) = 336;
  PROPERTY(int32_t, mm_image_resample) = 0;
  PROPERTY(bool, mm_image_do_rescale) = false;
  PROPERTY(double, mm_image_rescale_factor) = 0;
  PROPERTY(bool, mm_image_do_normalize) = false;
  PROPERTY(std::vector<double>, mm_image_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_image_normalize_std) = {};
};

// Model-defined sliding-window attention behavior and layer coverage.
class SlidingWindowArgs final {
 public:
  REFLECT_PROPERTIES(SlidingWindowArgs);

  PROPERTY(bool, use_sliding_window) = false;
  PROPERTY(int32_t, sliding_window) = -1;
  PROPERTY(bool, sliding_window_non_causal) = false;
  PROPERTY(int32_t, max_window_layers) = 0;
};

// DSpark's checkpoint-defined confidence head and optional Markov input.
class ConfidenceHeadArgs final {
 public:
  REFLECT_PROPERTIES(ConfidenceHeadArgs);

  // Registers confidence_head.proj for acceptance-probability estimation.
  PROPERTY(bool, enable_confidence_head) = false;
  // Applies the head to concat(hidden, markov_embedding[prev]) rather than
  // hidden alone, as configured in released DSpark checkpoints.
  PROPERTY(bool, confidence_head_with_markov) = false;
};

// Checkpoint-defined local-convolution and selector geometry for DFlash2.
class DFlash2Args final {
 public:
  REFLECT_PROPERTIES(DFlash2Args);

  PROPERTY(int32_t, dflash2_block_size) = 0;
  PROPERTY(int32_t, dflash2_conv_group_size) = 0;
  PROPERTY(int32_t, dflash2_conv_kernel_size) = 0;
  PROPERTY(int32_t, dflash2_selector_rank) = 0;
  PROPERTY(int32_t, dflash2_selector_top_k) = 0;
};

// Core transformer architecture and sequence-position configuration.
class ModelArchitectureArgs final {
 public:
  REFLECT_PROPERTIES(ModelArchitectureArgs);

  PROPERTY(std::string, model_type);
  PROPERTY(std::string, dtype);
  PROPERTY(int64_t, hidden_size) = 0;
  PROPERTY(std::string, hidden_act);
  PROPERTY(int64_t, intermediate_size) = 0;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, n_encoder_layers) = 0;
  PROPERTY(int64_t, max_position_embeddings) = 0;
  PROPERTY(bool, use_absolute_position_embedding) = false;
  PROPERTY(float, initializer_range) = 0.02f;
  PROPERTY(float, rms_norm_eps) = 0.0f;
  PROPERTY(float, layer_norm_eps) = 0.0f;
  PROPERTY(std::vector<std::string>, mlp_layer_types) = {};
  PROPERTY(int64_t, in_channels) = -1;
  PROPERTY(int64_t, out_channels) = -1;
};

// Backend-independent KV cache geometry and model capacity limits.
class KVCacheArgs final {
 public:
  REFLECT_PROPERTIES(KVCacheArgs);

  PROPERTY(int64_t, kv_cache_storage_head_dim) = 0;
  PROPERTY(int64_t, max_batch_size) = 0;
  PROPERTY(int64_t, max_seq_len) = 0;
};

// Attention normalization, projection, and model-specific bias switches.
class AttentionConfigArgs final {
 public:
  REFLECT_PROPERTIES(AttentionConfigArgs);

  PROPERTY(bool, use_qk_norm) = false;
  PROPERTY(std::optional<float>, attn_scalar);
  PROPERTY(bool, no_bias) = false;
  PROPERTY(bool, qkv_bias) = false;
  PROPERTY(bool, attention_bias) = false;
  PROPERTY(float, attention_dropout) = 0.0f;
  PROPERTY(bool, attn_output_gate) = false;
};

// Speculative model structure and target-side admission controls.
class SpeculativeArgs final {
 public:
  REFLECT_PROPERTIES(SpeculativeArgs);

  PROPERTY(int64_t, draft_vocab_size) = 0;
  PROPERTY(int64_t, num_speculative_tokens) = 0;
  PROPERTY(bool, enable_embedded_eagle3_draft) = false;
  PROPERTY(int32_t, max_concurrent_prefills_per_dp) = 0;
  PROPERTY(std::vector<int32_t>, layers_to_capture) = {};
  PROPERTY(bool, norm_before_fc) = false;
  PROPERTY(bool, norm_before_residual) = false;
  PROPERTY(bool, norm_output) = false;
  PROPERTY(bool, fc_norm) = false;
};

// DeepSeek sparse indexer and router metadata.
class IndexerArgs final {
 public:
  REFLECT_PROPERTIES(IndexerArgs);

  PROPERTY(int32_t, index_head_dim) = 0;
  PROPERTY(int32_t, index_n_heads) = 0;
  PROPERTY(int32_t, index_topk) = 0;
  PROPERTY(int32_t, index_kpool) = 1;
  PROPERTY(bool, index_kpool_compress) = false;
  PROPERTY(bool, index_kpool_always_select_tail) = false;
  PROPERTY(bool, indexer_rope_interleave) = false;
  PROPERTY(int32_t, index_topk_freq) = 1;
  PROPERTY(std::string, index_topk_pattern);
  PROPERTY(int32_t, index_skip_topk_offset) = 0;
  PROPERTY(bool, index_share_for_mtp_iteration) = false;
  PROPERTY(std::vector<std::string>, indexer_types) = {};
};

// DeepSeek-V4 low-rank attention, routing, and hashing geometry.
class DeepseekV4Args final {
 public:
  REFLECT_PROPERTIES(DeepseekV4Args);

  PROPERTY(int32_t, rope_head_dim) = 0;
  PROPERTY(int32_t, o_lora_rank) = 0;
  PROPERTY(int32_t, o_groups) = 0;
  PROPERTY(std::vector<int32_t>, compress_ratios);
  PROPERTY(float, compress_rope_theta) = 0.0f;
  PROPERTY(int32_t, window_size) = 0;
  PROPERTY(int32_t, n_activated_experts) = 0;
  PROPERTY(int32_t, n_hash_layers) = 0;
  PROPERTY(float, factor) = 0.0f;
  PROPERTY(float, beta_fast) = 0.0f;
  PROPERTY(float, beta_slow) = 0.0f;
  PROPERTY(std::string, scale_fmt);
};

// Hyperconnection topology and sinkhorn normalization configuration.
class HyperconnectionArgs final {
 public:
  REFLECT_PROPERTIES(HyperconnectionArgs);

  PROPERTY(int32_t, hc_mult) = 0;
  PROPERTY(int32_t, hc_sinkhorn_iters) = 0;
  PROPERTY(float, hc_eps) = 1e-6f;
};

// DSpark-specific Markov and sparse-attention runtime-independent geometry.
class DSparkArgs final {
 public:
  REFLECT_PROPERTIES(DSparkArgs);

  PROPERTY(int64_t, markov_rank) = 0;
  PROPERTY(int32_t, dspark_num_layers) = 0;
  PROPERTY(int32_t, dspark_block_size) = 0;
  PROPERTY(bool, dspark_use_native_sas) = false;
};

// Vision encoder, projector, and model-defined generation configuration.
class VisionArgs final {
 public:
  REFLECT_PROPERTIES(VisionArgs);

  PROPERTY(int32_t, query_num) = 0;
  PROPERTY(std::string, vision_custom_adapter);
  PROPERTY(int32_t, vision_max_slice_nums) = 0;
  PROPERTY(float, mm_dropout) = 0.0f;
  PROPERTY(std::string, mm_hidden_act);
  PROPERTY(int64_t, mm_hidden_size) = 0;
  PROPERTY(int64_t, mm_image_size) = 0;
  PROPERTY(int64_t, mm_intermediate_size) = 0;
  PROPERTY(int64_t, mm_num_channels) = 0;
  PROPERTY(float, mm_initializer_range) = 0.0f;
  PROPERTY(float, mm_layer_norm_eps) = 0.0f;
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
  PROPERTY(int, mm_window_size) = 0;
  PROPERTY(std::vector<int64_t>, mm_fullatt_block_indexes);
  PROPERTY(std::vector<int64_t>, mm_deepstack_visual_indexes);
  PROPERTY(int, mm_tokens_per_second) = 0;
  PROPERTY(int, mm_temporal_patch_size) = 0;
  PROPERTY(std::string, mm_projector_type);
  PROPERTY(int64_t, mm_num_position_embeddings);
  PROPERTY(std::string, mm_projector_hidden_act);
  PROPERTY(int64_t, mm_projector_n_layers) = 0;
  PROPERTY(int64_t, mm_vision_feature_layer) = 0;
  PROPERTY(std::string, mm_vision_feature_select_strategy);
  PROPERTY(int64_t, mm_init_pos_emb_width) = 64;
  PROPERTY(int64_t, mm_init_pos_emb_height) = 64;
  PROPERTY(int64_t, mm_init_pos_emb_time) = 4;
};

// Image and video processor settings beyond the common image transform.
class ImageVideoArgs final {
 public:
  REFLECT_PROPERTIES(ImageVideoArgs);

  PROPERTY(int64_t, mm_km_in_patch_limit) = 16384;
  PROPERTY(int64_t, mm_km_patch_size) = 14;
  PROPERTY(std::vector<int64_t>, mm_km_image_mean) = {};
  PROPERTY(std::vector<int64_t>, mm_km_image_std) = {};
  PROPERTY(int64_t, mm_km_merge_kernel_size) = 2;
  PROPERTY(int64_t, mm_km_fixed_output_tokens) = -1;
  PROPERTY(int64_t, mm_km_patch_limit_on_one_side) = 512;
  PROPERTY(int64_t, mm_km_in_patch_limit_each_frame) = 4096;
  PROPERTY(int64_t, mm_km_in_patch_limit_video) = 200;
  PROPERTY(float, mm_km_sample_fps) = 2.0f;
  PROPERTY(int64_t, mm_km_max_num_frames_each_video) = 2;
  PROPERTY(int64_t, mm_km_temporal_merge_kernel_size) = 4;
  PROPERTY(std::string, mm_km_timestamp_mode) = "hh:mm:ss.fff";
  PROPERTY(bool, mm_video_do_rescale) = false;
  PROPERTY(std::vector<double>, mm_video_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_video_normalize_std) = {};
  PROPERTY(int, mm_image_min_pixels) = 0;
  PROPERTY(int, mm_image_max_pixels) = 0;
  PROPERTY(int64_t, mm_image_shortest_edge) = 0;
  PROPERTY(int64_t, mm_image_longest_edge) = 0;
  PROPERTY(int64_t, mm_video_shortest_edge) = 0;
  PROPERTY(int64_t, mm_video_longest_edge) = 0;
  PROPERTY(int, mm_image_patch_size) = 0;
  PROPERTY(int, mm_image_temporal_patch_size) = 0;
  PROPERTY(int, mm_image_merge_size) = 0;
  PROPERTY(int, mm_image_min_tokens) = 0;
  PROPERTY(int, mm_image_max_tokens) = 0;
  PROPERTY(int, mm_video_patch_size) = 0;
  PROPERTY(int, mm_video_temporal_patch_size) = 0;
  PROPERTY(int, mm_video_merge_size) = 0;
  PROPERTY(int, mm_video_min_tokens) = 0;
  PROPERTY(int, mm_video_max_tokens) = 0;
  PROPERTY(int, mm_image_feature_size) = 0;
  PROPERTY(int, mm_scale_resolution) = 0;
  PROPERTY(bool, mm_slice_mode) = false;
  PROPERTY(bool, mm_use_image_id) = false;
  PROPERTY(int64_t, mm_image_token_index) = 0;
  PROPERTY(int64_t, mm_pad_token_id) = 0;
};

// Audio encoder and CTC configuration used by multimodal models.
class AudioArgs final {
 public:
  REFLECT_PROPERTIES(AudioArgs);

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

// VAE architecture and codec-specific configuration.
class VaeArgs final {
 public:
  REFLECT_PROPERTIES(VaeArgs);

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
  PROPERTY(float, batch_norm_eps) = 1e-4f;
  PROPERTY(float, batch_norm_momentum) = 0.1f;
  PROPERTY(std::vector<int64_t>, ae_patch_size) = {};
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
  PROPERTY(int64_t, base_dim) = 0;
  PROPERTY(int64_t, z_dim) = 0;
  PROPERTY(std::vector<int64_t>, dim_mult) = {};
  PROPERTY(std::vector<double>, attn_scales) = {};
  PROPERTY(std::vector<bool>, temperal_downsample) = {};
  PROPERTY(int64_t, num_res_blocks) = 0;
  PROPERTY(double, dropout) = 0.0;
  PROPERTY(std::vector<double>, latents_mean) = {};
  PROPERTY(std::vector<double>, latents_std) = {};
  PROPERTY(std::vector<int64_t>, spatial_downsample_factors) = {};
  PROPERTY(std::vector<int64_t>, temporal_downsample_factors) = {};
  PROPERTY(std::string, spatial_padding_mode) = "reflect";
  PROPERTY(int64_t, decoder_num_register_tokens) = 0;
  PROPERTY(int64_t, decoder_ffn_mult) = 0;
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
};

// Diffusion transformer architecture and checkpoint-defined conditioning.
class DitArgs final {
 public:
  REFLECT_PROPERTIES(DitArgs);

  PROPERTY(int64_t, joint_attention_dim) = 0;
  PROPERTY(int64_t, pooled_projection_dim) = 0;
  PROPERTY(bool, guidance_embeds) = true;
  PROPERTY(std::vector<int64_t>, axes_dims_rope) = {};
  PROPERTY(int64_t, num_single_layers) = 0;
  PROPERTY(float, mlp_ratio) = 3.0f;
  PROPERTY(int, timestep_guidance_channels) = 256;
  PROPERTY(int64_t, patch_size) = 1;
  PROPERTY(std::vector<int64_t>, wan_patch_size) = { 1, 2, 2 };
  PROPERTY(bool, cross_attn_norm) = true;
  PROPERTY(double, eps) = 1e-6;
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
  PROPERTY(double, mlp_width_ratio) = 4.0;
  PROPERTY(int64_t, text_dim) = 4096;
  PROPERTY(std::vector<int64_t>, rope_dim_list) = { 16, 56, 56 };
  PROPERTY(int64_t, rope_theta_dit) = 10000;
};

// T5 encoder architecture and relative-position configuration.
class T5Args final {
 public:
  REFLECT_PROPERTIES(T5Args);

  PROPERTY(int64_t, d_model) = 0;
  PROPERTY(int64_t, num_layers) = 0;
  PROPERTY(int64_t, d_kv) = 0;
  PROPERTY(int64_t, d_ff) = 0;
  PROPERTY(std::string, act_fn) = "";
  PROPERTY(bool, is_gated_act) = true;
  PROPERTY(int64_t, relative_attention_num_buckets) = 0;
  PROPERTY(int64_t, relative_attention_max_distance) = 0;
};

// Diffusion scheduler configuration independent of device/runtime options.
class SchedulerArgs final {
 public:
  REFLECT_PROPERTIES(SchedulerArgs);

  PROPERTY(int64_t, num_train_timesteps) = 0;
  PROPERTY(int64_t, shift) = 0;
  PROPERTY(bool, use_dynamic_shifting) = false;
  PROPERTY(float, base_shift) = 0.0f;
  PROPERTY(float, max_shift) = 0.0f;
  PROPERTY(int64_t, base_image_seq_len) = 0;
  PROPERTY(int64_t, max_image_seq_len) = 0;
  PROPERTY(float, shift_terminal) = 0.0f;
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

struct ModelArgs {
  // Expose every plain-data field to the generic property reflection layer so
  // the embedded Python model executor can receive the full, already-parsed
  // config without a hand-maintained field whitelist (see property_reflect.h).
  REFLECT_PROPERTIES(ModelArgs);

  PROPERTY(ModelArchitectureArgs, model_architecture_args);

  FORWARD_PROPERTY(std::string, model_type, model_architecture_args);
  FORWARD_PROPERTY(std::string, dtype, model_architecture_args);
  FORWARD_PROPERTY(int64_t, hidden_size, model_architecture_args);
  FORWARD_PROPERTY(std::string, hidden_act, model_architecture_args);
  FORWARD_PROPERTY(int64_t, intermediate_size, model_architecture_args);
  FORWARD_PROPERTY(int64_t, n_layers, model_architecture_args);
  FORWARD_PROPERTY(int64_t, n_encoder_layers, model_architecture_args);
  FORWARD_PROPERTY(int64_t, max_position_embeddings, model_architecture_args);
  FORWARD_PROPERTY(bool,
                   use_absolute_position_embedding,
                   model_architecture_args);
  FORWARD_PROPERTY(float, initializer_range, model_architecture_args);
  FORWARD_PROPERTY(float, rms_norm_eps, model_architecture_args);
  FORWARD_PROPERTY(float, layer_norm_eps, model_architecture_args);
  FORWARD_PROPERTY(std::vector<std::string>,
                   mlp_layer_types,
                   model_architecture_args);
  FORWARD_PROPERTY(int64_t, in_channels, model_architecture_args);
  FORWARD_PROPERTY(int64_t, out_channels, model_architecture_args);

  PROPERTY(TokenEmbeddingArgs, token_embedding_args);

  // Keep the established accessors while token embedding fields move into
  // their functional parameter group.
  FORWARD_PROPERTY(int64_t, vocab_size, token_embedding_args);
  FORWARD_PROPERTY(bool, tie_word_embeddings, token_embedding_args);
  FORWARD_PROPERTY(bool, encoder_embedding_mode, token_embedding_args);
  FORWARD_PROPERTY(bool, embedding_mode, token_embedding_args);

  PROPERTY(AttentionHeadArgs, attention_head_args);

  FORWARD_PROPERTY(int64_t, head_dim, attention_head_args);
  FORWARD_PROPERTY(int64_t, decoder_head_dim, attention_head_args);
  FORWARD_PROPERTY(int64_t, n_heads, attention_head_args);
  FORWARD_PROPERTY(int64_t, decoder_n_heads, attention_head_args);
  FORWARD_PROPERTY(int64_t, actual_n_heads, attention_head_args);
  FORWARD_PROPERTY(int64_t, num_attention_heads, attention_head_args);
  FORWARD_PROPERTY(std::optional<int64_t>, n_kv_heads, attention_head_args);
  FORWARD_PROPERTY(std::optional<int64_t>,
                   decoder_n_kv_heads,
                   attention_head_args);

  PROPERTY(KVCacheArgs, kv_cache_args);

  // Storage elements per full-attention K/V head, including backend padding.
  // Zero reuses the logical head_dim.
  FORWARD_PROPERTY(int64_t, kv_cache_storage_head_dim, kv_cache_args);

  PROPERTY(SpeculativeArgs, speculative_args);

  FORWARD_PROPERTY(int64_t, draft_vocab_size, speculative_args);

  PROPERTY(DSparkArgs, dspark_args);

  FORWARD_PROPERTY(int64_t, markov_rank, dspark_args);
  FORWARD_PROPERTY(int32_t, dspark_num_layers, dspark_args);
  FORWARD_PROPERTY(int32_t, dspark_block_size, dspark_args);
  FORWARD_PROPERTY(bool, dspark_use_native_sas, dspark_args);

  PROPERTY(ConfidenceHeadArgs, confidence_head_args);

  FORWARD_PROPERTY(bool, enable_confidence_head, confidence_head_args);
  FORWARD_PROPERTY(bool, confidence_head_with_markov, confidence_head_args);

  PROPERTY(DFlash2Args, dflash2_args);

  FORWARD_PROPERTY(int32_t, dflash2_block_size, dflash2_args);
  FORWARD_PROPERTY(int32_t, dflash2_conv_group_size, dflash2_args);
  FORWARD_PROPERTY(int32_t, dflash2_conv_kernel_size, dflash2_args);
  FORWARD_PROPERTY(int32_t, dflash2_selector_rank, dflash2_args);
  FORWARD_PROPERTY(int32_t, dflash2_selector_top_k, dflash2_args);

  PROPERTY(AttentionConfigArgs, attention_config_args);

  FORWARD_PROPERTY(bool, use_qk_norm, attention_config_args);
  FORWARD_PROPERTY(std::optional<float>, attn_scalar, attention_config_args);
  FORWARD_PROPERTY(bool, no_bias, attention_config_args);
  FORWARD_PROPERTY(bool, qkv_bias, attention_config_args);
  FORWARD_PROPERTY(bool, attention_bias, attention_config_args);
  FORWARD_PROPERTY(float, attention_dropout, attention_config_args);
  FORWARD_PROPERTY(bool, attn_output_gate, attention_config_args);

  PROPERTY(RopeArgs, rope_args);

  FORWARD_PROPERTY(int64_t, rotary_dim, rope_args);
  FORWARD_PROPERTY(float, rope_theta, rope_args);
  FORWARD_PROPERTY(std::string, rope_scaling_rope_type, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_factor, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_low_freq_factor, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_high_freq_factor, rope_args);
  FORWARD_PROPERTY(float, rope_extrapolation_factor, rope_args);
  FORWARD_PROPERTY(int64_t,
                   rope_scaling_original_max_position_embeddings,
                   rope_args);
  FORWARD_PROPERTY(int64_t, rope_scaling_beta_fast, rope_args);
  FORWARD_PROPERTY(int64_t, rope_scaling_beta_slow, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_attn_factor, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_mscale, rope_args);
  FORWARD_PROPERTY(float, rope_scaling_mscale_all_dim, rope_args);
  FORWARD_PROPERTY(std::vector<int64_t>, rope_scaling_mrope_section, rope_args);
  FORWARD_PROPERTY(bool, rope_scaling_mrope_interleaved, rope_args);
  FORWARD_PROPERTY(int32_t, rope_scaling, rope_args);
  FORWARD_PROPERTY(float, partial_rotary_factor, rope_args);

  PROPERTY(SpecialTokenArgs, special_token_args);

  FORWARD_PROPERTY(int32_t, bos_token_id, special_token_args);
  FORWARD_PROPERTY(int32_t, eos_token_id, special_token_args);
  FORWARD_PROPERTY(std::vector<int32_t>, eos_token_id_vec, special_token_args);
  FORWARD_PROPERTY(int32_t, pad_token_id, special_token_args);
  FORWARD_PROPERTY(std::unordered_set<int32_t>,
                   stop_token_ids,
                   special_token_args);

  // DeepSeek v2/v3 MoE configuration is kept with the shared routing group.

  PROPERTY(MoEArgs, moe_args);

  FORWARD_PROPERTY(std::string, topk_method, moe_args);
  FORWARD_PROPERTY(int32_t, n_routed_experts, moe_args);
  FORWARD_PROPERTY(int32_t, n_shared_experts, moe_args);
  FORWARD_PROPERTY(int32_t, num_experts_per_tok, moe_args);
  FORWARD_PROPERTY(int32_t, moe_intermediate_size, moe_args);
  FORWARD_PROPERTY(float, routed_scaling_factor, moe_args);
  FORWARD_PROPERTY(bool, norm_topk_prob, moe_args);
  FORWARD_PROPERTY(int32_t, n_group, moe_args);
  FORWARD_PROPERTY(int32_t, topk_group, moe_args);
  FORWARD_PROPERTY(std::string, scoring_func, moe_args);
  FORWARD_PROPERTY(int32_t, first_k_dense_replace, moe_args);
  FORWARD_PROPERTY(int32_t, moe_layer_freq, moe_args);
  FORWARD_PROPERTY(bool, use_moe, moe_args);
  FORWARD_PROPERTY(std::string, moe_score_func, moe_args);
  FORWARD_PROPERTY(float, moe_route_scale, moe_args);
  FORWARD_PROPERTY(bool, moe_use_shared_experts, moe_args);
  FORWARD_PROPERTY(float, swiglu_limit, moe_args);
  FORWARD_PROPERTY(int32_t, decoder_sparse_step, moe_args);
  FORWARD_PROPERTY(std::vector<int32_t>, mlp_only_layers, moe_args);
  FORWARD_PROPERTY(int32_t, num_experts, moe_args);
  FORWARD_PROPERTY(bool, output_router_logits, moe_args);
  FORWARD_PROPERTY(float, router_aux_loss_coef, moe_args);
  FORWARD_PROPERTY(int32_t, shared_expert_intermediate_size, moe_args);

  PROPERTY(AttentionArgs, attention_args);

  FORWARD_PROPERTY(bool, enable_mla, attention_args);

  PROPERTY(MlaArgs, mla_args);

  FORWARD_PROPERTY(int32_t, qk_nope_head_dim, mla_args);
  FORWARD_PROPERTY(int32_t, qk_rope_head_dim, mla_args);
  FORWARD_PROPERTY(int32_t, v_head_dim, mla_args);
  FORWARD_PROPERTY(int32_t, q_lora_rank, mla_args);
  FORWARD_PROPERTY(int32_t, kv_lora_rank, mla_args);

  PROPERTY(MtpArgs, mtp_args);

  FORWARD_PROPERTY(int32_t, num_nextn_predict_layers, mtp_args);
  FORWARD_PROPERTY(int64_t, mtp_start_layer_idx, mtp_args);
  FORWARD_PROPERTY(std::string, mtp_mlp_type, mtp_args);

  PROPERTY(IndexerArgs, indexer_args);

  // DeepSeek v3.2 indexer.
  FORWARD_PROPERTY(int32_t, index_head_dim, indexer_args);
  FORWARD_PROPERTY(int32_t, index_n_heads, indexer_args);
  FORWARD_PROPERTY(int32_t, index_topk, indexer_args);
  FORWARD_PROPERTY(int32_t, index_kpool, indexer_args);
  FORWARD_PROPERTY(bool, index_kpool_compress, indexer_args);
  FORWARD_PROPERTY(bool, index_kpool_always_select_tail, indexer_args);
  FORWARD_PROPERTY(bool, indexer_rope_interleave, indexer_args);
  FORWARD_PROPERTY(int32_t, index_topk_freq, indexer_args);
  FORWARD_PROPERTY(std::string, index_topk_pattern, indexer_args);
  FORWARD_PROPERTY(int32_t, index_skip_topk_offset, indexer_args);
  FORWARD_PROPERTY(bool, index_share_for_mtp_iteration, indexer_args);
  FORWARD_PROPERTY(std::vector<std::string>, indexer_types, indexer_args);

  PROPERTY(DeepseekV4Args, deepseek_v4_args);

  FORWARD_PROPERTY(int32_t, rope_head_dim, deepseek_v4_args);
  FORWARD_PROPERTY(int32_t, o_lora_rank, deepseek_v4_args);
  FORWARD_PROPERTY(int32_t, o_groups, deepseek_v4_args);
  FORWARD_PROPERTY(std::vector<int32_t>, compress_ratios, deepseek_v4_args);
  FORWARD_PROPERTY(float, compress_rope_theta, deepseek_v4_args);
  FORWARD_PROPERTY(int32_t, window_size, deepseek_v4_args);
  FORWARD_PROPERTY(int32_t, n_activated_experts, deepseek_v4_args);
  FORWARD_PROPERTY(int32_t, n_hash_layers, deepseek_v4_args);
  FORWARD_PROPERTY(float, factor, deepseek_v4_args);
  FORWARD_PROPERTY(float, beta_fast, deepseek_v4_args);
  FORWARD_PROPERTY(float, beta_slow, deepseek_v4_args);
  FORWARD_PROPERTY(std::string, scale_fmt, deepseek_v4_args);

  PROPERTY(HyperconnectionArgs, hyperconnection_args);

  FORWARD_PROPERTY(int32_t, hc_mult, hyperconnection_args);
  FORWARD_PROPERTY(int32_t, hc_sinkhorn_iters, hyperconnection_args);
  FORWARD_PROPERTY(float, hc_eps, hyperconnection_args);

  FORWARD_PROPERTY(int64_t, max_batch_size, kv_cache_args);
  FORWARD_PROPERTY(int64_t, max_seq_len, kv_cache_args);

  PROPERTY(MultimodalTokenArgs, multimodal_token_args);

  FORWARD_PROPERTY(int32_t, vision_start_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, vision_end_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, vision_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, image_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, video_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, audio_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, image_start_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, image_end_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, video_start_token_id, multimodal_token_args);
  FORWARD_PROPERTY(int32_t, video_end_token_id, multimodal_token_args);

  PROPERTY(VisionArgs, vision_args);

  FORWARD_PROPERTY(int32_t, query_num, vision_args);
  FORWARD_PROPERTY(std::string, vision_custom_adapter, vision_args);
  FORWARD_PROPERTY(int32_t, vision_max_slice_nums, vision_args);
  FORWARD_PROPERTY(float, mm_dropout, vision_args);
  FORWARD_PROPERTY(std::string, mm_hidden_act, vision_args);
  FORWARD_PROPERTY(int64_t, mm_hidden_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_image_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_intermediate_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_channels, vision_args);
  FORWARD_PROPERTY(float, mm_initializer_range, vision_args);
  FORWARD_PROPERTY(float, mm_layer_norm_eps, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_attention_heads, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_beam_groups, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_beams, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_hidden_layers, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_return_sequences, vision_args);
  FORWARD_PROPERTY(bool, mm_output_attentions, vision_args);
  FORWARD_PROPERTY(bool, mm_output_hidden_states, vision_args);
  FORWARD_PROPERTY(bool, mm_output_scores, vision_args);
  FORWARD_PROPERTY(int64_t, mm_patch_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_projection_dim, vision_args);
  FORWARD_PROPERTY(int64_t, mm_projector_hidden_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_spatial_merge_size, vision_args);
  FORWARD_PROPERTY(int64_t, mm_spatial_patch_size, vision_args);
  FORWARD_PROPERTY(bool, mm_remove_invalid_values, vision_args);
  FORWARD_PROPERTY(float, mm_repetition_penalty, vision_args);
  FORWARD_PROPERTY(bool, mm_return_dict, vision_args);
  FORWARD_PROPERTY(bool, mm_return_dict_in_generate, vision_args);
  FORWARD_PROPERTY(float, mm_temperature, vision_args);
  FORWARD_PROPERTY(bool, mm_tie_encoder_decoder, vision_args);
  FORWARD_PROPERTY(bool, mm_tie_word_embeddings, vision_args);
  FORWARD_PROPERTY(int64_t, mm_top_k, vision_args);
  FORWARD_PROPERTY(float, mm_top_p, vision_args);
  FORWARD_PROPERTY(bool, mm_torchscript, vision_args);
  FORWARD_PROPERTY(bool, mm_use_bfloat16, vision_args);
  FORWARD_PROPERTY(int64_t, mm_head_dim, vision_args);
  FORWARD_PROPERTY(int64_t, mm_vocab_size, vision_args);
  FORWARD_PROPERTY(int, mm_window_size, vision_args);
  FORWARD_PROPERTY(std::vector<int64_t>, mm_fullatt_block_indexes, vision_args);
  FORWARD_PROPERTY(std::vector<int64_t>,
                   mm_deepstack_visual_indexes,
                   vision_args);
  FORWARD_PROPERTY(int, mm_tokens_per_second, vision_args);
  FORWARD_PROPERTY(int, mm_temporal_patch_size, vision_args);
  FORWARD_PROPERTY(std::string, mm_projector_type, vision_args);
  FORWARD_PROPERTY(int64_t, mm_num_position_embeddings, vision_args);
  FORWARD_PROPERTY(std::string, mm_projector_hidden_act, vision_args);
  FORWARD_PROPERTY(int64_t, mm_projector_n_layers, vision_args);
  FORWARD_PROPERTY(int64_t, mm_vision_feature_layer, vision_args);
  FORWARD_PROPERTY(std::string, mm_vision_feature_select_strategy, vision_args);

  PROPERTY(HybridAttentionArgs, hybrid_attention_args);

  FORWARD_PROPERTY(int32_t, full_attention_interval, hybrid_attention_args);
  FORWARD_PROPERTY(std::vector<int32_t>,
                   full_attn_layers,
                   hybrid_attention_args);
  FORWARD_PROPERTY(std::vector<std::string>,
                   layer_types,
                   hybrid_attention_args);

  PROPERTY(LinearAttentionArgs, linear_attention_args);

  FORWARD_PROPERTY(int32_t, linear_conv_kernel_dim, linear_attention_args);
  FORWARD_PROPERTY(int32_t, linear_key_head_dim, linear_attention_args);
  FORWARD_PROPERTY(int32_t, linear_value_head_dim, linear_attention_args);
  FORWARD_PROPERTY(int64_t, linear_num_key_heads, linear_attention_args);
  FORWARD_PROPERTY(int32_t, linear_num_value_heads, linear_attention_args);
  FORWARD_PROPERTY(float, linear_lower_bound, linear_attention_args);
  FORWARD_PROPERTY(std::string, mamba_ssm_dtype, linear_attention_args);

  PROPERTY(ImagePreprocessArgs, image_preprocess_args);

  FORWARD_PROPERTY(bool, mm_image_do_center_crop, image_preprocess_args);
  FORWARD_PROPERTY(int32_t, mm_image_crop_height_size, image_preprocess_args);
  FORWARD_PROPERTY(int32_t, mm_image_crop_width_size, image_preprocess_args);
  FORWARD_PROPERTY(bool, mm_image_do_resize, image_preprocess_args);
  FORWARD_PROPERTY(int32_t,
                   mm_image_resize_shortest_edge,
                   image_preprocess_args);
  FORWARD_PROPERTY(int32_t, mm_image_resample, image_preprocess_args);
  FORWARD_PROPERTY(bool, mm_image_do_rescale, image_preprocess_args);
  FORWARD_PROPERTY(double, mm_image_rescale_factor, image_preprocess_args);
  FORWARD_PROPERTY(bool, mm_image_do_normalize, image_preprocess_args);
  FORWARD_PROPERTY(std::vector<double>,
                   mm_image_normalize_mean,
                   image_preprocess_args);
  FORWARD_PROPERTY(std::vector<double>,
                   mm_image_normalize_std,
                   image_preprocess_args);

  PROPERTY(ImageVideoArgs, image_video_args);

  FORWARD_PROPERTY(int64_t, mm_init_pos_emb_width, vision_args);
  FORWARD_PROPERTY(int64_t, mm_init_pos_emb_height, vision_args);
  FORWARD_PROPERTY(int64_t, mm_init_pos_emb_time, vision_args);
  FORWARD_PROPERTY(int64_t, mm_km_in_patch_limit, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_patch_size, image_video_args);
  FORWARD_PROPERTY(std::vector<int64_t>, mm_km_image_mean, image_video_args);
  FORWARD_PROPERTY(std::vector<int64_t>, mm_km_image_std, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_merge_kernel_size, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_fixed_output_tokens, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_patch_limit_on_one_side, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_in_patch_limit_each_frame, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_in_patch_limit_video, image_video_args);
  FORWARD_PROPERTY(float, mm_km_sample_fps, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_max_num_frames_each_video, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_km_temporal_merge_kernel_size, image_video_args);
  FORWARD_PROPERTY(std::string, mm_km_timestamp_mode, image_video_args);
  FORWARD_PROPERTY(bool, mm_video_do_rescale, image_video_args);
  FORWARD_PROPERTY(std::vector<double>,
                   mm_video_normalize_mean,
                   image_video_args);
  FORWARD_PROPERTY(std::vector<double>,
                   mm_video_normalize_std,
                   image_video_args);
  FORWARD_PROPERTY(int, mm_image_min_pixels, image_video_args);
  FORWARD_PROPERTY(int, mm_image_max_pixels, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_image_shortest_edge, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_image_longest_edge, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_video_shortest_edge, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_video_longest_edge, image_video_args);
  FORWARD_PROPERTY(int, mm_image_patch_size, image_video_args);
  FORWARD_PROPERTY(int, mm_image_temporal_patch_size, image_video_args);
  FORWARD_PROPERTY(int, mm_image_merge_size, image_video_args);
  FORWARD_PROPERTY(int, mm_image_min_tokens, image_video_args);
  FORWARD_PROPERTY(int, mm_image_max_tokens, image_video_args);
  FORWARD_PROPERTY(int, mm_video_patch_size, image_video_args);
  FORWARD_PROPERTY(int, mm_video_temporal_patch_size, image_video_args);
  FORWARD_PROPERTY(int, mm_video_merge_size, image_video_args);
  FORWARD_PROPERTY(int, mm_video_min_tokens, image_video_args);
  FORWARD_PROPERTY(int, mm_video_max_tokens, image_video_args);
  FORWARD_PROPERTY(int, mm_image_feature_size, image_video_args);
  FORWARD_PROPERTY(int, mm_scale_resolution, image_video_args);
  FORWARD_PROPERTY(bool, mm_slice_mode, image_video_args);
  FORWARD_PROPERTY(bool, mm_use_image_id, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_image_token_index, image_video_args);
  FORWARD_PROPERTY(int64_t, mm_pad_token_id, image_video_args);

  PROPERTY(AudioArgs, audio_args);

  FORWARD_PROPERTY(int64_t, mm_audio_sampling_rate, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_num_mel_bins, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_frame_length, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_frame_shift, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_max_frames, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_downsample_rate, audio_args);
  FORWARD_PROPERTY(std::vector<double>, mm_audio_cmvn_means, audio_args);
  FORWARD_PROPERTY(std::vector<double>, mm_audio_cmvn_inverse_std, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_idim, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_d_model, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_n_layers, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_n_head, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_kernel_size, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_pe_maxlen, audio_args);
  FORWARD_PROPERTY(int64_t, mm_audio_ctc_pad_token_id, audio_args);

  PROPERTY(SlidingWindowArgs, sliding_window_args);

  FORWARD_PROPERTY(bool, use_sliding_window, sliding_window_args);
  FORWARD_PROPERTY(int32_t, sliding_window, sliding_window_args);
  FORWARD_PROPERTY(bool, sliding_window_non_causal, sliding_window_args);
  FORWARD_PROPERTY(int32_t, max_window_layers, sliding_window_args);

  FORWARD_PROPERTY(int64_t, num_speculative_tokens, speculative_args);
  FORWARD_PROPERTY(bool, enable_embedded_eagle3_draft, speculative_args);
  FORWARD_PROPERTY(int32_t, max_concurrent_prefills_per_dp, speculative_args);
  FORWARD_PROPERTY(std::vector<int32_t>, layers_to_capture, speculative_args);
  FORWARD_PROPERTY(bool, norm_before_fc, speculative_args);
  FORWARD_PROPERTY(bool, norm_before_residual, speculative_args);
  FORWARD_PROPERTY(bool, norm_output, speculative_args);
  FORWARD_PROPERTY(bool, fc_norm, speculative_args);

  PROPERTY(VaeArgs, vae_args);

  FORWARD_PROPERTY(std::vector<std::string>, down_block_types, vae_args);
  FORWARD_PROPERTY(std::vector<std::string>, up_block_types, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, block_out_channels, vae_args);
  FORWARD_PROPERTY(int64_t, layers_per_block, vae_args);
  FORWARD_PROPERTY(int64_t, latent_channels, vae_args);
  FORWARD_PROPERTY(int64_t, norm_num_groups, vae_args);
  FORWARD_PROPERTY(int64_t, sample_size, vae_args);
  FORWARD_PROPERTY(float, scale_factor, vae_args);
  FORWARD_PROPERTY(float, shift_factor, vae_args);
  FORWARD_PROPERTY(bool, mid_block_add_attention, vae_args);
  FORWARD_PROPERTY(bool, force_upcast, vae_args);
  FORWARD_PROPERTY(bool, use_quant_conv, vae_args);
  FORWARD_PROPERTY(bool, use_post_quant_conv, vae_args);
  FORWARD_PROPERTY(int64_t, vae_scale_factor_temporal, vae_args);
  FORWARD_PROPERTY(int64_t, vae_scale_factor_spatial, vae_args);
  FORWARD_PROPERTY(bool, vae_is_residual, vae_args);
  FORWARD_PROPERTY(float, batch_norm_eps, vae_args);
  FORWARD_PROPERTY(float, batch_norm_momentum, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, ae_patch_size, vae_args);
  FORWARD_PROPERTY(int64_t, vae_dim, vae_args);
  FORWARD_PROPERTY(int64_t, vae_num_heads, vae_args);
  FORWARD_PROPERTY(int64_t, encoder_num_blocks, vae_args);
  FORWARD_PROPERTY(int64_t, decoder_num_blocks, vae_args);
  FORWARD_PROPERTY(int64_t, shared_heads_kv, vae_args);
  FORWARD_PROPERTY(int64_t, vae_rope_theta, vae_args);
  FORWARD_PROPERTY(int64_t, vae_block_size, vae_args);
  FORWARD_PROPERTY(int64_t, vae_patch_size, vae_args);
  FORWARD_PROPERTY(bool, encoder_last_ln, vae_args);
  FORWARD_PROPERTY(float, shifting_factor, vae_args);
  FORWARD_PROPERTY(float, scaling_factor, vae_args);
  FORWARD_PROPERTY(bool, use_variation, vae_args);
  FORWARD_PROPERTY(int64_t, base_dim, vae_args);
  FORWARD_PROPERTY(int64_t, z_dim, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, dim_mult, vae_args);
  FORWARD_PROPERTY(std::vector<double>, attn_scales, vae_args);
  FORWARD_PROPERTY(std::vector<bool>, temperal_downsample, vae_args);
  FORWARD_PROPERTY(int64_t, num_res_blocks, vae_args);
  FORWARD_PROPERTY(double, dropout, vae_args);
  FORWARD_PROPERTY(std::vector<double>, latents_mean, vae_args);
  FORWARD_PROPERTY(std::vector<double>, latents_std, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, spatial_downsample_factors, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, temporal_downsample_factors, vae_args);
  FORWARD_PROPERTY(std::string, spatial_padding_mode, vae_args);
  FORWARD_PROPERTY(int64_t, decoder_num_register_tokens, vae_args);
  FORWARD_PROPERTY(int64_t, decoder_ffn_mult, vae_args);
  FORWARD_PROPERTY(float, decoder_rope_theta, vae_args);
  FORWARD_PROPERTY(float, decoder_rope_dim_ratio, vae_args);
  FORWARD_PROPERTY(float, decoder_norm_eps, vae_args);
  FORWARD_PROPERTY(int64_t, encoder_dim, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, encoder_rates, vae_args);
  FORWARD_PROPERTY(int64_t, decoder_dim, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, decoder_rates, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, decoder_kernel_sizes, vae_args);
  FORWARD_PROPERTY(std::vector<int64_t>, resblock_kernel_sizes, vae_args);
  FORWARD_PROPERTY(int64_t, sampling_rate, vae_args);
  FORWARD_PROPERTY(int64_t, clip_length, vae_args);
  FORWARD_PROPERTY(int64_t, token_drop, vae_args);

  PROPERTY(DitArgs, dit_args);

  FORWARD_PROPERTY(int64_t, joint_attention_dim, dit_args);
  FORWARD_PROPERTY(int64_t, pooled_projection_dim, dit_args);
  FORWARD_PROPERTY(bool, guidance_embeds, dit_args);
  FORWARD_PROPERTY(std::vector<int64_t>, axes_dims_rope, dit_args);
  FORWARD_PROPERTY(int64_t, num_single_layers, dit_args);
  FORWARD_PROPERTY(float, mlp_ratio, dit_args);
  FORWARD_PROPERTY(int, timestep_guidance_channels, dit_args);
  FORWARD_PROPERTY(int64_t, patch_size, dit_args);
  FORWARD_PROPERTY(std::vector<int64_t>, wan_patch_size, dit_args);
  FORWARD_PROPERTY(bool, cross_attn_norm, dit_args);
  FORWARD_PROPERTY(double, eps, dit_args);
  FORWARD_PROPERTY(int64_t, ffn_dim, dit_args);
  FORWARD_PROPERTY(int64_t, time_freq_dim, dit_args);
  FORWARD_PROPERTY(int64_t, dit_in_channels, dit_args);
  FORWARD_PROPERTY(int64_t, dit_out_channels, dit_args);
  FORWARD_PROPERTY(std::string, qk_norm, dit_args);
  FORWARD_PROPERTY(int64_t, rope_max_seq_len, dit_args);
  FORWARD_PROPERTY(int64_t, text_embed_dim, dit_args);
  FORWARD_PROPERTY(int64_t, image_embed_dim, dit_args);
  FORWARD_PROPERTY(int64_t, added_kv_proj_dim, dit_args);
  FORWARD_PROPERTY(int64_t, pos_embed_seq_len, dit_args);
  FORWARD_PROPERTY(int64_t, num_refiner_layers, dit_args);
  FORWARD_PROPERTY(int64_t, audio_in_channels, dit_args);
  FORWARD_PROPERTY(int64_t, time_embed_hidden_dim, dit_args);
  FORWARD_PROPERTY(int64_t, time_embed_dim, dit_args);
  FORWARD_PROPERTY(int64_t, rope_freq_dim, dit_args);
  FORWARD_PROPERTY(float, qk_norm_eps, dit_args);
  FORWARD_PROPERTY(float, final_norm_eps, dit_args);
  FORWARD_PROPERTY(int64_t, txt_dim, dit_args);
  FORWARD_PROPERTY(int64_t, txt_in_channels, dit_args);
  FORWARD_PROPERTY(int64_t, txt_out_channels, dit_args);
  FORWARD_PROPERTY(int64_t, emb_dim, dit_args);
  FORWARD_PROPERTY(int64_t, heads, dit_args);
  FORWARD_PROPERTY(int64_t, rope_dim, dit_args);
  FORWARD_PROPERTY(int64_t, expand_ratio, dit_args);
  FORWARD_PROPERTY(int64_t, block_size, dit_args);
  FORWARD_PROPERTY(int64_t, latent_dim, dit_args);
  FORWARD_PROPERTY(bool, qk_bias, dit_args);
  FORWARD_PROPERTY(float, norm_eps, dit_args);
  FORWARD_PROPERTY(bool, zero_cond_t, dit_args);
  FORWARD_PROPERTY(bool, use_additional_t_cond, dit_args);
  FORWARD_PROPERTY(bool, use_layer3d_rope, dit_args);
  FORWARD_PROPERTY(double, mlp_width_ratio, dit_args);
  FORWARD_PROPERTY(int64_t, text_dim, dit_args);
  FORWARD_PROPERTY(std::vector<int64_t>, rope_dim_list, dit_args);
  FORWARD_PROPERTY(int64_t, rope_theta_dit, dit_args);

  PROPERTY(T5Args, t5_args);

  FORWARD_PROPERTY(int64_t, d_model, t5_args);
  FORWARD_PROPERTY(int64_t, num_layers, t5_args);
  FORWARD_PROPERTY(int64_t, d_kv, t5_args);
  FORWARD_PROPERTY(int64_t, d_ff, t5_args);
  FORWARD_PROPERTY(std::string, act_fn, t5_args);
  FORWARD_PROPERTY(bool, is_gated_act, t5_args);
  FORWARD_PROPERTY(int64_t, relative_attention_num_buckets, t5_args);
  FORWARD_PROPERTY(int64_t, relative_attention_max_distance, t5_args);

  PROPERTY(SchedulerArgs, scheduler_args);

  FORWARD_PROPERTY(int64_t, num_train_timesteps, scheduler_args);
  FORWARD_PROPERTY(int64_t, shift, scheduler_args);
  FORWARD_PROPERTY(bool, use_dynamic_shifting, scheduler_args);
  FORWARD_PROPERTY(float, base_shift, scheduler_args);
  FORWARD_PROPERTY(float, max_shift, scheduler_args);
  FORWARD_PROPERTY(int64_t, base_image_seq_len, scheduler_args);
  FORWARD_PROPERTY(int64_t, max_image_seq_len, scheduler_args);
  FORWARD_PROPERTY(float, shift_terminal, scheduler_args);
  FORWARD_PROPERTY(float, beta_start, scheduler_args);
  FORWARD_PROPERTY(float, beta_end, scheduler_args);
  FORWARD_PROPERTY(std::string, beta_schedule, scheduler_args);
  FORWARD_PROPERTY(std::vector<float>, trained_betas, scheduler_args);
  FORWARD_PROPERTY(int64_t, solver_order, scheduler_args);
  FORWARD_PROPERTY(std::string, prediction_type, scheduler_args);
  FORWARD_PROPERTY(bool, thresholding, scheduler_args);
  FORWARD_PROPERTY(float, dynamic_thresholding_ratio, scheduler_args);
  FORWARD_PROPERTY(float, sample_max_value, scheduler_args);
  FORWARD_PROPERTY(bool, predict_x0, scheduler_args);
  FORWARD_PROPERTY(std::string, solver_type, scheduler_args);
  FORWARD_PROPERTY(bool, lower_order_final, scheduler_args);
  FORWARD_PROPERTY(std::vector<int64_t>, disable_corrector, scheduler_args);
  FORWARD_PROPERTY(bool, use_karras_sigmas, scheduler_args);
  FORWARD_PROPERTY(bool, use_exponential_sigmas, scheduler_args);
  FORWARD_PROPERTY(bool, use_beta_sigmas, scheduler_args);
  FORWARD_PROPERTY(bool, use_flow_sigmas, scheduler_args);
  FORWARD_PROPERTY(float, flow_shift, scheduler_args);
  FORWARD_PROPERTY(std::string, timestep_spacing, scheduler_args);
  FORWARD_PROPERTY(int64_t, steps_offset, scheduler_args);
  FORWARD_PROPERTY(std::string, final_sigmas_type, scheduler_args);
  FORWARD_PROPERTY(bool, rescale_betas_zero_snr, scheduler_args);
  FORWARD_PROPERTY(std::string, time_shift_type, scheduler_args);
};

// Qwen hybrid models may describe full-attention layers explicitly via
// layer_types or implicitly via full_attention_interval.
inline bool is_full_attention_layer(const ModelArgs& args, int64_t layer_id) {
  const auto& hybrid_layer_types = args.layer_types();
  if (layer_id >= 0 &&
      layer_id < static_cast<int64_t>(hybrid_layer_types.size())) {
    const auto& layer_type = hybrid_layer_types[layer_id];
    // deepseek_sparse_attention (glm5_next DSA layers) is a full-attention
    // variant: it uses paged KV, not the linear (conv/ssm) state cache. Treat
    // it as full attention here so capacity estimation and per-layer dispatch
    // classify it correctly (mirrors is_linear_attention_layer's exclusion).
    return layer_type == "full_attention" || layer_type == "attention" ||
           layer_type == "deepseek_sparse_attention";
  }

  int32_t attention_interval = args.full_attention_interval();
  if (attention_interval <= 1) {
    return true;
  }
  return (layer_id + 1) % attention_interval == 0;
}

inline bool has_linear_attention_layers(const ModelArgs& args) {
  const auto& hybrid_layer_types = args.layer_types();
  if (!hybrid_layer_types.empty()) {
    return std::any_of(hybrid_layer_types.begin(),
                       hybrid_layer_types.end(),
                       [](const std::string& layer_type) {
                         return layer_type != "full_attention" &&
                                layer_type != "attention" &&
                                layer_type != "deepseek_sparse_attention";
                       });
  }
  return args.full_attention_interval() > 1;
}

// Closed set by design: a new target variant must be enumerated here rather
// than matched by a "qwen3_5_" prefix, so draft bodies ("qwen3_5_mtp") are not
// silently promoted onto the target spec-verify path.
inline bool is_qwen3_5_target_model_type(std::string_view model_type) {
  return model_type == "qwen3_5" || model_type == "qwen3_5_moe" ||
         model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text";
}

// Reconciles the model vocab size against the tokenizer's: adopts the
// tokenizer size when unset, warns on unsafe mismatch.
inline void reconcile_vocab_size(ModelArgs& args,
                                 int64_t tokenizer_vocab_size) {
  const int64_t model_vocab_size = args.vocab_size();
  if (tokenizer_vocab_size == model_vocab_size) {
    return;
  }
  // use tokenizer vocab size if model vocab size is not set
  if (model_vocab_size <= 0) {
    LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab size: "
                 << tokenizer_vocab_size;
    args.vocab_size(tokenizer_vocab_size);
  } else if (tokenizer_vocab_size > model_vocab_size) {
    LOG(WARNING) << "Unsafe vocab mismatch: tokenizer: " << tokenizer_vocab_size
                 << ", model: " << model_vocab_size;
  } else {
    LOG(INFO) << "Tokenizer/model vocab differ: tokenizer="
              << tokenizer_vocab_size << ", model=" << model_vocab_size;
  }
}

inline bool is_glm5_next_target_model_type(std::string_view model_type) {
  return model_type == "glm5_3_flash" || model_type == "glm5_next";
}

inline std::ostream& operator<<(std::ostream& os, const ModelArgs& args) {
  os << "ModelArgs: [model_type: " << args.model_type();
  os << ", encoder_embedding_mode: " << args.encoder_embedding_mode();
  os << ", embedding_mode: " << args.embedding_mode();
  os << ", dtype: " << args.dtype();
  os << ", hidden_size: " << args.hidden_size();
  os << ", hidden_act: " << args.hidden_act();
  os << ", intermediate_size: " << args.intermediate_size();
  os << ", moe_intermediate_size: " << args.moe_intermediate_size();
  os << ", n_routed_experts: " << args.n_routed_experts();
  os << ", n_activated_experts: " << args.n_activated_experts();
  os << ", num_experts_per_tok: " << args.num_experts_per_tok();
  os << ", n_layers: " << args.n_layers();
  os << ", n_encoder_layers: " << args.n_encoder_layers();
  os << ", head_dim: " << args.head_dim();
  os << ", kv_cache_storage_head_dim: " << args.kv_cache_storage_head_dim();
  os << ", decoder_head_dim: " << args.decoder_head_dim();
  os << ", n_heads: " << args.n_heads();
  os << ", decoder_n_heads: " << args.decoder_n_heads();
  os << ", n_kv_heads: " << args.n_kv_heads().value_or(-1);
  os << ", decoder_n_kv_heads: " << args.decoder_n_kv_heads().value_or(-1);
  os << ", vocab_size: " << args.vocab_size();
  os << ", rms_norm_eps: " << args.rms_norm_eps();
  os << ", layer_norm_eps: " << args.layer_norm_eps();
  os << ", rotary_dim: " << args.rotary_dim();
  os << ", rope_theta: " << args.rope_theta();
  os << ", rope_scaling_rope_type: " << args.rope_scaling_rope_type();
  os << ", rope_scaling_factor: " << args.rope_scaling_factor();
  os << ", rope_scaling_low_freq_factor: "
     << args.rope_scaling_low_freq_factor();
  os << ", rope_scaling_high_freq_factor: "
     << args.rope_scaling_high_freq_factor();
  os << ", rope_scaling_original_max_position_embeddings: "
     << args.rope_scaling_original_max_position_embeddings();
  os << ", rope_scaling_mrope_section: [";
  for (const auto& sec : args.rope_scaling_mrope_section()) {
    os << sec << ", ";
  }
  os << "]";
  os << ", max_position_embeddings: " << args.max_position_embeddings();
  os << ", use_absolute_position_embedding: "
     << args.use_absolute_position_embedding();
  os << ", bos_token_id: " << args.bos_token_id();
  os << ", eos_token_id: " << args.eos_token_id();
  os << ", pad_token_id: " << args.pad_token_id();
  os << ", attn_scalar: " << args.attn_scalar().value_or(0.0f);
  os << ", no_bias: " << args.no_bias();
  os << ", qkv_bias: " << args.qkv_bias();
  os << ", stop_token_ids: [";
  for (const auto& id : args.stop_token_ids()) {
    os << id << ", ";
  }
  os << "]";
  os << ", vision_start_token_id: " << args.vision_start_token_id();
  os << ", vision_end_token_id: " << args.vision_end_token_id();
  os << ", vision_token_id: " << args.vision_token_id();
  os << ", image_token_id: " << args.image_token_id();
  os << ", video_token_id: " << args.video_token_id();
  os << ", vision_custom_adapter: " << args.vision_custom_adapter();
  os << ", vision_max_slice_nums: " << args.vision_max_slice_nums();
  os << ", mm_dropout: " << args.mm_dropout();
  os << ", mm_hidden_act: " << args.mm_hidden_act();
  os << ", mm_hidden_size: " << args.mm_hidden_size();
  os << ", mm_image_size: " << args.mm_image_size();
  os << ", mm_intermediate_size: " << args.mm_intermediate_size();
  os << ", mm_num_channels: " << args.mm_num_channels();
  os << ", mm_initializer_range: " << args.mm_initializer_range();
  os << ", mm_layer_norm_eps: " << args.mm_layer_norm_eps();
  os << ", mm_num_attention_heads: " << args.mm_num_attention_heads();
  os << ", mm_num_beam_groups: " << args.mm_num_beam_groups();
  os << ", mm_num_beams: " << args.mm_num_beams();
  os << ", mm_num_hidden_layers: " << args.mm_num_hidden_layers();
  os << ", mm_num_return_sequences: " << args.mm_num_return_sequences();
  os << ", mm_output_attentions: " << args.mm_output_attentions();
  os << ", mm_output_hidden_states: " << args.mm_output_hidden_states();
  os << ", mm_output_scores: " << args.mm_output_scores();
  os << ", mm_patch_size: " << args.mm_patch_size();
  os << ", mm_projection_dim: " << args.mm_projection_dim();
  os << ", mm_spatial_merge_size: " << args.mm_spatial_merge_size();
  os << ", mm_spatial_patch_size: " << args.mm_spatial_patch_size();
  os << ", mm_remove_invalid_values: " << args.mm_remove_invalid_values();
  os << ", mm_repetition_penalty: " << args.mm_repetition_penalty();
  os << ", mm_return_dict: " << args.mm_return_dict();
  os << ", mm_return_dict_in_generate: " << args.mm_return_dict_in_generate();
  os << ", mm_temperature: " << args.mm_temperature();
  os << ", mm_tie_encoder_decoder: " << args.mm_tie_encoder_decoder();
  os << ", mm_tie_word_embeddings: " << args.mm_tie_word_embeddings();
  os << ", mm_top_k: " << args.mm_top_k();
  os << ", mm_top_p: " << args.mm_top_p();
  os << ", mm_torchscript: " << args.mm_torchscript();
  os << ", mm_use_bfloat16: " << args.mm_use_bfloat16();
  os << ", mm_head_dim: " << args.mm_head_dim();
  os << ", mm_vocab_size: " << args.mm_vocab_size();
  os << ", mm_window_size: " << args.mm_window_size();
  os << ", mm_fullatt_block_indexes: [";
  for (auto& index : args.mm_fullatt_block_indexes()) {
    os << index << ",";
  }
  os << "]";
  os << ", mm_deepstack_visual_indexes: [";
  for (auto& index : args.mm_deepstack_visual_indexes()) {
    os << index << ",";
  }
  os << "]";
  os << ", mm_tokens_per_second: " << args.mm_tokens_per_second();
  os << ", mm_temporal_patch_size: " << args.mm_temporal_patch_size();
  os << ", mm_projector_type: " << args.mm_projector_type();
  os << ", mm_projector_hidden_act: " << args.mm_projector_hidden_act();
  os << ", mm_projector_n_layers: " << args.mm_projector_n_layers();
  os << ", mm_vision_feature_layer: " << args.mm_vision_feature_layer();
  os << ", mm_vision_feature_select_strategy: "
     << args.mm_vision_feature_select_strategy();
  os << ", mm_image_do_center_crop: " << args.mm_image_do_center_crop();
  os << ", mm_image_crop_height_size: " << args.mm_image_crop_height_size();
  os << ", mm_image_crop_width_size: " << args.mm_image_crop_width_size();
  os << ", mm_image_do_resize: " << args.mm_image_do_resize();
  os << ", mm_image_resize_shortest_edge: "
     << args.mm_image_resize_shortest_edge();
  os << ", mm_image_resample: " << args.mm_image_resample();
  os << ", mm_image_do_rescale: " << args.mm_image_do_rescale();
  os << ", mm_video_do_rescale: " << args.mm_video_do_rescale();
  os << ", mm_image_rescale_factor: " << args.mm_image_rescale_factor();
  os << ", mm_image_do_normalize: " << args.mm_image_do_normalize();
  os << ", mm_image_normalize_mean: [";
  for (const auto& mean : args.mm_image_normalize_mean()) {
    os << mean << ", ";
  }
  os << "], mm_image_normalize_std: [";
  for (const auto& std : args.mm_image_normalize_std()) {
    os << std << ", ";
  }
  os << "]";
  os << ", mm_image_shortest_edge: " << args.mm_image_shortest_edge();
  os << ", mm_image_longest_edge: " << args.mm_image_longest_edge();
  os << ", mm_image_min_pixels: " << args.mm_image_min_pixels();
  os << ", mm_image_max_pixels: " << args.mm_image_max_pixels();
  os << ", mm_image_patch_size: " << args.mm_image_patch_size();
  os << ", mm_image_temporal_patch_size: "
     << args.mm_image_temporal_patch_size();
  os << ", mm_image_merge_size: " << args.mm_image_merge_size();
  os << ", mm_image_min_tokens: " << args.mm_image_min_tokens();
  os << ", mm_image_max_tokens: " << args.mm_image_max_tokens();
  os << ", mm_image_token_index: " << args.mm_image_token_index();
  os << ", mm_video_normalize_mean: [";
  for (const auto& mean : args.mm_video_normalize_mean()) {
    os << mean << ", ";
  }
  os << "], mm_video_normalize_std: [";
  for (const auto& std : args.mm_video_normalize_std()) {
    os << std << ", ";
  }
  os << "]";
  os << ", mm_video_shortest_edge: " << args.mm_video_shortest_edge();
  os << ", mm_video_longest_edge: " << args.mm_video_longest_edge();
  os << ", mm_video_patch_size: " << args.mm_video_patch_size();
  os << ", mm_video_temporal_patch_size: "
     << args.mm_video_temporal_patch_size();
  os << ", mm_video_merge_size: " << args.mm_video_merge_size();
  os << ", mm_video_min_tokens: " << args.mm_video_min_tokens();
  os << ", mm_video_max_tokens: " << args.mm_video_max_tokens();
  os << ", mm_pad_token_id: " << args.mm_pad_token_id();
  os << ", tie_word_embeddings: " << args.tie_word_embeddings();
  os << ", use_sliding_window: " << args.use_sliding_window();
  os << ", sliding_window: " << args.sliding_window();
  os << ", max_window_layers: " << args.max_window_layers();
  os << ", query_num: " << args.query_num();
  os << ", num_speculative_tokens: " << args.num_speculative_tokens();
  os << ", in_channels: " << args.in_channels();
  os << ", out_channels: " << args.out_channels();
  os << ", down_block_types: [";
  for (const auto& type : args.down_block_types()) {
    os << type << ", ";
  }
  os << "]";
  os << ", up_block_types: [";
  for (const auto& type : args.up_block_types()) {
    os << type << ", ";
  }
  os << "]";
  os << ", block_out_channels: [";
  for (const auto& channel : args.block_out_channels()) {
    os << channel << ", ";
  }
  os << "]";
  os << ", layers_per_block: " << args.layers_per_block();
  os << ", latent_channels: " << args.latent_channels();
  os << ", norm_num_groups: " << args.norm_num_groups();
  os << ", sample_size: " << args.sample_size();
  os << ", scale_factor: " << args.scale_factor();
  os << ", shift_factor: " << args.shift_factor();
  os << ", mid_block_add_attention: " << args.mid_block_add_attention();
  os << ", force_upcast: " << args.force_upcast();
  os << ", use_quant_conv: " << args.use_quant_conv();
  os << ", use_post_quant_conv: " << args.use_post_quant_conv();
  os << ", joint_attention_dim: " << args.joint_attention_dim();
  os << ", pooled_projection_dim: " << args.pooled_projection_dim();
  os << ", guidance_embeds: " << args.guidance_embeds();
  os << ", axes_dims_rope: [";
  for (const auto& dim : args.axes_dims_rope()) {
    os << dim << ", ";
  }
  os << "]";
  os << ", num_single_layers: " << args.num_single_layers();
  os << ", d_model: " << args.d_model();
  os << ", num_layers: " << args.num_layers();
  os << ", d_kv: " << args.d_kv();
  os << ", d_ff: " << args.d_ff();
  os << ", act_fn: " << args.act_fn();
  os << ", is_gated_act: " << args.is_gated_act();
  os << ", relative_attention_num_buckets: "
     << args.relative_attention_num_buckets();
  os << ", relative_attention_max_distance: "
     << args.relative_attention_max_distance();
  os << ", num_train_timesteps: " << args.num_train_timesteps();
  os << ", shift: " << args.shift();
  os << ", use_dynamic_shifting: " << args.use_dynamic_shifting();
  os << ", base_shift: " << args.base_shift();
  os << ", max_shift: " << args.max_shift();
  os << ", base_image_seq_len: " << args.base_image_seq_len();
  os << ", max_image_seq_len: " << args.max_image_seq_len();
  os << "]";
  return os;
}

// Speculative decoding extends decode from one token to
// num_speculative_tokens + 1 rows per sequence; the dummy forward of an idle
// shard must mirror that width to keep DP collectives row-aligned.
inline int32_t dummy_token_count(const ModelArgs& args) {
  return args.num_speculative_tokens() + 1;
}

}  // namespace xllm
