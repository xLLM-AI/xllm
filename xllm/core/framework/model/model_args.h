/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"

namespace xllm {

struct ModelArgs {
  // Expose every plain-data field to the generic property reflection layer so
  // the embedded Python model executor can receive the full, already-parsed
  // config without a hand-maintained field whitelist (see property_reflect.h).
  REFLECT_PROPERTIES(ModelArgs);

  PROPERTY(std::string, model_type);

  PROPERTY(std::string, dtype);

  PROPERTY(int64_t, hidden_size) = 0;

  PROPERTY(std::string, hidden_act);

  // intermediate size
  PROPERTY(int64_t, intermediate_size) = 0;

  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, n_encoder_layers) = 0;

  // attn head dim
  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, decoder_head_dim) = 0;

  // attn head num
  PROPERTY(int64_t, n_heads) = 0;
  PROPERTY(int64_t, decoder_n_heads) = 0;

  PROPERTY(int64_t, actual_n_heads) = 0;

  // attn head num for key/value
  PROPERTY(std::optional<int64_t>, n_kv_heads);
  PROPERTY(std::optional<int64_t>, decoder_n_kv_heads);

  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(int64_t, draft_vocab_size) = 0;

  PROPERTY(bool, use_qk_norm) = false;
  PROPERTY(float, rms_norm_eps) = 0.0f;

  PROPERTY(float, layer_norm_eps) = 0.0f;

  PROPERTY(int64_t, rotary_dim) = 0;

  // the base period of the rotary position embeddings.
  PROPERTY(float, rope_theta) = 10000.0f;

  // rope_scaling related args
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

  // the maximum sequence length to use for rotary position embeddings.
  PROPERTY(int64_t, max_position_embeddings) = 0;
  PROPERTY(bool, use_absolute_position_embedding) = false;

  // token id for beginning of sentence.
  PROPERTY(int32_t, bos_token_id) = 0;

  // token id for end of sentence.
  PROPERTY(int32_t, eos_token_id) = -1;

  // token id vector for end of sentence.
  PROPERTY(std::vector<int32_t>, eos_token_id_vec);

  // token id for pad of sentence.
  PROPERTY(int32_t, pad_token_id) = 0;

  // scaling factor used on the attention scores
  PROPERTY(std::optional<float>, attn_scalar);

  // whether to use bias. only used for mpt models
  PROPERTY(bool, no_bias) = false;

  // whether to use bias for qkv.
  PROPERTY(bool, qkv_bias) = false;

  // Stop token ids for decoding.
  PROPERTY(std::unordered_set<int32_t>, stop_token_ids);

  // deepseek v2/v3
  PROPERTY(int32_t, first_k_dense_replace) = 0;
  PROPERTY(int32_t, moe_layer_freq) = 0;
  // deepseek v2/v3 MoE
  PROPERTY(bool, use_moe) = false;
  PROPERTY(std::string, moe_score_func);
  PROPERTY(float, moe_route_scale) = 1.0f;
  PROPERTY(bool, moe_use_shared_experts) = false;
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
  PROPERTY(float, swiglu_limit) = 0.0f;
  // deepseek v2/v3 MLA
  PROPERTY(bool, enable_mla) = false;
  PROPERTY(int32_t, qk_nope_head_dim) = 0;
  PROPERTY(int32_t, qk_rope_head_dim) = 0;
  PROPERTY(int32_t, v_head_dim) = 0;
  PROPERTY(int32_t, q_lora_rank) = 0;
  PROPERTY(int32_t, kv_lora_rank) = 0;
  // deepseek v3/v3.2 MTP
  PROPERTY(int32_t, num_nextn_predict_layers) = 0;
  PROPERTY(std::string, mtp_mlp_type) = "moe";

  // deepseek v3.2 indexer
  PROPERTY(int32_t, index_head_dim) = 0;
  PROPERTY(int32_t, index_n_heads) = 0;
  PROPERTY(int32_t, index_topk) = 0;
  PROPERTY(bool, indexer_rope_interleave) = false;
  // IndexCache: https://arxiv.org/abs/2603.12201
  PROPERTY(int32_t, index_topk_freq) = 1;
  PROPERTY(std::string, index_topk_pattern);
  PROPERTY(int32_t, index_skip_topk_offset) = 0;
  PROPERTY(bool, index_share_for_mtp_iteration) = false;

  // deepseek v4
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
  PROPERTY(int32_t, hc_mult) = 0;
  PROPERTY(int32_t, hc_sinkhorn_iters) = 0;
  PROPERTY(float, hc_eps) = 1e-6f;
  PROPERTY(int64_t, max_batch_size) = 0;
  PROPERTY(int64_t, max_seq_len) = 0;

  PROPERTY(int32_t, vision_start_token_id) = 0;
  PROPERTY(int32_t, vision_end_token_id) = 0;
  PROPERTY(int32_t, vision_token_id) = 0;
  PROPERTY(int32_t, image_token_id) = 0;
  PROPERTY(int32_t, video_token_id) = 0;

  // glm4v moe
  PROPERTY(int32_t, image_start_token_id) = 0;
  PROPERTY(int32_t, image_end_token_id) = 0;
  PROPERTY(int32_t, video_start_token_id) = 0;
  PROPERTY(int32_t, video_end_token_id) = 0;

  PROPERTY(std::string, vision_custom_adapter);
  PROPERTY(int32_t, vision_max_slice_nums) = 0;

  // qwen3 moe
  PROPERTY(bool, attention_bias) = false;
  PROPERTY(float, attention_dropout) = 0.0f;
  PROPERTY(int32_t, decoder_sparse_step) = 1;
  PROPERTY(float, initializer_range) = 0.02f;
  PROPERTY(std::vector<int32_t>, mlp_only_layers) = {};
  PROPERTY(int64_t, num_attention_heads) = 32;
  PROPERTY(int32_t, num_experts) = 128;
  PROPERTY(bool, output_router_logits) = false;
  PROPERTY(int32_t, rope_scaling) = -1;
  PROPERTY(float, router_aux_loss_coef) = 0.001f;

  // qwen3 next initialized with 0, and will be loaded in model file
  PROPERTY(bool, attn_output_gate) = false;
  PROPERTY(int32_t, full_attention_interval) = 0;
  PROPERTY(int32_t, linear_conv_kernel_dim) = 0;
  PROPERTY(int32_t, linear_key_head_dim) = 0;
  PROPERTY(int32_t, linear_value_head_dim) = 0;
  PROPERTY(int64_t, linear_num_key_heads) = 0;
  PROPERTY(int32_t, linear_num_value_heads) = 0;
  PROPERTY(std::string, mamba_ssm_dtype);
  PROPERTY(int32_t, shared_expert_intermediate_size) = 0;
  PROPERTY(float, partial_rotary_factor) = 0.0f;
  PROPERTY(std::vector<std::string>, layer_types) = {};

  // Vision model's dropout
  PROPERTY(float, mm_dropout) = 0.0f;

  // Vision model's hidden_act
  PROPERTY(std::string, mm_hidden_act);

  // Vision model's mm_hidden_size
  PROPERTY(int64_t, mm_hidden_size) = 0;

  // Vision model's mm_image_size
  PROPERTY(int64_t, mm_image_size) = 0;

  // Vision model's mm_intermediate_size
  PROPERTY(int64_t, mm_intermediate_size) = 0;

  // Vision model's mm_num_channels
  PROPERTY(int64_t, mm_num_channels) = 0;

  // Vision model's mm_initializer_range
  PROPERTY(float, mm_initializer_range) = 0.0f;

  // Vision model's mm_layer_norm_eps
  PROPERTY(float, mm_layer_norm_eps) = 0;

  // Vision model's mm_num_attention_heads
  PROPERTY(int64_t, mm_num_attention_heads) = 0;

  // Vision model's mm_num_beam_groups
  PROPERTY(int64_t, mm_num_beam_groups) = 0;

  // Vision model's mm_num_beams
  PROPERTY(int64_t, mm_num_beams) = 0;

  // Vision model's mm_num_hidden_layers
  PROPERTY(int64_t, mm_num_hidden_layers) = 0;

  // Vision model's mm_num_return_sequences
  PROPERTY(int64_t, mm_num_return_sequences) = 0;

  // Vision model's mm_output_attentions
  PROPERTY(bool, mm_output_attentions) = false;

  // Vision model's mm_output_hidden_states
  PROPERTY(bool, mm_output_hidden_states) = false;

  // Vision model's mm_output_scores
  PROPERTY(bool, mm_output_scores) = false;

  // Vision model's mm_patch_size
  PROPERTY(int64_t, mm_patch_size) = 0;

  // Vision model's mm_projection_dim
  PROPERTY(int64_t, mm_projection_dim) = 0;

  // Vision model's mm_projector_hidden_size
  PROPERTY(int64_t, mm_projector_hidden_size) = 0;

  PROPERTY(int64_t, mm_spatial_merge_size) = 0;
  PROPERTY(int64_t, mm_spatial_patch_size) = 0;

  // Vision model's mm_remove_invalid_values
  PROPERTY(bool, mm_remove_invalid_values) = false;

  // Vision model's mm_repetition_penalty
  PROPERTY(float, mm_repetition_penalty) = 0.0f;

  // Vision model's mm_return_dict
  PROPERTY(bool, mm_return_dict) = false;

  // Vision model's mm_return_dict_in_generate
  PROPERTY(bool, mm_return_dict_in_generate) = false;

  // Vision model's mm_temperature
  PROPERTY(float, mm_temperature) = 0.0f;

  // Vision model's mm_tie_encoder_decoder
  PROPERTY(bool, mm_tie_encoder_decoder) = false;

  // Vision model's mm_tie_word_embeddings
  PROPERTY(bool, mm_tie_word_embeddings) = false;

  // Vision model's mm_top_k
  PROPERTY(int64_t, mm_top_k) = 0;

  // Vision model's mm_top_p
  PROPERTY(float, mm_top_p) = 0.0f;

  // Vision model's mm_torchscript
  PROPERTY(bool, mm_torchscript) = false;

  // Vision model's mm_use_bfloat16
  PROPERTY(bool, mm_use_bfloat16) = false;

  // Vision model's mm_head_dim
  PROPERTY(int64_t, mm_head_dim) = 0;

  // Vision model's mm_vocab_size
  PROPERTY(int64_t, mm_vocab_size) = 0;

  PROPERTY(int, mm_window_size) = 0;
  PROPERTY(std::vector<int64_t>, mm_fullatt_block_indexes);
  PROPERTY(std::vector<int64_t>, mm_deepstack_visual_indexes);
  PROPERTY(int, mm_tokens_per_second) = 0;
  PROPERTY(int, mm_temporal_patch_size) = 0;

  // VLM model projector's mm_projector_type
  PROPERTY(std::string, mm_projector_type);

  //
  PROPERTY(int64_t, mm_num_position_embeddings);
  // VLM model projector's mm_projector_hidden_act
  PROPERTY(std::string, mm_projector_hidden_act);

  // VLM model projector's mm_projector_n_layers
  PROPERTY(int64_t, mm_projector_n_layers) = 0;

  // VLM model projector's mm_vision_feature_layer
  PROPERTY(int64_t, mm_vision_feature_layer) = 0;

  // VLM model projector's mm_vision_feature_select_strategy
  PROPERTY(std::string, mm_vision_feature_select_strategy);

  // mm image begin
  // VLM image preprocessor centor crop
  PROPERTY(bool, mm_image_do_center_crop) = false;
  PROPERTY(int, mm_image_crop_height_size) = 336;
  PROPERTY(int, mm_image_crop_width_size) = 336;

  // VLM image preprocessor resize
  PROPERTY(bool, mm_image_do_resize) = false;
  PROPERTY(int, mm_image_resize_shortest_edge) = 336;

  PROPERTY(int, mm_image_resample) = 0;

  // VLM image preprocessor resize
  PROPERTY(bool, mm_image_do_rescale) = false;
  PROPERTY(double, mm_image_rescale_factor) = 0;

  // VLM image preprocessor normalization
  PROPERTY(bool, mm_image_do_normalize) = false;
  PROPERTY(std::vector<double>, mm_image_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_image_normalize_std) = {};

  // KIMI_K25
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

  // GLM
  PROPERTY(bool, mm_video_do_rescale) = false;
  PROPERTY(std::vector<double>, mm_video_normalize_mean) = {};
  PROPERTY(std::vector<double>, mm_video_normalize_std) = {};

  PROPERTY(int, mm_image_min_pixels) = 0;
  PROPERTY(int, mm_image_max_pixels) = 0;

  PROPERTY(int64_t, mm_image_shortest_edge) = 0;
  PROPERTY(int64_t, mm_image_longest_edge) = 0;

  // GLM
  PROPERTY(int64_t, mm_video_shortest_edge) = 0;
  PROPERTY(int64_t, mm_video_longest_edge) = 0;

  PROPERTY(int, mm_image_patch_size) = 0;
  PROPERTY(int, mm_image_temporal_patch_size) = 0;
  PROPERTY(int, mm_image_merge_size) = 0;

  // GLM
  PROPERTY(int, mm_video_patch_size) = 0;
  PROPERTY(int, mm_video_temporal_patch_size) = 0;
  PROPERTY(int, mm_video_merge_size) = 0;

  PROPERTY(int, mm_image_feature_size) = 0;
  PROPERTY(int, mm_scale_resolution) = 0;
  PROPERTY(bool, mm_slice_mode) = false;
  PROPERTY(bool, mm_use_image_id) = false;

  // mm image end

  PROPERTY(int64_t, mm_image_token_index) = 0;
  PROPERTY(int64_t, mm_pad_token_id) = 0;

  // whether to tie weight embeddings
  PROPERTY(bool, tie_word_embeddings) = false;

  // sliding window for attention
  PROPERTY(bool, use_sliding_window) = false;
  PROPERTY(int32_t, sliding_window) = -1;
  PROPERTY(int32_t, max_window_layers) = 0;

  PROPERTY(int32_t, query_num) = 0;
  PROPERTY(bool, encoder_embedding_mode) = false;
  PROPERTY(bool, embedding_mode) = false;

  // number of speculative decoding tokens
  PROPERTY(int64_t, num_speculative_tokens) = 0;

  // Eagle3: layer indices (0-based) to capture aux hidden states, from config
  PROPERTY(std::vector<int32_t>, layers_to_capture) = {};

  // VAE related args
  PROPERTY(int64_t, in_channels) = -1;
  PROPERTY(int64_t, out_channels) = -1;
  PROPERTY(std::vector<std::string>, down_block_types) = {

  };
  PROPERTY(std::vector<std::string>, up_block_types) = {

  };
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

  // Wan_2.2_ VAE related args (base_dim, dim_mult, latents_mean, latents_std,
  // num_res_blocks, attn_scales, temporal_downsample, dropout are reused from
  // qwen_image_edit_2509 vae args above)
  PROPERTY(int64_t, vae_scale_factor_temporal) = 0;
  PROPERTY(int64_t, vae_scale_factor_spatial) = 0;
  PROPERTY(bool, vae_is_residual) = false;

  // dit related args
  PROPERTY(int64_t, joint_attention_dim) = 0;
  PROPERTY(int64_t, pooled_projection_dim) = 0;
  PROPERTY(bool, guidance_embeds) = true;
  PROPERTY(std::vector<int64_t>, axes_dims_rope) = {};
  PROPERTY(int64_t, num_single_layers) = 0;
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

  // cola-dlm dit related args
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

  // cola-dlm vae related args
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

  // t5 related args
  PROPERTY(int64_t, d_model) = 0;
  PROPERTY(int64_t, num_layers) = 0;
  PROPERTY(int64_t, d_kv) = 0;
  PROPERTY(int64_t, d_ff) = 0;
  PROPERTY(std::string, act_fn) = "";
  PROPERTY(bool, is_gated_act) = true;
  PROPERTY(int64_t, relative_attention_num_buckets) = 0;
  PROPERTY(int64_t, relative_attention_max_distance) = 0;

  // scheduler related args
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

  // qwen_image_edit_2509 vae related args
  PROPERTY(int64_t, base_dim) = 0;
  PROPERTY(int64_t, z_dim) = 0;
  PROPERTY(std::vector<int64_t>, dim_mult) = {};
  PROPERTY(std::vector<double>, attn_scales) = {};
  PROPERTY(std::vector<bool>, temperal_downsample) = {};
  PROPERTY(int64_t, num_res_blocks) = 0;
  PROPERTY(double, dropout) = 0;
  PROPERTY(std::vector<double>, latents_mean) = {};
  PROPERTY(std::vector<double>, latents_std) = {};

  // qwen_image_edit_2511 dit related args
  PROPERTY(bool, zero_cond_t) = false;
  PROPERTY(bool, use_additional_t_cond) = false;
  PROPERTY(bool, use_layer3d_rope) = false;
};

// Qwen hybrid models may describe full-attention layers explicitly via
// layer_types or implicitly via full_attention_interval.
inline bool is_full_attention_layer(const ModelArgs& args, int64_t layer_id) {
  const auto& hybrid_layer_types = args.layer_types();
  if (layer_id >= 0 &&
      layer_id < static_cast<int64_t>(hybrid_layer_types.size())) {
    const auto& layer_type = hybrid_layer_types[layer_id];
    return layer_type == "full_attention" || layer_type == "attention";
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
                                layer_type != "attention";
                       });
  }
  return args.full_attention_interval() > 1;
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

}  // namespace xllm
