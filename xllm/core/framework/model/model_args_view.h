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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm {

// The views deliberately expose const references to the existing ModelArgs.
// This keeps the loader and reflection representation flat while giving model
// implementations a family-specific contract during the migration. The
// registry remains responsible for selecting the concrete view for a model;
// this low-level header does not add a second runtime family dispatcher.
#define XLLM_MODEL_ARGS_VIEW_FIELD(Type, field) \
  const Type& field() const noexcept { return args_.field(); }

#define XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(Class) \
  explicit Class(ModelArgs&&) = delete;                \
  explicit Class(const ModelArgs&&) = delete;

class TransformerModelArgsView final {
 public:
  explicit TransformerModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(TransformerModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, model_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, dtype)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, hidden_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, hidden_act)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, intermediate_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, n_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, n_encoder_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, kv_cache_storage_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, n_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_n_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, actual_n_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::optional<int64_t>, n_kv_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::optional<int64_t>, decoder_n_kv_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vocab_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_qk_norm)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rms_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, layer_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rotary_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_theta)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, rope_scaling_rope_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_low_freq_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_high_freq_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_extrapolation_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t,
                             rope_scaling_original_max_position_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_scaling_beta_fast)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_scaling_beta_slow)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_attn_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_mscale)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, rope_scaling_mscale_all_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, rope_scaling_mrope_section)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, rope_scaling_mrope_interleaved)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, max_position_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_absolute_position_embedding)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, bos_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, eos_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int32_t>, eos_token_id_vec)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, pad_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::optional<float>, attn_scalar)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, no_bias)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, qkv_bias)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::unordered_set<int32_t>, stop_token_ids)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, tie_word_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, attention_bias)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, attention_dropout)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, decoder_sparse_step)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, initializer_range)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int32_t>, mlp_only_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_attention_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, num_experts)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, output_router_logits)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, rope_scaling)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, router_aux_loss_coef)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, attn_output_gate)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, full_attention_interval)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int32_t>, full_attn_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, linear_conv_kernel_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, linear_key_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, linear_value_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, linear_num_key_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, linear_num_value_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, linear_lower_bound)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mamba_ssm_dtype)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, shared_expert_intermediate_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, partial_rotary_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<std::string>, layer_types)

 private:
  const ModelArgs& args_;
};

class LlmModelArgsView final {
 public:
  explicit LlmModelArgsView(const ModelArgs& args)
      : args_(args), transformer_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(LlmModelArgsView)

  const TransformerModelArgsView& transformer() const noexcept {
    return transformer_;
  }

  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, first_k_dense_replace)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, moe_layer_freq)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, draft_vocab_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, markov_rank)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dspark_num_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dspark_block_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, dspark_use_native_sas)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, enable_confidence_head)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, confidence_head_with_markov)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dflash2_block_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dflash2_conv_group_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dflash2_conv_kernel_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dflash2_selector_rank)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, dflash2_selector_top_k)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_moe)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, moe_score_func)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, moe_route_scale)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, moe_use_shared_experts)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, topk_method)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, n_routed_experts)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, n_shared_experts)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, num_experts_per_tok)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, moe_intermediate_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, routed_scaling_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, norm_topk_prob)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, n_group)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, topk_group)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, scoring_func)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, swiglu_limit)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, enable_mla)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, qk_nope_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, qk_rope_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, v_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, q_lora_rank)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, kv_lora_rank)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, num_nextn_predict_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mtp_start_layer_idx)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mtp_mlp_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_sliding_window)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, sliding_window)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, max_window_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_speculative_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, enable_embedded_eagle3_draft)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, max_concurrent_prefills_per_dp)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int32_t>, layers_to_capture)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_n_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_topk)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_kpool)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, index_kpool_compress)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, index_kpool_always_select_tail)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, indexer_rope_interleave)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_topk_freq)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, index_topk_pattern)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, index_skip_topk_offset)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, index_share_for_mtp_iteration)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<std::string>, indexer_types)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<std::string>, mlp_layer_types)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, rope_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, o_lora_rank)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, o_groups)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int32_t>, compress_ratios)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, compress_rope_theta)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, window_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, n_activated_experts)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, n_hash_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, beta_fast)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, beta_slow)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, scale_fmt)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, hc_mult)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, hc_sinkhorn_iters)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, hc_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, max_batch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, max_seq_len)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, norm_before_fc)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, norm_before_residual)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, norm_output)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, fc_norm)

 private:
  const ModelArgs& args_;
  TransformerModelArgsView transformer_;
};

class VlmVisionModelArgsView final {
 public:
  explicit VlmVisionModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(VlmVisionModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, vision_start_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, vision_end_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, vision_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, image_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, video_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, audio_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, image_start_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, image_end_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, video_start_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, video_end_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, vision_custom_adapter)
  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, vision_max_slice_nums)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_dropout)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mm_hidden_act)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_hidden_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_image_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_intermediate_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_initializer_range)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_layer_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_attention_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_beam_groups)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_beams)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_hidden_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_return_sequences)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_output_attentions)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_output_hidden_states)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_output_scores)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_projection_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_projector_hidden_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_spatial_merge_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_spatial_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_remove_invalid_values)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_repetition_penalty)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_return_dict)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_return_dict_in_generate)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_temperature)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_tie_encoder_decoder)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_tie_word_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_top_k)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_top_p)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_torchscript)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_use_bfloat16)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_head_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_vocab_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_window_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, mm_fullatt_block_indexes)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, mm_deepstack_visual_indexes)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_tokens_per_second)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_temporal_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mm_projector_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_num_position_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mm_projector_hidden_act)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_projector_n_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_vision_feature_layer)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mm_vision_feature_select_strategy)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_image_do_center_crop)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_crop_height_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_crop_width_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_image_do_resize)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_resize_shortest_edge)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_resample)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_image_do_rescale)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_video_do_rescale)
  XLLM_MODEL_ARGS_VIEW_FIELD(double, mm_image_rescale_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_image_do_normalize)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_image_normalize_mean)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_image_normalize_std)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_init_pos_emb_width)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_init_pos_emb_height)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_init_pos_emb_time)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_in_patch_limit)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, mm_km_image_mean)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, mm_km_image_std)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_merge_kernel_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_fixed_output_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_patch_limit_on_one_side)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_in_patch_limit_each_frame)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_in_patch_limit_video)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mm_km_sample_fps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_max_num_frames_each_video)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_km_temporal_merge_kernel_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, mm_km_timestamp_mode)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_video_normalize_mean)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_video_normalize_std)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_min_pixels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_max_pixels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_image_shortest_edge)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_image_longest_edge)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_video_shortest_edge)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_video_longest_edge)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_temporal_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_merge_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_min_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_max_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_video_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_video_temporal_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_video_merge_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_video_min_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_video_max_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_image_feature_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, mm_scale_resolution)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_slice_mode)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mm_use_image_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_image_token_index)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_pad_token_id)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_sampling_rate)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_num_mel_bins)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_frame_length)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_frame_shift)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_max_frames)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_downsample_rate)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_audio_cmvn_means)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, mm_audio_cmvn_inverse_std)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_idim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_d_model)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_n_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_n_head)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_kernel_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_pe_maxlen)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, mm_audio_ctc_pad_token_id)

 private:
  const ModelArgs& args_;
};

class VlmModelArgsView final {
 public:
  explicit VlmModelArgsView(const ModelArgs& args)
      : transformer_(args), vision_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(VlmModelArgsView)

  const TransformerModelArgsView& transformer() const noexcept {
    return transformer_;
  }
  const VlmVisionModelArgsView& vision() const noexcept { return vision_; }

 private:
  TransformerModelArgsView transformer_;
  VlmVisionModelArgsView vision_;
};

class RecModelArgsView final {
 public:
  explicit RecModelArgsView(const ModelArgs& args)
      : args_(args), transformer_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(RecModelArgsView)

  const TransformerModelArgsView& transformer() const noexcept {
    return transformer_;
  }

  XLLM_MODEL_ARGS_VIEW_FIELD(int32_t, query_num)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, encoder_embedding_mode)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, embedding_mode)

 private:
  const ModelArgs& args_;
  TransformerModelArgsView transformer_;
};

class DiTTransformerModelArgsView final {
 public:
  explicit DiTTransformerModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(DiTTransformerModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, in_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, out_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, latent_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, joint_attention_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, pooled_projection_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, guidance_embeds)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, axes_dims_rope)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_single_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, mlp_ratio)
  XLLM_MODEL_ARGS_VIEW_FIELD(int, timestep_guidance_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, wan_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, cross_attn_norm)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, ffn_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, time_freq_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, dit_in_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, dit_out_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, qk_norm)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_max_seq_len)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, text_embed_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, image_embed_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, added_kv_proj_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, pos_embed_seq_len)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_refiner_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, audio_in_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, time_embed_hidden_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, time_embed_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_freq_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, qk_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, final_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(double, mlp_width_ratio)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, text_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, rope_dim_list)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_theta_dit)
  XLLM_MODEL_ARGS_VIEW_FIELD(double, eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, txt_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, txt_in_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, txt_out_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, emb_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, rope_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, expand_ratio)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, block_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, latent_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, qk_bias)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, zero_cond_t)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_additional_t_cond)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_layer3d_rope)

 private:
  const ModelArgs& args_;
};

class DiTVaeModelArgsView final {
 public:
  explicit DiTVaeModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(DiTVaeModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, in_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, out_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<std::string>, down_block_types)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<std::string>, up_block_types)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, block_out_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, layers_per_block)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, latent_channels)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, norm_num_groups)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, sample_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, scale_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, shift_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, mid_block_add_attention)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, force_upcast)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_quant_conv)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_post_quant_conv)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_scale_factor_temporal)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_scale_factor_spatial)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, vae_is_residual)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, batch_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, batch_norm_momentum)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, ae_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, base_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, z_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, dim_mult)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, latents_mean)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, latents_std)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<double>, attn_scales)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<bool>, temperal_downsample)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_res_blocks)
  XLLM_MODEL_ARGS_VIEW_FIELD(double, dropout)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, spatial_downsample_factors)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, temporal_downsample_factors)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, spatial_padding_mode)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_num_register_tokens)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_ffn_mult)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, decoder_rope_theta)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, decoder_rope_dim_ratio)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, decoder_norm_eps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, encoder_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, encoder_rates)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, decoder_rates)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, decoder_kernel_sizes)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, resblock_kernel_sizes)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, sampling_rate)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, clip_length)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, token_drop)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_dim)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_num_heads)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, encoder_num_blocks)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, decoder_num_blocks)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, shared_heads_kv)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_rope_theta)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_block_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, vae_patch_size)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, encoder_last_ln)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, shifting_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, scaling_factor)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_variation)

 private:
  const ModelArgs& args_;
};

class DiTTextEncoderModelArgsView final {
 public:
  explicit DiTTextEncoderModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(DiTTextEncoderModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, d_model)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, max_position_embeddings)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_layers)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, d_kv)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, d_ff)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, act_fn)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, is_gated_act)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, relative_attention_num_buckets)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, relative_attention_max_distance)

 private:
  const ModelArgs& args_;
};

class DiTSchedulerModelArgsView final {
 public:
  explicit DiTSchedulerModelArgsView(const ModelArgs& args) : args_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(DiTSchedulerModelArgsView)

  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, num_train_timesteps)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, shift)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_dynamic_shifting)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, base_shift)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, max_shift)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, base_image_seq_len)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, max_image_seq_len)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, shift_terminal)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, beta_start)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, beta_end)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, beta_schedule)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<float>, trained_betas)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, solver_order)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, prediction_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, thresholding)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, dynamic_thresholding_ratio)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, sample_max_value)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, predict_x0)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, solver_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, lower_order_final)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::vector<int64_t>, disable_corrector)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_karras_sigmas)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_exponential_sigmas)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_beta_sigmas)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, use_flow_sigmas)
  XLLM_MODEL_ARGS_VIEW_FIELD(float, flow_shift)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, timestep_spacing)
  XLLM_MODEL_ARGS_VIEW_FIELD(int64_t, steps_offset)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, final_sigmas_type)
  XLLM_MODEL_ARGS_VIEW_FIELD(bool, rescale_betas_zero_snr)
  XLLM_MODEL_ARGS_VIEW_FIELD(std::string, time_shift_type)

 private:
  const ModelArgs& args_;
};

class DiTModelArgsView final {
 public:
  explicit DiTModelArgsView(const ModelArgs& args)
      : transformer_(args), vae_(args), text_encoder_(args), scheduler_(args) {}
  XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES(DiTModelArgsView)

  const DiTTransformerModelArgsView& transformer() const noexcept {
    return transformer_;
  }
  const DiTVaeModelArgsView& vae() const noexcept { return vae_; }
  const DiTTextEncoderModelArgsView& text_encoder() const noexcept {
    return text_encoder_;
  }
  const DiTSchedulerModelArgsView& scheduler() const noexcept {
    return scheduler_;
  }

 private:
  DiTTransformerModelArgsView transformer_;
  DiTVaeModelArgsView vae_;
  DiTTextEncoderModelArgsView text_encoder_;
  DiTSchedulerModelArgsView scheduler_;
};

inline TransformerModelArgsView make_transformer_model_args_view(
    const ModelArgs& args) {
  return TransformerModelArgsView(args);
}
inline TransformerModelArgsView make_transformer_model_args_view(ModelArgs&&) =
    delete;
inline TransformerModelArgsView make_transformer_model_args_view(
    const ModelArgs&&) = delete;

inline LlmModelArgsView make_llm_model_args_view(const ModelArgs& args) {
  return LlmModelArgsView(args);
}
inline LlmModelArgsView make_llm_model_args_view(ModelArgs&&) = delete;
inline LlmModelArgsView make_llm_model_args_view(const ModelArgs&&) = delete;

inline VlmModelArgsView make_vlm_model_args_view(const ModelArgs& args) {
  return VlmModelArgsView(args);
}
inline VlmModelArgsView make_vlm_model_args_view(ModelArgs&&) = delete;
inline VlmModelArgsView make_vlm_model_args_view(const ModelArgs&&) = delete;

inline RecModelArgsView make_rec_model_args_view(const ModelArgs& args) {
  return RecModelArgsView(args);
}
inline RecModelArgsView make_rec_model_args_view(ModelArgs&&) = delete;
inline RecModelArgsView make_rec_model_args_view(const ModelArgs&&) = delete;

inline DiTModelArgsView make_dit_model_args_view(const ModelArgs& args) {
  return DiTModelArgsView(args);
}
inline DiTModelArgsView make_dit_model_args_view(ModelArgs&&) = delete;
inline DiTModelArgsView make_dit_model_args_view(const ModelArgs&&) = delete;

#undef XLLM_MODEL_ARGS_VIEW_FIELD
#undef XLLM_MODEL_ARGS_VIEW_REJECT_TEMPORARIES

}  // namespace xllm
