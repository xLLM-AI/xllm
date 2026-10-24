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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/model/raw_model_args.h"

namespace xllm {

// Each capability owns its fields. Family arguments compose these values and
// expose references to them; RawModelArgs remains the loading configuration.
class AttentionModelParams final {
 public:
  AttentionModelParams() = default;
  explicit AttentionModelParams(const RawModelArgs& args)
      : head_dim_(args.head_dim()),
        n_heads_(args.n_heads()),
        n_kv_heads_(args.n_kv_heads()),
        decoder_head_dim_(args.decoder_head_dim()),
        decoder_n_heads_(args.decoder_n_heads()),
        decoder_n_kv_heads_(args.decoder_n_kv_heads()),
        actual_n_heads_(args.actual_n_heads()),
        rotary_dim_(args.rotary_dim()),
        partial_rotary_factor_(args.partial_rotary_factor()),
        use_qk_norm_(args.use_qk_norm()),
        rms_norm_eps_(args.rms_norm_eps()),
        layer_norm_eps_(args.layer_norm_eps()),
        attention_bias_(args.attention_bias()),
        qkv_bias_(args.qkv_bias()),
        attn_scalar_(args.attn_scalar()),
        use_sliding_window_(args.use_sliding_window()),
        sliding_window_(args.sliding_window()),
        max_window_layers_(args.max_window_layers()),
        enable_mla_(args.enable_mla()),
        qk_nope_head_dim_(args.qk_nope_head_dim()),
        qk_rope_head_dim_(args.qk_rope_head_dim()),
        v_head_dim_(args.v_head_dim()),
        num_attention_heads_(args.num_attention_heads()),
        q_lora_rank_(args.q_lora_rank()),
        kv_lora_rank_(args.kv_lora_rank()),
        rope_theta_(args.rope_theta()),
        rope_scaling_rope_type_(args.rope_scaling_rope_type()),
        rope_scaling_factor_(args.rope_scaling_factor()),
        rope_scaling_low_freq_factor_(args.rope_scaling_low_freq_factor()),
        rope_scaling_high_freq_factor_(args.rope_scaling_high_freq_factor()),
        rope_extrapolation_factor_(args.rope_extrapolation_factor()),
        rope_scaling_original_max_position_embeddings_(
            args.rope_scaling_original_max_position_embeddings()),
        rope_scaling_beta_fast_(args.rope_scaling_beta_fast()),
        rope_scaling_beta_slow_(args.rope_scaling_beta_slow()),
        rope_scaling_attn_factor_(args.rope_scaling_attn_factor()),
        rope_scaling_mscale_(args.rope_scaling_mscale()),
        rope_scaling_mscale_all_dim_(args.rope_scaling_mscale_all_dim()),
        rope_scaling_mrope_section_(args.rope_scaling_mrope_section()),
        rope_scaling_mrope_interleaved_(args.rope_scaling_mrope_interleaved()),
        attention_dropout_(args.attention_dropout()),
        attn_output_gate_(args.attn_output_gate()),
        use_absolute_position_embedding_(
            args.use_absolute_position_embedding()),
        relative_attention_num_buckets_(args.relative_attention_num_buckets()),
        relative_attention_max_distance_(
            args.relative_attention_max_distance()) {}

  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, n_heads) = 0;
  PROPERTY(std::optional<int64_t>, n_kv_heads) = {};
  PROPERTY(int64_t, decoder_head_dim) = 0;
  PROPERTY(int64_t, decoder_n_heads) = 0;
  PROPERTY(std::optional<int64_t>, decoder_n_kv_heads) = {};
  PROPERTY(int64_t, actual_n_heads) = 0;
  PROPERTY(int64_t, rotary_dim) = 0;
  PROPERTY(float, partial_rotary_factor) = 0.0f;
  PROPERTY(bool, use_qk_norm) = false;
  PROPERTY(float, rms_norm_eps) = 0.0f;
  PROPERTY(float, layer_norm_eps) = 0.0f;
  PROPERTY(bool, attention_bias) = false;
  PROPERTY(bool, qkv_bias) = false;
  PROPERTY(std::optional<float>, attn_scalar) = {};
  PROPERTY(bool, use_sliding_window) = false;
  PROPERTY(int32_t, sliding_window) = -1;
  PROPERTY(int32_t, max_window_layers) = 0;
  PROPERTY(bool, enable_mla) = false;
  PROPERTY(int32_t, qk_nope_head_dim) = 0;
  PROPERTY(int32_t, qk_rope_head_dim) = 0;
  PROPERTY(int32_t, v_head_dim) = 0;
  PROPERTY(int64_t, num_attention_heads) = 32;
  PROPERTY(int32_t, q_lora_rank) = 0;
  PROPERTY(int32_t, kv_lora_rank) = 0;
  PROPERTY(float, rope_theta) = 10000.0f;
  PROPERTY(std::string, rope_scaling_rope_type) = {};
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
  PROPERTY(std::vector<int64_t>, rope_scaling_mrope_section) = {};
  PROPERTY(bool, rope_scaling_mrope_interleaved) = false;
  PROPERTY(float, attention_dropout) = 0.0f;
  PROPERTY(bool, attn_output_gate) = false;
  PROPERTY(bool, use_absolute_position_embedding) = false;
  PROPERTY(int64_t, relative_attention_num_buckets) = 0;
  PROPERTY(int64_t, relative_attention_max_distance) = 0;
};

class LinearAttentionModelParams final {
 public:
  LinearAttentionModelParams() = default;
  explicit LinearAttentionModelParams(const RawModelArgs& args)
      : full_attention_interval_(args.full_attention_interval()),
        full_attn_layers_(args.full_attn_layers()),
        layer_types_(args.layer_types()),
        linear_conv_kernel_dim_(args.linear_conv_kernel_dim()),
        linear_key_head_dim_(args.linear_key_head_dim()),
        linear_value_head_dim_(args.linear_value_head_dim()),
        linear_num_key_heads_(args.linear_num_key_heads()),
        linear_num_value_heads_(args.linear_num_value_heads()),
        linear_lower_bound_(args.linear_lower_bound()),
        mamba_ssm_dtype_(args.mamba_ssm_dtype()) {}

  PROPERTY(int32_t, full_attention_interval) = 0;
  PROPERTY(std::vector<int32_t>, full_attn_layers) = {};
  PROPERTY(std::vector<std::string>, layer_types) = {};
  PROPERTY(int32_t, linear_conv_kernel_dim) = 0;
  PROPERTY(int32_t, linear_key_head_dim) = 0;
  PROPERTY(int32_t, linear_value_head_dim) = 0;
  PROPERTY(int64_t, linear_num_key_heads) = 0;
  PROPERTY(int32_t, linear_num_value_heads) = 0;
  PROPERTY(float, linear_lower_bound) = -5.0f;
  PROPERTY(std::string, mamba_ssm_dtype) = {};

  bool has_linear_attention_layers() const noexcept {
    if (!layer_types_.empty()) {
      for (const auto& layer_type : layer_types_) {
        if (layer_type != "full_attention" && layer_type != "attention" &&
            layer_type != "deepseek_sparse_attention") {
          return true;
        }
      }
      return false;
    }
    return full_attention_interval_ > 1;
  }

  bool is_full_attention_layer(int64_t layer_id) const noexcept {
    if (layer_id >= 0 && layer_id < static_cast<int64_t>(layer_types_.size())) {
      const auto& layer_type = layer_types_[static_cast<size_t>(layer_id)];
      return layer_type == "full_attention" || layer_type == "attention" ||
             layer_type == "deepseek_sparse_attention";
    }
    if (full_attention_interval_ <= 1) {
      return true;
    }
    return (layer_id + 1) % full_attention_interval_ == 0;
  }
};

class KVCacheModelParams final {
 public:
  KVCacheModelParams() = default;
  explicit KVCacheModelParams(const RawModelArgs& args)
      : kv_cache_storage_head_dim_(args.kv_cache_storage_head_dim()),
        window_size_(args.window_size()),
        max_batch_size_(args.max_batch_size()),
        max_seq_len_(args.max_seq_len()),
        compress_ratios_(args.compress_ratios()),
        scale_fmt_(args.scale_fmt()),
        hc_mult_(args.hc_mult()),
        hc_sinkhorn_iters_(args.hc_sinkhorn_iters()),
        hc_eps_(args.hc_eps()),
        index_head_dim_(args.index_head_dim()),
        index_n_heads_(args.index_n_heads()),
        index_topk_(args.index_topk()),
        index_kpool_(args.index_kpool()),
        index_kpool_compress_(args.index_kpool_compress()),
        index_kpool_always_select_tail_(args.index_kpool_always_select_tail()),
        indexer_rope_interleave_(args.indexer_rope_interleave()),
        index_topk_freq_(args.index_topk_freq()),
        index_topk_pattern_(args.index_topk_pattern()),
        index_skip_topk_offset_(args.index_skip_topk_offset()),
        index_share_for_mtp_iteration_(args.index_share_for_mtp_iteration()),
        indexer_types_(args.indexer_types()),
        rope_head_dim_(args.rope_head_dim()),
        o_lora_rank_(args.o_lora_rank()),
        o_groups_(args.o_groups()),
        compress_rope_theta_(args.compress_rope_theta()),
        factor_(args.factor()),
        beta_fast_(args.beta_fast()),
        beta_slow_(args.beta_slow()) {}

  PROPERTY(int64_t, kv_cache_storage_head_dim) = 0;
  PROPERTY(int32_t, window_size) = 0;
  PROPERTY(int64_t, max_batch_size) = 0;
  PROPERTY(int64_t, max_seq_len) = 0;
  PROPERTY(std::vector<int32_t>, compress_ratios) = {};
  PROPERTY(std::string, scale_fmt) = {};
  PROPERTY(int32_t, hc_mult) = 0;
  PROPERTY(int32_t, hc_sinkhorn_iters) = 0;
  PROPERTY(float, hc_eps) = 1e-6f;
  PROPERTY(int32_t, index_head_dim) = 0;
  PROPERTY(int32_t, index_n_heads) = 0;
  PROPERTY(int32_t, index_topk) = 0;
  PROPERTY(int32_t, index_kpool) = 1;
  PROPERTY(bool, index_kpool_compress) = false;
  PROPERTY(bool, index_kpool_always_select_tail) = false;
  PROPERTY(bool, indexer_rope_interleave) = false;
  PROPERTY(int32_t, index_topk_freq) = 1;
  PROPERTY(std::string, index_topk_pattern) = {};
  PROPERTY(int32_t, index_skip_topk_offset) = 0;
  PROPERTY(bool, index_share_for_mtp_iteration) = false;
  PROPERTY(std::vector<std::string>, indexer_types) = {};
  PROPERTY(int32_t, rope_head_dim) = 0;
  PROPERTY(int32_t, o_lora_rank) = 0;
  PROPERTY(int32_t, o_groups) = 0;
  PROPERTY(float, compress_rope_theta) = 0.0f;
  PROPERTY(float, factor) = 0.0f;
  PROPERTY(float, beta_fast) = 0.0f;
  PROPERTY(float, beta_slow) = 0.0f;
};

class MoEModelParams final {
 public:
  MoEModelParams() = default;
  explicit MoEModelParams(const RawModelArgs& args)
      : use_moe_(args.use_moe()),
        first_k_dense_replace_(args.first_k_dense_replace()),
        moe_layer_freq_(args.moe_layer_freq()),
        moe_score_func_(args.moe_score_func()),
        moe_route_scale_(args.moe_route_scale()),
        moe_use_shared_experts_(args.moe_use_shared_experts()),
        topk_method_(args.topk_method()),
        n_routed_experts_(args.n_routed_experts()),
        n_shared_experts_(args.n_shared_experts()),
        num_experts_per_tok_(args.num_experts_per_tok()),
        moe_intermediate_size_(args.moe_intermediate_size()),
        routed_scaling_factor_(args.routed_scaling_factor()),
        norm_topk_prob_(args.norm_topk_prob()),
        n_group_(args.n_group()),
        topk_group_(args.topk_group()),
        scoring_func_(args.scoring_func()),
        swiglu_limit_(args.swiglu_limit()),
        num_experts_(args.num_experts()),
        n_activated_experts_(args.n_activated_experts()),
        shared_expert_intermediate_size_(
            args.shared_expert_intermediate_size()),
        output_router_logits_(args.output_router_logits()),
        router_aux_loss_coef_(args.router_aux_loss_coef()),
        mlp_only_layers_(args.mlp_only_layers()),
        n_hash_layers_(args.n_hash_layers()),
        mlp_layer_types_(args.mlp_layer_types()),
        decoder_sparse_step_(args.decoder_sparse_step()) {}

  PROPERTY(bool, use_moe) = false;
  PROPERTY(int32_t, first_k_dense_replace) = 0;
  PROPERTY(int32_t, moe_layer_freq) = 0;
  PROPERTY(std::string, moe_score_func) = {};
  PROPERTY(float, moe_route_scale) = 1.0f;
  PROPERTY(bool, moe_use_shared_experts) = false;
  PROPERTY(std::string, topk_method) = {};
  PROPERTY(int32_t, n_routed_experts) = 0;
  PROPERTY(int32_t, n_shared_experts) = 0;
  PROPERTY(int32_t, num_experts_per_tok) = 0;
  PROPERTY(int32_t, moe_intermediate_size) = 0;
  PROPERTY(float, routed_scaling_factor) = 0.0f;
  PROPERTY(bool, norm_topk_prob) = false;
  PROPERTY(int32_t, n_group) = 0;
  PROPERTY(int32_t, topk_group) = 0;
  PROPERTY(std::string, scoring_func) = {};
  PROPERTY(float, swiglu_limit) = 0.0f;
  PROPERTY(int32_t, num_experts) = 128;
  PROPERTY(int32_t, n_activated_experts) = 0;
  PROPERTY(int32_t, shared_expert_intermediate_size) = 0;
  PROPERTY(bool, output_router_logits) = false;
  PROPERTY(float, router_aux_loss_coef) = 0.001f;
  PROPERTY(std::vector<int32_t>, mlp_only_layers) = {};
  PROPERTY(int32_t, n_hash_layers) = 0;
  PROPERTY(std::vector<std::string>, mlp_layer_types) = {};
  PROPERTY(int32_t, decoder_sparse_step) = 1;
};

class SpeculativeModelParams final {
 public:
  SpeculativeModelParams() = default;
  explicit SpeculativeModelParams(const RawModelArgs& args)
      : draft_vocab_size_(args.draft_vocab_size()),
        num_speculative_tokens_(args.num_speculative_tokens()),
        enable_embedded_eagle3_draft_(args.enable_embedded_eagle3_draft()),
        layers_to_capture_(args.layers_to_capture()),
        num_nextn_predict_layers_(args.num_nextn_predict_layers()),
        mtp_start_layer_idx_(args.mtp_start_layer_idx()),
        mtp_mlp_type_(args.mtp_mlp_type()),
        markov_rank_(args.markov_rank()),
        dspark_num_layers_(args.dspark_num_layers()),
        dspark_block_size_(args.dspark_block_size()),
        dspark_use_native_sas_(args.dspark_use_native_sas()),
        enable_confidence_head_(args.enable_confidence_head()),
        confidence_head_with_markov_(args.confidence_head_with_markov()),
        norm_before_fc_(args.norm_before_fc()),
        norm_before_residual_(args.norm_before_residual()),
        norm_output_(args.norm_output()),
        fc_norm_(args.fc_norm()),
        dflash2_block_size_(args.dflash2_block_size()),
        dflash2_conv_group_size_(args.dflash2_conv_group_size()),
        dflash2_conv_kernel_size_(args.dflash2_conv_kernel_size()),
        dflash2_selector_rank_(args.dflash2_selector_rank()),
        dflash2_selector_top_k_(args.dflash2_selector_top_k()) {}

  PROPERTY(int64_t, draft_vocab_size) = 0;
  PROPERTY(int64_t, num_speculative_tokens) = 0;
  PROPERTY(bool, enable_embedded_eagle3_draft) = false;
  PROPERTY(std::vector<int32_t>, layers_to_capture) = {};
  PROPERTY(int32_t, num_nextn_predict_layers) = 0;
  PROPERTY(int64_t, mtp_start_layer_idx) = -1;
  PROPERTY(std::string, mtp_mlp_type) = "moe";
  PROPERTY(int64_t, markov_rank) = 0;
  PROPERTY(int32_t, dspark_num_layers) = 0;
  PROPERTY(int32_t, dspark_block_size) = 0;
  PROPERTY(bool, dspark_use_native_sas) = false;
  PROPERTY(bool, enable_confidence_head) = false;
  PROPERTY(bool, confidence_head_with_markov) = false;
  PROPERTY(bool, norm_before_fc) = false;
  PROPERTY(bool, norm_before_residual) = false;
  PROPERTY(bool, norm_output) = false;
  PROPERTY(bool, fc_norm) = false;
  PROPERTY(int32_t, dflash2_block_size) = 0;
  PROPERTY(int32_t, dflash2_conv_group_size) = 0;
  PROPERTY(int32_t, dflash2_conv_kernel_size) = 0;
  PROPERTY(int32_t, dflash2_selector_rank) = 0;
  PROPERTY(int32_t, dflash2_selector_top_k) = 0;
};

class EmbeddingModelParams final {
 public:
  EmbeddingModelParams() = default;
  explicit EmbeddingModelParams(const RawModelArgs& args)
      : query_num_(args.query_num()),
        encoder_embedding_mode_(args.encoder_embedding_mode()),
        embedding_mode_(args.embedding_mode()) {}

  PROPERTY(int32_t, query_num) = 0;
  PROPERTY(bool, encoder_embedding_mode) = false;
  PROPERTY(bool, embedding_mode) = false;
};

class ExecutionModelParams final {
 public:
  ExecutionModelParams() = default;
  explicit ExecutionModelParams(const RawModelArgs& args)
      : model_type_(args.model_type()),
        dtype_(args.dtype()),
        hidden_size_(args.hidden_size()),
        hidden_act_(args.hidden_act()),
        intermediate_size_(args.intermediate_size()),
        n_layers_(args.n_layers()),
        n_encoder_layers_(args.n_encoder_layers()),
        max_position_embeddings_(args.max_position_embeddings()),
        vocab_size_(args.vocab_size()),
        bos_token_id_(args.bos_token_id()),
        eos_token_id_(args.eos_token_id()),
        eos_token_id_vec_(args.eos_token_id_vec()),
        pad_token_id_(args.pad_token_id()),
        stop_token_ids_(args.stop_token_ids()),
        tie_word_embeddings_(args.tie_word_embeddings()),
        no_bias_(args.no_bias()),
        initializer_range_(args.initializer_range()),
        rope_scaling_(args.rope_scaling()),
        max_concurrent_prefills_per_dp_(args.max_concurrent_prefills_per_dp()) {
  }

  PROPERTY(std::string, model_type) = {};
  PROPERTY(std::string, dtype) = {};
  PROPERTY(int64_t, hidden_size) = 0;
  PROPERTY(std::string, hidden_act) = {};
  PROPERTY(int64_t, intermediate_size) = 0;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, n_encoder_layers) = 0;
  PROPERTY(int64_t, max_position_embeddings) = 0;
  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(int32_t, bos_token_id) = 0;
  PROPERTY(int32_t, eos_token_id) = -1;
  PROPERTY(std::vector<int32_t>, eos_token_id_vec) = {};
  PROPERTY(int32_t, pad_token_id) = 0;
  PROPERTY(std::unordered_set<int32_t>, stop_token_ids) = {};
  PROPERTY(bool, tie_word_embeddings) = false;
  PROPERTY(bool, no_bias) = false;
  PROPERTY(float, initializer_range) = 0.02f;
  PROPERTY(int32_t, rope_scaling) = -1;
  PROPERTY(int32_t, max_concurrent_prefills_per_dp) = 0;
};

}  // namespace xllm
