/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "deepseek_v2_attention.h"

#include <tuple>

#include "core/framework/config/parallel_config.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

DeepseekV2AttentionImpl::DeepseekV2AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const OptimizationConfig& optimization_config)
    : q_lora_rank_(args.q_lora_rank()),
      kv_lora_rank_(args.kv_lora_rank()),
      qk_nope_head_dim_(args.qk_nope_head_dim()),
      qk_rope_head_dim_(args.qk_rope_head_dim()),
      enable_lighting_indexer_(args.index_n_heads() > 0),
      index_topk_(args.index_topk()),
      v_head_dim_(args.v_head_dim()),
      eps_(args.rms_norm_eps()),
      interleaved_(true) {
  use_full_replicated_attention_weights_ =
      ::xllm::ParallelConfig::get_instance().enable_prefill_sp();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();
  int64_t max_position_embeddings = args.max_position_embeddings();

  qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  tp_heads_ = {num_heads / tp_size, num_heads / tp_size};
  full_heads_ = {num_heads, num_heads};
  float scaling = std::pow(qk_head_dim_, -0.5f);

  ProcessGroup* weight_group =
      use_replicated_attn_weights() &&
              parallel_args.single_rank_group_ != nullptr
          ? parallel_args.single_rank_group_
          : parallel_args.tp_group_;
  const LinearExtraArgs attention_linear_extra_args("none", false);

  if (q_lora_rank_ > 0) {
    q_a_proj_ = register_module(
        "q_a_proj",
        ReplicatedLinear(
            hidden_size, q_lora_rank_, false, QuantArgs(), options));
    q_a_layernorm_ =
        register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));
    q_b_proj_ = register_module(
        "q_b_proj",
        ColumnParallelLinear(q_lora_rank_,
                             full_heads().proj_width(qk_head_dim_),
                             false,
                             false,
                             quant_args,
                             weight_group,
                             options,
                             attention_linear_extra_args));
  } else {
    q_proj_ = register_module(
        "q_proj",
        ColumnParallelLinear(hidden_size,
                             full_heads().proj_width(qk_head_dim_),
                             false,
                             false,
                             quant_args,
                             weight_group,
                             options,
                             attention_linear_extra_args));
  }

  kv_a_proj_with_mqa_ =
      register_module("kv_a_proj_with_mqa",
                      ReplicatedLinear(hidden_size,
                                       kv_lora_rank_ + qk_rope_head_dim_,
                                       false,
                                       QuantArgs(),
                                       options));
  kv_a_layernorm_ =
      register_module("kv_a_layernorm", RMSNorm(kv_lora_rank_, eps_, options));
  kv_b_proj_ = register_module(
      "kv_b_proj",
      ColumnParallelLinear(
          kv_lora_rank_,
          full_heads().proj_width(qk_nope_head_dim_ + v_head_dim_),
          false,
          false,
          QuantArgs(),
          weight_group,
          options,
          attention_linear_extra_args));

  auto kv_b_proj_weight = kv_b_proj_->weight();
  auto weights =
      kv_b_proj_weight.unflatten(0, {-1, qk_nope_head_dim_ + v_head_dim_});
  w_kc_ = weights.slice(1, 0, qk_nope_head_dim_);
  w_vc_ = weights.slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

  rotary_emb_ =
      register_module("rotary_emb",
                      create_mla_rotary_embedding(args,
                                                  qk_rope_head_dim_,
                                                  max_position_embeddings,
                                                  interleaved_,
                                                  options));

  // indexer rotary embedding for lighting indexer
  indexer_rotary_emb_ = register_module(
      "indexer_rotary_emb",
      create_mla_rotary_embedding(args,
                                  qk_rope_head_dim_,
                                  max_position_embeddings,
                                  args.indexer_rope_interleave(),
                                  options));

  if (args.rope_scaling_rope_type() == "deepseek_yarn") {
    float mscale = layer::rotary::yarn_get_mscale(
        args.rope_scaling_factor(), args.rope_scaling_mscale_all_dim());
    scaling *= mscale * mscale;
  }

  if (enable_lighting_indexer_) {
    indexer_ =
        register_module("indexer",
                        Indexer(hidden_size,
                                args.index_n_heads(),
                                args.index_head_dim(),
                                qk_rope_head_dim_,
                                args.index_topk(),
                                q_lora_rank_,
                                optimization_config.enable_fused_indexer_qk,
                                indexer_rotary_emb_,
                                quant_args,
                                parallel_args,
                                options));
  }

  use_fused_mla_qkv_ = optimization_config.enable_fused_mla_kernel;

  attn_ = register_module("attn",
                          Attention(active_heads().attn,
                                    kv_lora_rank_ + qk_rope_head_dim_,
                                    /*num_local_heads=*/1,
                                    kv_lora_rank_,
                                    args.sliding_window(),
                                    scaling,
                                    use_fused_mla_qkv_,
                                    enable_lighting_indexer_,
                                    args.enable_mla()));

  o_proj_ =
      register_module("o_proj",
                      RowParallelLinear(full_heads().proj_width(v_head_dim_),
                                        hidden_size,
                                        false,
                                        /*input_is_parallelized=*/true,
                                        /*reduce=*/false,
                                        quant_args,
                                        weight_group,
                                        options,
                                        attention_linear_extra_args));
}

DeepseekV2AttentionImpl::QueryPrep DeepseekV2AttentionImpl::prep_query(
    const torch::Tensor& hidden_states,
    const HeadInfo& heads) {
  QueryPrep out;
  if (q_lora_rank_ > 0) {
    out.q = q_a_proj_(hidden_states);
    auto q_a = std::get<0>(q_a_layernorm_(out.q));
    out.q_norm = q_a;
    out.q = q_b_proj_->forward(q_a).view({-1, heads.proj, qk_head_dim_});
  } else {
    out.q =
        q_proj_->forward(hidden_states).view({-1, heads.proj, qk_head_dim_});
  }
  return out;
}

void DeepseekV2AttentionImpl::fill_q_input(
    torch::Tensor& q_input,
    const torch::Tensor& q,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    bool use_prompt_rope) {
  const int32_t dim = -1;
  auto q_vec = q.split({qk_nope_head_dim_, qk_rope_head_dim_}, dim);
  auto q_nope = q_vec[0];
  auto q_pe = q_vec[1];
  // bmm(q_nope, w_kc_)
  auto q_nope_transposed = q_nope.transpose(0, 1);
  auto q_input_slice = q_input.slice(dim, 0, kv_lora_rank_).transpose(0, 1);
  torch::Tensor w_kc_for_runtime = w_kc_;
  torch::bmm_out(q_input_slice, q_nope_transposed, w_kc_for_runtime);
  rotary_emb_->forward(q_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       use_prompt_rope);
  q_input.slice(dim, kv_lora_rank_) = q_pe;
}

void DeepseekV2AttentionImpl::decode_kv_pre_base(
    torch::Tensor& latent_cache,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    bool use_prompt_rope) {
  auto v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  // pass the output address so that the output can be written to the address
  // directly
  v_input = std::get<0>(kv_a_layernorm_(v_input,
                                        /*residual=*/std::nullopt,
                                        v_input));
  auto k_pe = latent_cache.slice(-1, kv_lora_rank_).unsqueeze(1);
  rotary_emb_->forward(k_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       use_prompt_rope);
}

void DeepseekV2AttentionImpl::decode_qkv_pre_fused(
    torch::Tensor& q,
    torch::Tensor& q_norm,
    torch::Tensor& q_input,
    torch::Tensor& latent_cache,
    torch::Tensor& kv_cache,
    std::optional<torch::Tensor> k_cache_scale,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    bool use_prompt_rope) {
  // forward_decoder_fused_mla_q
  // fused_mla_q: q_a_layernorm + q_b_proj + split + bmm + rotary_emb
  if (q_lora_rank_ > 0) {
    q_norm = torch::empty_like(q);
    if (q.dim() == 2) {
      q = q.unsqueeze(1);
    }
    q_input = q_input.view(
        {q.size(0), q.size(1), q_input.size(-2), q_input.size(-1)});
    kernel::FusedMlaQParams fused_mla_q_params;
    fused_mla_q_params.q = q;
    fused_mla_q_params.output = q_input;
    fused_mla_q_params.output_norm = q_norm.view(q.sizes());
    fused_mla_q_params.gamma = q_a_layernorm_->weight();
    fused_mla_q_params.smooth_quant_scale = q_b_proj_->smooth();
    fused_mla_q_params.weight_b = q_b_proj_->weight();
    fused_mla_q_params.weight_b_scale = q_b_proj_->per_channel_scale();
    fused_mla_q_params.weight_c = weight_c_;
    fused_mla_q_params.sin = rotary_emb_->get_sin_cache();
    fused_mla_q_params.cos = rotary_emb_->get_cos_cache();
    fused_mla_q_params.position_id = positions;
    fused_mla_q_params.quant_mode = "none";
    fused_mla_q_params.eps = eps_;
    fused_mla_q_params.interleaved = interleaved_;
    kernel::fused_mla_q(fused_mla_q_params);
  } else {
    fill_q_input(q_input, q, positions, attn_metadata, use_prompt_rope);
  }

  // forward_decoder_fused_mla_kv
  // fused_mla_kv: kv_a_layernorm + rotary_emb + reshape_paged_cache
  if (latent_cache.dim() == 2) {
    latent_cache = latent_cache.unsqueeze(1);
  }
  int32_t batch = latent_cache.size(0);
  int32_t seq = latent_cache.size(1);
  int32_t head_num = 1;
  latent_cache =
      latent_cache.view({batch, seq, head_num, latent_cache.size(-1)});
  kernel::FusedMlaKVParams fused_mla_kv_params;
  fused_mla_kv_params.input_kv = latent_cache;
  fused_mla_kv_params.sin = rotary_emb_->get_sin_cache();
  fused_mla_kv_params.cos = rotary_emb_->get_cos_cache();
  fused_mla_kv_params.position_id = positions;
  fused_mla_kv_params.gamma = kv_a_layernorm_->weight();
  fused_mla_kv_params.kv_cache = kv_cache;
  fused_mla_kv_params.kv_cache_scale = k_cache_scale;
  fused_mla_kv_params.slot_mapping =
      attn_metadata.slot_mapping.view({batch, seq});
  fused_mla_kv_params.cache_bs_id = std::nullopt;
  fused_mla_kv_params.cache_seq_offset = std::nullopt;
  fused_mla_kv_params.quant_mode =
      k_cache_scale.has_value() ? "dynamic_per_token" : "none";
  fused_mla_kv_params.eps = eps_;
  fused_mla_kv_params.interleaved = interleaved_;
  kernel::fused_mla_kv(fused_mla_kv_params);
}

void DeepseekV2AttentionImpl::prepare_mla_inputs(
    torch::Tensor& q,
    torch::Tensor& q_norm,
    torch::Tensor& q_input,
    torch::Tensor& latent_cache,
    const torch::Tensor& hidden_states,
    torch::Tensor& k_cache,
    std::optional<torch::Tensor> k_cache_scale,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    bool enable_fused_qkv,
    bool use_prompt_rope) {
  const auto& heads = active_heads();
  latent_cache = kv_a_proj_with_mqa_(hidden_states);
  if (enable_fused_qkv) {
    if (q_lora_rank_ > 0) {
      q = q_a_proj_(hidden_states);
    } else {
      q = q_proj_->forward(hidden_states).view({-1, heads.proj, qk_head_dim_});
    }
    decode_qkv_pre_fused(q,
                         q_norm,
                         q_input,
                         latent_cache,
                         k_cache,
                         k_cache_scale,
                         positions,
                         attn_metadata,
                         use_prompt_rope);
  } else {
    auto query_prep = prep_query(hidden_states, heads);
    q = query_prep.q;
    q_norm = query_prep.q_norm;
    fill_q_input(q_input, q, positions, attn_metadata, use_prompt_rope);
    decode_kv_pre_base(latent_cache, positions, attn_metadata, use_prompt_rope);
  }
}

AttentionMetadata DeepseekV2AttentionImpl::build_mla_attention_metadata(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& k_input,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    std::optional<torch::Tensor> k_cache_scale,
    bool is_prefill_phase,
    const std::optional<torch::Tensor>& slot_mapping,
    const std::optional<torch::Tensor>& new_block_tables,
    const std::optional<torch::Tensor>& new_context_lens) {
  // reshape_paged_cache before attn
  // since the reshape_paged_cache and indexer_ does not involve any
  // communication, we will skip them if it is dummy run in data parallel
  AttentionMetadata attn_indexer_metadata = attn_metadata;
  if (!attn_metadata.is_dummy) {
    // mla prefill save cache before flashattn
    if (is_prefill_phase) {
      auto key = k_input.unsqueeze(1);
      xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
      reshape_paged_cache_params.key = key;
      reshape_paged_cache_params.k_cache = kv_cache.get_k_cache();
      reshape_paged_cache_params.slot_mapping =
          slot_mapping.value_or(attn_metadata.slot_mapping);
      if (k_cache_scale.has_value()) {
        // Use quant_to_paged_cache for INT8 quantization
        reshape_paged_cache_params.k_cache_scale = k_cache_scale;
        xllm::kernel::quant_to_paged_cache(reshape_paged_cache_params);
      } else {
        // Use standard reshape_paged_cache
        xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
      }
    }
    // indexer and update index params for attn
    attn_indexer_metadata = attn_metadata;
    attn_indexer_metadata.compute_dtype = "half";
    if (new_block_tables.has_value() && new_context_lens.has_value()) {
      attn_indexer_metadata.block_table = new_block_tables.value();
      attn_indexer_metadata.kv_seq_lens = new_context_lens.value();
      attn_indexer_metadata.max_seq_len = index_topk_;
    } else if (enable_lighting_indexer_) {
      auto index_cache = kv_cache.get_index_cache();
      auto [new_block_tables, new_context_lens] = indexer_(hidden_states,
                                                           q_norm,
                                                           positions,
                                                           index_cache,
                                                           attn_metadata,
                                                           is_prefill_phase,
                                                           std::nullopt);
      attn_indexer_metadata.block_table = new_block_tables;
      attn_indexer_metadata.kv_seq_lens = new_context_lens;
      attn_indexer_metadata.max_seq_len = index_topk_;
    }
  }
  return attn_indexer_metadata;
}

torch::Tensor DeepseekV2AttentionImpl::project_output(
    const torch::Tensor& attn_output,
    const HeadInfo& heads) {
  // bmm(attn_out, w_vc_)
  auto attn_output_view = attn_output.view({-1, heads.attn, kv_lora_rank_});
  auto attn_bmm_output = torch::empty(
      {attn_output.size(0), heads.proj, v_head_dim_}, attn_output.options());
  auto attn_bmm_trans_out = attn_bmm_output.transpose(0, 1);
  torch::Tensor w_vc_for_runtime = w_vc_;
  torch::bmm_out(
      attn_bmm_trans_out, attn_output_view.transpose(0, 1), w_vc_for_runtime);
  auto proj_input = attn_bmm_output.flatten(1, 2);
  return o_proj_->forward(proj_input);
}

torch::Tensor DeepseekV2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const v32_sp::DeepseekV32SPContext* sp_ctx) {
  bool is_prefill_or_chunked_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (sp_ctx != nullptr && can_use_sp()) {
    return forward_sp(positions,
                      hidden_states,
                      attn_metadata,
                      *sp_ctx,
                      kv_cache,
                      is_prefill_or_chunked_prefill);
  }
  return forward_normal_tp(positions,
                           hidden_states,
                           attn_metadata,
                           kv_cache,
                           is_prefill_or_chunked_prefill);
}

torch::Tensor DeepseekV2AttentionImpl::forward_normal_tp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill) {
  const auto& heads = active_heads();
  torch::Tensor q, q_norm;
  torch::Tensor q_input = torch::empty(
      {hidden_states.size(0), heads.attn, kv_lora_rank_ + qk_rope_head_dim_},
      hidden_states.options());
  auto latent_cache = torch::Tensor();
  auto k_cache = kv_cache.get_k_cache();
  auto k_cache_scale = kv_cache.get_k_cache_scale();
  const bool enable_fused_qkv =
      use_fused_mla_qkv_ && !is_prefill_or_chunked_prefill;
  const bool use_prompt_rope = attn_metadata.is_prefill;

  prepare_mla_inputs(q,
                     q_norm,
                     q_input,
                     latent_cache,
                     hidden_states,
                     k_cache,
                     k_cache_scale,
                     positions,
                     attn_metadata,
                     enable_fused_qkv,
                     use_prompt_rope);

  // reshape q,k,v
  auto v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  auto k_input = latent_cache;
  q_input = q_input.view({q_input.size(0), -1});
  k_input = k_input.view({k_input.size(0), -1});
  v_input = v_input.view({v_input.size(0), -1});

  AttentionMetadata attn_indexer_metadata =
      build_mla_attention_metadata(positions,
                                   hidden_states,
                                   q_norm,
                                   k_input,
                                   attn_metadata,
                                   kv_cache,
                                   k_cache_scale,
                                   is_prefill_or_chunked_prefill);

  // mla forward
  auto [attn_output, output_lse] =
      attn_(attn_indexer_metadata, q_input, k_input, v_input, kv_cache);

  return project_output(attn_output, heads);
}

void DeepseekV2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  // load q proj weights
  if (q_proj_) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
  } else {
    q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_a_proj."));
    q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_b_proj."));
    q_a_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("q_a_layernorm."));
  }

  // load kv proj weights
  kv_a_layernorm_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_layernorm."));
  kv_a_proj_with_mqa_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_proj_with_mqa."));
  kv_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("kv_b_proj."));

  // load indexer weights
  if (enable_lighting_indexer_) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }

  // load o proj weights
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));

  // transpose before forward
  if (kv_b_proj_->is_weight_loaded() && !has_trans_) {
    if (use_fused_mla_qkv_) {
      weight_c_ = w_kc_.transpose(1, 2).contiguous();
    }
    w_vc_ = w_vc_.transpose(1, 2);
    has_trans_ = true;
  }
}

}  // namespace layer
}  // namespace xllm
