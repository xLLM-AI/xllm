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

#include "layers/dcu/deepseek_v2_attention.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/dcu/flash_mla_adapter.h"
#include "layers/common/rotary_embedding.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

namespace {

torch::Tensor to_deepseek_rope_layout(const torch::Tensor& tensor) {
  std::vector<int64_t> view_shape = tensor.sizes().vec();
  const int64_t last_dim = view_shape.back();
  CHECK_EQ(last_dim % 2, 0)
      << "DeepSeek RoPE dimension must be even, tensor: " << tensor.sizes();
  view_shape.back() = last_dim / 2;
  view_shape.emplace_back(2);
  return tensor.view(view_shape)
      .transpose(-1, -2)
      .reshape_as(tensor)
      .contiguous();
}

}  // namespace

DeepseekV2AttentionImpl::DeepseekV2AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : q_lora_rank_(args.q_lora_rank()),
      kv_lora_rank_(args.kv_lora_rank()),
      qk_nope_head_dim_(args.qk_nope_head_dim()),
      qk_rope_head_dim_(args.qk_rope_head_dim()),
      v_head_dim_(args.v_head_dim()),
      eps_(args.rms_norm_eps()),
      interleaved_(false) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t num_heads = args.n_heads();
  const int64_t hidden_size = args.hidden_size();
  const int64_t max_position_embeddings = args.max_position_embeddings();
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  tp_heads_ = num_heads / tp_size;
  qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;

  ProcessGroup* weight_group = parallel_args.tp_group_;
  const LinearExtraArgs attention_linear_extra_args("none", false);

  if (q_lora_rank_ > 0) {
    q_a_proj_ = register_module(
        "q_a_proj",
        ReplicatedLinear(
            hidden_size, q_lora_rank_, /*bias=*/false, QuantArgs(), options));
    q_a_layernorm_ =
        register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));
    q_b_proj_ =
        register_module("q_b_proj",
                        ColumnParallelLinear(q_lora_rank_,
                                             num_heads * qk_head_dim_,
                                             /*bias=*/false,
                                             /*gather_output=*/false,
                                             quant_args,
                                             weight_group,
                                             options,
                                             attention_linear_extra_args));
  } else {
    q_proj_ =
        register_module("q_proj",
                        ColumnParallelLinear(hidden_size,
                                             num_heads * qk_head_dim_,
                                             /*bias=*/false,
                                             /*gather_output=*/false,
                                             quant_args,
                                             weight_group,
                                             options,
                                             attention_linear_extra_args));
  }

  kv_a_proj_with_mqa_ =
      register_module("kv_a_proj_with_mqa",
                      ReplicatedLinear(hidden_size,
                                       kv_lora_rank_ + qk_rope_head_dim_,
                                       /*bias=*/false,
                                       QuantArgs(),
                                       options));
  kv_a_layernorm_ =
      register_module("kv_a_layernorm", RMSNorm(kv_lora_rank_, eps_, options));
  kv_b_proj_ = register_module(
      "kv_b_proj",
      ColumnParallelLinear(kv_lora_rank_,
                           num_heads * (qk_nope_head_dim_ + v_head_dim_),
                           /*bias=*/false,
                           /*gather_output=*/false,
                           QuantArgs(),
                           weight_group,
                           options,
                           attention_linear_extra_args));

  torch::Tensor weights = kv_b_proj_->weight().unflatten(
      0, {tp_heads_, qk_nope_head_dim_ + v_head_dim_});
  w_kc_ = weights.slice(1, 0, qk_nope_head_dim_);
  w_vc_ = weights.slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

  rotary_emb_ =
      register_module("rotary_emb",
                      create_mla_rotary_embedding(args,
                                                  qk_rope_head_dim_,
                                                  max_position_embeddings,
                                                  interleaved_,
                                                  options));

  o_proj_ = register_module("o_proj",
                            RowParallelLinear(num_heads * v_head_dim_,
                                              hidden_size,
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/false,
                                              quant_args,
                                              weight_group,
                                              options,
                                              attention_linear_extra_args));

  softmax_scale_ = static_cast<float>(std::pow(
      static_cast<double>(qk_nope_head_dim_ + qk_rope_head_dim_), -0.5));
  if (args.rope_scaling_rope_type() == "deepseek_yarn") {
    const float mscale = layer::rotary::yarn_get_mscale(
        args.rope_scaling_factor(), args.rope_scaling_mscale_all_dim());
    softmax_scale_ *= mscale * mscale;
  }
}

torch::Tensor DeepseekV2AttentionImpl::prepare_query(
    const torch::Tensor& hidden_states) {
  torch::Tensor q;
  if (q_lora_rank_ > 0) {
    q = q_a_proj_(hidden_states);
    torch::Tensor q_a = std::get<0>(q_a_layernorm_(q));
    q = q_b_proj_->forward(q_a);
  } else {
    q = q_proj_->forward(hidden_states);
  }
  return q.view({-1, tp_heads_, qk_head_dim_});
}

void DeepseekV2AttentionImpl::store_latent_cache(
    const torch::Tensor& latent_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& k_cache) {
  const int64_t dim = k_cache.size(-1);
  torch::Tensor k_cache_rows = k_cache.view({-1, dim});
  k_cache_rows.index_copy_(
      /*dim=*/0, slot_mapping.to(torch::kInt64), latent_cache);
}

torch::Tensor DeepseekV2AttentionImpl::project_output(
    const torch::Tensor& attn_latent) {
  torch::Tensor attn_bmm =
      torch::bmm(attn_latent.transpose(0, 1), w_vc_);  // [tp_heads, tokens, v]
  attn_bmm = attn_bmm.transpose(0, 1);                 // [tokens, tp_heads, v]
  torch::Tensor proj_input = attn_bmm.flatten(1, 2);   // [tokens, tp_heads*v]
  return o_proj_->forward(proj_input);
}

torch::Tensor DeepseekV2AttentionImpl::decode_flash_mla(
    const torch::Tensor& q_nope_absorbed,
    const torch::Tensor& q_pe,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  const int64_t batch = q_nope_absorbed.size(0);
  kernel::dcu::flash_mla::DenseDecodeParams params;
  params.q_nope = q_nope_absorbed.view({batch, 1, tp_heads_, kv_lora_rank_});
  params.q_pe = q_pe.view({batch, 1, tp_heads_, qk_rope_head_dim_});
  params.k_cache = kv_cache.get_k_cache();
  params.seqlens_k = attn_metadata.kv_seq_lens;
  params.block_table = attn_metadata.block_table;
  params.head_size_v = kv_lora_rank_;
  params.softmax_scale = softmax_scale_;
  params.is_causal = attn_metadata.is_causal;
  params.kind = kernel::dcu::flash_mla::DenseDecodeKind::kQNopePe;

  torch::Tensor attn_latent =
      kernel::dcu::flash_mla::dense_decode(params);  // [B, 1, H, kv_lora]
  attn_latent = attn_latent.view({batch, tp_heads_, kv_lora_rank_});
  return project_output(attn_latent);
}

torch::Tensor DeepseekV2AttentionImpl::prefill_sdpa(
    const torch::Tensor& q_nope_absorbed,
    const torch::Tensor& q_pe,
    const torch::Tensor& latent_cache,
    const AttentionMetadata& attn_metadata) {
  const int64_t head_dim = kv_lora_rank_ + qk_rope_head_dim_;
  torch::Tensor q_input = torch::cat({q_nope_absorbed, q_pe}, /*dim=*/-1);
  const int64_t num_tokens = q_input.size(0);

  torch::Tensor q_cu = attn_metadata.q_cu_seq_lens.cpu().to(torch::kInt64);
  torch::Tensor kv_cu = attn_metadata.kv_cu_seq_lens.cpu().to(torch::kInt64);
  const int64_t batch_size = q_cu.size(0) - 1;

  torch::Tensor out_latent =
      torch::empty({num_tokens, tp_heads_, kv_lora_rank_}, q_input.options());

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t q_start = q_cu[i].item<int64_t>();
    const int64_t q_end = q_cu[i + 1].item<int64_t>();
    const int64_t kv_start = kv_cu[i].item<int64_t>();
    const int64_t kv_end = kv_cu[i + 1].item<int64_t>();
    const int64_t sq = q_end - q_start;
    const int64_t sk = kv_end - kv_start;
    if (sk == 0) {
      continue;
    }

    torch::Tensor q_seq = q_input.slice(0, q_start, q_end);          // [sq,H,D]
    torch::Tensor kv_seq = latent_cache.slice(0, kv_start, kv_end);  // [sk,D]

    torch::Tensor q4 = q_seq.permute({1, 0, 2}).unsqueeze(0).contiguous();
    torch::Tensor kv4 = kv_seq.view({1, 1, sk, head_dim});

    torch::Tensor attn = torch::scaled_dot_product_attention(
        q4,
        kv4,
        kv4,
        std::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/attn_metadata.is_causal,
        /*scale=*/static_cast<double>(softmax_scale_),
        /*enable_gqa=*/true);                // [1, H, sq, D]
    attn = attn.slice(-1, 0, kv_lora_rank_)  // keep o_latent cols
               .squeeze(0)
               .permute({1, 0, 2})
               .contiguous();  // [sq, H, kv_lora]
    out_latent.slice(0, q_start, q_end).copy_(attn);
  }

  return project_output(out_latent);
}

torch::Tensor DeepseekV2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  const bool is_prefill = attn_metadata.is_prefill;
  CHECK(!attn_metadata.is_chunked_prefill)
      << "DCU DeepSeek-V2 chunked prefill is not supported yet.";
  CHECK_EQ(positions.numel(), hidden_states.size(0))
      << "DCU DeepSeek-V2 position/token mismatch, positions: "
      << positions.sizes() << ", hidden_states: " << hidden_states.sizes()
      << ", q_cu_seq_lens: " << attn_metadata.q_cu_seq_lens.sizes()
      << ", kv_cu_seq_lens: " << attn_metadata.kv_cu_seq_lens.sizes()
      << ", is_prefill: " << attn_metadata.is_prefill
      << ", is_chunked_prefill: " << attn_metadata.is_chunked_prefill;

  torch::Tensor latent_cache =
      kv_a_proj_with_mqa_(hidden_states);  // [tokens, kv_lora+rope]
  torch::Tensor c_kv = latent_cache.slice(-1, 0, kv_lora_rank_);
  torch::Tensor c_kv_normed = std::get<0>(kv_a_layernorm_(c_kv));
  torch::Tensor k_pe = latent_cache.slice(-1, kv_lora_rank_).contiguous();
  torch::Tensor k_pe_3d = to_deepseek_rope_layout(k_pe.unsqueeze(1));
  rotary_emb_->forward(k_pe_3d,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       /*is_prompt=*/is_prefill);
  k_pe = k_pe_3d.squeeze(1).contiguous();
  torch::Tensor latent_normed = torch::cat({c_kv_normed, k_pe}, /*dim=*/-1);

  const torch::Tensor k_cache = kv_cache.get_k_cache();
  if (k_cache.defined() && attn_metadata.slot_mapping.defined()) {
    store_latent_cache(latent_normed, attn_metadata.slot_mapping, k_cache);
  }

  torch::Tensor q = prepare_query(hidden_states);  // [tokens, H, qk_head_dim]
  std::vector<torch::Tensor> q_vec =
      q.split({qk_nope_head_dim_, qk_rope_head_dim_}, /*dim=*/-1);
  torch::Tensor q_nope = q_vec[0].contiguous();  // [tokens, H, qk_nope]
  torch::Tensor q_pe = q_vec[1].contiguous();    // [tokens, H, qk_rope]
  q_pe = to_deepseek_rope_layout(q_pe);
  rotary_emb_->forward(q_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       /*is_prompt=*/is_prefill);
  torch::Tensor q_nope_absorbed =
      torch::bmm(q_nope.transpose(0, 1), w_kc_).transpose(0, 1);

  if (is_prefill) {
    return prefill_sdpa(q_nope_absorbed, q_pe, latent_normed, attn_metadata);
  }
  return decode_flash_mla(q_nope_absorbed, q_pe, attn_metadata, kv_cache);
}

void DeepseekV2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  if (q_proj_) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
  } else {
    q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_a_proj."));
    q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_b_proj."));
    q_a_layernorm_->load_state_dict(
        state_dict.get_dict_with_prefix("q_a_layernorm."));
  }
  kv_a_proj_with_mqa_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_proj_with_mqa."));
  kv_a_layernorm_->load_state_dict(
      state_dict.get_dict_with_prefix("kv_a_layernorm."));
  kv_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("kv_b_proj."));
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));

  if (kv_b_proj_->is_weight_loaded() && !has_trans_) {
    w_vc_ = w_vc_.transpose(1, 2)
                .contiguous();  // [H, v_head, kv_lora] -> [H, kv_lora, v_head]
    has_trans_ = true;
  }
}

}  // namespace layer
}  // namespace xllm
