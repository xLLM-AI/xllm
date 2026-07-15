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
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/dcu/dcu_ops_api.h"
#include "kernels/dcu/flash_mla_adapter.h"
#include "layers/common/rotary_embedding.h"
#include "layers/common/rotary_embedding_util.h"
#include "layers/dcu/flash_attention.h"

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

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

bool has_prefill_context(const AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_chunked_prefill) {
    return true;
  }
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "DeepSeek MHA prefill requires q_cu_seq_lens.";
  CHECK(attn_metadata.kv_cu_seq_lens.defined())
      << "DeepSeek MHA prefill requires kv_cu_seq_lens.";
  return !torch::equal(attn_metadata.q_cu_seq_lens,
                       attn_metadata.kv_cu_seq_lens);
}

void check_first_phase_prefill_metadata(
    const AttentionMetadata& attn_metadata) {
  CHECK(attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)
      << "DeepSeek MHA prefill must run in prefill/chunked prefill phase.";
  CHECK(!attn_metadata.is_chunked_prefill)
      << "DCU DeepSeek MHA prefill phase 1 does not support chunked "
         "prefill/context yet.";
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "DeepSeek MHA prefill requires q_cu_seq_lens.";
  CHECK(attn_metadata.kv_cu_seq_lens.defined())
      << "DeepSeek MHA prefill requires kv_cu_seq_lens.";
  CHECK(torch::equal(attn_metadata.q_cu_seq_lens, attn_metadata.kv_cu_seq_lens))
      << "DCU DeepSeek MHA prefill phase 1 only supports no prefix/cache "
         "context; q_cu_seq_lens must equal kv_cu_seq_lens.";
}

void check_context_prefill_metadata(const AttentionMetadata& attn_metadata) {
  CHECK(attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)
      << "DeepSeek context prefill must run in prefill/chunked prefill phase.";
  CHECK(attn_metadata.q_cu_seq_lens.defined())
      << "DeepSeek context prefill requires q_cu_seq_lens.";
  CHECK(attn_metadata.kv_cu_seq_lens.defined())
      << "DeepSeek context prefill requires kv_cu_seq_lens.";
  CHECK(attn_metadata.kv_seq_lens.defined() ||
        attn_metadata.kv_cu_seq_lens.defined())
      << "DeepSeek context prefill requires KV sequence lengths.";
  CHECK_GT(attn_metadata.total_kv_len, 0)
      << "DeepSeek context prefill requires total KV length.";
  CHECK(attn_metadata.block_table.defined())
      << "DeepSeek context prefill requires block_table.";
  CHECK(attn_metadata.slot_mapping.defined())
      << "DeepSeek context prefill requires slot_mapping.";
  CHECK_GE(attn_metadata.max_seq_len, attn_metadata.max_query_len)
      << "context prefill expects max_seq_len >= max_query_len.";
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
      interleaved_(false),
      runtime_dtype_(c10::typeMetaToScalarType(options.dtype())) {
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
            hidden_size, q_lora_rank_, /*bias=*/false, quant_args, options));
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
                                       quant_args,
                                       options));
  kv_a_layernorm_ =
      register_module("kv_a_layernorm", RMSNorm(kv_lora_rank_, eps_, options));
  kv_b_proj_ = register_module(
      "kv_b_proj",
      ColumnParallelLinear(kv_lora_rank_,
                           num_heads * (qk_nope_head_dim_ + v_head_dim_),
                           /*bias=*/false,
                           /*gather_output=*/false,
                           quant_args,
                           weight_group,
                           options,
                           attention_linear_extra_args));

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
  CHECK(w_vc_.defined()) << "DeepSeek MLA absorbed value weight is not loaded.";
  torch::Tensor attn_bmm =
      torch::bmm(attn_latent.transpose(0, 1), w_vc_);  // [tp_heads, tokens, v]
  attn_bmm = attn_bmm.transpose(0, 1);                 // [tokens, tp_heads, v]
  torch::Tensor proj_input =
      attn_bmm.flatten(1, 2).contiguous();  // [tokens, tp_heads*v]
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

torch::Tensor DeepseekV2AttentionImpl::prefill_mha(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& c_kv_normed,
    const torch::Tensor& k_pe,
    const AttentionMetadata& attn_metadata) {
  check_first_phase_prefill_metadata(attn_metadata);

  torch::Tensor kv =
      kv_b_proj_->forward(c_kv_normed)
          .view({-1, tp_heads_, qk_nope_head_dim_ + v_head_dim_});
  std::vector<torch::Tensor> kv_vec =
      kv.split({qk_nope_head_dim_, v_head_dim_}, /*dim=*/-1);
  torch::Tensor k_nope = kv_vec[0].contiguous();
  torch::Tensor value = kv_vec[1].contiguous();

  torch::Tensor query = torch::cat({q_nope, q_pe}, /*dim=*/-1).contiguous();
  torch::Tensor key =
      torch::cat({k_nope, k_pe.unsqueeze(1).expand({-1, tp_heads_, -1})},
                 /*dim=*/-1)
          .contiguous();

  torch::Tensor cu_seqlens =
      attn_metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();
  std::vector<torch::Tensor> result = flash_attention_varlen_forward(
      query,
      key,
      value,
      /*num_heads=*/tp_heads_,
      /*num_kv_heads=*/tp_heads_,
      cu_seqlens,
      cu_seqlens,
      /*max_seqlen_q=*/attn_metadata.max_query_len,
      /*max_seqlen_k=*/attn_metadata.max_query_len,
      /*softmax_scale=*/softmax_scale_,
      /*is_causal=*/attn_metadata.is_causal,
      /*window_size_left=*/-1,
      /*window_size_right=*/attn_metadata.is_causal ? 0 : -1,
      /*is_bf16_output=*/query.scalar_type() == torch::kBFloat16);

  CHECK(!result.empty()) << "DeepSeek MHA prefill returned no output.";
  torch::Tensor proj_input = result[0].flatten(1, 2).contiguous();
  return o_proj_->forward(proj_input);
}

torch::Tensor DeepseekV2AttentionImpl::prefill_mha_with_full_context(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& k_cache) {
  check_context_prefill_metadata(attn_metadata);

  const int64_t batch_size = attn_metadata.block_table.size(0);
  CHECK_GT(batch_size, 0) << "DeepSeek context prefill batch size must be > 0.";
  const int64_t required_blocks =
      (attn_metadata.max_seq_len + k_cache.size(1) - 1) / k_cache.size(1);
  CHECK_LE(required_blocks, attn_metadata.block_table.size(1))
      << "DeepSeek context prefill block_table does not contain enough blocks.";
  torch::Tensor latent_cache =
      kernel::dcu::gather_mla_latent_cache(k_cache,
                                           attn_metadata.block_table,
                                           attn_metadata.kv_cu_seq_lens,
                                           attn_metadata.total_kv_len,
                                           attn_metadata.max_seq_len);
  torch::Tensor c_kv_normed =
      latent_cache.slice(/*dim=*/-1, /*start=*/0, kv_lora_rank_).contiguous();
  torch::Tensor k_pe =
      latent_cache.slice(/*dim=*/-1, /*start=*/kv_lora_rank_).contiguous();

  torch::Tensor kv =
      kv_b_proj_->forward(c_kv_normed)
          .view({-1, tp_heads_, qk_nope_head_dim_ + v_head_dim_});
  std::vector<torch::Tensor> kv_vec =
      kv.split({qk_nope_head_dim_, v_head_dim_}, /*dim=*/-1);
  torch::Tensor k_nope = kv_vec[0].contiguous();
  torch::Tensor value = kv_vec[1].contiguous();

  torch::Tensor query = torch::cat({q_nope, q_pe}, /*dim=*/-1).contiguous();
  torch::Tensor key =
      torch::cat({k_nope, k_pe.unsqueeze(1).expand({-1, tp_heads_, -1})},
                 /*dim=*/-1)
          .contiguous();

  torch::Tensor q_cu_seqlens =
      attn_metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();
  torch::Tensor kv_cu_seqlens =
      attn_metadata.kv_cu_seq_lens.to(torch::kInt32).contiguous();
  std::vector<torch::Tensor> result = flash_attention_varlen_forward(
      query,
      key,
      value,
      /*num_heads=*/tp_heads_,
      /*num_kv_heads=*/tp_heads_,
      q_cu_seqlens,
      kv_cu_seqlens,
      /*max_seqlen_q=*/attn_metadata.max_query_len,
      /*max_seqlen_k=*/attn_metadata.max_seq_len,
      /*softmax_scale=*/softmax_scale_,
      /*is_causal=*/attn_metadata.is_causal,
      /*window_size_left=*/-1,
      /*window_size_right=*/attn_metadata.is_causal ? 0 : -1,
      /*is_bf16_output=*/query.scalar_type() == torch::kBFloat16);

  CHECK(!result.empty())
      << "DeepSeek full-context MHA prefill returned no output.";
  torch::Tensor proj_input = result[0].flatten(1, 2).contiguous();
  return o_proj_->forward(proj_input);
}

torch::Tensor DeepseekV2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  const bool use_prefill_attention =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
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
                       /*is_prompt=*/use_prefill_attention);
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
                       /*is_prompt=*/use_prefill_attention);

  if (use_prefill_attention) {
    if (has_prefill_context(attn_metadata)) {
      return prefill_mha_with_full_context(
          q_nope, q_pe, attn_metadata, k_cache);
    }
    return prefill_mha(q_nope, q_pe, c_kv_normed, k_pe, attn_metadata);
  }

  CHECK(w_kc_.defined()) << "DeepSeek MLA absorbed key weight is not loaded.";
  torch::Tensor q_nope_absorbed =
      torch::bmm(q_nope.transpose(0, 1), w_kc_).transpose(0, 1);

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

  if (kv_b_proj_->is_weight_loaded()) {
    refresh_kv_b_proj_weights();
  }
}

void DeepseekV2AttentionImpl::refresh_kv_b_proj_weights() {
  torch::Tensor kv_b_weight = kv_b_proj_->weight();
  CHECK(kv_b_weight.defined())
      << "DeepSeek MLA kv_b_proj weight must be loaded.";
  if (is_fp8_dtype(kv_b_weight.scalar_type())) {
    torch::Tensor weight_scale = kv_b_proj_->weight_scale();
    CHECK(weight_scale.defined())
        << "DeepSeek MLA FP8 kv_b_proj requires weight_scale.";
    torch::Tensor scale = weight_scale;
    if (scale.dim() == 2) {
      CHECK_EQ(scale.size(1), 1)
          << "DeepSeek MLA kv_b_proj weight_scale must be [N] or [N,1].";
      scale = scale.squeeze(1);
    }
    CHECK_EQ(scale.dim(), 1)
        << "DeepSeek MLA kv_b_proj weight_scale must be [N] or [N,1].";
    CHECK_EQ(scale.size(0), kv_b_weight.size(0))
        << "DeepSeek MLA kv_b_proj weight_scale output dim mismatch.";
    kv_b_weight = (kv_b_weight.to(runtime_dtype_) *
                   scale.to(kv_b_weight.device(), runtime_dtype_).view({-1, 1}))
                      .contiguous();
  }

  torch::Tensor weights =
      kv_b_weight.unflatten(0, {tp_heads_, qk_nope_head_dim_ + v_head_dim_});
  w_kc_ = weights.slice(1, 0, qk_nope_head_dim_).contiguous();
  w_vc_ = weights.slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_)
              .transpose(1, 2)
              .contiguous();
  has_trans_ = true;
}

}  // namespace layer
}  // namespace xllm
