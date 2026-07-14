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

#include "layers/mlu/deepseek_v4/compressor.h"

#include <glog/logging.h>

#include <cmath>
#include <optional>

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
#include "util/linalg.h"

namespace {

void write_cache(const torch::Tensor& key,
                 torch::Tensor& cache,
                 const torch::Tensor& slot_mapping) {
  xllm::kernel::ReshapePagedCacheParams params;
  params.key = key.contiguous();
  params.value = std::nullopt;
  params.k_cache = cache;
  params.v_cache = std::nullopt;
  params.slot_mapping = slot_mapping.contiguous();
  params.direction = false;
  xllm::kernel::reshape_paged_cache(params);
}

void apply_rotary(torch::Tensor& kv,
                  const torch::Tensor& sin_table,
                  const torch::Tensor& cos_table,
                  const torch::Tensor& positions,
                  int64_t rope_head_dim) {
  if (rope_head_dim == 0 || kv.numel() == 0) {
    return;
  }

  torch::Tensor position_ids =
      positions.slice(/*dim=*/0, /*start=*/0, /*end=*/kv.size(0));
  torch::Tensor rope_part =
      kv.slice(/*dim=*/-1, /*start=*/kv.size(-1) - rope_head_dim).unsqueeze(1);
  xllm::kernel::RotaryParams params;
  params.q = rope_part;
  params.k = torch::Tensor();
  params.sin = sin_table;
  params.cos = cos_table;
  params.position_ids = position_ids;
  params.cu_query_lens = std::nullopt;
  params.interleaved = true;
  params.discrete = true;
  params.dynamic_ntk = false;
  params.max_query_len = kv.size(0);
  xllm::kernel::apply_rotary(params);
  rope_part.copy_(params.q);
}

const torch::Tensor& compressed_positions(const xllm::layer::DSAMetadata& dsa,
                                          int64_t compress_ratio) {
  if (compress_ratio == 4) {
    return dsa.c4_pad_positions;
  }
  return dsa.c128_pad_positions;
}

torch::Tensor empty_output(const torch::Tensor& hidden_states,
                           int64_t head_dim) {
  return torch::empty({0, head_dim}, hidden_states.options());
}

}  // namespace

namespace xllm {
namespace layer {

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t hidden_dim,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               bool rotate,
                               double norm_eps,
                               const torch::TensorOptions& options,
                               const ModelArgs& args,
                               const QuantArgs& quant_args)
    : compress_ratio_(compress_ratio),
      hidden_dim_(hidden_dim),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      rotate_(rotate),
      eps_(norm_eps),
      overlap_(compress_ratio == 4 ? true : false),
      coff_(compress_ratio == 4 ? 2 : 1) {
  compress_len_ = compress_ratio_ * coff_;
  wkv_ = register_module("wkv",
                         ReplicatedLinear(hidden_dim_,
                                          coff_ * head_dim_,
                                          /*bias=*/false,
                                          quant_args,
                                          options));
  wgate_ = register_module("wgate",
                           ReplicatedLinear(hidden_dim_,
                                            coff_ * head_dim_,
                                            /*bias=*/false,
                                            quant_args,
                                            options));
  norm_ = register_module(
      "norm", RMSNorm(head_dim_, eps_, options.dtype(torch::kFloat32)));
  ape_ = register_parameter("ape",
                            torch::empty({compress_ratio_, coff_ * head_dim_},
                                         options.dtype(torch::kFloat32)),
                            /*requires_grad=*/false);

  if (rotate_) {
    const double log_dim = std::ceil(std::log2(static_cast<double>(head_dim_)));
    const int64_t padded_dim =
        static_cast<int64_t>(1ull << static_cast<uint64_t>(log_dim));
    hadamard_matrix_ = util::create_hadamard_matrix(
        padded_dim, torch::kFloat32, torch::Device(torch::kCPU), true);
    hadamard_matrix_ =
        hadamard_matrix_.to(options.device(), options.dtype().toScalarType());
  }
}

torch::Tensor CompressorImpl::forward_prefill(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    torch::Tensor& state_cache,
    const torch::Tensor& state_block_table,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table,
    std::optional<torch::Tensor> projected_kv,
    std::optional<torch::Tensor> projected_score) {
  // When the caller (DeepseekV4Attention fused GEMM) has already projected
  // hidden_states, reuse the precomputed tensors instead of redoing the GEMM.
  torch::Tensor kv_pack = projected_kv.has_value()
                              ? projected_kv.value()
                              : wkv_->forward(hidden_states);
  torch::Tensor score_pack = projected_score.has_value()
                                 ? projected_score.value()
                                 : wgate_->forward(hidden_states);
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;

  const int64_t num_compressed_rows = slot_mapping.numel();
  torch::Tensor compressed_kv =
      torch::empty({num_compressed_rows, head_dim_}, hidden_states.options());

  xllm::kernel::mlu::fused_compress_multi_kv(
      /*kv=*/kv_pack,
      /*score=*/score_pack,
      /*state_cache=*/state_cache,
      /*state_block_table=*/state_block_table.contiguous(),
      /*cu_seqlens=*/attn_metadata.q_cu_seq_lens,
      /*positions=*/dsa.input_positions.to(torch::kInt32).contiguous(),
      /*ape=*/ape_,
      /*max_seqlen=*/attn_metadata.max_query_len,
      /*overlap=*/overlap_,
      /*compressed_kv=*/compressed_kv);

  if (num_compressed_rows == 0) {
    return empty_output(hidden_states, head_dim_);
  }

  auto kv = compressed_kv.to(torch::kFloat32);
  auto output = std::get<0>(norm_->forward(kv));
  output = output.to(hidden_states.scalar_type());
  apply_rotary(output,
               compressed_sin_table,
               compressed_cos_table,
               compressed_positions(dsa, compress_ratio_),
               rope_head_dim_);
  if (rotate_) {
    output = util::rotate_activation(output, hadamard_matrix_);
  }

  write_cache(output.unsqueeze(1), kv_cache, slot_mapping);
  return output;
}

torch::Tensor CompressorImpl::forward_decode(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    torch::Tensor& state_cache,
    const torch::Tensor& state_block_table,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table,
    std::optional<torch::Tensor> projected_kv,
    std::optional<torch::Tensor> projected_score) {
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  const int64_t state_block_size = state_cache.size(1);

  // Reuse the fused projection output when provided; otherwise compute it.
  torch::Tensor kv_pack = projected_kv.has_value()
                              ? projected_kv.value()
                              : wkv_->forward(hidden_states);
  torch::Tensor score_pack = projected_score.has_value()
                                 ? projected_score.value()
                                 : wgate_->forward(hidden_states);
  // The rewritten operator takes a 2D kv/score ([total_tokens, coff_dim]).
  if (kv_pack.dim() == 3) {
    kv_pack = kv_pack.reshape({-1, kv_pack.size(-1)});
  }
  if (score_pack.dim() == 3) {
    score_pack = score_pack.reshape({-1, score_pack.size(-1)});
  }

  torch::Tensor positions = dsa.input_positions.to(torch::kInt32).contiguous();
  torch::Tensor kv_cache_view = kv_cache.squeeze(/*dim=*/1);
  const int64_t coff_dim = coff_ * head_dim_;
  std::optional<torch::Tensor> hadamard_matrix = std::nullopt;
  if (rotate_) {
    hadamard_matrix = hadamard_matrix_;
  }

  // The MLU backend expands spec tokens into the batch dimension during MTP
  // inference, so K selection (K > 0) is not needed here.
  xllm::kernel::mlu::fused_compress_single_kv(
      /*kv=*/kv_pack,
      /*score=*/score_pack,
      /*position=*/positions,
      /*ape=*/ape_,
      /*gamma=*/norm_->weight(),
      /*sin=*/compressed_sin_table,
      /*cos=*/compressed_cos_table,
      /*hadamard_matrix=*/hadamard_matrix,
      /*slot_mapping=*/slot_mapping,
      /*kv_cache=*/kv_cache_view,
      /*kv_cache_scale=*/std::nullopt,
      /*eps=*/eps_,
      /*overlap=*/overlap_,
      /*state_cache=*/state_cache,
      /*state_bt=*/state_block_table.contiguous(),
      /*state_width=*/coff_dim,
      /*state_block_size=*/state_block_size,
      /*cu_query_len=*/dsa.q_cu_seq_lens,
      /*K=*/0);

  return empty_output(hidden_states, head_dim_);
}

torch::Tensor CompressorImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    torch::Tensor& state_cache,
    const torch::Tensor& state_block_table,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table,
    std::optional<torch::Tensor> projected_kv,
    std::optional<torch::Tensor> projected_score) {
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (is_prefill) {
    return forward_prefill(attn_metadata,
                           hidden_states,
                           kv_cache,
                           slot_mapping,
                           state_cache,
                           state_block_table,
                           compressed_sin_table,
                           compressed_cos_table,
                           projected_kv,
                           projected_score);
  }
  return forward_decode(attn_metadata,
                        hidden_states,
                        kv_cache,
                        slot_mapping,
                        state_cache,
                        state_block_table,
                        compressed_sin_table,
                        compressed_cos_table,
                        projected_kv,
                        projected_score);
}

void CompressorImpl::load_state_dict(const StateDict& state_dict,
                                     bool skip_proj_weights) {
  if (state_dict.size() == 0) {
    return;
  }
  // When the projection weights have been merged into the attention layer's
  // fused GEMM, skip loading them here; norm/ape are always loaded.
  if (!skip_proj_weights) {
    wkv_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
    wgate_->load_state_dict(state_dict.get_dict_with_prefix("wgate."));
  }
  norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  LOAD_WEIGHT(ape);
}

}  // namespace layer
}  // namespace xllm
