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

#include "layers/mlu/deepseek_v4/deepseek_v4_indexer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "common/global_flags.h"
#include "kernels/ops_api.h"
#include "util/linalg.h"

namespace {

constexpr int64_t kCompressRatio = 4;

void apply_last_dim_rope(torch::Tensor& tensor,
                         const torch::Tensor& sin_table,
                         const torch::Tensor& cos_table,
                         const torch::Tensor& input_positions,
                         int64_t rope_dim) {
  if (rope_dim == 0 || tensor.numel() == 0) {
    return;
  }
  torch::Tensor rope_part =
      tensor.slice(/*dim=*/-1, /*start=*/tensor.size(-1) - rope_dim);
  xllm::kernel::RotaryParams params;
  params.q = rope_part;
  params.k = torch::Tensor();
  params.sin = sin_table;
  params.cos = cos_table;
  params.position_ids = input_positions;
  params.cu_query_lens = std::nullopt;
  params.interleaved = true;
  params.discrete = true;
  params.dynamic_ntk = false;
  params.max_query_len = tensor.size(0);
  xllm::kernel::apply_rotary(params);
}

torch::Tensor gather_c4_cache(torch::Tensor& index_cache,
                              const torch::Tensor& block_table,
                              const torch::Tensor& c4_seq_lens,
                              int64_t total_c4_len,
                              int64_t max_c4_len) {
  torch::Tensor dense =
      torch::empty({total_c4_len, index_cache.size(-1)},
                   index_cache.options().dtype(index_cache.scalar_type()));
  if (total_c4_len == 0) {
    return dense;
  }

  xllm::kernel::ReshapeFromCacheParams params;
  params.key = dense.unsqueeze(1);
  params.value = std::nullopt;
  params.key_cache = index_cache;
  params.value_cache = std::nullopt;
  params.context_lengths = c4_seq_lens;
  params.max_context_len = max_c4_len;
  params.block_tables = block_table;
  params.context_seq_offset = std::nullopt;
  params.cache_seq_offset = std::nullopt;
  xllm::kernel::reshape_from_cache(params);
  return dense;
}

}  // namespace

namespace xllm {
namespace layer {

DeepseekV4IndexerImpl::DeepseekV4IndexerImpl(
    int64_t dim,
    int64_t index_n_heads,
    int64_t index_head_dim,
    int64_t rope_head_dim,
    int64_t index_topk,
    int64_t q_lora_rank,
    double norm_eps,
    const torch::TensorOptions& options,
    const ModelArgs& args,
    const QuantArgs& quant_args)
    : dim_(dim),
      n_heads_(index_n_heads),
      head_dim_(index_head_dim),
      rope_head_dim_(rope_head_dim),
      index_topk_(index_topk),
      q_lora_rank_(q_lora_rank),
      softmax_scale_(static_cast<float>(
          std::pow(static_cast<double>(index_head_dim), -0.5) *
          std::pow(static_cast<double>(index_n_heads), -0.5))) {
  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(q_lora_rank_,
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  weights_proj_ = register_module("weights_proj",
                                  ReplicatedLinear(dim_,
                                                   n_heads_,
                                                   /*bias=*/false,
                                                   quant_args,
                                                   options));
  compressor_ = register_module("compressor",
                                Compressor(kCompressRatio,
                                           dim_,
                                           head_dim_,
                                           rope_head_dim_,
                                           /*rotate=*/true,
                                           norm_eps,
                                           options,
                                           args,
                                           quant_args));

  const double log_dim = std::ceil(std::log2(static_cast<double>(head_dim_)));
  const int64_t padded_dim =
      static_cast<int64_t>(1ull << static_cast<uint64_t>(log_dim));
  hadamard_matrix_ = util::create_hadamard_matrix(padded_dim,
                                                  torch::kFloat32,
                                                  torch::Device(torch::kCPU),
                                                  /*normalize=*/true);
  hadamard_matrix_ =
      hadamard_matrix_.to(options.device(), options.dtype().toScalarType());
}

torch::Tensor DeepseekV4IndexerImpl::preprocess_q(
    const torch::Tensor& qr,
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table) {
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  torch::Tensor q = wq_b_->forward(qr).view({qr.size(0), n_heads_, head_dim_});
  apply_last_dim_rope(q,
                      compressed_sin_table,
                      compressed_cos_table,
                      dsa.input_positions,
                      rope_head_dim_);
  q = util::rotate_activation(q, hadamard_matrix_);
  return q;
}

torch::Tensor DeepseekV4IndexerImpl::preprocess_weights(
    const torch::Tensor& x,
    std::optional<torch::Tensor> projected_weights) {
  if (projected_weights.has_value()) {
    return projected_weights.value();
  }
  return weights_proj_->forward(x);
}

torch::Tensor DeepseekV4IndexerImpl::compress_kv(
    torch::Tensor& hidden_states,
    torch::Tensor& compress_index_state,
    const AttentionMetadata& attn_metadata,
    torch::Tensor& index_cache,
    const DeepseekV4IndexerCacheRefs& cache_refs,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table,
    std::optional<torch::Tensor> projected_kv,
    std::optional<torch::Tensor> projected_score) {
  auto output = compressor_->forward(attn_metadata,
                                     hidden_states,
                                     index_cache,
                                     cache_refs.index_slot_mapping,
                                     compress_index_state,
                                     cache_refs.index_state_block_table,
                                     compressed_sin_table,
                                     compressed_cos_table,
                                     projected_kv,
                                     projected_score);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4IndexerImpl::select_topk(
    const torch::Tensor& q,
    const torch::Tensor& weights,
    const torch::Tensor& current_kv,
    torch::Tensor& index_cache,
    const AttentionMetadata& attn_metadata,
    const DeepseekV4IndexerCacheRefs& cache_refs,
    bool is_prefill) {
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  const int64_t batch_size = attn_metadata.kv_seq_lens.size(0);
  torch::Tensor k_source;
  std::optional<torch::Tensor> k_cache_block_table = std::nullopt;
  bool select_prefill = is_prefill;
  if (is_prefill) {
    if (attn_metadata.is_chunked_prefill) {
      k_source = gather_c4_cache(index_cache,
                                 cache_refs.index_block_table,
                                 dsa.index_c4_seq_lens,
                                 dsa.index_total_c4_len,
                                 dsa.index_max_c4_len);
    } else {
      k_source = current_kv;
    }
  } else {
    const int64_t seq_len = q.size(0) / batch_size;
    k_source = index_cache;
    k_cache_block_table = cache_refs.index_block_table;
    select_prefill = false;
    (void)seq_len;
  }

  torch::Tensor new_block_tables;
  if (select_prefill) {
    new_block_tables =
        torch::empty({q.size(0), index_topk_},
                     torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(cache_refs.index_block_table.device()));
  } else {
    const int64_t seq_len = q.size(0) / batch_size;
    new_block_tables =
        torch::empty({batch_size, seq_len, index_topk_},
                     torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(cache_refs.index_block_table.device()));
  }
  torch::Tensor new_context_lens =
      torch::empty({q.size(0)},
                   torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(cache_refs.index_block_table.device()));
  new_block_tables.fill_(-1);
  new_context_lens.zero_();
  if (dsa.index_total_c4_len == 0) {
    if (!select_prefill) {
      new_block_tables = new_block_tables.view({-1, index_topk_});
    }
    return {new_block_tables, new_context_lens};
  }

  xllm::kernel::MaskedIndexerSelectPagedKVParams params;
  params.query =
      select_prefill
          ? q
          : q.view({batch_size, q.size(0) / batch_size, n_heads_, head_dim_});
  params.k_cache = k_source;
  params.weights =
      select_prefill
          ? weights
          : weights.view({batch_size, q.size(0) / batch_size, n_heads_});
  params.kv_cache_block_table = cache_refs.index_block_table;
  params.cu_seq_q_lens = select_prefill
                             ? std::optional<torch::Tensor>(
                                   attn_metadata.q_cu_seq_lens.contiguous())
                             : std::nullopt;
  params.cu_seq_k_lens = select_prefill
                             ? std::optional<torch::Tensor>(
                                   attn_metadata.kv_cu_seq_lens.contiguous())
                             : std::nullopt;
  params.k_context_lens = select_prefill
                              ? std::nullopt
                              : std::optional<torch::Tensor>(
                                    attn_metadata.kv_seq_lens.contiguous());
  params.k_cache_block_table = k_cache_block_table;
  params.is_prefill = select_prefill;
  params.index_topk = index_topk_;
  params.kv_cache_block_size = index_cache.size(2);
  params.softmax_scale = softmax_scale_;
  params.q_scale = std::nullopt;
  params.k_scale_cache = std::nullopt;
  params.sparse_block_table = new_block_tables;
  params.sparse_context_lens = new_context_lens;
  params.is_score_float = false;
  params.compress_ratio = kCompressRatio;
  xllm::kernel::masked_indexer_select_paged_kv(params);

  if (!select_prefill) {
    new_block_tables = new_block_tables.view({-1, index_topk_});
  }
  return {new_block_tables, new_context_lens};
}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4IndexerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    torch::Tensor& index_cache,
    torch::Tensor& compress_index_state,
    const AttentionMetadata& attn_metadata,
    const DeepseekV4IndexerCacheRefs& cache_refs,
    bool is_prefill,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table,
    std::optional<torch::Tensor> projected_weights,
    std::optional<torch::Tensor> projected_kv,
    std::optional<torch::Tensor> projected_score) {
  torch::Tensor q = preprocess_q(
      qr, attn_metadata, compressed_sin_table, compressed_cos_table);
  torch::Tensor hidden_states = x;
  torch::Tensor current_kv = compress_kv(hidden_states,
                                         compress_index_state,
                                         attn_metadata,
                                         index_cache,
                                         cache_refs,
                                         compressed_sin_table,
                                         compressed_cos_table,
                                         projected_kv,
                                         projected_score);
  torch::Tensor weights = preprocess_weights(x, projected_weights);
  return select_topk(q,
                     weights,
                     current_kv,
                     index_cache,
                     attn_metadata,
                     cache_refs,
                     is_prefill);
}

void DeepseekV4IndexerImpl::load_state_dict(const StateDict& state_dict,
                                            bool skip_proj_weights) {
  if (state_dict.size() == 0) {
    return;
  }
  // wq_b_ projects qr (not hidden), so it never participates in the attention
  // layer's fused GEMM and must always be loaded. Only weights_proj_ (hidden
  // -> n_heads) is skipped when its weights have been merged into the fused
  // projection.
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  if (!skip_proj_weights) {
    weights_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("weights_proj."));
  }
  compressor_->load_state_dict(state_dict.get_dict_with_prefix("compressor."),
                               /*skip_proj_weights=*/skip_proj_weights);
}

}  // namespace layer
}  // namespace xllm
