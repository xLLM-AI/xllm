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

#include "core/layers/npu_torch/minimax_m3_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/layers/common/rotary_embedding_util.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

constexpr char kQuantMethodMxfp8[] = "mxfp8";
constexpr int64_t kSparseAttentionQueryChunkSize = 256;

bool is_load_time_dequant_method(const std::string& quant_method) {
  return quant_method == kQuantMethodFp8 || quant_method == kQuantMethodMxfp8;
}

bool is_minimax_sparse_layer(const ModelArgs& args, int32_t layer_id) {
  if (!args.use_sparse_attention()) {
    return false;
  }
  const std::vector<int32_t>& sparse_freq = args.sparse_attention_freq();
  return layer_id >= 0 && layer_id < static_cast<int32_t>(sparse_freq.size()) &&
         sparse_freq[static_cast<size_t>(layer_id)] != 0;
}

int64_t max_seq_len_from_metadata(
    const layer::AttentionMetadata& attn_metadata) {
  if (!attn_metadata.kv_seq_lens_vec.empty()) {
    const auto max_it = std::max_element(attn_metadata.kv_seq_lens_vec.begin(),
                                         attn_metadata.kv_seq_lens_vec.end());
    return static_cast<int64_t>(*max_it);
  }
  return attn_metadata.max_seq_len;
}

std::vector<int64_t> to_int64_vector(const std::vector<int32_t>& values) {
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (int32_t value : values) {
    result.emplace_back(static_cast<int64_t>(value));
  }
  return result;
}

}  // namespace

MiniMaxM3AttentionImpl::MiniMaxM3AttentionImpl(const ModelContext& context,
                                               int32_t layer_id) {
  const ModelArgs& args = context.get_model_args();
  QuantArgs quant_args = context.get_quant_args();
  if (is_load_time_dequant_method(quant_args.quant_method())) {
    quant_args.quant_method("");
  }
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  tp_rank_ = parallel_args.tp_group_->rank();
  total_num_heads_ = args.n_heads();
  total_num_kv_heads_ = args.n_kv_heads().value_or(args.n_heads());

  CHECK(total_num_heads_ % tp_size == 0);
  num_heads_ = total_num_heads_ / tp_size;
  if (total_num_kv_heads_ >= tp_size) {
    CHECK(total_num_kv_heads_ % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads_ / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads_ == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads_;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  CHECK_EQ(num_heads_ % num_kv_heads_, 0)
      << "MiniMax-M3 local attention heads must be divisible by local kv heads";
  scaling_ = std::sqrt(1.0f / static_cast<float>(head_dim_));

  qkv_proj_ = register_module("qkv_proj",
                              layer::QKVParallelLinear(args.hidden_size(),
                                                       num_heads_,
                                                       num_kv_heads_,
                                                       head_dim_,
                                                       num_kv_head_replicas_,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       parallel_args,
                                                       options));

  use_sparse_attention_ = is_minimax_sparse_layer(args, layer_id);
  if (use_sparse_attention_) {
    sparse_index_dim_ =
        args.sparse_index_dim() > 0 ? args.sparse_index_dim() : head_dim_;
    total_sparse_index_heads_ = args.sparse_num_index_heads() > 0
                                    ? args.sparse_num_index_heads()
                                    : total_num_kv_heads_;
    sparse_topk_blocks_ =
        args.sparse_topk_blocks() > 0 ? args.sparse_topk_blocks() : 16;
    sparse_block_size_ =
        args.sparse_block_size() > 0 ? args.sparse_block_size() : 128;
    sparse_init_block_ = std::max<int64_t>(args.sparse_init_block(), 0);
    sparse_local_block_ = std::max<int64_t>(args.sparse_local_block(), 0);
    sparse_count_ = sparse_topk_blocks_ * sparse_block_size_;
    sparse_total_blocks_ =
        sparse_topk_blocks_ + sparse_init_block_ + sparse_local_block_;
    CHECK_GT(total_sparse_index_heads_, 0);
    CHECK_GT(sparse_index_dim_, 0);
    CHECK_GT(sparse_topk_blocks_, 0);
    CHECK_GT(sparse_block_size_, 0);

    const int64_t heads_per_kv = total_num_heads_ / total_num_kv_heads_;
    const int64_t q_start = tp_rank_ * num_heads_;
    const int64_t q_end = q_start + num_heads_ - 1;
    const int64_t first_group = q_start / heads_per_kv;
    const int64_t last_group = q_end / heads_per_kv;
    local_sparse_index_heads_.reserve(
        static_cast<size_t>(last_group - first_group + 1));
    for (int64_t group = first_group; group <= last_group; ++group) {
      local_sparse_index_heads_.emplace_back(group % total_sparse_index_heads_);
    }
    local_sparse_index_head_start_ = local_sparse_index_heads_.front();
    local_sparse_index_head_count_ =
        static_cast<int64_t>(local_sparse_index_heads_.size());
    for (int64_t i = 0; i < local_sparse_index_head_count_; ++i) {
      if (local_sparse_index_heads_[static_cast<size_t>(i)] !=
          local_sparse_index_head_start_ + i) {
        local_sparse_index_heads_contiguous_ = false;
        break;
      }
    }

    sparse_index_head_per_q_head_.reserve(static_cast<size_t>(num_heads_));
    for (int64_t local_q_head = 0; local_q_head < num_heads_; ++local_q_head) {
      const int64_t sparse_head =
          ((q_start + local_q_head) / heads_per_kv) % total_sparse_index_heads_;
      auto it = std::find(local_sparse_index_heads_.begin(),
                          local_sparse_index_heads_.end(),
                          sparse_head);
      CHECK(it != local_sparse_index_heads_.end());
      const int64_t local_sparse_head = static_cast<int64_t>(
          std::distance(local_sparse_index_heads_.begin(), it));
      sparse_index_head_per_q_head_.emplace_back(local_sparse_head);
    }

    index_q_proj_ = register_module(
        "index_q_proj",
        layer::ReplicatedLinear(args.hidden_size(),
                                total_sparse_index_heads_ * sparse_index_dim_,
                                /*bias=*/false,
                                quant_args,
                                options));
    index_k_proj_ = register_module("index_k_proj",
                                    layer::ReplicatedLinear(args.hidden_size(),
                                                            sparse_index_dim_,
                                                            /*bias=*/false,
                                                            quant_args,
                                                            options));
    index_q_norm_ =
        register_module("index_q_norm",
                        layer::Qwen3NextRMSNorm(
                            sparse_index_dim_, args.rms_norm_eps(), options));
    index_k_norm_ =
        register_module("index_k_norm",
                        layer::Qwen3NextRMSNorm(
                            sparse_index_dim_, args.rms_norm_eps(), options));
  }

  o_proj_ =
      register_module("o_proj",
                      layer::RowParallelLinear(total_num_heads_ * head_dim_,
                                               args.hidden_size(),
                                               /*bias=*/false,
                                               /*input_is_parallelized=*/true,
                                               /*enable_result_reduction=*/true,
                                               quant_args,
                                               parallel_args.tp_group_,
                                               options));

  q_norm_ = register_module(
      "q_norm",
      layer::Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
  k_norm_ = register_module(
      "k_norm",
      layer::Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  const int64_t rotary_dim =
      args.rotary_dim() > 0 ? args.rotary_dim() : args.head_dim();
  const torch::Tensor inv_freq =
      layer::rotary::compute_inv_freq(rotary_dim, args.rope_theta(), options);
  rotary_emb_ = register_module(
      "rope",
      std::make_shared<RotaryEmbeddingGeneric>(rotary_dim,
                                               args.max_position_embeddings(),
                                               inv_freq,
                                               /*interleaved=*/false,
                                               options));
  attn_ = register_module("attn",
                          layer::Attention(num_heads_,
                                           head_dim_,
                                           scaling_,
                                           num_kv_heads_,
                                           args.sliding_window()));
}

std::vector<int64_t> MiniMaxM3AttentionImpl::get_query_cu_lens(
    const layer::AttentionMetadata& attn_metadata,
    int64_t num_query_tokens,
    const std::vector<int64_t>& seq_lens) const {
  const int64_t batch_size = static_cast<int64_t>(seq_lens.size());
  if (static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size()) == batch_size) {
    std::vector<int64_t> q_cu_lens;
    q_cu_lens.reserve(static_cast<size_t>(batch_size));
    int64_t total = 0;
    for (int32_t q_len : attn_metadata.q_seq_lens_vec) {
      total += static_cast<int64_t>(q_len);
      q_cu_lens.emplace_back(total);
    }
    if (total == num_query_tokens) {
      return q_cu_lens;
    }
  }

  if (!attn_metadata.q_cu_seq_lens_host_vec.empty()) {
    const std::vector<int64_t>& raw = attn_metadata.q_cu_seq_lens_host_vec;
    if (static_cast<int64_t>(raw.size()) == batch_size + 1 &&
        raw.front() == 0 && raw.back() == num_query_tokens) {
      return std::vector<int64_t>(raw.begin() + 1, raw.end());
    }
    if (static_cast<int64_t>(raw.size()) == batch_size &&
        raw.back() == num_query_tokens) {
      return raw;
    }
  }

  if (batch_size == 1) {
    return std::vector<int64_t>{num_query_tokens};
  }

  if (num_query_tokens == batch_size) {
    std::vector<int64_t> q_cu_lens;
    q_cu_lens.reserve(static_cast<size_t>(batch_size));
    for (int64_t i = 1; i <= batch_size; ++i) {
      q_cu_lens.emplace_back(i);
    }
    return q_cu_lens;
  }

  int64_t seq_total = 0;
  for (int64_t seq_len : seq_lens) {
    seq_total += seq_len;
  }
  if (attn_metadata.is_prefill && seq_total == num_query_tokens) {
    std::vector<int64_t> q_cu_lens;
    q_cu_lens.reserve(static_cast<size_t>(batch_size));
    int64_t total = 0;
    for (int64_t seq_len : seq_lens) {
      total += seq_len;
      q_cu_lens.emplace_back(total);
    }
    return q_cu_lens;
  }

  LOG(FATAL) << "Cannot derive MiniMax-M3 sparse attention query lengths: "
             << "num_query_tokens=" << num_query_tokens
             << ", batch_size=" << batch_size
             << ", q_seq_lens=" << attn_metadata.q_seq_lens_vec
             << ", kv_seq_lens=" << attn_metadata.kv_seq_lens_vec;
}

torch::Tensor MiniMaxM3AttentionImpl::select_local_sparse_index_heads(
    const torch::Tensor& index_q) const {
  if (local_sparse_index_head_count_ == total_sparse_index_heads_) {
    return index_q.contiguous();
  }
  if (local_sparse_index_heads_contiguous_) {
    return index_q
        .narrow(/*dim=*/1,
                local_sparse_index_head_start_,
                local_sparse_index_head_count_)
        .contiguous();
  }

  torch::Tensor index = torch::tensor(
      local_sparse_index_heads_,
      torch::TensorOptions().dtype(torch::kInt64).device(index_q.device()));
  return index_q.index_select(/*dim=*/1, index).contiguous();
}

void MiniMaxM3AttentionImpl::write_kv_cache(
    const torch::Tensor& key,
    const torch::Tensor& value,
    KVCache& kv_cache,
    const layer::AttentionMetadata& attn_metadata) const {
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key =
      key.view({-1, num_kv_heads_, head_dim_}).contiguous();
  reshape_paged_cache_params.value =
      value.view({-1, num_kv_heads_, head_dim_}).contiguous();
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
}

void MiniMaxM3AttentionImpl::write_index_cache(
    const torch::Tensor& index_key,
    KVCache& kv_cache,
    const layer::AttentionMetadata& attn_metadata) const {
  torch::Tensor index_cache = kv_cache.get_index_cache();
  CHECK(index_cache.defined())
      << "MiniMax-M3 sparse attention requires index cache.";
  torch::Tensor slot_mapping = attn_metadata.slot_mapping.to(torch::kInt64);
  torch::Tensor valid_positions =
      torch::nonzero(slot_mapping >= 0).flatten().to(torch::kInt64);
  if (valid_positions.numel() == 0) {
    return;
  }

  torch::Tensor valid_slots =
      slot_mapping.index_select(/*dim=*/0, valid_positions);
  torch::Tensor valid_index_key =
      index_key.index_select(/*dim=*/0, valid_positions)
          .view({-1, 1, sparse_index_dim_})
          .contiguous();
  torch::Tensor flat_index_cache = index_cache.view({-1, 1, sparse_index_dim_});
  flat_index_cache.index_copy_(/*dim=*/0, valid_slots, valid_index_key);
}

torch::Tensor MiniMaxM3AttentionImpl::logical_cache_slice(
    const torch::Tensor& cache,
    const torch::Tensor& block_table,
    int64_t seq_idx,
    int64_t seq_len) const {
  if (seq_len == 0) {
    return cache.new_empty({0, cache.size(2), cache.size(3)});
  }

  const int64_t block_size = cache.size(1);
  torch::Tensor logical = torch::arange(
      seq_len,
      torch::TensorOptions().dtype(torch::kInt64).device(cache.device()));
  torch::Tensor block_ids = logical.div(block_size, "floor");
  torch::Tensor physical_blocks = block_table.index({seq_idx})
                                      .to(torch::kInt64)
                                      .index_select(
                                          /*dim=*/0, block_ids);
  torch::Tensor offsets = logical.remainder(block_size);
  return cache.index({physical_blocks, offsets});
}

torch::Tensor MiniMaxM3AttentionImpl::merge_minimax_sparse_blocks(
    const torch::Tensor& topk_blocks,
    const torch::Tensor& query_positions,
    int64_t num_blocks) const {
  if (sparse_init_block_ <= 0 && sparse_local_block_ <= 0) {
    return topk_blocks;
  }

  const int64_t q_len = query_positions.size(0);
  const int64_t num_index_heads = topk_blocks.size(1);
  std::vector<torch::Tensor> forced_parts;
  if (sparse_init_block_ > 0) {
    forced_parts.emplace_back(
        torch::arange(sparse_init_block_,
                      torch::TensorOptions()
                          .dtype(topk_blocks.scalar_type())
                          .device(topk_blocks.device()))
            .view({1, 1, -1})
            .expand({q_len, num_index_heads, -1}));
  }
  if (sparse_local_block_ > 0) {
    torch::Tensor local_offsets =
        torch::arange(sparse_local_block_,
                      torch::TensorOptions()
                          .dtype(query_positions.scalar_type())
                          .device(query_positions.device()));
    torch::Tensor block_ids = query_positions.div(sparse_block_size_, "floor");
    torch::Tensor first_local_block =
        (block_ids - sparse_local_block_ + 1).clamp_min(0);
    forced_parts.emplace_back(
        (first_local_block.unsqueeze(1) + local_offsets.unsqueeze(0))
            .to(topk_blocks.scalar_type())
            .view({q_len, 1, -1})
            .expand({q_len, num_index_heads, -1}));
  }

  if (forced_parts.empty()) {
    return topk_blocks;
  }

  forced_parts.emplace_back(topk_blocks);
  torch::Tensor candidates = torch::cat(forced_parts, /*dim=*/-1);
  torch::Tensor valid = (candidates >= 0) & (candidates < num_blocks);
  valid = valid &
          (candidates * sparse_block_size_ <=
           query_positions.view({q_len, 1, 1}).to(candidates.scalar_type()));

  torch::Tensor invalid_value = torch::full_like(candidates, num_blocks);
  torch::Tensor sorted_candidates =
      std::get<0>(torch::sort(torch::where(valid, candidates, invalid_value),
                              /*dim=*/-1));
  torch::Tensor sorted_valid = sorted_candidates < num_blocks;
  torch::Tensor previous = torch::cat(
      {torch::full_like(sorted_candidates.slice(/*dim=*/-1, 0, 1), -1),
       sorted_candidates.slice(/*dim=*/-1, 0, -1)},
      /*dim=*/-1);
  torch::Tensor keep = sorted_valid & (sorted_candidates != previous);
  torch::Tensor ranks = torch::cumsum(keep.to(torch::kInt32), /*dim=*/-1) - 1;

  torch::Tensor output =
      torch::full({q_len, num_index_heads, sparse_total_blocks_ + 1},
                  -1,
                  topk_blocks.options());
  torch::Tensor overflow_rank = torch::full_like(ranks, sparse_total_blocks_);
  torch::Tensor scatter_index =
      torch::where(keep & (ranks < sparse_total_blocks_), ranks, overflow_rank)
          .to(torch::kInt64);
  torch::Tensor scatter_src = torch::where(keep, sorted_candidates, -1);
  output.scatter_(/*dim=*/2, scatter_index, scatter_src);
  return output.slice(/*dim=*/2, /*start=*/0, /*end=*/sparse_total_blocks_);
}

torch::Tensor MiniMaxM3AttentionImpl::select_minimax_sparse_blocks(
    const torch::Tensor& index_query,
    const torch::Tensor& index_key,
    const torch::Tensor& query_positions,
    int64_t seq_len) const {
  const int64_t num_blocks =
      (seq_len + sparse_block_size_ - 1) / sparse_block_size_;
  const int64_t topk_blocks = std::min(sparse_topk_blocks_, num_blocks);
  torch::Tensor scores = torch::einsum("qhd,kd->qhk", {index_query, index_key})
                             .to(torch::kFloat32);

  const int64_t padded_tokens = num_blocks * sparse_block_size_;
  if (padded_tokens != seq_len) {
    torch::Tensor pad =
        torch::full({scores.size(0), scores.size(1), padded_tokens - seq_len},
                    -1.0e30f,
                    scores.options());
    scores = torch::cat({scores, pad}, /*dim=*/-1);
  }

  torch::Tensor key_positions =
      torch::arange(padded_tokens,
                    torch::TensorOptions()
                        .dtype(query_positions.scalar_type())
                        .device(query_positions.device()));
  torch::Tensor valid =
      (key_positions.view({1, -1}) < seq_len) &
      (key_positions.view({1, -1}) <= query_positions.view({-1, 1}));
  scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e30f);
  torch::Tensor block_scores = scores
                                   .view({index_query.size(0),
                                          index_query.size(1),
                                          num_blocks,
                                          sparse_block_size_})
                                   .amax(/*dim=*/-1);
  torch::Tensor blocks =
      std::get<1>(torch::topk(block_scores, topk_blocks, /*dim=*/-1))
          .to(torch::kInt32);
  if (topk_blocks < sparse_topk_blocks_) {
    torch::Tensor pad = torch::full(
        {blocks.size(0), blocks.size(1), sparse_topk_blocks_ - topk_blocks},
        -1,
        blocks.options());
    blocks = torch::cat({blocks, pad}, /*dim=*/-1);
  }
  return merge_minimax_sparse_blocks(blocks, query_positions, num_blocks);
}

torch::Tensor MiniMaxM3AttentionImpl::expand_sparse_blocks_to_tokens(
    const torch::Tensor& block_indices,
    int64_t seq_len) const {
  torch::Tensor offsets = torch::arange(sparse_block_size_,
                                        torch::TensorOptions()
                                            .dtype(block_indices.scalar_type())
                                            .device(block_indices.device()));
  torch::Tensor token_indices =
      block_indices.unsqueeze(-1) * sparse_block_size_ + offsets;
  token_indices = token_indices.flatten(/*start_dim=*/2);
  torch::Tensor valid_blocks =
      (block_indices.flatten(/*start_dim=*/2) >= 0)
          .repeat_interleave(sparse_block_size_, /*dim=*/-1);
  torch::Tensor valid_tokens = valid_blocks & (token_indices < seq_len);
  return torch::where(
      valid_tokens, token_indices, torch::full_like(token_indices, -1));
}

torch::Tensor MiniMaxM3AttentionImpl::apply_sparse_attention_for_index_head(
    const torch::Tensor& query,
    const torch::Tensor& key_seq,
    const torch::Tensor& value_seq,
    const torch::Tensor& token_indices,
    const torch::Tensor& query_positions) const {
  const int64_t q_len = query.size(0);
  const int64_t seq_len = key_seq.size(0);
  torch::Tensor output = torch::empty_like(query);
  for (int64_t q_start = 0; q_start < q_len;
       q_start += kSparseAttentionQueryChunkSize) {
    const int64_t q_end =
        std::min(q_start + kSparseAttentionQueryChunkSize, q_len);
    torch::Tensor query_chunk = query.slice(/*dim=*/0, q_start, q_end);
    torch::Tensor token_indices_chunk =
        token_indices.slice(/*dim=*/0, q_start, q_end);
    torch::Tensor query_positions_chunk =
        query_positions.slice(/*dim=*/0, q_start, q_end);
    const int64_t q_chunk_len = q_end - q_start;

    torch::Tensor valid =
        (token_indices_chunk >= 0) & (token_indices_chunk < seq_len);
    valid =
        valid & (token_indices_chunk <= query_positions_chunk.view({-1, 1}));
    torch::Tensor safe_idx =
        token_indices_chunk.clamp(0, std::max<int64_t>(seq_len - 1, 0))
            .to(torch::kInt64);
    torch::Tensor flat_idx = safe_idx.reshape({-1});
    torch::Tensor k_selected = key_seq.index_select(/*dim=*/0, flat_idx)
                                   .view({q_chunk_len, -1, head_dim_});
    torch::Tensor v_selected = value_seq.index_select(/*dim=*/0, flat_idx)
                                   .view({q_chunk_len, -1, head_dim_});
    torch::Tensor scores =
        torch::einsum("qhd,qkd->qhk", {query_chunk, k_selected})
            .to(torch::kFloat32);
    scores = scores * scaling_;
    scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e30f);
    torch::Tensor probs = torch::softmax(scores, /*dim=*/-1);
    output.slice(/*dim=*/0, q_start, q_end)
        .copy_(torch::einsum("qhk,qkd->qhd",
                             {probs.to(v_selected.scalar_type()), v_selected}));
  }
  return output;
}

torch::Tensor MiniMaxM3AttentionImpl::apply_full_attention_for_sequence(
    const torch::Tensor& query,
    const torch::Tensor& key_seq,
    const torch::Tensor& value_seq,
    const torch::Tensor& query_positions) const {
  const int64_t seq_len = key_seq.size(0);
  torch::Tensor key_positions =
      torch::arange(seq_len,
                    torch::TensorOptions()
                        .dtype(query_positions.scalar_type())
                        .device(query_positions.device()));
  torch::Tensor valid =
      key_positions.view({1, -1}) <= query_positions.view({-1, 1});
  torch::Tensor scores =
      torch::einsum("qhd,kd->qhk", {query, key_seq}).to(torch::kFloat32);
  scores = scores * scaling_;
  scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e30f);
  torch::Tensor probs = torch::softmax(scores, /*dim=*/-1);
  return torch::einsum("qhk,kd->qhd",
                       {probs.to(value_seq.scalar_type()), value_seq});
}

torch::Tensor MiniMaxM3AttentionImpl::sparse_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& index_query,
    const torch::Tensor& index_key,
    const torch::Tensor& positions,
    KVCache& kv_cache,
    const layer::AttentionMetadata& attn_metadata) const {
  CHECK_EQ(num_kv_heads_, 1)
      << "MiniMax-M3 sparse attention currently requires one local KV head. "
      << "Use TP >= total KV heads for the sparse path.";

  const std::vector<int64_t> seq_lens =
      to_int64_vector(attn_metadata.kv_seq_lens_vec);
  CHECK(!seq_lens.empty())
      << "MiniMax-M3 sparse attention requires host kv_seq_lens.";
  const std::vector<int64_t> query_cu_lens =
      get_query_cu_lens(attn_metadata, query.size(0), seq_lens);
  CHECK_EQ(query_cu_lens.size(), seq_lens.size());

  torch::Tensor output = torch::empty_like(query);
  torch::Tensor flat_positions = positions.reshape({-1}).to(torch::kInt64);
  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();
  torch::Tensor index_cache = kv_cache.get_index_cache();
  torch::Tensor block_table = attn_metadata.block_table;

  int64_t q_start = 0;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(seq_lens.size());
       ++seq_idx) {
    const size_t seq_offset = static_cast<size_t>(seq_idx);
    const int64_t q_end = query_cu_lens[seq_offset];
    if (q_end <= q_start) {
      continue;
    }

    const int64_t query_len = q_end - q_start;
    int64_t seq_len = seq_lens[seq_offset];
    torch::Tensor k_seq;
    torch::Tensor v_seq;
    torch::Tensor index_k_seq;
    const bool use_direct_prefill =
        attn_metadata.is_prefill && seq_len == query_len;
    if (use_direct_prefill) {
      k_seq = key.slice(/*dim=*/0, q_start, q_end).select(/*dim=*/1, 0);
      v_seq = value.slice(/*dim=*/0, q_start, q_end).select(/*dim=*/1, 0);
      index_k_seq =
          index_key.slice(/*dim=*/0, q_start, q_end).select(/*dim=*/1, 0);
    } else {
      CHECK(block_table.defined())
          << "MiniMax-M3 sparse attention requires block_table outside "
             "prefill.";
      k_seq = logical_cache_slice(k_cache, block_table, seq_idx, seq_len)
                  .select(/*dim=*/1, 0);
      v_seq = logical_cache_slice(v_cache, block_table, seq_idx, seq_len)
                  .select(/*dim=*/1, 0);
      index_k_seq =
          logical_cache_slice(index_cache, block_table, seq_idx, seq_len)
              .select(/*dim=*/1, 0);
    }

    if (seq_len == 0) {
      output.slice(/*dim=*/0, q_start, q_end).zero_();
      q_start = q_end;
      continue;
    }

    torch::Tensor q_seq = query.slice(/*dim=*/0, q_start, q_end);
    torch::Tensor query_positions =
        flat_positions.slice(/*dim=*/0, q_start, q_end).to(torch::kInt64);
    if (seq_len <= sparse_count_) {
      output.slice(/*dim=*/0, q_start, q_end)
          .copy_(apply_full_attention_for_sequence(
              q_seq, k_seq, v_seq, query_positions));
      q_start = q_end;
      continue;
    }

    torch::Tensor sparse_blocks = select_minimax_sparse_blocks(
        index_query.slice(/*dim=*/0, q_start, q_end),
        index_k_seq,
        query_positions,
        seq_len);
    torch::Tensor sparse_tokens =
        expand_sparse_blocks_to_tokens(sparse_blocks, seq_len);
    torch::Tensor seq_output = torch::empty_like(q_seq);
    int64_t head_start = 0;
    while (head_start < num_heads_) {
      const int64_t index_head =
          sparse_index_head_per_q_head_[static_cast<size_t>(head_start)];
      int64_t head_end = head_start + 1;
      while (head_end < num_heads_ &&
             sparse_index_head_per_q_head_[static_cast<size_t>(head_end)] ==
                 index_head) {
        ++head_end;
      }
      seq_output.slice(/*dim=*/1, head_start, head_end)
          .copy_(apply_sparse_attention_for_index_head(
              q_seq.slice(/*dim=*/1, head_start, head_end),
              k_seq,
              v_seq,
              sparse_tokens.index({torch::indexing::Slice(), index_head}),
              query_positions));
      head_start = head_end;
    }
    output.slice(/*dim=*/0, q_start, q_end).copy_(seq_output);
    q_start = q_end;
  }
  return output;
}

torch::Tensor MiniMaxM3AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const layer::AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }

  torch::Tensor qkv = qkv_proj_->forward(hidden_states);
  torch::Tensor q = qkv.slice(/*dim=*/-1, 0, q_size_);
  torch::Tensor k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  torch::Tensor v =
      qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t num_tokens = q.size(0);
  torch::Tensor q_heads = q.view({num_tokens, num_heads_, head_dim_});
  torch::Tensor k_heads = k.view({num_tokens, num_kv_heads_, head_dim_});
  torch::Tensor v_heads = v.view({num_tokens, num_kv_heads_, head_dim_});

  q_heads = std::get<0>(q_norm_->forward(q_heads));
  k_heads = std::get<0>(k_norm_->forward(k_heads));

  std::tie(q_heads, k_heads) =
      rotary_emb_->forward(q_heads, k_heads, positions);

  torch::Tensor q_flat = q_heads.reshape({num_tokens, q_size_}).contiguous();
  torch::Tensor k_flat = k_heads.reshape({num_tokens, kv_size_}).contiguous();
  torch::Tensor v_flat = v_heads.reshape({num_tokens, kv_size_}).contiguous();

  if (use_sparse_attention_) {
    torch::Tensor index_k = index_k_proj_->forward(hidden_states);
    torch::Tensor index_k_heads =
        index_k.view({num_tokens, 1, sparse_index_dim_});
    index_k_heads = std::get<0>(index_k_norm_->forward(index_k_heads));

    if (max_seq_len_from_metadata(attn_metadata) <= sparse_count_) {
      std::tuple<torch::Tensor, torch::Tensor> rotated_index_k =
          rotary_emb_->forward(index_k_heads, index_k_heads, positions);
      index_k_heads = std::get<1>(rotated_index_k).contiguous();
      write_index_cache(index_k_heads, kv_cache, attn_metadata);
      torch::Tensor out = std::get<0>(
          attn_->forward(attn_metadata, q_flat, k_flat, v_flat, kv_cache));
      return o_proj_->forward(out);
    }

    torch::Tensor index_q = index_q_proj_->forward(hidden_states);
    torch::Tensor index_q_heads = index_q.view(
        {num_tokens, total_sparse_index_heads_, sparse_index_dim_});
    index_q_heads = std::get<0>(index_q_norm_->forward(index_q_heads));
    std::tie(index_q_heads, index_k_heads) =
        rotary_emb_->forward(index_q_heads, index_k_heads, positions);
    index_q_heads = select_local_sparse_index_heads(index_q_heads);
    index_k_heads = index_k_heads.contiguous();

    write_index_cache(index_k_heads, kv_cache, attn_metadata);
    write_kv_cache(k_flat, v_flat, kv_cache, attn_metadata);
    torch::Tensor sparse_out = sparse_attention_forward(q_heads,
                                                        k_heads,
                                                        v_heads,
                                                        index_q_heads,
                                                        index_k_heads,
                                                        positions,
                                                        kv_cache,
                                                        attn_metadata);
    return o_proj_->forward(
        sparse_out.reshape({num_tokens, q_size_}).contiguous());
  }

  torch::Tensor out = std::get<0>(
      attn_->forward(attn_metadata, q_flat, k_flat, v_flat, kv_cache));
  return o_proj_->forward(out);
}

void MiniMaxM3AttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  if (use_sparse_attention_) {
    index_q_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("index_q_proj."));
    index_k_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("index_k_proj."));
    index_q_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("index_q_norm."));
    index_k_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("index_k_norm."));
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
}

}  // namespace layer
}  // namespace xllm
