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

#pragma once

#include <memory>
#include <vector>

#include "core/framework/model_context.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/npu/rotary_embedding.h"
#include "core/layers/npu_torch/attention.h"

namespace xllm {
namespace layer {

class MiniMaxM3AttentionImpl final : public torch::nn::Module {
 public:
  MiniMaxM3AttentionImpl(const ModelContext& context, int32_t layer_id);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  std::vector<int64_t> get_query_cu_lens(
      const layer::AttentionMetadata& attn_metadata,
      int64_t num_query_tokens,
      const std::vector<int64_t>& seq_lens) const;
  torch::Tensor select_local_sparse_index_heads(
      const torch::Tensor& index_q) const;
  void write_kv_cache(const torch::Tensor& key,
                      const torch::Tensor& value,
                      KVCache& kv_cache,
                      const layer::AttentionMetadata& attn_metadata) const;
  void write_index_cache(const torch::Tensor& index_key,
                         KVCache& kv_cache,
                         const layer::AttentionMetadata& attn_metadata) const;
  torch::Tensor logical_cache_slice(const torch::Tensor& cache,
                                    const torch::Tensor& block_table,
                                    int64_t seq_idx,
                                    int64_t seq_len) const;
  torch::Tensor merge_minimax_sparse_blocks(
      const torch::Tensor& topk_blocks,
      const torch::Tensor& query_positions,
      int64_t num_blocks) const;
  torch::Tensor select_minimax_sparse_blocks(
      const torch::Tensor& index_query,
      const torch::Tensor& index_key,
      const torch::Tensor& query_positions,
      int64_t seq_len) const;
  torch::Tensor expand_sparse_blocks_to_tokens(
      const torch::Tensor& block_indices,
      int64_t seq_len) const;
  torch::Tensor apply_sparse_attention_for_index_head(
      const torch::Tensor& query,
      const torch::Tensor& key_seq,
      const torch::Tensor& value_seq,
      const torch::Tensor& token_indices,
      const torch::Tensor& query_positions) const;
  torch::Tensor apply_full_attention_for_sequence(
      const torch::Tensor& query,
      const torch::Tensor& key_seq,
      const torch::Tensor& value_seq,
      const torch::Tensor& query_positions) const;
  torch::Tensor sparse_attention_forward(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& value,
      const torch::Tensor& index_query,
      const torch::Tensor& index_key,
      const torch::Tensor& positions,
      KVCache& kv_cache,
      const layer::AttentionMetadata& attn_metadata) const;

  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  float scaling_ = 1.0f;
  int64_t total_num_heads_ = 0;
  int64_t total_num_kv_heads_ = 0;
  int64_t tp_rank_ = 0;
  int64_t sparse_index_dim_ = 0;
  int64_t total_sparse_index_heads_ = 0;
  int64_t local_sparse_index_head_start_ = 0;
  int64_t local_sparse_index_head_count_ = 0;
  int64_t sparse_topk_blocks_ = 0;
  int64_t sparse_block_size_ = 0;
  int64_t sparse_init_block_ = 0;
  int64_t sparse_local_block_ = 0;
  int64_t sparse_count_ = 0;
  int64_t sparse_total_blocks_ = 0;
  bool use_sparse_attention_ = false;
  bool local_sparse_index_heads_contiguous_ = true;
  std::vector<int64_t> local_sparse_index_heads_;
  std::vector<int64_t> sparse_index_head_per_q_head_;
  layer::QKVParallelLinear qkv_proj_{nullptr};
  layer::ReplicatedLinear index_q_proj_{nullptr};
  layer::ReplicatedLinear index_k_proj_{nullptr};
  layer::RowParallelLinear o_proj_{nullptr};
  layer::Qwen3NextRMSNorm q_norm_{nullptr};
  layer::Qwen3NextRMSNorm k_norm_{nullptr};
  layer::Qwen3NextRMSNorm index_q_norm_{nullptr};
  layer::Qwen3NextRMSNorm index_k_norm_{nullptr};
  std::shared_ptr<NpuRotaryEmbedding> rotary_emb_;
  layer::Attention attn_{nullptr};
};
TORCH_MODULE(MiniMaxM3Attention);

}  // namespace layer
}  // namespace xllm
