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

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "attention.h"
#include "compressor.h"
#include "deepseek_v4_indexer.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {
namespace layer {

inline int64_t deepseek_v4_ori_window_left(int64_t window_size,
                                           int64_t dspark_block_size,
                                           bool use_native_dspark_sas) {
  if (use_native_dspark_sas && dspark_block_size > 0) {
    return std::max<int64_t>(window_size + dspark_block_size - 1, 0);
  }
  // The compatibility fallback accepted by CANN 9.0 uses window_size - 1
  // (127 for DSV4). DSpark block visibility is encoded by expanding the block
  // into q_len=1 rows that share the same kv end.
  return std::max<int64_t>(window_size - 1, 0);
}

torch::Tensor build_dspark_swa_indices(const torch::Tensor& block_table,
                                       const torch::Tensor& query_cu_seq_lens,
                                       const torch::Tensor& seq_lens,
                                       int64_t window_size,
                                       int64_t dspark_block_size,
                                       int64_t cache_block_size);

// DSA kv state aligned with Python:
// (ori_kv, compressor_kv_state, compressor_score_state,
//  c4_indexer_kv_state, c4_indexer_score_state)
using KVState = std::tuple<torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor>;

class DSAttentionImpl : public torch::nn::Module {
 public:
  DSAttentionImpl() = default;
  explicit DSAttentionImpl(const ModelContext& context, int32_t layer_id = -1);
  DSAttentionImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options,
                  int32_t layer_id = -1);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const DSAMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      KVCache& kv_cache,
      KVState& kv_state,
      bool is_prefill,
      bool is_chunked_prefill,
      const std::
          tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&
              compress_metadata);

  // Shared-KV write path: no query attention, no FFN.
  void write_context_kv(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        const torch::Tensor& slot_mapping,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);
  int64_t non_registered_weight_bytes() const;

 private:
  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t n_kv_heads_;
  int64_t sliding_window_;
  int64_t head_dim_;

  int64_t n_local_heads_;
  int64_t q_lora_rank_;
  int64_t o_lora_rank_;
  int64_t o_groups_;

  int64_t rope_head_dim_;
  int64_t nope_head_dim_;
  int64_t window_size_;
  double compress_ratio_;

  double softmax_scale_;
  double eps_ = 1e-6;
  int64_t qk_head_dim_;
  int64_t n_local_groups_;
  int64_t tp_rank_ = 0;
  int64_t tp_size_ = 1;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 0;
  int64_t dspark_block_size_ = 0;
  bool dspark_use_native_sas_ = false;

  double rope_theta_ = 10000.0;
  double compress_rope_theta_ = 40000.0;

  ReplicatedLinear q_a_proj_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  RMSNorm q_layernorm_{nullptr};

  ReplicatedLinear kv_proj_{nullptr};
  RMSNorm kv_layernorm_{nullptr};

  ColumnParallelLinear o_a_proj_{nullptr};
  RowParallelLinear o_b_proj_{nullptr};

  torch::Tensor attn_sink_;
  bool attn_sink_loaded_ = false;

  Attention attn_{nullptr};
  DeepseekScalingRotaryEmbedding rotary_emb_{nullptr};
  DeepseekV4Indexer indexer_{nullptr};
  Compressor compressor_{nullptr};

  torch::Tensor q_rms_gamma_;
};
TORCH_MODULE(DSAttention);

}  // namespace layer
}  // namespace xllm
