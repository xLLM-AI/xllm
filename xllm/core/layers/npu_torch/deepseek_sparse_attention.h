/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "attention.h"
#include "compressor.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "deepseek_v4_indexer.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quantization/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {

class ModelParamsView;

namespace layer {

void build_deepseek_v4_sparse_metadata(DSAMetadata& dsa,
                                       const ModelParamsView& params,
                                       const ModelArgs& args,
                                       int64_t dp_local_tp_size,
                                       bool is_prefill);

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
  DSAttentionImpl(const ModelContext& context, int32_t layer_id);

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

 private:
  int64_t num_heads_;
  int64_t head_dim_;

  int64_t n_local_heads_;
  int64_t q_lora_rank_;
  int64_t o_lora_rank_;
  int64_t o_groups_;

  int64_t rope_head_dim_;
  int64_t nope_head_dim_;
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
  int64_t ori_win_left_ = 0;

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
