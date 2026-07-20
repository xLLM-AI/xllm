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

#include <optional>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v4/compressor.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_indexer.h"
#include "layers/mlu/deepseek_v4/dsa_cache_mapping.h"

namespace xllm {
namespace layer {

class DeepseekV4AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV4AttentionImpl() = default;

  DeepseekV4AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          int32_t layer_id);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

  void set_cache_mapping(const DSACacheMapping& mapping);

 private:
  // q_down / kv_down are the precomputed fused-projection slices for the q_a
  // and kv projections (segments 0 and 1 of hidden_proj); the attention layer
  // merges the hidden->* GEMMs into a single fused_wqa_wkv_ call and splits.
  torch::Tensor project_q(torch::Tensor& q_down, torch::Tensor& qr);

  torch::Tensor project_kv(torch::Tensor& kv_down);

  torch::Tensor project_output(torch::Tensor& attn_output);

  void apply_last_rope(torch::Tensor& tensor,
                       const torch::Tensor& sin_table,
                       const torch::Tensor& cos_table,
                       const torch::Tensor& input_positions,
                       int64_t rope_dim);

  int32_t layer_id_ = 0;
  int64_t hidden_dim_ = 0;
  int64_t q_lora_rank_ = 0;
  int64_t n_heads_ = 0;
  int64_t n_local_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t nope_head_dim_ = 0;
  int64_t o_groups_ = 0;
  int64_t o_lora_rank_ = 0;
  int64_t n_local_groups_ = 0;
  int64_t window_size_ = 0;
  int64_t compress_ratio_ = 1;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 0;
  int64_t tp_rank_ = 0;
  int64_t tp_size_ = 1;
  double eps_ = 1e-6;
  float scale_ = 1.0f;
  bool attn_sink_loaded_ = false;

  ReplicatedLinear fused_wqa_wkv_{nullptr};  // merged hidden->{wq_a,wkv,
                                             // compressor.wkv/wgate,
                                             // indexer.weights_proj,
                                             // indexer.compressor.wkv/wgate}
  std::vector<int64_t> hidden_proj_sizes_;   // per-segment output dims, drives
                                             // both split() and checkpoint cat
  RMSNorm q_norm_{nullptr};
  ColumnParallelLinear wq_b_{nullptr};
  RMSNorm kv_norm_{nullptr};
  ColumnParallelLinear wo_a_{nullptr};
  RowParallelLinear wo_b_{nullptr};
  Compressor compressor_{nullptr};
  DeepseekV4Indexer indexer_{nullptr};
  torch::Tensor attn_sink_;
  DSACacheMapping cache_mapping_;
};

TORCH_MODULE(DeepseekV4Attention);

}  // namespace layer
}  // namespace xllm
