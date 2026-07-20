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

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/mlu/deepseek_v4/compressor.h"

namespace xllm {
namespace layer {

struct DeepseekV4IndexerCacheRefs {
  torch::Tensor index_block_table;
  torch::Tensor index_slot_mapping;
  // Single shared block table for the merged compress_index_state (kv and
  // score share the same SLIDING_WINDOW block table).
  torch::Tensor index_state_block_table;
};

class DeepseekV4IndexerImpl : public torch::nn::Module {
 public:
  DeepseekV4IndexerImpl() = default;

  DeepseekV4IndexerImpl(
      int64_t dim,
      int64_t index_n_heads,
      int64_t index_head_dim,
      int64_t rope_head_dim,
      int64_t index_topk,
      int64_t q_lora_rank,
      double norm_eps,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU),
      const ModelArgs& args = ModelArgs{},
      const QuantArgs& quant_args = QuantArgs{});

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      torch::Tensor& index_cache,
      torch::Tensor& compress_index_state,
      const AttentionMetadata& attn_metadata,
      const DeepseekV4IndexerCacheRefs& cache_refs,
      bool is_prefill,
      const torch::Tensor& compressed_sin_table,
      const torch::Tensor& compressed_cos_table,
      std::optional<torch::Tensor> projected_weights = std::nullopt,
      std::optional<torch::Tensor> projected_kv = std::nullopt,
      std::optional<torch::Tensor> projected_score = std::nullopt);

  void load_state_dict(const StateDict& state_dict,
                       bool skip_proj_weights = false);

 private:
  torch::Tensor preprocess_q(const torch::Tensor& qr,
                             const AttentionMetadata& attn_metadata,
                             const torch::Tensor& compressed_sin_table,
                             const torch::Tensor& compressed_cos_table);

  torch::Tensor preprocess_weights(
      const torch::Tensor& x,
      std::optional<torch::Tensor> projected_weights = std::nullopt);

  torch::Tensor compress_kv(
      torch::Tensor& hidden_states,
      torch::Tensor& compress_index_state,
      const AttentionMetadata& attn_metadata,
      torch::Tensor& index_cache,
      const DeepseekV4IndexerCacheRefs& cache_refs,
      const torch::Tensor& compressed_sin_table,
      const torch::Tensor& compressed_cos_table,
      std::optional<torch::Tensor> projected_kv = std::nullopt,
      std::optional<torch::Tensor> projected_score = std::nullopt);

  std::tuple<torch::Tensor, torch::Tensor> select_topk(
      const torch::Tensor& q,
      const torch::Tensor& weights,
      const torch::Tensor& current_kv,
      torch::Tensor& index_cache,
      const AttentionMetadata& attn_metadata,
      const DeepseekV4IndexerCacheRefs& cache_refs,
      bool is_prefill);

  int64_t dim_ = 0;
  int64_t n_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t index_topk_ = 0;
  int64_t q_lora_rank_ = 0;
  float softmax_scale_ = 1.0f;

  ReplicatedLinear wq_b_{nullptr};
  ReplicatedLinear weights_proj_{nullptr};
  Compressor compressor_{nullptr};
  torch::Tensor hadamard_matrix_;
};

TORCH_MODULE(DeepseekV4Indexer);

}  // namespace layer
}  // namespace xllm
