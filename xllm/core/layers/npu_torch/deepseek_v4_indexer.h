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

#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/npu_torch/compressor.h"

namespace xllm {
namespace layer {

class DeepseekV4IndexerImpl : public torch::nn::Module {
 public:
  DeepseekV4IndexerImpl() = default;

  DeepseekV4IndexerImpl(int64_t dim,
                        int64_t index_n_heads,
                        int64_t index_head_dim,
                        int64_t rope_head_dim,
                        int64_t index_topk,
                        int64_t q_lora_rank,
                        int64_t compress_ratio,
                        double norm_eps,
                        const QuantArgs& quant_args,
                        const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x,
                                                   const torch::Tensor& qr);

  torch::Tensor select_qli(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const std::optional<torch::Tensor>& qr_pertoken_scale,
      torch::Tensor& index_cache,
      torch::Tensor* quant_index_cache,
      const AttentionMetadata& attn_metadata,
      const std::optional<torch::Tensor>& cos = std::nullopt,
      const std::optional<torch::Tensor>& sin = std::nullopt,
      const std::optional<torch::Tensor>& compressed_cos = std::nullopt,
      const std::optional<torch::Tensor>& compressed_sin = std::nullopt,
      const std::optional<torch::Tensor>& actual_seq_lengths_query =
          std::nullopt,
      const std::optional<torch::Tensor>& actual_seq_lengths_key = std::nullopt,
      const std::optional<torch::Tensor>& qli_metadata = std::nullopt,
      bool with_prefill = false,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_states = nullptr,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables =
          nullptr,
      // Under prefill CP, `x` serves two conflicting roles: compress_kv(x)
      // writes the index cache and must see all tokens, while build_weights(x)
      // emits one weight row per query and must match this rank's query shard.
      // x_kv carries the global-ordered hidden; undefined (the default) makes
      // both paths use `x`, keeping non-CP callers unchanged. Appended last so
      // existing positional calls keep binding to the same parameters.
      const torch::Tensor& x_kv = torch::Tensor(),
      // Cumulative query lengths describing x_kv's row axis: (batch+1,) with a
      // leading 0, the same layout the compressor expects from
      // actual_seq_lengths_query. Must be supplied whenever x_kv is, because
      // actual_seq_lengths_query has been localized to this rank's shard and
      // would under-count x_kv's rows. Do not substitute
      // actual_seq_lengths_key: that one is per-sequence, not cumulative.
      const std::optional<torch::Tensor>& x_kv_cu_seq_lens = std::nullopt);

  torch::Tensor select_qli(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const std::optional<torch::Tensor>& qr_pertoken_scale,
      torch::Tensor& index_cache,
      const AttentionMetadata& attn_metadata,
      const std::optional<torch::Tensor>& cos = std::nullopt,
      const std::optional<torch::Tensor>& sin = std::nullopt,
      const std::optional<torch::Tensor>& compressed_cos = std::nullopt,
      const std::optional<torch::Tensor>& compressed_sin = std::nullopt,
      const std::optional<torch::Tensor>& actual_seq_lengths_query =
          std::nullopt,
      const std::optional<torch::Tensor>& actual_seq_lengths_key = std::nullopt,
      const std::optional<torch::Tensor>& qli_metadata = std::nullopt,
      bool with_prefill = false,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_states = nullptr,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables =
          nullptr,
      // Forwarded to the full overload; see its declaration above.
      const torch::Tensor& x_kv = torch::Tensor(),
      const std::optional<torch::Tensor>& x_kv_cu_seq_lens = std::nullopt);

  torch::Tensor build_query(const torch::Tensor& qr);
  torch::Tensor build_query(
      const torch::Tensor& qr,
      const std::optional<torch::Tensor>& qr_pertoken_scale);

  torch::Tensor build_weights(const torch::Tensor& x);

  torch::Tensor compress_kv(
      const torch::Tensor& x,
      const AttentionMetadata& attn_metadata,
      const std::optional<torch::Tensor>& compressed_cos,
      const std::optional<torch::Tensor>& compressed_sin,
      const std::optional<torch::Tensor>& actual_seq_lengths_query,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_states,
      std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables);

  void load_state_dict(const StateDict& state_dict);

  int64_t dim() const { return dim_; }
  int64_t n_heads() const { return n_heads_; }
  int64_t head_dim() const { return head_dim_; }
  int64_t rope_head_dim() const { return rope_head_dim_; }
  int64_t index_topk() const { return index_topk_; }
  int64_t q_lora_rank() const { return q_lora_rank_; }
  int64_t compress_ratio() const { return compress_ratio_; }
  double softmax_scale() const { return softmax_scale_; }

  ReplicatedLinear wq_b() const { return wq_b_; }
  ReplicatedLinear weights_proj() const { return weights_proj_; }

 private:
  int64_t dim_ = 0;
  int64_t n_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t index_topk_ = 0;
  int64_t q_lora_rank_ = 0;
  int64_t compress_ratio_ = 1;

  double softmax_scale_ = 1.0;
  double indexer_softmax_mul_head_dim_sqrt_ = 1.0;
  double hadamard_scale_ = 1.0;
  int64_t index_head_dim_padded_ = 1;

  torch::Tensor hadamard_matrix_;

  ReplicatedLinear wq_b_{nullptr};
  ReplicatedLinear weights_proj_{nullptr};
  Compressor compressor_{nullptr};
};

TORCH_MODULE(DeepseekV4Indexer);

}  // namespace layer
}  // namespace xllm
