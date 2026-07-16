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

#include <cstdint>
#include <optional>
#include <tuple>

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/dsa_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"

namespace xllm {
namespace layer {

class CompressorImpl : public torch::nn::Module {
 public:
  CompressorImpl() = default;

  CompressorImpl(
      int64_t compress_ratio,
      int64_t hidden_dim,
      int64_t head_dim,
      int64_t rope_head_dim,
      bool rotate,
      double norm_eps,
      const torch::TensorOptions& options =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU),
      const ModelArgs& args = ModelArgs{},
      const QuantArgs& quant_args = QuantArgs{});

  torch::Tensor forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      torch::Tensor& kv_cache,
      const torch::Tensor& slot_mapping,
      torch::Tensor& state_cache,
      const torch::Tensor& state_block_table,
      const torch::Tensor& compressed_sin_table,
      const torch::Tensor& compressed_cos_table,
      std::optional<torch::Tensor> projected_kv = std::nullopt,
      std::optional<torch::Tensor> projected_score = std::nullopt);

  torch::Tensor forward_decode(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      torch::Tensor& kv_cache,
      const torch::Tensor& slot_mapping,
      torch::Tensor& state_cache,
      const torch::Tensor& state_block_table,
      const torch::Tensor& compressed_sin_table,
      const torch::Tensor& compressed_cos_table,
      std::optional<torch::Tensor> projected_kv = std::nullopt,
      std::optional<torch::Tensor> projected_score = std::nullopt);

  torch::Tensor forward_prefill(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& hidden_states,
      torch::Tensor& kv_cache,
      const torch::Tensor& slot_mapping,
      torch::Tensor& state_cache,
      const torch::Tensor& state_block_table,
      const torch::Tensor& compressed_sin_table,
      const torch::Tensor& compressed_cos_table,
      std::optional<torch::Tensor> projected_kv = std::nullopt,
      std::optional<torch::Tensor> projected_score = std::nullopt);

  void load_state_dict(const StateDict& state_dict,
                       bool skip_proj_weights = false);

 private:
  ReplicatedLinear wkv_{nullptr};
  ReplicatedLinear wgate_{nullptr};
  RMSNorm norm_{nullptr};
  DEFINE_WEIGHT(ape);
  torch::Tensor hadamard_matrix_;

  int64_t compress_ratio_ = 0;
  int64_t hidden_dim_ = 0;
  int64_t head_dim_ = 0;
  int64_t rope_head_dim_ = 0;
  int64_t compress_len_ = 0;
  bool rotate_ = false;
  bool overlap_ = false;
  double eps_ = 1e-6;
  int64_t coff_ = 1;
};

TORCH_MODULE(Compressor);

}  // namespace layer
}  // namespace xllm
