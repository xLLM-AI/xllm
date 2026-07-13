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
#include <string>
#include <tuple>
#include <utility>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm_gated.h"

namespace xllm {
namespace layer {

class Qwen3GatedDeltaNetBaseImpl : public torch::nn::Module {
 public:
  Qwen3GatedDeltaNetBaseImpl() = default;
  Qwen3GatedDeltaNetBaseImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 protected:
  virtual std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual std::optional<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
  project_prefill_split_inputs(const torch::Tensor& hidden_states,
                               const AttentionMetadata& attn_metadata) {
    return std::nullopt;
  }
  virtual bool use_fla_ssm_state_layout() const { return false; }

  void load_common_state_dict(const StateDict& state_dict);
  void verify_common_loaded_weights(const std::string& prefix) const;

  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;

  std::pair<torch::Tensor, torch::Tensor> project_padded_inputs(
      const torch::Tensor& hidden_states,
      const AttentionMetadata& attn_metadata);

  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata,
                                   const torch::Tensor& padded_qkvz) const;

  torch::Tensor reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata,
                                      const torch::Tensor& qkvz) const;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_mixed_qkv(
      torch::Tensor& mixed_qkv) const;

  int64_t num_k_heads_ = 0;
  int64_t num_v_heads_ = 0;
  int64_t head_k_dim_ = 0;
  int64_t head_v_dim_ = 0;
  int64_t k_size_ = 0;
  int64_t v_size_ = 0;
  int64_t tp_size_ = 1;
  int64_t rank_ = 0;
  int32_t conv_kernel_size_ = 0;

  ColumnParallelLinear conv1d_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RmsNormGated norm_{nullptr};

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};

}  // namespace layer
}  // namespace xllm
