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

#include <torch/torch.h>

#include <string>
#include <tuple>
#include <utility>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/mlu/chunk_gated_delta_rule.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm_gated.h"

namespace xllm {
namespace layer {

class Qwen3_5GatedDeltaNetImpl : public torch::nn::Module {
 public:
  Qwen3_5GatedDeltaNetImpl() = default;
  Qwen3_5GatedDeltaNetImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;

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
  ColumnParallelLinear in_proj_qkv_{nullptr};
  ColumnParallelLinear in_proj_z_{nullptr};
  ColumnParallelLinear in_proj_b_{nullptr};
  ColumnParallelLinear in_proj_a_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RmsNormGated norm_{nullptr};
  xllm::kernel::mlu::ChunkGatedDeltaRule chunk_gated_delta_rule_{nullptr};

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};

TORCH_MODULE(Qwen3_5GatedDeltaNet);

}  // namespace layer
}  // namespace xllm