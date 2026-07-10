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
#include <utility>

#include "layers/dcu/qwen3_gated_delta_net_base.h"

namespace xllm {
namespace layer {

class Qwen3_5GatedDeltaNetImpl : public Qwen3GatedDeltaNetBaseImpl {
 public:
  Qwen3_5GatedDeltaNetImpl() = default;
  Qwen3_5GatedDeltaNetImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 protected:
  // Stubs: not used by DCU forward (uses raw projections)
  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor&) override {
    return {};
  }
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor&) override {
    return {};
  }

 private:
  void load_projection_state_dict(const StateDict& state_dict);
  void verify_projection_weights(const std::string& prefix) const;

  torch::Tensor merge_qkvz_from_split_activations(const torch::Tensor& qkv,
                                                  const torch::Tensor& z) const;
  torch::Tensor merge_ba_from_split_activations(const torch::Tensor& b,
                                                const torch::Tensor& a) const;

  ColumnParallelLinear in_proj_qkv_{nullptr};
  ColumnParallelLinear in_proj_z_{nullptr};
  ColumnParallelLinear in_proj_b_{nullptr};
  ColumnParallelLinear in_proj_a_{nullptr};
};
TORCH_MODULE(Qwen3_5GatedDeltaNet);

}  // namespace layer
}  // namespace xllm
