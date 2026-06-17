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

#include "qwen3_gated_delta_net_base.h"

namespace xllm {
namespace layer {

class Qwen3NextGatedDeltaNetImpl : public Qwen3GatedDeltaNetBaseImpl {
 public:
  Qwen3NextGatedDeltaNetImpl() = default;
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

 protected:
  Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options,
                             bool init_projections);

  std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) override;
  std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) override;

  virtual void load_projection_state_dict(const StateDict& state_dict);
  virtual void verify_projection_weights(const std::string& prefix) const;

  void init_next_projections(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

 private:
  ColumnParallelLinear qkvz_proj_{nullptr};
  ColumnParallelLinear ba_proj_{nullptr};
};
TORCH_MODULE(Qwen3NextGatedDeltaNet);

}  // namespace layer
}  // namespace xllm
