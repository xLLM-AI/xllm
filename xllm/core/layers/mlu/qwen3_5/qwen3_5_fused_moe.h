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

#include "layers/mlu/fused_moe.h"

namespace xllm {
namespace layer {

class Qwen3_5FusedMoEImpl final : public FusedMoEImpl {
 public:
  Qwen3_5FusedMoEImpl() = default;

  Qwen3_5FusedMoEImpl(const ModelArgs& model_args,
                      const FusedMoEArgs& moe_args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict) override;

 protected:
  void final_comm_allreduce(torch::Tensor& final_hidden_states,
                            const torch::Tensor& hidden_states,
                            torch::Tensor& shared_expert_output) override;

 private:
  void load_experts(const StateDict& state_dict);
  torch::nn::Linear shared_expert_gate_{nullptr};
};

TORCH_MODULE(Qwen3_5FusedMoE);
}  // namespace layer
}  // namespace xllm
