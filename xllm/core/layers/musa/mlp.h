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

#include "framework/state_dict/state_dict.h"
#include "layers/musa/layer_base.h"

namespace xllm {
namespace layer {
class MusaMLPImpl : public MUSALayerBaseImpl {
 public:
  MusaMLPImpl(int32_t hidden_size,
              int32_t intermediate_size,
              bool is_gated,
              bool has_bias,
              const std::string& hidden_act,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options,
              float rms_eps);
  ~MusaMLPImpl() {};

  torch::Tensor forward(torch::Tensor& input,
                        ForwardParams& fwd_params) override;

  void load_state_dict(StateDict const& state_dict) override;

 private:
  // todo: add member
  int32_t hidden_size_;
  int32_t intermediate_size_;
  float rms_eps;
  constexpr static int32_t weight_num_ = 4;
};
TORCH_MODULE(MusaMLP);

}  // namespace layer
}  // namespace xllm