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

#include <functional>

#include "common/dense_mlp.h"
#include "common/oxygen_vision_attention.h"
#include "common/rms_norm.h"
#include "framework/model/model_args.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class OxygenVisionLayerImpl : public torch::nn::Module {
 public:
  OxygenVisionLayerImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec,
                        int node_id);

 private:
  OxygenVisionAttention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm norm1_{nullptr};
  RMSNorm norm2_{nullptr};
};
TORCH_MODULE(OxygenVisionLayer);

}  // namespace layer
}  // namespace xllm
