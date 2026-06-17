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

#include "framework/model_context.h"
#include "qwen2_vision_attention.h"

namespace xllm {
namespace layer {

class OxygenVisionAttentionImpl : public Qwen2VisionAttentionImpl {
 public:
  OxygenVisionAttentionImpl() = default;
  OxygenVisionAttentionImpl(const ModelContext& context);

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& m_cos_pos,
                        torch::Tensor& m_sin_pos,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec) override;
};
TORCH_MODULE(OxygenVisionAttention);

}  // namespace layer
}  // namespace xllm
