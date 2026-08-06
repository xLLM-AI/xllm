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

#include "qwen3_vision_layer.h"

namespace xllm {
namespace layer {

Qwen3_VisionLayerImpl::Qwen3_VisionLayerImpl(const ModelContext& context)
    : Qwen2_5_VisionLayerImpl(context, true) {}

void Qwen3_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"linear_fc1."}, "linear_fc2.");
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
}

}  // namespace layer
}  // namespace xllm