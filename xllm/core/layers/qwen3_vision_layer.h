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

#include <string>

#include "core/layers/qwen2_5_vision_layer.h"

namespace xllm {
namespace layer {

class Qwen3_VisionLayerImpl final : public Qwen2_5_VisionLayerImpl {
 public:
  explicit Qwen3_VisionLayerImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 private:
  bool norm1_weight_loaded_ = false;
  bool norm1_bias_loaded_ = false;
  bool norm2_weight_loaded_ = false;
  bool norm2_bias_loaded_ = false;
  bool attention_qkv_weight_loaded_ = false;
  bool attention_qkv_bias_loaded_ = false;
  bool attention_proj_weight_loaded_ = false;
  bool attention_proj_bias_loaded_ = false;
  bool mlp_fc1_weight_loaded_ = false;
  bool mlp_fc1_bias_loaded_ = false;
  bool mlp_fc2_weight_loaded_ = false;
  bool mlp_fc2_bias_loaded_ = false;
};
TORCH_MODULE(Qwen3_VisionLayer);

}  // namespace layer
}  // namespace xllm
