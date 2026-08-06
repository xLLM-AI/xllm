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

#include <map>
#include <vector>

#include "core/layers/npu/npu_base_layer.h"

namespace xllm {
namespace layer {

class Qwen3DecoderLoader : public BaseLoader {
 public:
  Qwen3DecoderLoader(uint64_t weight_count,
                     const ModelContext& context,
                     bool enableAddNorm,
                     LoadMode mode = LoadMode::kEager);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  bool down_proj_quantized() const { return down_proj_quantized_; }
  bool o_proj_quantized() const { return o_proj_quantized_; }

 protected:
  void merge_host_at_weights() override;

  torch::Tensor at_placeholder_;
  bool enableAddNorm_;
  bool down_proj_quantized_ = false;
  bool o_proj_quantized_ = true;  // default true for W8A8 models
  int rank_id_;
};

}  // namespace layer
}  // namespace xllm
