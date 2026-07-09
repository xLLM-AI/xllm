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

#include "layers/musa/qwen3_next_gated_delta_net.h"
#include "layers/musa/qwen3_next_hybrid_decoder_layer_base.h"

namespace xllm {
namespace layer {

class Qwen3NextDecoderLayerImpl : public Qwen3HybridDecoderLayerImplBase {
 public:
  explicit Qwen3NextDecoderLayerImpl(const ModelContext& context,
                                     int32_t layer_id);

 protected:
  Qwen3NextDecoderLayerImpl(
      const ModelContext& context,
      int32_t layer_id,
      std::shared_ptr<Qwen3GatedDeltaNetBaseImpl> linear_attention_module);
};
TORCH_MODULE(Qwen3NextDecoderLayer);

}  // namespace layer
}  // namespace xllm
