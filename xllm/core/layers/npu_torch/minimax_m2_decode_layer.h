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

#include <string>

#include "core/framework/model_context.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu_torch/fused_moe.h"
#include "minimax_m2_attention.h"

namespace xllm {
namespace layer {

class MiniMaxM2DecoderLayerImpl : public torch::nn::Module {
 public:
  MiniMaxM2DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

 private:
  MiniMaxM2Attention attention_{nullptr};
  layer::FusedMoE moe_{nullptr};
  layer::RMSNorm input_norm_{nullptr};
  layer::RMSNorm post_norm_{nullptr};
};
TORCH_MODULE(MiniMaxM2DecoderLayer);

}  // namespace layer
}  // namespace xllm
