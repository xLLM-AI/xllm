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

#include <array>
#include <optional>

#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/npu_torch/fused_moe.h"
#include "core/layers/npu_torch/minimax_m3_attention.h"

namespace xllm {
namespace layer {

class MiniMaxM3DecoderLayerImpl final : public torch::nn::Module {
 public:
  MiniMaxM3DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

 private:
  MiniMaxM3Attention attention_{nullptr};
  layer::DenseMLP mlp_{nullptr};
  layer::FusedMoE moe_{nullptr};
  layer::Qwen3NextRMSNorm input_norm_{nullptr};
  layer::Qwen3NextRMSNorm post_norm_{nullptr};
  std::array<int64_t, 2> weight_block_size_ = {1, 32};
  bool is_moe_layer_ = false;
  bool enable_weight_dequant_ = false;
  bool use_e8m0_scale_ = false;
};
TORCH_MODULE(MiniMaxM3DecoderLayer);

}  // namespace layer
}  // namespace xllm
