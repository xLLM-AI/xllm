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

#if defined(USE_MLU)
#include "layers/mlu/fused_moe.h"
#elif defined(USE_NPU)
#include "npu_torch/fused_moe.h"
#elif defined(USE_ILU)
#include "layers/ilu/fused_moe.h"
#elif defined(USE_CUDA)
#include "layers/cuda/fused_moe.h"
#elif defined(USE_DCU)
#include "layers/dcu/fused_moe.h"
#else
#include "layers/common/fused_moe.h"
#endif
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "minimax_m2_attention.h"

namespace xllm {
namespace layer {

class MiniMaxM2DecoderLayerImpl : public torch::nn::Module {
 public:
  MiniMaxM2DecoderLayerImpl() = default;
  MiniMaxM2DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);

 private:
  MiniMaxM2Attention attention_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  FusedMoE moe_{nullptr};
};
TORCH_MODULE(MiniMaxM2DecoderLayer);

}  // namespace layer
}  // namespace xllm
