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

#include <optional>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/rms_norm.h"
#include "layers/dcu/deepseek_v2_attention.h"
#include "layers/dcu/fused_moe.h"

namespace xllm {
namespace layer {

// DCU DeepSeek-V2 decoder layer.
class DeepseekV2DecoderLayerImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV2DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights() const {}

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  DeepseekV2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  FusedMoE moe_mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  ParallelArgs parallel_args_;
  bool is_moe_layer_;
};

TORCH_MODULE(DeepseekV2DecoderLayer);

}  // namespace layer
}  // namespace xllm
