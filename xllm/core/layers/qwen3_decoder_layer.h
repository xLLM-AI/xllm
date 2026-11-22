/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <optional>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/qwen3_attention.h"
#include "core/layers/common/rms_norm.h"

namespace xllm {

namespace layer {

class Qwen3DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen3DecoderLayerImpl(const ModelContext& context,
                                 int32_t layer_id = -1);
  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;
  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

 protected:
  Qwen3Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
};
TORCH_MODULE(Qwen3DecoderLayer);

}  // namespace layer
}  // namespace xllm
