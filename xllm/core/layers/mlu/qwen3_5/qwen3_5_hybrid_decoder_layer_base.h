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

#include <memory>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

// MLU-side abstract interface for a Qwen3 hybrid decoder layer. Mirrors the
// NPU `Qwen3HybridDecoderLayerModule` contract so the shared
// `qwen3_5_mtp.h` draft model can hold layers of either backend behind the
// same pointer type. The MLU attention/GDN path does not consume the
// `mrope_cos_sin` split, so concrete layers accept the argument and discard
// it; the trailing default keeps the target path's 6-arg call valid.
class Qwen3HybridDecoderLayerModule : public torch::nn::Module {
 public:
  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;
  virtual torch::Tensor forward(torch::Tensor& x,
                                std::optional<torch::Tensor>& residual,
                                torch::Tensor& positions,
                                const AttentionMetadata& attn_metadata,
                                KVCache& kv_cache,
                                const ModelInputParams& input_params,
                                const torch::Tensor& mrope_cos_sin = {}) = 0;
  virtual torch::Tensor build_mrope_cos_sin(
      const torch::Tensor& positions) const {
    return {};
  }
};

using Qwen3HybridDecoderLayerModulePtr =
    std::shared_ptr<Qwen3HybridDecoderLayerModule>;

}  // namespace layer
}  // namespace xllm
