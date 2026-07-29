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
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/qwen3_next_rms_norm.h"
#if !defined(USE_MUSA)
#include "layers/cuda/fused_moe.h"
#endif
#include "layers/musa/qwen3_gated_delta_net_base.h"
#include "layers/musa/qwen3_next_attention.h"

namespace xllm {
namespace layer {

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

class Qwen3HybridDecoderLayerImplBase : public Qwen3HybridDecoderLayerModule {
 public:
  explicit Qwen3HybridDecoderLayerImplBase(
      const ModelContext& context,
      int32_t layer_id,
      std::shared_ptr<Qwen3GatedDeltaNetBaseImpl> linear_attention_module);

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const override;

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        const torch::Tensor& mrope_cos_sin = {}) override;

  torch::Tensor build_mrope_cos_sin(
      const torch::Tensor& positions) const override;

 protected:
  Qwen3NextAttention attention_{nullptr};
  std::shared_ptr<Qwen3GatedDeltaNetBaseImpl> linear_attention_;

  DenseMLP mlp_{nullptr};
#if !defined(USE_MUSA)
  FusedMoE moe_mlp_{nullptr};
#endif

  Qwen3NextRMSNorm input_norm_{nullptr};
  Qwen3NextRMSNorm post_norm_{nullptr};
};

}  // namespace layer
}  // namespace xllm
