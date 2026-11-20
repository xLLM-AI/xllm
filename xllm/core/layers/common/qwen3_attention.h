/* Copyright 2026 The xLLM Authors.

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

#include "core/layers/common/qwen2_attention.h"
#include "core/layers/common/rms_norm.h"

namespace xllm {
namespace layer {

class Qwen3AttentionImpl final : public Qwen2AttentionImpl {
 public:
  explicit Qwen3AttentionImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const override;

 private:
  void apply_qk_transform(torch::Tensor& qkv,
                          torch::Tensor& q,
                          torch::Tensor& k,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata) override;

  RMSNorm q_norm_{nullptr};
  RMSNorm k_norm_{nullptr};
#if defined(USE_CUDA) || defined(USE_DCU)
  bool can_use_fused_qk_norm_rope_ = false;
#endif
};
TORCH_MODULE(Qwen3Attention);

}  // namespace layer
}  // namespace xllm
