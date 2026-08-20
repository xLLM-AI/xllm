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
#include <tuple>

#include "common/dense_mlp.h"
#include "common/qwen2_attention.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class Qwen2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen2DecoderLayerImpl(const ModelContext& context,
                                 int32_t layer_id = -1);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  Qwen2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  ParallelArgs parallel_args_;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> apply_norm(
      RMSNorm& norm,
      torch::Tensor& input,
      std::optional<torch::Tensor>& residual,
      const std::optional<torch::Tensor>& fp8_scale);
};
TORCH_MODULE(Qwen2DecoderLayer);

}  // namespace layer
}  // namespace xllm
