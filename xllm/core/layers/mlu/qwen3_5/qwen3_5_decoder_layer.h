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

#include <optional>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/mlu/qwen3_5/qwen3_5_attention.h"
#include "layers/mlu/qwen3_5/qwen3_5_fused_moe.h"
#include "layers/mlu/qwen3_5/qwen3_5_gated_delta_net.h"
#include "layers/mlu/qwen3_5/qwen3_next_rms_norm.h"

namespace xllm {
namespace layer {

class Qwen3_5DecoderLayerImpl final : public torch::nn::Module {
 public:
  Qwen3_5DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  std::tuple<torch::Tensor, std::optional<torch::Tensor>> apply_norm(
      Qwen3NextRMSNormMlu& norm,
      torch::Tensor& input,
      std::optional<torch::Tensor>& residual);

  torch::Tensor run_moe(torch::Tensor x, const ModelInputParams& input_params);

  std::string layer_type_;
  Qwen3_5Attention full_attention_{nullptr};
  Qwen3_5GatedDeltaNet linear_attention_{nullptr};
  DenseMLP mlp_{nullptr};
  Qwen3_5FusedMoE moe_mlp_{nullptr};
  Qwen3NextRMSNormMlu input_norm_{nullptr};
  Qwen3NextRMSNormMlu post_norm_{nullptr};
  ParallelArgs parallel_args_;
  bool enable_deep_ep_ = false;
};

TORCH_MODULE(Qwen3_5DecoderLayer);

}  // namespace layer
}  // namespace xllm
