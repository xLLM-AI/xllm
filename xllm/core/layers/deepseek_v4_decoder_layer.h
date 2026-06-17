/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <optional>
#include <tuple>

#include "common/attention_metadata.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "npu_torch/deepseek_sparse_attention.h"
#include "npu_torch/deepseek_v4_gate.h"
#include "npu_torch/fused_moe.h"

namespace xllm {
namespace layer {

class DeepseekV4DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DeepseekV4DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  torch::Tensor forward(
      torch::Tensor& x,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_pre(
      const torch::Tensor& x,
      const torch::Tensor& hc_fn,
      const torch::Tensor& hc_scale,
      const torch::Tensor& hc_base);

  torch::Tensor hc_post(const torch::Tensor& x,
                        const torch::Tensor& residual,
                        const torch::Tensor& post,
                        const torch::Tensor& comb);

  DSAttention attention_{nullptr};
  FusedMoE moe_mlp_{nullptr};
  DeepseekV4Gate gate_{nullptr};
  RMSNorm attn_norm_{nullptr};
  RMSNorm ffn_norm_{nullptr};

  int64_t hc_mult_ = 1;
  int64_t hc_sinkhorn_iters_ = 0;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  DEFINE_WEIGHT(hc_attn_fn);
  DEFINE_WEIGHT(hc_ffn_fn);
  DEFINE_WEIGHT(hc_attn_base);
  DEFINE_WEIGHT(hc_ffn_base);
  DEFINE_WEIGHT(hc_attn_scale);
  DEFINE_WEIGHT(hc_ffn_scale);
};
TORCH_MODULE(DeepseekV4DecoderLayer);

}  // namespace layer
}  // namespace xllm
