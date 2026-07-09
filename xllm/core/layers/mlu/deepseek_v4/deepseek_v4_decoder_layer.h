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

#include <cstdint>
#include <optional>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"

namespace xllm {
namespace layer {

class DeepseekV4DecoderLayerImpl final : public torch::nn::Module {
 public:
  DeepseekV4DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  void set_cache_mapping(const DSACacheMapping& mapping);

  torch::Tensor forward(
      torch::Tensor& x,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

 private:
  std::optional<torch::Tensor> route_input_ids(
      const torch::Tensor& ffn_input,
      const std::optional<torch::Tensor>& input_ids) const;

  int32_t layer_id_ = 0;
  bool use_hash_ = false;

  DeepseekV4HCPre attn_hc_pre_{nullptr};
  DeepseekV4HCPre ffn_hc_pre_{nullptr};
  DeepseekV4HCPost hc_post_{nullptr};
  RMSNorm attn_norm_{nullptr};
  RMSNorm ffn_norm_{nullptr};
  DeepseekV4Attention attention_{nullptr};
  DeepseekV4SparseMoEBlock sparse_moe_{nullptr};
};

TORCH_MODULE(DeepseekV4DecoderLayer);

}  // namespace layer
}  // namespace xllm
