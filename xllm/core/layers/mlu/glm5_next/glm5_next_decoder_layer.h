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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/dsa_topk_share_plan.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"
#include "layers/mlu/glm5_next/glm5_next_kda.h"
#include "layers/mlu/hyper_connection.h"

namespace xllm {
namespace layer {

class DsaTopkRelay;

class Glm5NextDecoderLayerImpl final : public torch::nn::Module {
 public:
  Glm5NextDecoderLayerImpl(const ModelContext& context,
                           int32_t layer_id,
                           const DsaTopkSharePlan& topk_share_plan);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  torch::Tensor forward(torch::Tensor& hidden_states,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        DsaTopkRelay* topk_relay = nullptr);

 private:
  std::string weight_prefix_;
  DsaTopkShareDecision topk_share_decision_;
  ProcessGroup* tp_group_ = nullptr;

  MHCPre attn_hc_pre_{nullptr};
  MHCPre ffn_hc_pre_{nullptr};
  MHCPost hc_post_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  Glm5NextKDA kda_{nullptr};
  DeepseekV2Attention dsa_{nullptr};
  DenseMLP dense_mlp_{nullptr};
  DeepseekV4SparseMoEBlock sparse_moe_{nullptr};
};

TORCH_MODULE(Glm5NextDecoderLayer);

}  // namespace layer
}  // namespace xllm
