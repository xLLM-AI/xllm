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

#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/layers/mlu/dsa_topk_relay.h"
#include "models/llm/glm5.h"
#include "platform/platform.h"

// GLM5.2 shares model_type "glm_moe_dsa" with GLM5.0/5.1 and is told apart by
// the resolved indexer top-k share plan. GLM5.0/5.1 configs carry no reuse, so
// every layer stays non-sharing and the relay below bypasses itself entirely --
// Glm52 then behaves exactly like Glm5. glm_moe_dsa is therefore registered
// once here on Glm52ForCausalLM, while REGISTER_MODEL_ARGS(glm_moe_dsa, ...)
// stays in glm5.h.

namespace xllm {

class Glm52ModelImpl : public Glm5ModelImpl {
 public:
  explicit Glm52ModelImpl(const ModelContext& context)
      : Glm5ModelImpl(context, create_decoder_layer_factory(context)),
        dsa_topk_share_plan_(context.get_model_args()) {
    const bool enable_prefill_cp = context.get_parallel_args().cp_size() > 1 &&
                                   Platform::uses_model_cp_partition();
    if (layer::cp_conflicts_with_dsa_topk_share(enable_prefill_cp,
                                                dsa_topk_share_plan_)) {
      LOG(FATAL) << "Prefill CP is not supported together with GLM5.2 "
                    "DSA cross-layer top-k sharing. Disable one of them: unset "
                    "cp_size, or run a model whose DSA top-k plan contains "
                    "no Shared layers.";
    }
  }

 protected:
  // The relay is reset at layer 0 so its state remains forward-scoped. Decoder
  // layers own role interpretation and the attention transfer protocol.
  torch::Tensor forward_decoder_layer(
      size_t layer_id,
      layer::DeepseekV2DecoderLayer& layer,
      torch::Tensor& hidden_states,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      layer::AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params) override {
    if (layer_id == 0) {
      topk_relay_.reset();
    }
    return layer(hidden_states,
                 residual,
                 positions,
                 attn_metadata,
                 kv_cache,
                 input_params,
                 /*input_ids=*/std::nullopt,
                 &topk_relay_);
  }

 private:
  static DecoderLayerFactory create_decoder_layer_factory(
      const ModelContext& context) {
    const layer::DsaTopkSharePlan topk_share_plan(context.get_model_args());
    return
        [topk_share_plan](const ModelContext& layer_context, int32_t layer_id) {
          return layer::DeepseekV2DecoderLayer(
              layer_context, layer_id, topk_share_plan);
        };
  }

  layer::DsaTopkSharePlan dsa_topk_share_plan_;
  layer::DsaTopkRelay topk_relay_;
};
TORCH_MODULE(Glm52Model);

class Glm52ForCausalLMImpl : public LlmForCausalLMImplBase<Glm52Model> {
 public:
  explicit Glm52ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm52Model>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    LlmForCausalLMImplBase<Glm52Model>::load_model(std::move(loader), prefix);
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm52ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm_moe_dsa, Glm52ForCausalLM);

}  // namespace xllm
