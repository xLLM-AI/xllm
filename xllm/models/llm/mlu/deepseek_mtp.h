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

#include <string>
#include <vector>

#include "core/layers/mlu/deepseek_v2_decoder_layer_impl.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/mtp_model_base.h"
#include "models/model_registry.h"

namespace xllm::mlu::model {

class DeepseekMultiTokenPredictorLayerImpl
    : public MluMtpDecoderLayerImplBase<layer::DeepseekV2DecoderLayer> {
 public:
  DeepseekMultiTokenPredictorLayerImpl(const ModelContext& context,
                                       const int32_t layer_index)
      : MluMtpDecoderLayerImplBase<layer::DeepseekV2DecoderLayer>(context,
                                                                  layer_index) {
  }
};
TORCH_MODULE(DeepseekMultiTokenPredictorLayer);

class DeepseekMtpModelImpl
    : public MluMtpModelImplBase<DeepseekMultiTokenPredictorLayer> {
 public:
  DeepseekMtpModelImpl(const ModelContext& context)
      : MluMtpModelImplBase<DeepseekMultiTokenPredictorLayer>(context) {}
};
TORCH_MODULE(DeepseekMtpModel);

class DeepseekMtpForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekMtpModel> {
 public:
  DeepseekMtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekMtpModel>(context,
                                                 /*share_lm_head=*/true) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    // no need to load lm_head since it shares the same weights with main models
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(DeepseekMtpForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v3_mtp, DeepseekMtpForCausalLM);
REGISTER_CAUSAL_MODEL(glm_moe_dsa_mtp, DeepseekMtpForCausalLM);
REGISTER_CAUSAL_MODEL(joyai_llm_flash_mtp, DeepseekMtpForCausalLM);

}  // namespace xllm::mlu::model
