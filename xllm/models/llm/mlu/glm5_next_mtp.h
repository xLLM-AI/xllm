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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/deepseek_mtp.h"
#include "models/llm/mlu/glm5_next_graph.h"
#include "models/model_registry.h"

namespace xllm::mlu::model {

class Glm5NextMtpForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekMtpModel> {
 public:
  explicit Glm5NextMtpForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekMtpModel>(
            [&context]() -> const ModelContext& {
              const auto& args = context.get_model_args();
              CHECK_GE(args.mtp_start_layer_idx(), 0)
                  << "Native MLU GLM5 MTP requires a full target checkpoint "
                     "with an appended MTP layer; exported draft checkpoints "
                     "are not supported.";
              return context;
            }()) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    const auto mtp_layer_idx = loader->model_args().mtp_start_layer_idx();
    const std::string mtp_layer_prefix =
        "layers." + std::to_string(mtp_layer_idx) + ".";
    const std::string head_suffix =
        tie_word_embeddings_ ? "embed_tokens." : "shared_head.head.";
    const std::vector<std::string> head_prefixes{
        "model.language_model." + mtp_layer_prefix + head_suffix,
        prefix + mtp_layer_prefix + head_suffix};
    const std::vector<std::string> prefixes{"model.language_model.", prefix};
    for (const auto& state_dict : loader->get_state_dicts()) {
      lm_head_->load_state_dict(
          state_dict->get_dict_with_prefix(head_prefixes));
      StateDict model_state = state_dict->get_dict_with_prefix(prefixes);
      this->model_->load_state_dict(model_state);
      this->model_->get_word_embedding()->load_state_dict(
          model_state.get_dict_with_prefix(mtp_layer_prefix + "embed_tokens."));
    }
    CHECK(this->model_->get_word_embedding()->is_weight_loaded())
        << "Native MLU GLM5 MTP checkpoint is missing "
           "appended-layer embed_tokens.weight";
    CHECK(lm_head_->is_weight_loaded())
        << "Native MLU GLM5 MTP checkpoint is missing appended-layer "
           "shared_head.head (lm_head) weights";
    this->model_->verify_loaded_weights();
  }

  bool requires_graph_forward_metadata() override { return true; }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    return std::make_unique<Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelParamsView& params) override {
    auto* graph_state = dynamic_cast<Glm5NextGraphMetadataState*>(state);
    CHECK(graph_state != nullptr) << "GLM5 requires per-graph metadata state";
    graph_state->prepare(positions, params);
  }
};
TORCH_MODULE(Glm5NextMtpForCausalLM);

REGISTER_CAUSAL_MODEL(glm5_next_mtp, Glm5NextMtpForCausalLM);

}  // namespace xllm::mlu::model
