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

// Keep the appended MTP layer's checkpoint index; vocabulary weights are
// shared with the target.
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
              CHECK_EQ(args.n_layers(), 1);
              CHECK_EQ(args.num_nextn_predict_layers(), 1);
              return context;
            }()) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    const std::vector<std::string> prefixes{"model.language_model.", prefix};
    for (const auto& state_dict : loader->get_state_dicts()) {
      // The appended layer lives inside the target checkpoint, so try the
      // multimodal language-model root before the plain model root.
      StateDict model_state = state_dict->get_dict_with_prefix(prefixes);
      this->model_->load_state_dict(model_state);
    }
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

const bool glm5_next_mtp_capabilities_registered = []() {
  MtpModelCapabilities capabilities;
  capabilities.native_draft_requires_accepted_span_replay = true;
  capabilities.python_draft_owns_embedding_and_lm_head = true;
  ModelRegistry::register_mtp_capabilities("glm5_next_mtp", capabilities);
  return true;
}();

}  // namespace xllm::mlu::model
