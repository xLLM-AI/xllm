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
#include "core/util/env_var.h"
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
              CHECK_EQ(args.n_layers(), 1);
              CHECK_EQ(args.num_nextn_predict_layers(), 1);
              return context;
            }()),
        mtp_layer_prefix_(
            "layers." +
            std::to_string(context.get_model_args().mtp_start_layer_idx()) +
            ".") {}

  static constexpr const char* kDecoderCheckpointPrefix = "";

  void load_state_dict(const StateDict& state_dict) override {
    const std::vector<std::string> prefixes{"model.language_model.",
                                            kWeightPrefix};
    StateDict model_state = state_dict.get_dict_with_prefix(prefixes);
    this->model_->load_state_dict(model_state);
    StateDict embed_tokens =
        model_state.get_dict_with_prefix(mtp_layer_prefix_ + "embed_tokens.");
    this->model_->get_word_embedding()->load_state_dict(embed_tokens);
  }

  StateDict select_output_weights(const StateDict& state_dict) {
    const std::string suffix =
        tie_word_embeddings_ ? "embed_tokens." : "shared_head.head.";
    const std::vector<std::string> prefixes{
        "model.language_model." + mtp_layer_prefix_ + suffix,
        kWeightPrefix + mtp_layer_prefix_ + suffix};
    return state_dict.get_dict_with_prefix(prefixes);
  }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelParamsView& params) {
    auto* graph_state = dynamic_cast<Glm5NextGraphMetadataState*>(state);
    CHECK(graph_state != nullptr) << "GLM5 requires per-graph metadata state";
    graph_state->prepare(positions, params);
  }

 private:
  std::string mtp_layer_prefix_;
};
TORCH_MODULE(Glm5NextMtpForCausalLM);

inline int64_t fixed_graph_capacity(const ModelArgs& args) {
  constexpr int kDefaultGraphHistoryCapacity = 32768;
  int64_t capacity = xllm::util::get_int_env("XLLM_GRAPH_INDEX_HISTORY_MAX_KV",
                                             kDefaultGraphHistoryCapacity);
  CHECK_GT(capacity, 0) << "XLLM_GRAPH_INDEX_HISTORY_MAX_KV must be positive";
  if (args.max_position_embeddings() > 0) {
    capacity = std::min(capacity, args.max_position_embeddings());
  }
  return capacity;
}

const bool glm5_next_mtp_capabilities_registered = []() {
  ModelRegistry::register_graph_kv_capacity("glm5_next", &fixed_graph_capacity);
  ModelRegistry::register_graph_kv_capacity("glm5_next_mtp",
                                            &fixed_graph_capacity);
  return true;
}();

REGISTER_CAUSAL_MODEL(glm5_next_mtp, Glm5NextMtpForCausalLM);

}  // namespace xllm::mlu::model
