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
#include <tuple>
#include <vector>

#include "core/layers/common/rotary_embedding_util.h"
#include "models/llm/qwen3_5_mtp_base.h"
#include "models/model_registry.h"

namespace xllm {

namespace qwen3_5_mtp {

inline void configure_mtp_args(ModelArgs& args, bool is_python_model) {
  if (is_python_model) {
    return;
  }
  const int32_t mtp_layers =
      std::max<int32_t>(args.num_nextn_predict_layers(), 1);
  args.num_nextn_predict_layers(mtp_layers);
  args.n_layers(mtp_layers);
  args.layer_types(std::vector<std::string>(static_cast<size_t>(mtp_layers),
                                            "full_attention"));
  args.full_attention_interval(1);
}

}  // namespace qwen3_5_mtp

const bool qwen3_5_mtp_adapter_registered = []() {
  for (const char* name : {"qwen3_5",
                           "qwen3_5_moe",
                           "qwen3_5_text",
                           "qwen3_5_moe_text",
                           "qwen3_5_mtp",
                           "qwen3_5_moe_mtp"}) {
    ModelRegistry::register_mtp_args_adapter(name,
                                             &qwen3_5_mtp::configure_mtp_args);
  }
  return true;
}();

class Qwen3_5MtpModelImpl final : public Qwen3_5MtpModelImplBase {
 public:
  explicit Qwen3_5MtpModelImpl(const ModelContext& context)
      : Qwen3_5MtpModelImplBase(context),
        mrope_section_(context.get_model_args().rope_scaling_mrope_section()) {
    if (mrope_section_.empty()) {
      return;
    }

    const ModelArgs& args = context.get_model_args();
    const int64_t rotary_dim =
        static_cast<int64_t>(args.head_dim() * args.partial_rotary_factor());
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        rotary_dim,
        args.max_position_embeddings(),
        args.rope_theta(),
        context.get_tensor_options());
  }

 protected:
  void prepare_mrope(const torch::Tensor& positions,
                     layer::AttentionMetadata& attn_metadata) const override {
    if (mrope_section_.empty()) {
      return;
    }

    std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
        layer::rotary::apply_mrope(cos_sin_, positions, mrope_section_);
  }

 private:
  torch::Tensor cos_sin_;
  std::vector<int64_t> mrope_section_;
};

class Qwen3_5MtpForCausalLMImpl final : public Qwen3_5MtpForCausalLMImplBase {
 public:
  explicit Qwen3_5MtpForCausalLMImpl(const ModelContext& context)
      : Qwen3_5MtpForCausalLMImplBase(
            context,
            std::make_shared<Qwen3_5MtpModelImpl>(context)) {}
};
TORCH_MODULE(Qwen3_5MtpForCausalLM);

REGISTER_CAUSAL_MODEL(qwen3_5_mtp, Qwen3_5MtpForCausalLM);
REGISTER_CAUSAL_MODEL(qwen3_5_moe_mtp, Qwen3_5MtpForCausalLM);

REGISTER_MODEL_ARGS_LOADER(qwen3_5_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return qwen3_5_mtp::load_model_args(
                                 json, args, "qwen3_5_text", "qwen3_5_mtp");
                           });

REGISTER_MODEL_ARGS_LOADER(qwen3_5_moe_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return qwen3_5_mtp::load_model_args(
                                 json,
                                 args,
                                 "qwen3_5_moe_text",
                                 "qwen3_5_moe_mtp");
                           });

const bool qwen3_5_mtp_contract_registered = []() {
  MtpModelCapabilities target;
  target.replay_family = MtpReplayFamily::QWEN35;
  target.supports_expanded_replay_target = true;
  target.supports_grouped_mtp_graph = true;
  target.graph_history = MtpGraphHistoryPolicy::QWEN_FULL_ATTENTION;
  MtpModelCapabilities draft;
  draft.replay_family = MtpReplayFamily::QWEN35;
  draft.native_draft_requires_accepted_span_replay = true;
  draft.allow_tail_fallback = true;
  draft.supports_grouped_mtp_graph = true;
  draft.graph_history = MtpGraphHistoryPolicy::QWEN_FULL_ATTENTION;
  for (const char* name :
       {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}) {
    ModelRegistry::register_mtp_capabilities(name, target);
  }
  for (const char* name : {"qwen3_5_mtp", "qwen3_5_moe_mtp"}) {
    ModelRegistry::register_mtp_capabilities(name, draft);
  }
  return true;
}();

}  // namespace xllm
