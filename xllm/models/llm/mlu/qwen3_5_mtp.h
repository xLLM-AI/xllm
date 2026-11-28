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

const bool qwen3_5_mtp_metadata_registered = [] {
  const auto prepare_args = [](ModelArgs& args, bool is_python_model) {
    if (is_qwen3_5_target_model_type(args.model_type()) &&
        (!is_python_model || args.num_nextn_predict_layers() != 0)) {
      const bool is_moe = args.model_type() == "qwen3_5_moe" ||
                          args.model_type() == "qwen3_5_moe_text";
      args.model_type(is_moe ? "qwen3_5_moe_mtp" : "qwen3_5_mtp");
    }
    if (is_python_model) {
      return;
    }
    const int32_t mtp_layers =
        std::max<int32_t>(args.num_nextn_predict_layers(), /*b=*/1);
    args.num_nextn_predict_layers(mtp_layers);
    args.n_layers(mtp_layers);
    args.layer_types(std::vector<std::string>(static_cast<size_t>(mtp_layers),
                                              /*value=*/"full_attention"));
    args.full_attention_interval(/*value=*/1);
  };
  const MtpModelMetadata target{
      .prepare_args_ = prepare_args,
      .capabilities_ =
          {
              .model_family = MtpModelFamily::QWEN35,
              .supports_expanded_replay = true,
              .supports_grouped_graph = true,
              .graph_context_policy = GraphContextPolicy::QWEN_FULL_ATTENTION,
          },
  };
  const MtpModelMetadata draft{
      .prepare_args_ = prepare_args,
      .capabilities_ =
          {
              .native_draft_requires_accepted_span_replay = true,
              .allow_tail_fallback = true,
              .model_family = MtpModelFamily::QWEN35,
              .supports_grouped_graph = true,
              .graph_context_policy = GraphContextPolicy::QWEN_FULL_ATTENTION,
          },
  };
  for (const char* name :
       {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}) {
    ModelRegistry::register_mtp_metadata(name, target);
  }
  for (const char* name : {"qwen3_5_mtp", "qwen3_5_moe_mtp"}) {
    ModelRegistry::register_mtp_metadata(name, draft);
  }
  return true;
}();

}  // namespace xllm
