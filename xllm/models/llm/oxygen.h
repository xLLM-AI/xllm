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
#include "core/framework/model/model_output.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "llm_model_base.h"
#include "qwen3.h"

namespace xllm {

class OxygenModelImpl : public QWen3ModelImpl {
 public:
  OxygenModelImpl(const ModelContext& context) : QWen3ModelImpl(context) {}

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) override {
    const bool use_deepstack = input_params.has_multimodal() &&
                               !input_params.multimodal().deep_stacks.empty();
    ModelParamsView& input_params_new =
        const_cast<ModelParamsView&>(input_params);
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }
    if (use_deepstack) {
      deep_stacks = input_params.multimodal()
                        .deep_stacks;  // [num_deepstack, hidden_size]
    }

    auto& dp_token_nums = input_params_new.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!input_params_new.attn_metadata) {
      input_params_new.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(input_params_new,
                                                     model_args_.enable_mla()));
    }

    auto& attn_metadata = *(input_params_new.attn_metadata);
    bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer->forward(h, residual, positions, attn_metadata, kv_caches[i]);
      if (!input_params_new.record_layer(static_cast<uint32_t>(i),
                                         h.device())) {
        return ModelOutput();
      }

      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = h + deep_stacks[i];
        }
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

 protected:
  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto sections = mrope_section_;
      sections.insert(sections.end(), sections.begin(), sections.end());

      auto vec = x.split(sections, -1);
      std::vector<torch::Tensor> selects;
      selects.reserve(vec.size());

      for (int64_t i = 0; i < vec.size(); ++i) {
        auto m = vec[i];
        selects.push_back(m[i % mrope_section_.size()]);
      }
      return torch::cat(selects, -1);
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }
};
TORCH_MODULE(OxygenModel);

class OxygenForCausalLMImpl : public LlmForCausalLMImplBase<OxygenModel> {
 public:
  OxygenForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<OxygenModel>(context) {}
};
TORCH_MODULE(OxygenForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(oxygenvlm_text, OxygenForCausalLM);

}  // namespace xllm
