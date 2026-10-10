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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context/model_context.h"
#include "core/kernels/mlu/mlu_ops_api.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/dsa_topk_relay.h"
#include "core/layers/mlu/glm5_next/glm5_next_decoder_layer.h"
#include "models/llm/llm_model_base.h"
#include "models/model_registry.h"

namespace xllm {
namespace mlu {
namespace model {

class Glm5NextModelImpl final
    : public LlmModelImplBase<layer::Glm5NextDecoderLayer> {
 public:
  explicit Glm5NextModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Glm5NextDecoderLayer>("glm5_next",
                                                      context.get_model_args()),
        device_(context.get_tensor_options().device()),
        hc_mult_(context.get_model_args().hc_mult()) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    CHECK_EQ(parallel_args.cp_size(), 1)
        << "GLM5-Next MLU does not yet support context parallelism.";
    CHECK_EQ(args.num_speculative_tokens(), 0)
        << "GLM5-Next MLU speculative decoding is not supported yet.";
    CHECK_GT(hc_mult_, 0) << "GLM5-Next requires hc_mult > 0.";

    const int64_t tp_size = std::max<int64_t>(parallel_args.tp_size(), 1);
    CHECK_EQ(args.linear_num_key_heads() % tp_size, 0)
        << "GLM5-Next KDA head count must be divisible by TP size.";

    auto blocks = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(static_cast<size_t>(args.n_layers()));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    const layer::DsaTopkSharePlan topk_share_plan(args);
    for (int32_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
      layer::Glm5NextDecoderLayer decoder_layer(
          context, layer_id, topk_share_plan);
      layers_.emplace_back(decoder_layer);
      blocks->push_back(decoder_layer);
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    CHECK_EQ(kv_caches.size(), layers_.size())
        << "GLM5-Next requires one layer-specific cache object per layer.";
    CHECK(!input_params.enable_graph)
        << "GLM5-Next MLU currently requires eager execution.";
    CHECK(!input_params.is_spec_verify)
        << "GLM5-Next MLU speculative verification is not supported yet.";

    if (tokens.numel() == 0) {
      tokens = torch::ones(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::zeros(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    torch::Tensor hidden_states =
        input_params.embedding.input_embedding.defined()
            ? input_params.embedding.input_embedding
            : embed_tokens_(tokens);
    if (hidden_states.dim() == 2) {
      hidden_states =
          hidden_states.unsqueeze(1).repeat({1, hc_mult_, 1}).contiguous();
    }
    CHECK(hidden_states.dim() == 3 && hidden_states.size(1) == hc_mult_)
        << "GLM5-Next expects [tokens, hc_mult, hidden] input embeddings.";

    auto modified_input_params = input_params.clone().view();
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla(),
                                                     /*compute_dtype=*/"half",
                                                     /*attn_mask=*/std::nullopt,
                                                     /*device=*/device_));
    }
    // Convolution uses 64-token blocks; KDA builds its own chunk indices.
    layer::AttentionMetadataBuilder::build_linear_prefill(
        *modified_input_params.attn_metadata,
        kernel::mlu::kCausalConv1dPrefillBlockSize);
    const layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;

    std::optional<torch::Tensor> residual;
    layer::DsaTopkRelay topk_relay;
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      if (!modified_input_params.synchronize_layer(
              static_cast<uint32_t>(layer_id))) {
        return ModelOutput();
      }
      hidden_states = layers_[layer_id]->forward(hidden_states,
                                                 residual,
                                                 positions,
                                                 attn_metadata,
                                                 kv_caches[layer_id],
                                                 modified_input_params,
                                                 &topk_relay);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(layer_id),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }

    CHECK_EQ(hidden_states.dim(), 3)
        << "GLM5-Next mHC output must retain the residual-stream dimension.";
    CHECK_EQ(hidden_states.size(-2), hc_mult_)
        << "GLM5-Next mHC residual-stream count changed unexpectedly.";
    hidden_states = hidden_states.mean(/*dim=*/-2);
    auto [normalized, residual_out] = norm_(hidden_states, std::nullopt);
    return ModelOutput(normalized, residual_out);
  }

  void verify_loaded_weights() const {
    CHECK(embed_tokens_->is_weight_loaded()) << "Missing embed_tokens.weight";
    norm_->verify_loaded_weights("norm.");
    for (const auto& layer : layers_) {
      layer->verify_loaded_weights();
    }
  }

 private:
  torch::Device device_;
  int64_t hc_mult_ = 1;
};
TORCH_MODULE(Glm5NextModel);

class Glm5NextForCausalLMImpl final
    : public LlmForCausalLMImplBase<Glm5NextModel> {
 public:
  explicit Glm5NextForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5NextModel>(context) {}

  bool is_hybrid_linear_attention() { return true; }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<Glm5NextModel>::load_model(std::move(loader),
                                                      std::move(prefix));
    model_->verify_loaded_weights();
    if (!embedding_mode_) {
      CHECK(lm_head_->is_weight_loaded()) << "Missing lm_head.weight";
    }
  }
};
TORCH_MODULE(Glm5NextForCausalLM);

REGISTER_CAUSAL_MODEL(glm5_next, Glm5NextForCausalLM);

}  // namespace model
}  // namespace mlu
}  // namespace xllm
