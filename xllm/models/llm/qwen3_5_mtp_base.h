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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/layers/common/linear.h"
#include "core/layers/qwen3_5_decoder_layer.h"
#include "models/llm/qwen3_next_hybrid_base.h"
#include "models/model_registry.h"

namespace xllm {

namespace qwen3_5_mtp {

inline StateDict get_lm_head_dict(const StateDict& state_dict) {
  static const std::vector<std::string> kLmHeadPrefixes = {
      "lm_head.",
      "model.lm_head.",
      "language_model.lm_head.",
      "model.language_model.lm_head."};
  for (const std::string& prefix : kLmHeadPrefixes) {
    StateDict sub_dict = state_dict.get_dict_with_prefix(prefix);
    if (sub_dict.get_tensor("weight").defined() ||
        sub_dict.get_tensor("qweight").defined()) {
      return sub_dict;
    }
  }
  return StateDict({}, "");
}

inline StateDict get_embed_tokens_dict(const StateDict& state_dict) {
  static const std::vector<std::string> kEmbeddingPrefixes = {
      "model.language_model.embed_tokens.",
      "language_model.model.embed_tokens.",
      "model.embed_tokens.",
      "embed_tokens."};
  return state_dict.get_dict_with_prefix(kEmbeddingPrefixes);
}

inline bool load_model_args(const JsonReader& json,
                            ModelArgs* args,
                            const std::string& base_type,
                            const std::string& mtp_type) {
  ModelArgsLoader base_loader = ModelRegistry::get_model_args_loader(base_type);
  if (base_loader == nullptr || base_loader(json, args) == false) {
    return false;
  }

  int32_t mtp_num_layers = args->num_nextn_predict_layers();
  if (mtp_num_layers <= 0) {
    mtp_num_layers = 1;
  }
  args->model_type(mtp_type);
  args->num_nextn_predict_layers(mtp_num_layers);
  args->n_layers(mtp_num_layers);
  args->layer_types(std::vector<std::string>(
      static_cast<size_t>(mtp_num_layers), "full_attention"));
  return true;
}

}  // namespace qwen3_5_mtp

class Qwen3_5MtpModelImplBase : public Qwen3HybridModelImplBase {
 public:
  explicit Qwen3_5MtpModelImplBase(const ModelContext& context)
      : Qwen3HybridModelImplBase(context) {
    const torch::TensorOptions& options = context.get_tensor_options();
    const int32_t n_layers =
        std::max<int32_t>(static_cast<int32_t>(model_args_.n_layers()), 1);

    pre_fc_norm_embedding_ = register_module(
        "pre_fc_norm_embedding",
        layer::Qwen3NextRMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));
    pre_fc_norm_hidden_ = register_module(
        "pre_fc_norm_hidden",
        layer::Qwen3NextRMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));
    fc_ = register_module("fc",
                          layer::ReplicatedLinear(model_args_.hidden_size() * 2,
                                                  model_args_.hidden_size(),
                                                  /*bias=*/false,
                                                  QuantArgs(),
                                                  options));

    layers_.reserve(n_layers);
    for (int32_t layer_id = 0; layer_id < n_layers; ++layer_id) {
      add_decoder_layer(
          std::make_shared<layer::Qwen3_5DecoderLayerImpl>(context, layer_id));
    }
  }

  torch::Tensor prepare_inputs_embeds(
      const torch::Tensor& tokens,
      const torch::Tensor& input_embedding) override {
    auto embedding = embed_tokens_(tokens);
    auto hidden = input_embedding.defined() ? input_embedding : embedding;
    embedding = std::get<0>(pre_fc_norm_embedding_->forward(embedding));
    hidden = std::get<0>(pre_fc_norm_hidden_->forward(hidden));
    return fc_(torch::cat({embedding, hidden}, /*dim=*/-1));
  }

  ModelOutput forward_from_embeddings(
      torch::Tensor mtp_hidden,
      torch::Tensor positions,
      std::vector<KVCache>& kv_caches,
      const ModelParamsView& input_params) override {
    layer::AttentionMetadataBuildOptions metadata_build_options;
#if defined(USE_NPU)
    // Native NPU GDN consumes the canonical host mask directly. Avoid
    // materializing the unused device bool tensor inside ACL graph capture.
    metadata_build_options.materialize_linear_state_validity =
        !input_params.enable_graph;
#endif
    layer::AttentionMetadata attn_metadata =
        layer::AttentionMetadataBuilder::build(
            input_params,
            model_args_.enable_mla(),
            build_attention_mask(input_params),
            /*device=*/device_,
            metadata_build_options);
    prepare_mrope(positions, attn_metadata);

    CHECK_EQ(kv_caches.size(), layers_.size());
    torch::Tensor mrope_cos_sin;
    for (const layer::Qwen3HybridDecoderLayerModulePtr& layer : layers_) {
      mrope_cos_sin = layer->build_mrope_cos_sin(positions);
      if (mrope_cos_sin.defined()) {
        break;
      }
    }

    std::optional<torch::Tensor> residual = std::nullopt;
    if (!input_params.parallel.synchronize_draft_layer()) {
      return ModelOutput();
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
      mtp_hidden = layers_[i]->forward(mtp_hidden,
                                       residual,
                                       positions,
                                       attn_metadata,
                                       kv_caches[i],
                                       input_params,
                                       mrope_cos_sin);
#if defined(USE_NPU)
      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->record_event(
              static_cast<int64_t>(i), device_.index())) {
        return ModelOutput();
      }
#endif
    }
    auto [new_mtp_hidden, new_res] = norm_->forward(mtp_hidden, residual);
    mtp_hidden = new_mtp_hidden;
    return ModelOutput(mtp_hidden);
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        qwen3_5_mtp::get_embed_tokens_dict(state_dict));
    static const std::vector<std::string> kMtpPrefixes = {"mtp.", "model.mtp."};
    StateDict mtp_dict = state_dict.get_dict_with_prefix(kMtpPrefixes);
    pre_fc_norm_embedding_->load_state_dict(
        mtp_dict.get_dict_with_prefix("pre_fc_norm_embedding."));
    pre_fc_norm_hidden_->load_state_dict(
        mtp_dict.get_dict_with_prefix("pre_fc_norm_hidden."));
    fc_->load_state_dict(mtp_dict.get_dict_with_prefix("fc."));
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          mtp_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(mtp_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(embed_tokens_->is_weight_loaded())
        << "Failed to find embedding weights for qwen3.5 mtp draft "
           "model";
    CHECK(pre_fc_norm_embedding_->is_weight_loaded())
        << "Failed to find mtp pre_fc_norm_embedding weights for qwen3.5 mtp "
           "draft model";
    CHECK(pre_fc_norm_hidden_->is_weight_loaded())
        << "Failed to find mtp pre_fc_norm_hidden weights for qwen3.5 mtp "
           "draft model";
    CHECK(fc_->is_weight_loaded())
        << "Failed to find mtp fc weights for qwen3.5 mtp draft "
           "model";
    CHECK(norm_->is_weight_loaded())
        << "Failed to find mtp norm weights for qwen3.5 mtp draft model";
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 protected:
  virtual void prepare_mrope(const torch::Tensor& positions,
                             layer::AttentionMetadata& attn_metadata) const {
    UNUSED_PARAMETER(positions);
    UNUSED_PARAMETER(attn_metadata);
  }

 private:
  layer::Qwen3NextRMSNorm pre_fc_norm_embedding_{nullptr};
  layer::Qwen3NextRMSNorm pre_fc_norm_hidden_{nullptr};
  layer::ReplicatedLinear fc_{nullptr};
};

class Qwen3_5MtpForCausalLMImplBase : public Qwen3HybridForCausalLMImplBase {
 public:
  void load_model(std::unique_ptr<ModelLoader> loader) {
    Qwen3HybridForCausalLMImplBase::load_lm_head(loader);
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      model_->load_state_dict(*state_dict);
    }

    model_->verify_loaded_weights("mtp.");
    CHECK(lm_head_->is_weight_loaded())
        << "Failed to find lm_head weights for qwen3.5 mtp draft model";
  }

  void load_lm_head(const StateDict& state_dict,
                    const std::string& /*prefix*/) override {
    lm_head_->load_state_dict(
        tie_word_embeddings_ ? qwen3_5_mtp::get_embed_tokens_dict(state_dict)
                             : qwen3_5_mtp::get_lm_head_dict(state_dict));
  }

 protected:
  Qwen3_5MtpForCausalLMImplBase(
      const ModelContext& context,
      std::shared_ptr<Qwen3_5MtpModelImplBase> mtp_model)
      : Qwen3HybridForCausalLMImplBase(context, std::move(mtp_model)) {}
};

}  // namespace xllm
