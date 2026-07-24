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

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/layers/common/linear.h"
#include "models/model_registry.h"
#include "qwen3_5.h"

namespace xllm {

namespace {

StateDict find_lm_head_state_dict(const StateDict& state_dict) {
  static const std::vector<std::string> kLmHeadPrefixes = {
      "lm_head.",
      "model.lm_head.",
      "language_model.lm_head.",
      "model.language_model.lm_head."};
  for (const auto& prefix : kLmHeadPrefixes) {
    auto sub_dict = state_dict.get_dict_with_prefix(prefix);
    if (sub_dict.get_tensor("weight").defined() ||
        sub_dict.get_tensor("qweight").defined()) {
      return sub_dict;
    }
  }
  return StateDict({}, "");
}

bool load_qwen3_5_mtp_model_args(const JsonReader& json,
                                 ModelArgs* args,
                                 const std::string& base_model_type,
                                 const std::string& mtp_model_type) {
  auto base_loader = ModelRegistry::get_model_args_loader(base_model_type);
  if (base_loader == nullptr || base_loader(json, args) == false) {
    return false;
  }

  int32_t mtp_num_layers = args->num_nextn_predict_layers();
  if (mtp_num_layers <= 0) {
    mtp_num_layers = 1;
  }
  args->model_type(mtp_model_type);
  args->num_nextn_predict_layers(mtp_num_layers);
  args->n_layers(mtp_num_layers);
  args->layer_types(std::vector<std::string>(
      static_cast<size_t>(mtp_num_layers), "full_attention"));
  return true;
}

}  // namespace

class Qwen3_5MtpModelImpl : public Qwen3HybridModelImplBase {
 public:
  explicit Qwen3_5MtpModelImpl(const ModelContext& context)
      : Qwen3HybridModelImplBase(context) {
    const auto& options = context.get_tensor_options();
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

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;

    if (dp_size_ > 1 && tokens.sizes() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
      positions = torch::tensor({0}).to(torch::kInt32).to(device_);
    }

    auto attn_metadata = layer::AttentionMetadataBuilder::build(
        input_params,
        model_args_.enable_mla(),
        build_attention_mask(input_params));

    torch::Tensor embedding = embed_tokens_(tokens);
    torch::Tensor hidden = input_params.embedding.input_embedding;
    if (hidden.defined() == false) {
      hidden = embedding;
    }

    embedding = std::get<0>(pre_fc_norm_embedding_->forward(embedding));
    hidden = std::get<0>(pre_fc_norm_hidden_->forward(hidden));
    torch::Tensor mtp_hidden = fc_(torch::cat({embedding, hidden}, -1));

    CHECK_EQ(kv_caches.size(), layers_.size());
    torch::Tensor mrope_cos_sin;
    for (const auto& layer : layers_) {
      mrope_cos_sin = layer->build_mrope_cos_sin(positions);
      if (mrope_cos_sin.defined()) {
        break;
      }
    }

    std::optional<torch::Tensor> residual = std::nullopt;
    for (size_t i = 0; i < layers_.size(); ++i) {
      if (!input_params.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
      mtp_hidden = layers_[i]->forward(mtp_hidden,
                                       residual,
                                       positions,
                                       attn_metadata,
                                       kv_caches[i],
                                       input_params,
                                       mrope_cos_sin);
    }
    auto [new_mtp_hidden, new_res] = norm_->forward(mtp_hidden, residual);
    mtp_hidden = new_mtp_hidden;
    return ModelOutput(mtp_hidden);
  }

  void load_state_dict(const StateDict& state_dict) override {
    load_shared_embeddings(state_dict);
    load_mtp_state_dict(state_dict);
  }

  void load_shared_embeddings(const StateDict& state_dict) {
    auto embedding_state_dict =
        state_dict.get_dict_with_prefix("embed_tokens.");
    if (embedding_state_dict.get_tensor("weight").defined()) {
      shared_embedding_loaded_ = true;
    }
    embed_tokens_->load_state_dict(embedding_state_dict);
  }

  void load_mtp_state_dict(const StateDict& state_dict) {
    if (state_dict.get_tensor("pre_fc_norm_embedding.weight").defined()) {
      pre_fc_norm_embedding_loaded_ = true;
    }
    if (state_dict.get_tensor("pre_fc_norm_hidden.weight").defined()) {
      pre_fc_norm_hidden_loaded_ = true;
    }
    if (state_dict.get_tensor("fc.weight").defined() ||
        state_dict.get_tensor("fc.qweight").defined()) {
      fc_loaded_ = true;
    }
    if (state_dict.get_tensor("norm.weight").defined()) {
      norm_loaded_ = true;
    }

    pre_fc_norm_embedding_->load_state_dict(
        state_dict.get_dict_with_prefix("pre_fc_norm_embedding."));
    pre_fc_norm_hidden_->load_state_dict(
        state_dict.get_dict_with_prefix("pre_fc_norm_hidden."));
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(shared_embedding_loaded_)
        << "Failed to find shared embedding weights for qwen3.5 mtp draft "
           "model";
    CHECK(pre_fc_norm_embedding_loaded_)
        << "Failed to find mtp pre_fc_norm_embedding weights for qwen3.5 mtp "
           "draft model";
    CHECK(pre_fc_norm_hidden_loaded_)
        << "Failed to find mtp pre_fc_norm_hidden weights for qwen3.5 mtp "
           "draft model";
    CHECK(fc_loaded_) << "Failed to find mtp fc weights for qwen3.5 mtp draft "
                         "model";
    CHECK(norm_loaded_)
        << "Failed to find mtp norm weights for qwen3.5 mtp draft model";
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  layer::Qwen3NextRMSNorm pre_fc_norm_embedding_{nullptr};
  layer::Qwen3NextRMSNorm pre_fc_norm_hidden_{nullptr};
  layer::ReplicatedLinear fc_{nullptr};
  bool shared_embedding_loaded_ = false;
  bool pre_fc_norm_embedding_loaded_ = false;
  bool pre_fc_norm_hidden_loaded_ = false;
  bool fc_loaded_ = false;
  bool norm_loaded_ = false;
};

class Qwen3_5MtpForCausalLMImpl : public Qwen3HybridForCausalLMImplBase {
 public:
  explicit Qwen3_5MtpForCausalLMImpl(const ModelContext& context)
      : Qwen3HybridForCausalLMImplBase(context) {
    mtp_model_ = std::make_shared<Qwen3_5MtpModelImpl>(context);
    set_model_module(mtp_model_);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    static const std::vector<std::string> kEmbeddingPrefixes = {
        "model.language_model.", "language_model.model.", "model.", ""};
    static const std::vector<std::string> kMtpPrefixes = {"mtp.", "model.mtp."};
    bool lm_head_loaded = false;

    for (const auto& state_dict : loader->get_state_dicts()) {
      auto shared_embedding_state_dict =
          state_dict->get_dict_with_prefix(kEmbeddingPrefixes);
      auto mtp_state_dict = state_dict->get_dict_with_prefix(kMtpPrefixes);

      mtp_model_->load_shared_embeddings(shared_embedding_state_dict);
      mtp_model_->load_mtp_state_dict(mtp_state_dict);

      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            shared_embedding_state_dict.get_dict_with_prefix("embed_tokens."));
        if (shared_embedding_state_dict.get_tensor("embed_tokens.weight")
                .defined()) {
          lm_head_loaded = true;
        }
      } else {
        auto lm_head_state_dict = find_lm_head_state_dict(*state_dict);
        lm_head_->load_state_dict(lm_head_state_dict);
        if (lm_head_state_dict.get_tensor("weight").defined() ||
            lm_head_state_dict.get_tensor("qweight").defined()) {
          lm_head_loaded = true;
        }
      }
    }

    CHECK(lm_head_loaded)
        << "Failed to find lm_head weights for qwen3.5 mtp draft model";
    mtp_model_->verify_loaded_weights("mtp.");
  }

 private:
  std::shared_ptr<Qwen3_5MtpModelImpl> mtp_model_;
};
TORCH_MODULE(Qwen3_5MtpForCausalLM);

REGISTER_CAUSAL_MODEL(qwen3_5_mtp, Qwen3_5MtpForCausalLM);
REGISTER_CAUSAL_MODEL(qwen3_5_moe_mtp, Qwen3_5MtpForCausalLM);

REGISTER_MODEL_ARGS_LOADER(qwen3_5_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return load_qwen3_5_mtp_model_args(
                                 json, args, "qwen3_5_text", "qwen3_5_mtp");
                           });

REGISTER_MODEL_ARGS_LOADER(qwen3_5_moe_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             return load_qwen3_5_mtp_model_args(
                                 json,
                                 args,
                                 "qwen3_5_moe_text",
                                 "qwen3_5_moe_mtp");
                           });

}  // namespace xllm
