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

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model_context/model_context.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "models/model_registry.h"

namespace xllm {

template <typename DecoderLayerType>
class LlmModelImplBase : public torch::nn::Module {
 public:
  explicit LlmModelImplBase(const ModelArgs& args) : model_args_(args) {
    mrope_section_ = args.rope_scaling_mrope_section();
  }

  virtual std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) {
    return std::make_pair(torch::Tensor(), torch::Tensor());
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& input_params) {
    auto local_params = input_params.clone().view();
    return forward_impl(tokens, positions, kv_caches, local_params);
  }

  template <typename InputParams>
  ModelOutput forward_impl(torch::Tensor tokens,
                           torch::Tensor positions,
                           std::vector<KVCache>& kv_caches,
                           InputParams& input_params) {
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

    auto& dp_token_nums = input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!input_params.attn_metadata) {
      input_params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(input_params,
                                                 model_args_.enable_mla()));
    }
    auto& attn_metadata = *(input_params.attn_metadata);
    if (positions.dim() == 2) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (!input_params.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = layers_[i];
      auto& cache = kv_caches[i];
      if constexpr (requires {
                      layer->forward(h,
                                     residual,
                                     positions,
                                     attn_metadata,
                                     cache,
                                     input_params);
                    }) {
        h = layer->forward(
            h, residual, positions, attn_metadata, cache, input_params);
      } else {
        h = layer->forward(h, residual, positions, attn_metadata, cache);
      }
      if (!input_params.record_layer(static_cast<uint32_t>(i), h.device())) {
        return ModelOutput();
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

  // load the weight from the checkpoint
  virtual void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));

    // call each layer's load_state_dict function
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = embed_tokens_.is_empty()
                        ? this->register_module("embed_tokens", word_embedding)
                        : this->replace_module("embed_tokens", word_embedding);
  }

 protected:
  torch::Tensor cos_sin_;
  int32_t max_seq_len_ = 0;
  std::vector<int64_t> mrope_section_;
  ModelArgs model_args_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};

  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayerType> layers_;
};

template <typename LlmModelType>
class LlmForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit LlmForCausalLMImplBase(const ModelContext& context)
      : LlmForCausalLMImplBase(context, LlmModelType(context)) {}

  auto& model() { return *model_; }

  bool tie_word_embeddings() const { return tie_word_embeddings_; }

  torch::Tensor get_input_embeddings(const torch::Tensor& input_ids) {
    auto embedding = model_->get_word_embedding();
    return embedding.is_empty() ? torch::Tensor() : embedding(input_ids);
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  template <typename InputParams>
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const InputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  bool requires_graph_forward_metadata()
    requires requires(LlmModelType& model) {
      model->requires_graph_forward_metadata();
    }
  {
    return model_->requires_graph_forward_metadata();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
    requires requires(LlmModelType& model) {
      model->create_graph_forward_metadata_state();
    }
  {
    return model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelParamsView& parameters)
    requires requires(LlmModelType& model) {
      model->prepare_graph_forward_metadata(state, positions, parameters);
    }
  {
    model_->prepare_graph_forward_metadata(state, positions, parameters);
  }

  static constexpr const char* kWeightPrefix = "model.";

  virtual void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict);
  }

  virtual void verify_model_weights(const std::string& prefix) {
    if constexpr (requires { model_->verify_loaded_weights(prefix); }) {
      model_->verify_loaded_weights(prefix);
    } else if constexpr (requires { model_->verify_loaded_weights(); }) {
      model_->verify_loaded_weights();
    }
  }

  virtual void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}
  virtual void update_expert_weight(int32_t) {}

 protected:
  LlmForCausalLMImplBase(const ModelContext& context, LlmModelType model)
      : model_(register_module(/*name=*/"model", std::move(model))),
        tie_word_embeddings_(context.get_model_args().tie_word_embeddings()) {}

  LlmModelType model_{nullptr};
  bool tie_word_embeddings_ = false;
};

}  // namespace xllm
