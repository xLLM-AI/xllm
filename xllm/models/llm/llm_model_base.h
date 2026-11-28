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

  virtual ModelOutput forward_from_embeddings(
      torch::Tensor h,
      torch::Tensor positions,
      std::vector<KVCache>& kv_caches,
      const ModelParamsView& input_params) {
    return forward_impl(std::move(h),
                        std::move(positions),
                        kv_caches,
                        input_params.clone().view());
  }

 protected:
  ModelOutput forward_impl(torch::Tensor h,
                           torch::Tensor positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelParamsView& input_params) {
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

 public:
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

  virtual layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
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
  explicit LlmForCausalLMImplBase(const ModelContext& context,
                                  bool share_lm_head = false)
      : LlmForCausalLMImplBase(context, LlmModelType(context), share_lm_head) {}

  layer::LmHead get_lm_head() { return lm_head_; }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    auto selected = selected_idxes.defined()
                        ? hidden_states.index_select(/*dim=*/0, selected_idxes)
                        : hidden_states;
    return lm_head_(selected);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) {
    if (selected_idxes.defined()) {
      const auto idxes = selected_idxes.to(
          torch::dtype(torch::kLong).device(hidden_states.device()));
      out_hidden = hidden_states.index_select(/*dim=*/0, idxes);
    } else {
      out_hidden = hidden_states;
    }
    return logits(out_hidden, torch::Tensor());
  }

  void tie_word_embeddings(const layer::WordEmbedding& embedding) {
    if (!tie_word_embeddings_ || lm_head_.is_empty() || embedding.is_empty() ||
        !embedding->is_weight_loaded()) {
      return;
    }
    auto head_weight = lm_head_->weight();
    auto embedding_weight = embedding->weight();
    // Tensor-parallel head and embedding weights may use different shard axes.
    if (!head_weight.defined() || !embedding_weight.defined() ||
        head_weight.sizes() != embedding_weight.sizes() ||
        head_weight.scalar_type() != embedding_weight.scalar_type()) {
      return;
    }
    lm_head_->tie_weights(embedding_weight);
  }

  bool has_own_lm_head() const {
    return !lm_head_.is_empty() && lm_head_->is_weight_loaded();
  }

  void share_lm_head(layer::LmHead target_head) {
    if (embedding_mode_ || has_own_lm_head()) {
      return;
    }
    if (lm_head_.is_empty()) {
      lm_head_ = register_module("lm_head", std::move(target_head));
    } else {
      lm_head_ = replace_module("lm_head", std::move(target_head));
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return get_word_embedding()(input_ids);
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& input_params) {
    if constexpr (requires {
                    model_->forward(tokens, positions, kv_caches, input_params);
                  }) {
      return model_->forward(tokens, positions, kv_caches, input_params);
    } else {
      auto model_tokens = tokens;
      auto model_positions = positions;
      if (tokens.numel() == 0) {
        model_tokens = torch::tensor(/*values=*/{1}, /*options=*/torch::kInt32).to(tokens.device());
        model_positions = model_tokens;
      }
      return model_->forward_from_embeddings(
          prepare_inputs_embeds(model_tokens,
                                input_params.embedding.input_embedding),
          std::move(model_positions),
          kv_caches,
          input_params);
    }
  }

 protected:
  torch::Tensor prepare_inputs_embeds(const torch::Tensor& tokens,
                                      const torch::Tensor& input_embedding) {
    return input_embedding.defined() ? input_embedding
                                     : get_input_embeddings(tokens);
  }

 public:
  virtual void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) {
    this->load_lm_head(loader, prefix);
    for (const auto& state_dict : loader->get_state_dicts()) {
      // The same model_type may come from checkpoints with different top-level
      // weight prefixes. Try these candidate prefixes in order to improve
      // compatibility across such variants.
      model_->load_state_dict(state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.",
                                   "language_model.model.",
                                   prefix,
                                   "model.",
                                   ""}));
    }
  }

  virtual void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}
  virtual void update_expert_weight(int32_t) {}

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  LlmForCausalLMImplBase(const ModelContext& context,
                         LlmModelType model,
                         bool share_lm_head = false)
      : model_(register_module("model", std::move(model))) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    embedding_mode_ = args.embedding_mode();
    if (!embedding_mode_ && !share_lm_head) {
      lm_head_ = register_module("lm_head", layer::LmHead(context));
    }
  }

  void load_lm_head(const std::unique_ptr<ModelLoader>& loader,
                    const std::string& model_prefix = "model.") {
    if (lm_head_.is_empty()) {
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_lm_head(*state_dict, model_prefix);
    }
  }

  virtual void load_lm_head(const StateDict& state_dict,
                            const std::string& model_prefix) {
    const std::vector<std::string> prefixes =
        tie_word_embeddings_
            ? std::vector<std::string>{model_prefix + "embed_tokens.",
                                       "model.embed_tokens.",
                                       "embed_tokens.",
                                       "embed."}
            : std::vector<std::string>{"lm_head.",
                                       "model.lm_head.",
                                       "model.head.",
                                       "head.",
                                       model_prefix + "lm_head.",
                                       model_prefix + "head.",
                                       model_prefix};
    lm_head_->load_state_dict(state_dict.get_dict_with_prefix(prefixes));
  }

  LlmModelType model_{nullptr};
  layer::LmHead lm_head_{nullptr};
  bool tie_word_embeddings_ = false;
  bool embedding_mode_ = false;
};

}  // namespace xllm
