/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/model.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_traits.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

struct ModelGraphMetadataState {
  virtual ~ModelGraphMetadataState() = default;
};

struct DFlash2CandidateOutput {
  // Candidate vocabulary ids [batch, draft_steps, top_k]. Edge logits use
  // [batch, draft_steps, predecessor_top_k, successor_top_k]; at step zero
  // every predecessor entry represents the same anchor token.
  torch::Tensor candidate_ids;
  torch::Tensor edge_logits;
};

class DecoderModel {
 public:
  virtual ~DecoderModel() = default;

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& parameters) = 0;

  virtual bool requires_graph_forward_metadata() { return false; }

  virtual bool is_hybrid_linear_attention() { return false; }

  virtual bool supports_mla_graph_kv_bucketing() const { return false; }

  virtual std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return nullptr;
  }

  virtual void prepare_graph_forward_metadata(ModelGraphMetadataState*,
                                              const torch::Tensor&,
                                              ModelParamsView&) {}

  virtual void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}

  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}

  virtual void update_expert_weight(int32_t) {}

  virtual bool last_prepare_expert_weight_ok(int32_t /*layer_id*/) const {
    return true;
  }

  virtual ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                                       const torch::Tensor& positions,
                                       const torch::Tensor& device_cache_slots,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelParamsView& input_params) {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }

  // DSpark-specific low-rank Markov projection. The draft worker owns the
  // sequential sampling lifecycle; the model owns only the trained weights and
  // bias computation.
  virtual torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual layer::WordEmbedding get_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
};

#define FORWARD_IF_SUPPORTED(trait, hook, ret, sig, args) \
  ret hook sig override {                                 \
    if constexpr (detail::trait<ModelType>::value) {      \
      return model_->hook args;                           \
    } else {                                              \
      return DecoderModel::hook args;                     \
    }                                                     \
  }

template <typename ModelType>
class DecoderModelImpl final : public DecoderModel {
 public:
  explicit DecoderModelImpl(ModelType& model) : model_(model) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  FORWARD_IF_SUPPORTED(has_requires_graph_forward_metadata,
                       requires_graph_forward_metadata,
                       bool,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(has_is_hybrid_linear_attention,
                       is_hybrid_linear_attention,
                       bool,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(has_supports_mla_graph_kv_bucketing,
                       supports_mla_graph_kv_bucketing,
                       bool,
                       () const,
                       ())
  FORWARD_IF_SUPPORTED(has_create_graph_forward_metadata_state,
                       create_graph_forward_metadata_state,
                       std::unique_ptr<ModelGraphMetadataState>,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(has_prepare_graph_forward_metadata,
                       prepare_graph_forward_metadata,
                       void,
                       (ModelGraphMetadataState * state,
                        const torch::Tensor& positions,
                        ModelParamsView& parameters),
                       (state, positions, parameters))
  FORWARD_IF_SUPPORTED(
      has_write_context_kv,
      write_context_kv,
      ModelOutput,
      (const torch::Tensor& target_hidden,
       const torch::Tensor& positions,
       const torch::Tensor& device_cache_slots,
       std::vector<KVCache>& kv_caches,
       const ModelParamsView& input_params),
      (target_hidden, positions, device_cache_slots, kv_caches, input_params))
  FORWARD_IF_SUPPORTED(has_dflash2_candidates,
                       dflash2_candidates,
                       DFlash2CandidateOutput,
                       (const torch::Tensor& hidden_states,
                        const torch::Tensor& unary_logits,
                        const torch::Tensor& anchor_token_ids),
                       (hidden_states, unary_logits, anchor_token_ids))
  FORWARD_IF_SUPPORTED(has_dspark_markov_bias,
                       dspark_markov_bias,
                       torch::Tensor,
                       (const torch::Tensor& previous_token_ids),
                       (previous_token_ids))
  FORWARD_IF_SUPPORTED(has_prepare_expert_weight,
                       prepare_expert_weight,
                       void,
                       (int32_t layer_id,
                        const std::vector<int32_t>& expert_ids),
                       (layer_id, expert_ids))
  FORWARD_IF_SUPPORTED(has_start_expert_weight_transfer,
                       start_expert_weight_transfer,
                       void,
                       (int32_t layer_id),
                       (layer_id))
  FORWARD_IF_SUPPORTED(has_update_expert_weight,
                       update_expert_weight,
                       void,
                       (int32_t layer_id),
                       (layer_id))
  FORWARD_IF_SUPPORTED(has_last_prepare_expert_weight_ok,
                       last_prepare_expert_weight_ok,
                       bool,
                       (int32_t layer_id) const,
                       (layer_id))
  FORWARD_IF_SUPPORTED(has_get_word_embedding,
                       get_word_embedding,
                       layer::WordEmbedding,
                       (),
                       ())

 private:
  ModelType& model_;
};

#undef FORWARD_IF_SUPPORTED

class CausalLM : public torch::nn::Module, public Model {
 public:
  ~CausalLM() override = default;

  virtual DecoderModel& model() = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // out_hidden: [num_selected_tokens, hidden_size]
  // returns: [num_selected_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes,
                               torch::Tensor& out_hidden) {
    if (seleted_idxes.defined()) {
      const auto idxes = seleted_idxes.to(
          torch::dtype(torch::kLong).device(hidden_states.device()));
      out_hidden = hidden_states.index_select(/*dim=*/0, idxes);
    } else {
      out_hidden = hidden_states;
    }
    return logits(hidden_states, seleted_idxes);
  }

  const torch::Tensor& d2t() const { return d2t_; }

  virtual void share_lm_head(CausalLM&) {}
  virtual void share_embed_tokens(CausalLM&) {}

  virtual layer::LmHead get_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }

 protected:
  void load_d2t(const std::vector<std::unique_ptr<StateDict>>& state_dicts) {
    for (const auto& state_dict : state_dicts) {
      const auto offsets = state_dict->get_tensor(/*tensor_name=*/"d2t");
      if (offsets.defined()) {
        d2t_ = offsets.clone();
        return;
      }
    }
    d2t_ = torch::Tensor();
  }

 private:
  torch::Tensor d2t_;
};

template <typename ModelType>
class CausalLMImpl : public ModelImpl<ModelType, CausalLM> {
 public:
  using Base = ModelImpl<ModelType, CausalLM>;
  using DecoderType =
      std::remove_reference_t<decltype(std::declval<ModelType>()->model())>;

  CausalLMImpl(ModelType model, const torch::TensorOptions& options)
      : Base(std::move(model), options), decoder_(this->model_->model()) {}

  DecoderModel& model() override { return decoder_; }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    this->load_d2t(loader->get_state_dicts());
    Base::load_model(std::move(loader));
    if constexpr (detail::has_tie_word_embeddings<ModelType>::value &&
                  detail::has_get_word_embedding<DecoderType>::value) {
      this->model_->tie_word_embeddings(
          this->model_->model()->get_word_embedding());
    }
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    return this->model_->logits(hidden_states, selected_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_logits_with_hidden<ModelType>::value) {
      return this->model_->logits(hidden_states, selected_idxes, out_hidden);
    } else {
      return CausalLM::logits(hidden_states, selected_idxes, out_hidden);
    }
  }

  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<ModelType>::value) {
      return this->model_->get_lm_head();
    } else {
      return CausalLM::get_lm_head();
    }
  }

  void share_lm_head(CausalLM& target_model) override {
    if constexpr (detail::has_share_lm_head<ModelType>::value) {
      this->model_->share_lm_head(target_model.get_lm_head());
    }
  }

  void share_embed_tokens(CausalLM& target_model) override {
    if constexpr (detail::has_get_word_embedding<DecoderType>::value &&
                  detail::has_set_word_embedding<DecoderType>::value) {
      auto own = this->model_->model()->get_word_embedding();
      if (own.is_empty() || !own->is_weight_loaded()) {
        auto embedding = target_model.model().get_word_embedding();
        this->model_->model()->set_word_embedding(embedding);
      }
    }
  }

 private:
  DecoderModelImpl<DecoderType> decoder_;
};

}  // namespace xllm
