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
// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
#include "core/framework/model/model.h"
// clang-format on
#include <c10/core/Device.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_weight_loader.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/quantization/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
}

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

class CausalLM : public torch::nn::Module, public Model {
 public:
  ~CausalLM() override = default;

  // An empty key keeps the runtime's default executor selection.
  virtual std::string executor_type() const { return {}; }

  // tokens: [num_tokens]
  // positions: [num_tokens]
  // returns: [num_tokens, hidden_size]
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

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  virtual torch::Tensor logits_processor(const torch::Tensor& hidden_states) {
    return lm_head_(hidden_states);
  }

  virtual torch::nn::Module& model();

  virtual torch::Tensor get_input_embeddings(const torch::Tensor&) {
    return {};
  }

  layer::LmHead get_lm_head() const { return lm_head_; }

  const torch::Tensor& d2t() const { return d2t_; }

  virtual void share_lm_head(CausalLM&) {}
  virtual void share_embed_tokens(CausalLM&) {}

  virtual void prepare_expert_weight(
      int32_t layer_id,
      const std::vector<int32_t>& expert_ids) = 0;

  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}

  virtual void update_expert_weight(int32_t layer_id) = 0;

  // Returns whether the last prepare_expert_weight() call for the given layer
  // succeeded. Default is true so models that do not fail-report keep the old
  // behavior; DSV4 (npu_torch) overrides this to expose real prepare status
  // so EplbExecutor does not report a failed preparation as complete.
  virtual bool last_prepare_expert_weight_ok(int32_t /*layer_id*/) const {
    return true;
  }

  // DFlash-specific interface. Attention-free scatter-only pass that projects
  // target hidden into the draft's per-layer KV cache. Not part of forward()
  // because it has no attention and its shape doesn't match forward's decode
  // graph; sits outside the executor to avoid a per-flag branch in the eager
  // gate. Default: NOT_IMPLEMENTED. Overridden by DFlashDraftModel.
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

 protected:
  layer::LmHead lm_head_{nullptr};
  torch::Tensor d2t_;
};

template <typename ModelType>
class CausalLMImpl final : public CausalLM {
 public:
  explicit CausalLMImpl(const ModelContext& context, bool defer_lm_head = false)
      : CausalLMImpl(ModelType(context), context, defer_lm_head) {
    model_->eval();
  }

  CausalLMImpl(ModelType model,
               const ModelContext& context,
               bool defer_lm_head = false)
      : model_(std::move(model)), options_(context.get_tensor_options()) {
    const auto& args = context.get_model_args();
    if (!args.embedding_mode() && !args.encoder_embedding_mode()) {
      if (defer_lm_head) {
        lm_head_context_ = std::make_unique<ModelContext>(context);
      } else {
        lm_head_ = layer::LmHead(context);
        lm_head_->eval();
      }
    }
  }

  torch::nn::Module& model() override { return model_->model(); }

  torch::Tensor get_input_embeddings(const torch::Tensor& tokens) override {
    return model_->get_input_embeddings(tokens);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  bool requires_graph_forward_metadata() override {
    if constexpr (requires { model_->requires_graph_forward_metadata(); }) {
      return model_->requires_graph_forward_metadata();
    }
    return CausalLM::requires_graph_forward_metadata();
  }

  bool is_hybrid_linear_attention() override {
    if constexpr (requires { model_->is_hybrid_linear_attention(); }) {
      return model_->is_hybrid_linear_attention();
    }
    return CausalLM::is_hybrid_linear_attention();
  }

  bool supports_mla_graph_kv_bucketing() const override {
    if constexpr (requires { model_->supports_mla_graph_kv_bucketing(); }) {
      return model_->supports_mla_graph_kv_bucketing();
    }
    return CausalLM::supports_mla_graph_kv_bucketing();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    if constexpr (requires { model_->create_graph_forward_metadata_state(); }) {
      return model_->create_graph_forward_metadata_state();
    }
    return CausalLM::create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelParamsView& parameters) override {
    if constexpr (requires {
                    model_->prepare_graph_forward_metadata(
                        state, positions, parameters);
                  }) {
      model_->prepare_graph_forward_metadata(state, positions, parameters);
    }
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    d2t_ =
        load_model_weights(*model_, *loader, lm_head_, lm_head_context_.get());
    lm_head_context_.reset();
    if (model_->tie_word_embeddings() && !lm_head_.is_empty()) {
      auto embedding = model_->model().get_word_embedding();
      if (embedding.is_empty() || !embedding->is_weight_loaded()) {
        return;
      }
      auto head_weight = lm_head_->weight();
      auto embedding_weight = embedding->weight();
      // Tensor-parallel head and embedding weights can use different shard
      // axes.
      if (head_weight.defined() && embedding_weight.defined() &&
          head_weight.sizes() == embedding_weight.sizes() &&
          head_weight.scalar_type() == embedding_weight.scalar_type()) {
        lm_head_->tie_weights(embedding_weight);
      }
    }
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelParamsView& input_params) override {
    if constexpr (requires {
                    model_->write_context_kv(target_hidden,
                                             positions,
                                             device_cache_slots,
                                             kv_caches,
                                             input_params);
                  }) {
      return model_->write_context_kv(target_hidden,
                                      positions,
                                      device_cache_slots,
                                      kv_caches,
                                      input_params);
    }
    return CausalLM::write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

  DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) override {
    if constexpr (requires {
                    model_->dflash2_candidates(
                        hidden_states, unary_logits, anchor_token_ids);
                  }) {
      return model_->dflash2_candidates(
          hidden_states, unary_logits, anchor_token_ids);
    }
    return CausalLM::dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) override {
    if constexpr (requires {
                    model_->dspark_markov_bias(previous_token_ids);
                  }) {
      return model_->dspark_markov_bias(previous_token_ids);
    }
    return CausalLM::dspark_markov_bias(previous_token_ids);
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (requires { model_->lazy_load_model(std::move(loader)); }) {
      model_->lazy_load_model(std::move(loader));
    } else {
      CausalLM::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (requires { model_->free_model_weights(); }) {
      model_->free_model_weights();
    } else {
      CausalLM::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (requires { model_->reload_model_weights(); }) {
      model_->reload_model_weights();
    } else {
      CausalLM::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (requires { model_->reload_model_weights_from_device(); }) {
      model_->reload_model_weights_from_device();
    } else {
      CausalLM::reload_model_weights_from_device();
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    return model_->prepare_expert_weight(layer_id, expert_ids);
  }

  void start_expert_weight_transfer(int32_t layer_id) override {
    if constexpr (requires {
                    model_->start_expert_weight_transfer(layer_id);
                  }) {
      model_->start_expert_weight_transfer(layer_id);
    }
  }

  void update_expert_weight(int32_t layer_id) override {
    return model_->update_expert_weight(layer_id);
  }

  bool last_prepare_expert_weight_ok(int32_t layer_id) const override {
    if constexpr (requires {
                    model_->last_prepare_expert_weight_ok(layer_id);
                  }) {
      return model_->last_prepare_expert_weight_ok(layer_id);
    }
    return CausalLM::last_prepare_expert_weight_ok(layer_id);
  }

  void share_lm_head(CausalLM& target_model) override {
    if (lm_head_.is_empty() || !lm_head_->is_weight_loaded()) {
      auto head = target_model.get_lm_head();
      CHECK(!head.is_empty() && head->is_weight_loaded())
          << "Target model has no loaded native LM head";
      lm_head_ = std::move(head);
    }
  }

  void share_embed_tokens(CausalLM& target_model) override {
    auto& decoder = model_->model();
    auto own = decoder.get_word_embedding();
    if (own.is_empty() || !own->is_weight_loaded()) {
      const auto children = target_model.model().named_children();
      const auto* target_embedding = children.find(/*key=*/"embed_tokens");
      layer::WordEmbedding embedding(
          target_embedding == nullptr
              ? nullptr
              : std::dynamic_pointer_cast<layer::WordEmbeddingImpl>(
                    *target_embedding));
      CHECK(!embedding.is_empty() && embedding->is_weight_loaded())
          << "Target model has no loaded native word embedding";
      decoder.set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;
  torch::TensorOptions options_;
  std::unique_ptr<ModelContext> lm_head_context_;
};

}  // namespace xllm
