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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/model_loader/model_weight_loader.h"
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
  using LmHead = std::function<torch::Tensor(const torch::Tensor&)>;

 public:
  torch::Tensor logits_processor(const torch::Tensor& hidden_states,
                                 const torch::Tensor& lm_head_input_indices,
                                 torch::Tensor* output_hidden_states) const {
    if (output_hidden_states == nullptr) {
      return lm_head_(
          hidden_states.index_select(/*dim=*/0, lm_head_input_indices));
    }
    const auto indices = lm_head_input_indices.to(
        torch::dtype(torch::kLong).device(hidden_states.device()));
    auto sample_hidden_states = hidden_states.index_select(/*dim=*/0, indices);
    auto logits = lm_head_(sample_hidden_states);
    *output_hidden_states = std::move(sample_hidden_states);
    return logits;
  }

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

  virtual void share_lm_head(CausalLM& target_model) {
    auto* draft_head = lm_head_.target<layer::LmHead>();
    if (draft_head == nullptr ||
        (!draft_head->is_empty() && (*draft_head)->is_weight_loaded())) {
      return;
    }

    const auto* target_head = target_model.lm_head_.target<layer::LmHead>();
    CHECK(target_head != nullptr && !target_head->is_empty())
        << "Target model has no native LM head";
    *draft_head = *target_head;
  }

  const torch::Tensor& d2t() const { return d2t_; }

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
  explicit CausalLM(LmHead lm_head = {}) : lm_head_(std::move(lm_head)) {}

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

  LmHead lm_head_;

 private:
  torch::Tensor d2t_;
};

template <typename ModelType>
class CausalLMImpl : public CausalLM {
  using ModelLmHead = decltype(std::declval<ModelType&>()->make_lm_head(
      std::declval<const ModelContext&>()));

 public:
  CausalLMImpl(ModelType model, const ModelContext& context)
      : CausalLM(model->make_lm_head(context)),
        model_(std::move(model)),
        options_(context.get_tensor_options()) {}

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
    const auto& state_dicts = loader->get_state_dicts();
    load_d2t(state_dicts);
    auto& lm_head = *lm_head_.target<ModelLmHead>();
    load_model_weights(
        *model_, state_dicts, lm_head, ModelType::ContainedType::kWeightPrefix);
    if constexpr (requires {
                    model_->tie_word_embeddings();
                    lm_head->tie_weights(
                        model_->get_word_embedding()->weight());
                  }) {
      if (!model_->tie_word_embeddings() || lm_head.is_empty()) {
        return;
      }
      auto embedding = model_->get_word_embedding();
      if (embedding.is_empty() || !embedding->is_weight_loaded()) {
        return;
      }
      auto head_weight = lm_head->weight();
      auto embedding_weight = embedding->weight();
      // Tensor-parallel head and embedding weights can use different shard
      // axes.
      if (head_weight.defined() && embedding_weight.defined() &&
          head_weight.sizes() == embedding_weight.sizes() &&
          head_weight.scalar_type() == embedding_weight.scalar_type()) {
        lm_head->tie_weights(embedding_weight);
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

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (requires { model_->get_word_embedding(); }) {
      return model_->get_word_embedding();
    }
    return CausalLM::get_word_embedding();
  }

  void share_embed_tokens(CausalLM& target_model) override {
    if constexpr (requires(layer::WordEmbedding& embedding) {
                    model_->get_word_embedding();
                    model_->set_word_embedding(embedding);
                  }) {
      auto own = model_->get_word_embedding();
      if (own.is_empty() || !own->is_weight_loaded()) {
        auto embedding = target_model.get_word_embedding();
        model_->set_word_embedding(embedding);
      }
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
