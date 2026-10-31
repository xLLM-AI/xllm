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
#include "core/framework/model/model_traits.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/quantization/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
}

namespace xlite {
class XliteModelHolder;
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

#if defined(USE_NPU)
  // xlite runtime access, nullptr for non-xlite models.
  virtual xlite::XliteModelHolder* get_xlite_holder() { return nullptr; }
#endif

  virtual void share_lm_head(CausalLM& /*target*/) {}

  virtual void share_embed_tokens(CausalLM& /*target*/) {}

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

  // DSpark ConfidenceHead: acceptance-prob estimate for adaptive-speculative
  // pruning over the whole draft block. hidden_all [num_reqs, num_spec, H],
  // prev_matrix [num_reqs, num_spec]; returns [num_reqs, num_spec] fp32 in
  // [0, 1]. When the confidence head is absent (not built or feature disabled),
  // returns an undefined tensor; the worker falls back to the sampler-gathered
  // draft prob.
  virtual torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) {
    return {};
  }
  virtual bool has_dspark_confidence_head() const { return false; }
};

template <typename ModelType>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(ModelType model, const ModelContext& context)
      : model_(std::move(model)), options_(context.get_tensor_options()) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    embedding_mode_ = args.embedding_mode();
    // DFlash2 shares the target lm-head after loading; skip registering a
    // draft-local copy so the unused vocabulary projection does not stay
    // resident on device.
    const bool shares_target_lm_head =
        is_dflash2_draft_model_type(args.model_type());
    // Models that own a non-standard head (e.g. rwkv7's nn::Linear) provide
    // their own logits; skip registering a framework-owned lm_head for them.
    const bool owns_custom_head = detail::has_logits<ModelType>::value;
    if (!embedding_mode_ && !shares_target_lm_head && !owns_custom_head) {
#if defined(USE_NPU)
      if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
        npu_lm_head_ =
            register_module("npu_lm_head", layer::NpuLmHead(context));
      } else
#endif
      {
        lm_head_ = register_module("lm_head", layer::LmHead(context));
      }
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  bool requires_graph_forward_metadata() override {
    if constexpr (detail::has_requires_graph_forward_metadata<
                      ModelType>::value) {
      return model_->requires_graph_forward_metadata();
    } else {
      return CausalLM::requires_graph_forward_metadata();
    }
  }

  bool is_hybrid_linear_attention() override {
    if constexpr (detail::has_is_hybrid_linear_attention<ModelType>::value) {
      return model_->is_hybrid_linear_attention();
    } else {
      return CausalLM::is_hybrid_linear_attention();
    }
  }

  bool supports_mla_graph_kv_bucketing() const override {
    if constexpr (detail::has_supports_mla_graph_kv_bucketing<
                      ModelType>::value) {
      return model_->supports_mla_graph_kv_bucketing();
    }
    return CausalLM::supports_mla_graph_kv_bucketing();
  }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    if constexpr (detail::has_create_graph_forward_metadata_state<
                      ModelType>::value) {
      return model_->create_graph_forward_metadata_state();
    } else {
      return CausalLM::create_graph_forward_metadata_state();
    }
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelParamsView& parameters) override {
    if constexpr (detail::has_prepare_graph_forward_metadata<
                      ModelType>::value) {
      model_->prepare_graph_forward_metadata(state, positions, parameters);
    }
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (detail::has_logits<ModelType>::value) {
      return model_->logits(hidden_states, seleted_idxes);
    }
#if defined(USE_NPU)
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return npu_lm_head_(hidden_states, seleted_idxes, 0);
    }
#endif
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_logits<ModelType>::value) {
      return CausalLM::logits(hidden_states, seleted_idxes, out_hidden);
    }
#if defined(USE_NPU)
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return npu_lm_head_->forward_with_hidden(
          hidden_states, seleted_idxes, out_hidden, 0);
    }
#endif
    return lm_head_->forward_with_hidden(
        hidden_states, seleted_idxes, out_hidden, 0);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    if (!embedding_mode_ && !detail::has_logits<ModelType>::value) {
      for (const auto& state_dict : loader->get_state_dicts()) {
#if defined(USE_NPU)
        if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
          if (tie_word_embeddings_) {
            npu_lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix(std::vector<std::string>{
                    "model.embed_tokens.", "embed_tokens."}));
          } else {
            npu_lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix("lm_head."));
          }
        } else
#endif
        {
          if constexpr (detail::has_load_lm_head<ModelType>::value) {
            model_->load_lm_head(*state_dict, lm_head_);
          } else if (tie_word_embeddings_) {
            lm_head_->load_state_dict(state_dict->get_dict_with_prefix(
                std::vector<std::string>{"embed_tokens.", "embed."}));
          } else {
            lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix("lm_head."));
          }
        }
      }
    }
    model_->load_model(std::move(loader));
    if (!embedding_mode_ && !detail::has_logits<ModelType>::value) {
#if defined(USE_NPU)
      if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
        npu_lm_head_->verify_loaded_weights(
            tie_word_embeddings_ ? "embed_tokens." : "lm_head.");
        npu_lm_head_->merge_loaded_weights();
      }
#endif
    }
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelParamsView& input_params) override {
    if constexpr (detail::has_write_context_kv<ModelType>::value) {
      return model_->write_context_kv(target_hidden,
                                      positions,
                                      device_cache_slots,
                                      kv_caches,
                                      input_params);
    } else {
      return CausalLM::write_context_kv(target_hidden,
                                        positions,
                                        device_cache_slots,
                                        kv_caches,
                                        input_params);
    }
  }

  DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) override {
    if constexpr (detail::has_dflash2_candidates<ModelType>::value) {
      return model_->dflash2_candidates(
          hidden_states, unary_logits, anchor_token_ids);
    }
    return CausalLM::dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) override {
    if constexpr (detail::has_dspark_markov_bias<ModelType>::value) {
      return model_->dspark_markov_bias(previous_token_ids);
    }
    return CausalLM::dspark_markov_bias(previous_token_ids);
  }

  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) override {
    if constexpr (detail::has_dspark_confidence_probs<ModelType>::value) {
      return model_->dspark_confidence_probs(hidden_all, prev_matrix);
    }
    return CausalLM::dspark_confidence_probs(hidden_all, prev_matrix);
  }

  bool has_dspark_confidence_head() const override {
    if constexpr (detail::has_has_dspark_confidence_head<ModelType>::value) {
      return model_->has_dspark_confidence_head();
    }
    return false;
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<ModelType>::value) {
      model_->lazy_load_model(std::move(loader));
    } else {
      CausalLM::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (detail::has_free_model_weights<ModelType>::value) {
      model_->free_model_weights();
    } else {
      CausalLM::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (detail::has_reload_model_weights<ModelType>::value) {
      model_->reload_model_weights();
    } else {
      CausalLM::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (detail::has_reload_model_weights_from_device<
                      ModelType>::value) {
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
    if constexpr (detail::has_start_expert_weight_transfer<ModelType>::value) {
      model_->start_expert_weight_transfer(layer_id);
    }
  }

  void update_expert_weight(int32_t layer_id) override {
    return model_->update_expert_weight(layer_id);
  }

  bool last_prepare_expert_weight_ok(int32_t layer_id) const override {
    if constexpr (detail::has_last_prepare_expert_weight_ok<ModelType>::value) {
      return model_->last_prepare_expert_weight_ok(layer_id);
    } else {
      return CausalLM::last_prepare_expert_weight_ok(layer_id);
    }
  }

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return model_->get_npu_lm_head();
    }
    return npu_lm_head_;
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    if constexpr (detail::has_set_npu_lm_head<ModelType>::value) {
      model_->set_npu_lm_head(head);
      return;
    }
    npu_lm_head_ = head;
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<ModelType>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return CausalLM::get_npu_word_embedding();
    }
  }

  bool has_restored_npu_word_embedding() override {
    if constexpr (detail::has_restored_npu_word_embedding<ModelType>::value) {
      return model_->has_restored_npu_word_embedding();
    } else {
      return CausalLM::has_restored_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<ModelType>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      CausalLM::set_npu_word_embedding(embedding);
    }
  }

  void set_restored_npu_word_embedding(
      layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_restored_npu_word_embedding<
                      ModelType>::value) {
      model_->set_restored_npu_word_embedding(embedding);
    } else {
      CausalLM::set_restored_npu_word_embedding(embedding);
    }
  }

  bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                       Stream* compute_stream,
                                       int32_t num_cached_slots,
                                       int32_t requested_rolling_slots,
                                       const std::string& model_id) override {
    if constexpr (detail::has_init_or_refresh_rolling_runtime<
                      ModelType>::value) {
      return model_->init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
    }
    return CausalLM::init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
  }

  // Forward to inner ModelType.
  xlite::XliteModelHolder* get_xlite_holder() override {
    if constexpr (detail::has_get_xlite_holder<ModelType>::value) {
      return model_->get_xlite_holder();
    }
    return CausalLM::get_xlite_holder();
  }
#endif

  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<ModelType>::value) {
      return model_->get_lm_head();
    }
    return lm_head_;
  }

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<ModelType>::value) {
      model_->set_lm_head(head);
      return;
    }
    lm_head_ = head;
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<ModelType>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<ModelType>::value) {
      model_->set_word_embedding(embedding);
    } else {
      CausalLM::set_word_embedding(embedding);
    }
  }

  void share_lm_head(CausalLM& target) override {
#if defined(USE_NPU)
    if constexpr (detail::has_set_npu_lm_head<ModelType>::value) {
      if (get_npu_lm_head().is_empty() ||
          !get_npu_lm_head()->is_weight_loaded()) {
        auto head = target.get_npu_lm_head();
        set_npu_lm_head(head);
      }
      return;
    }
    // Framework-owned head with no model setter (e.g. DFlash2 shares the
    // target head and skips registering a draft-local copy, leaving the
    // holder empty until shared here). Models with a custom head own their
    // logits and must not share.
    if constexpr (!detail::has_logits<ModelType>::value) {
      if (npu_lm_head_.is_empty() || !npu_lm_head_->is_weight_loaded()) {
        npu_lm_head_ = target.get_npu_lm_head();
      }
    }
    return;
#endif
    if constexpr (detail::has_set_lm_head<ModelType>::value) {
      if (get_lm_head().is_empty() || !get_lm_head()->is_weight_loaded()) {
        auto head = target.get_lm_head();
        set_lm_head(head);
      }
      return;
    }
    // Framework-owned head with no model setter; share the target head when the
    // holder is empty (DFlash2) or not yet loaded. Models with a custom head
    // own their logits and must not share.
    if constexpr (!detail::has_logits<ModelType>::value) {
      if (lm_head_.is_empty() || !lm_head_->is_weight_loaded()) {
        lm_head_ = target.get_lm_head();
      }
    }
  }

  void share_embed_tokens(CausalLM& target) override {
#if defined(USE_NPU)
    if constexpr (detail::has_set_npu_word_embedding<ModelType>::value) {
      if (!get_npu_word_embedding().is_empty() &&
          get_npu_word_embedding()->is_weight_loaded()) {
        return;
      }
      auto word_embedding = target.get_npu_word_embedding();
      if (target.has_restored_npu_word_embedding()) {
        set_restored_npu_word_embedding(word_embedding);
        return;
      }
      set_npu_word_embedding(word_embedding);
      return;
    }
#endif
    if constexpr (detail::has_set_word_embedding<ModelType>::value) {
      if (!get_word_embedding().is_empty() &&
          get_word_embedding()->is_weight_loaded()) {
        return;
      }
      auto word_embedding = target.get_word_embedding();
      set_word_embedding(word_embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;

  torch::TensorOptions options_;

  bool tie_word_embeddings_{false};
  bool embedding_mode_{false};

#if defined(USE_NPU)
  layer::NpuLmHead npu_lm_head_{nullptr};
#endif
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm
