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
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/multimodal/mm_data_item.h"
#include "core/framework/quantization/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
class LmHead;
}  // namespace layer

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

  // Multimodal entry points, present only on VLM families.
  virtual MMDict encode(const ModelParamsView& parameters) {
    (void)parameters;
    LOG(FATAL) << "encode is only available on multimodal models.";
    return {};
  }

  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) {
    (void)input_ids;
    (void)input_params;
    LOG(FATAL) << "get_input_embeddings is only available on multimodal "
               << "models.";
    return {};
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    torch::Tensor pooled;
    gather_rows_if_defined(hidden_states, seleted_idxes, pooled);
    return pooled;
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens] (undefined = all rows)
  // out_hidden: [num_selected, hidden_size]; logits: [num_selected, vocab]
  virtual void project_lm_head(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes,
                               torch::Tensor& out_hidden,
                               torch::Tensor& logits) = 0;

  virtual layer::LmHead get_lm_head() { return nullptr; }

  virtual void share_lm_head(CausalLM& /*target*/) {}

  virtual void share_embed_tokens(CausalLM& /*target*/) {}

  virtual void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) {}

  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}

  virtual void update_expert_weight(int32_t /*layer_id*/) {}

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
};

template <typename ModelType>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(ModelType model, const ModelContext& context)
      : model_(std::move(model)), options_(context.get_tensor_options()) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_requires_graph_forward_metadata,
                        requires_graph_forward_metadata,
                        bool,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_is_hybrid_linear_attention,
                        is_hybrid_linear_attention,
                        bool,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_supports_mla_graph_kv_bucketing,
                        supports_mla_graph_kv_bucketing,
                        bool,
                        () const,
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_create_graph_forward_metadata_state,
                        create_graph_forward_metadata_state,
                        std::unique_ptr<ModelGraphMetadataState>,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_prepare_graph_forward_metadata,
                        prepare_graph_forward_metadata,
                        void,
                        (ModelGraphMetadataState * state,
                         const torch::Tensor& positions,
                         ModelParamsView& params),
                        (state, positions, params))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_pooler,
                        pooler,
                        torch::Tensor,
                        (const torch::Tensor& hidden_states,
                         const torch::Tensor& seleted_idxes),
                        (hidden_states, seleted_idxes))

  void project_lm_head(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden,
                       torch::Tensor& logits) override {
    if constexpr (detail::has_project_lm_head<ModelType>::value) {
      model_->project_lm_head(hidden_states, seleted_idxes, out_hidden, logits);
    } else {
      NOT_IMPLEMENTED();
    }
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
    // The base get_word_embedding() aborts for models without one, so both
    // conditions are required before tying.
    if constexpr (detail::has_tie_lm_head_weights<ModelType>::value &&
                  detail::has_get_word_embedding<ModelType>::value) {
      model_->tie_lm_head_weights(get_word_embedding());
    }
  }

  XLLM_FORWARD_OPTIONAL(
      ModelType,
      CausalLM,
      detail::has_write_context_kv,
      write_context_kv,
      ModelOutput,
      (const torch::Tensor& target_hidden,
       const torch::Tensor& positions,
       const torch::Tensor& device_cache_slots,
       std::vector<KVCache>& kv_caches,
       const ModelParamsView& input_params),
      (target_hidden, positions, device_cache_slots, kv_caches, input_params))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_dflash2_candidates,
                        dflash2_candidates,
                        DFlash2CandidateOutput,
                        (const torch::Tensor& hidden_states,
                         const torch::Tensor& unary_logits,
                         const torch::Tensor& anchor_token_ids),
                        (hidden_states, unary_logits, anchor_token_ids))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_dspark_markov_bias,
                        dspark_markov_bias,
                        torch::Tensor,
                        (const torch::Tensor& previous_token_ids),
                        (previous_token_ids))

  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_lazy_load_model,
                        lazy_load_model,
                        void,
                        (std::unique_ptr<ModelLoader> loader),
                        (std::move(loader)))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_free_model_weights,
                        free_model_weights,
                        void,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_reload_model_weights,
                        reload_model_weights,
                        void,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_reload_model_weights_from_device,
                        reload_model_weights_from_device,
                        void,
                        (),
                        ())

  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_prepare_expert_weight,
                        prepare_expert_weight,
                        void,
                        (int32_t layer_id,
                         const std::vector<int32_t>& expert_ids),
                        (layer_id, expert_ids))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_start_expert_weight_transfer,
                        start_expert_weight_transfer,
                        void,
                        (int32_t layer_id),
                        (layer_id))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_update_expert_weight,
                        update_expert_weight,
                        void,
                        (int32_t layer_id),
                        (layer_id))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_last_prepare_expert_weight_ok,
                        last_prepare_expert_weight_ok,
                        bool,
                        (int32_t layer_id) const,
                        (layer_id))

  // CPU-flavor head channel; the share_lm_head virtual below reads it to
  // adopt the target's head on hosts that own none of their own.
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_family_get_lm_head,
                        get_lm_head,
                        layer::LmHead,
                        (),
                        ())

  // The interface and backbone hook names differ, so the macro can't be used.
  MMDict encode(const ModelParamsView& parameters) override {
    if constexpr (detail::has_get_multimodal_embeddings<ModelType>::value) {
      return model_->get_multimodal_embeddings(parameters);
    }
    return CausalLM::encode(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) override {
    if constexpr (detail::has_vlm_get_input_embeddings<ModelType>::value) {
      return model_->get_input_embeddings(input_ids, input_params);
    }
    return CausalLM::get_input_embeddings(input_ids, input_params);
  }

  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_get_word_embedding,
                        get_word_embedding,
                        layer::WordEmbedding,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_set_word_embedding,
                        set_word_embedding,
                        void,
                        (layer::WordEmbedding & embedding),
                        (embedding))

  void share_lm_head(CausalLM& target) override {
    if constexpr (detail::has_adopt_lm_head<ModelType>::value) {
      model_->adopt_lm_head(*model_, target.get_lm_head());
      return;
    }
    CausalLM::share_lm_head(target);
  }

  void share_embed_tokens(CausalLM& target) override {
    if constexpr (detail::has_set_word_embedding<ModelType>::value) {
      auto own = get_word_embedding();
      if (!detail::weight_loaded(own)) {
        auto target_embed = target.get_word_embedding();
        set_word_embedding(target_embed);
      }
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 protected:
  ModelType model_;

 private:
  torch::TensorOptions options_;
};

namespace detail {

// The host must derive from torch::nn::Module.
class LmHeadOwner {
 public:
  void init_lm_head(torch::nn::Module& host,
                    const ModelContext& context,
                    bool defer_head = false) {
    read_model_args(context);
    if (embedding_mode_ || defer_head) {
      return;
    }
    lm_head_ = host.register_module("lm_head", layer::LmHead(context));
  }

  layer::LmHead get_lm_head() { return lm_head_; }

  void project_lm_head(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden,
                       torch::Tensor& logits) {
    logits = lm_head_->forward_with_hidden(
        hidden_states, selected_idxes, out_hidden);
  }

  bool head_weight_loaded() const { return weight_loaded(lm_head_); }

  virtual void load_lm_head_weights(const StateDict& state_dict,
                                    layer::LmHead& head,
                                    bool tie_word_embeddings) {
    if (tie_word_embeddings) {
      static const std::vector<std::string> kTorchTiedEmbeddingPrefixes = {
          "model.embed_tokens.", "embed_tokens.", "embed."};
      head->load_state_dict(
          state_dict.get_dict_with_prefix(kTorchTiedEmbeddingPrefixes));
    } else {
      static const std::vector<std::string> kTorchLmHeadPrefixes = {
          "lm_head.", "model.lm_head.", "model.head.", "head."};
      head->load_state_dict(
          state_dict.get_dict_with_prefix(kTorchLmHeadPrefixes));
    }
  }

  void load_lm_head_weights(const std::unique_ptr<ModelLoader>& loader) {
    if (embedding_mode_ || lm_head_.is_empty() || head_weight_loaded()) {
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_lm_head_weights(*state_dict, lm_head_, tie_word_embeddings_);
    }
  }

  // Bind the tied lm_head weight to embed_tokens.weight so they share one
  // storage. TP>1 shards them on orthogonal dims, so they cannot alias and the
  // head keeps its own copy.
  void tie_lm_head_weights(const layer::WordEmbedding& embed_tokens) {
    if (!tie_word_embeddings_ || embedding_mode_ || lm_head_.is_empty()) {
      return;
    }
    if (embed_tokens.is_empty()) {
      return;
    }
    auto head_weight = lm_head_->weight();
    auto embed_weight = embed_tokens->weight();
    if (!head_weight.defined() || !embed_weight.defined() ||
        head_weight.sizes() != embed_weight.sizes() ||
        head_weight.scalar_type() != embed_weight.scalar_type()) {
      return;
    }
    lm_head_->tie_weights(embed_weight);
  }

  void adopt_lm_head(torch::nn::Module& host, layer::LmHead target_head) {
    if (head_weight_loaded()) {
      return;
    }
    // replace_module (not assignment) so an already-allocated eager head
    // releases its vocab-sized device tensor.
    if (lm_head_.is_empty()) {
      lm_head_ = host.register_module("lm_head", std::move(target_head));
    } else {
      lm_head_ = host.replace_module("lm_head", std::move(target_head));
    }
  }

 protected:
  void read_model_args(const ModelContext& context) {
    const auto& model_args = context.get_model_args();
    tie_word_embeddings_ = model_args.tie_word_embeddings();
    embedding_mode_ = model_args.embedding_mode();
  }

  layer::LmHead lm_head_{nullptr};
  bool tie_word_embeddings_ = false;
  bool embedding_mode_ = false;
};

// VLM-wrapper mixin: delegates the language-tower surface via requires-clauses.
template <typename LanguageModel>
class VlmLanguageModelHost {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) {
    return language_model_->forward(tokens, positions, kv_caches, input_params);
  }

  void project_lm_head(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden,
                       torch::Tensor& logits) {
    language_model_->project_lm_head(
        hidden_states, seleted_idxes, out_hidden, logits);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->pooler(hidden_states, seleted_idxes);
  }

  layer::WordEmbedding get_word_embedding()
    requires has_get_word_embedding<LanguageModel>::value
  {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& embedding)
    requires has_set_word_embedding<LanguageModel>::value
  {
    language_model_->set_word_embedding(embedding);
  }

  bool supports_mla_graph_kv_bucketing() const
    requires has_supports_mla_graph_kv_bucketing<LanguageModel>::value
  {
    return language_model_->supports_mla_graph_kv_bucketing();
  }

  bool is_hybrid_linear_attention()
    requires has_is_hybrid_linear_attention<LanguageModel>::value
  {
    return language_model_->is_hybrid_linear_attention();
  }

 protected:
  LanguageModel language_model_{nullptr};
};

}  // namespace detail
}  // namespace xllm
