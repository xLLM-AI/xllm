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
#include "core/util/tensor_helper.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
class LmHead;
#if defined(USE_NPU)
class NpuLmHead;
#endif
}  // namespace layer

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
    torch::Tensor pooled;
    gather_rows_if_defined(hidden_states, seleted_idxes, pooled);
    return pooled;
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
    gather_rows_if_defined(hidden_states, seleted_idxes, out_hidden);
    return logits(hidden_states, seleted_idxes);
  }

#if defined(USE_NPU)
  // xlite runtime access, nullptr for non-xlite models.
  virtual xlite::XliteModelHolder* get_xlite_holder() { return nullptr; }

  // Head channel for cross-model sharing (draft adopts the target's head).
  // Null for models that host no vocabulary head.
  virtual layer::NpuLmHead get_npu_lm_head() { return nullptr; }
#endif

  // CPU-head channel, same contract as get_npu_lm_head().
  virtual layer::LmHead get_lm_head() { return nullptr; }

  // Adopts the target's vocabulary head when this model hosts none of its own
  // loaded. Default: nothing to adopt. Families with an LmHeadOwner host
  // implement this through CausalLMImpl, which adopts via the owner.
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

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    torch::Tensor out_hidden;
    return logits(hidden_states, seleted_idxes, out_hidden);
  }

  // The vocabulary head is owned by the family model (see
  // detail::LmHeadOwner); headless families never route here because the
  // runtime only calls logits() for generation.
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_lm_head_logits<ModelType>::value) {
      return model_->logits(hidden_states, seleted_idxes, out_hidden);
    }
    NOT_IMPLEMENTED();
    return {};
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
                        detail::has_dspark_confidence_probs,
                        dspark_confidence_probs,
                        torch::Tensor,
                        (const torch::Tensor& hidden_all,
                         const torch::Tensor& prev_matrix),
                        (hidden_all, prev_matrix))

  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_has_dspark_confidence_head,
                        has_dspark_confidence_head,
                        bool,
                        () const,
                        ())

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

#if defined(USE_NPU)
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_get_npu_word_embedding,
                        get_npu_word_embedding,
                        layer::NpuWordEmbedding,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_family_get_npu_lm_head,
                        get_npu_lm_head,
                        layer::NpuLmHead,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_set_npu_word_embedding,
                        set_npu_word_embedding,
                        void,
                        (layer::NpuWordEmbedding & embedding),
                        (embedding))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_init_or_refresh_rolling_runtime,
                        init_or_refresh_rolling_runtime,
                        bool,
                        (Stream * load_stream,
                         Stream* compute_stream,
                         int32_t num_cached_slots,
                         int32_t requested_rolling_slots,
                         const std::string& model_id),
                        (load_stream,
                         compute_stream,
                         num_cached_slots,
                         requested_rolling_slots,
                         model_id))
  // Forward to inner ModelType.
  XLLM_FORWARD_OPTIONAL(ModelType,
                        CausalLM,
                        detail::has_get_xlite_holder,
                        get_xlite_holder,
                        xlite::XliteModelHolder*,
                        (),
                        ())
#endif

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

  // Virtual head channel mirroring share_embed_tokens below: adopts the
  // target's head through the owner when this model hosts none of its own
  // loaded. An already-allocated eager head is replaced so its vocab-sized
  // device tensor is released instead of lingering as an orphaned child.
  // Works across model types because both endpoints speak CausalLM.
  void share_lm_head(CausalLM& target) override {
#if defined(USE_NPU)
    if constexpr (detail::has_adopt_npu_lm_head<ModelType>::value) {
      auto own = get_npu_lm_head();
      if (!detail::weight_loaded(own)) {
        model_->adopt_lm_head(*model_, target.get_npu_lm_head());
      }
      return;
    }
#endif
    if constexpr (detail::has_adopt_lm_head<ModelType>::value) {
      auto own = get_lm_head();
      if (!detail::weight_loaded(own)) {
        model_->adopt_lm_head(*model_, target.get_lm_head());
      }
      return;
    }
    CausalLM::share_lm_head(target);
  }

  void share_embed_tokens(CausalLM& target) override {
#if defined(USE_NPU)
    if constexpr (detail::has_set_npu_word_embedding<ModelType>::value) {
      auto own = get_npu_word_embedding();
      if (!detail::weight_loaded(own)) {
        auto target_embed = target.get_npu_word_embedding();
        set_npu_word_embedding(target_embed);
      }
      return;
    }
#endif
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

}  // namespace xllm
