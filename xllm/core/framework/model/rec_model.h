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

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/framework/model/model.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model/rec_model_kind.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/model_context/model_context.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/util/tensor_helper.h"

namespace xllm {

class RecModel : public torch::nn::Module, public Model {
 public:
  ~RecModel() override = default;

  virtual RecModelKind model_kind() const = 0;

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const RecModelParams& parameters) = 0;

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    torch::Tensor pooled;
    gather_rows_if_defined(hidden_states, selected_idxes, pooled);
    return pooled;
  }

  // hidden_states: [num_tokens, hidden_size]; selected_idxes: [num_tokens] or
  // undefined; out_hidden (optional export): [num_selected_tokens, hidden_size]
  // gathered rows; returns [num_selected_tokens, vocab_size].
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes,
                               torch::Tensor& out_hidden) = 0;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    torch::Tensor out_hidden;
    return logits(hidden_states, selected_idxes, out_hidden);
  }
};

namespace detail {

template <typename ModelType, typename RecType>
class RecModelImpl : public RecType {
  static_assert(std::is_base_of_v<RecModel, RecType>);

 public:
  RecModelImpl(ModelType model, const ModelContext& context)
      : model_(std::move(model)), options_(context.get_tensor_options()) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const RecModelParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_pooler,
                        pooler,
                        torch::Tensor,
                        (const torch::Tensor& hidden_states,
                         const torch::Tensor& selected_idxes),
                        (hidden_states, selected_idxes))

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_lm_head_logits<ModelType>::value) {
      return model_->logits(hidden_states, selected_idxes, out_hidden);
    }
    NOT_IMPLEMENTED();
    return {};
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_lazy_load_model,
                        lazy_load_model,
                        void,
                        (std::unique_ptr<ModelLoader> loader),
                        (std::move(loader)))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_free_model_weights,
                        free_model_weights,
                        void,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_reload_model_weights,
                        reload_model_weights,
                        void,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_reload_model_weights_from_device,
                        reload_model_weights_from_device,
                        void,
                        (),
                        ())

#if defined(USE_NPU)
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_get_npu_word_embedding,
                        get_npu_word_embedding,
                        layer::NpuWordEmbedding,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_set_npu_word_embedding,
                        set_npu_word_embedding,
                        void,
                        (layer::NpuWordEmbedding & embedding),
                        (embedding))
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
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
#endif

  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_get_word_embedding,
                        get_word_embedding,
                        layer::WordEmbedding,
                        (),
                        ())
  XLLM_FORWARD_OPTIONAL(ModelType,
                        RecType,
                        detail::has_set_word_embedding,
                        set_word_embedding,
                        void,
                        (layer::WordEmbedding & embedding),
                        (embedding))

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 protected:
  ModelType model_;

 private:
  torch::TensorOptions options_;
};

}  // namespace detail

}  // namespace xllm
