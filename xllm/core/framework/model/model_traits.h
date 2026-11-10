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

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace xllm {
class ModelParamsView;
struct ModelGraphMetadataState;
class KVCache;

namespace layer {
class LmHead;
class WordEmbedding;
}  // namespace layer

namespace detail {

// The macro must expand to a real virtual override: a member function
// template would hide the base virtual and silently dispatch base defaults
// through base pointers.
#define XLLM_FORWARD_OPTIONAL(ModelType, Base, trait, hook, ret, sig, args) \
  ret hook sig override {                                                   \
    if constexpr (trait<ModelType>::value) {                                \
      return model_->hook args;                                             \
    } else {                                                                \
      return Base::hook args;                                               \
    }                                                                       \
  }

template <typename T, typename = void>
struct has_project_lm_head : std::false_type {};

template <typename T>
struct has_project_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->project_lm_head(
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>(),
        std::declval<torch::Tensor&>(),
        std::declval<torch::Tensor&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_get_word_embedding : std::false_type {};

template <typename T>
struct has_get_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_family_get_lm_head : std::false_type {};

template <typename T>
struct has_family_get_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->get_lm_head())>> : std::true_type {
};

// adopt_lm_head(host, head) exists only on LmHeadOwner hosts.
template <typename T, typename = void>
struct has_adopt_lm_head : std::false_type {};

template <typename T>
struct has_adopt_lm_head<T,
                         std::void_t<decltype(std::declval<T>()->adopt_lm_head(
                             std::declval<torch::nn::Module&>(),
                             std::declval<layer::LmHead>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_word_embedding : std::false_type {};

template <typename T>
struct has_set_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_word_embedding(
        std::declval<layer::WordEmbedding&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_lazy_load_model : std::false_type {};

template <typename T>
struct has_lazy_load_model<
    T,
    std::void_t<decltype(std::declval<T>()->lazy_load_model(
        std::declval<std::unique_ptr<ModelLoader>>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_free_model_weights : std::false_type {};

template <typename T>
struct has_free_model_weights<
    T,
    std::void_t<decltype(std::declval<T>()->free_model_weights())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_reload_model_weights : std::false_type {};

template <typename T>
struct has_reload_model_weights<
    T,
    std::void_t<decltype(std::declval<T>()->reload_model_weights())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_reload_model_weights_from_device : std::false_type {};

template <typename T>
struct has_reload_model_weights_from_device<
    T,
    std::void_t<
        decltype(std::declval<T>()->reload_model_weights_from_device())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_requires_graph_forward_metadata : std::false_type {};

template <typename T>
struct has_requires_graph_forward_metadata<
    T,
    std::void_t<decltype(std::declval<T>()->requires_graph_forward_metadata())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_last_prepare_expert_weight_ok : std::false_type {};

template <typename T>
struct has_last_prepare_expert_weight_ok<
    T,
    std::void_t<decltype(std::declval<T>()->last_prepare_expert_weight_ok(
        std::declval<int32_t>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_start_expert_weight_transfer : std::false_type {};

template <typename T>
struct has_start_expert_weight_transfer<
    T,
    std::void_t<decltype(std::declval<T>()->start_expert_weight_transfer(
        std::declval<int32_t>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_prepare_expert_weight : std::false_type {};

template <typename T>
struct has_prepare_expert_weight<
    T,
    std::void_t<decltype(std::declval<T>()->prepare_expert_weight(
        std::declval<int32_t>(),
        std::declval<const std::vector<int32_t>&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_update_expert_weight : std::false_type {};

template <typename T>
struct has_update_expert_weight<
    T,
    std::void_t<decltype(std::declval<T>()->update_expert_weight(
        std::declval<int32_t>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_is_hybrid_linear_attention : std::false_type {};

template <typename T>
struct has_is_hybrid_linear_attention<
    T,
    std::void_t<decltype(std::declval<T>()->is_hybrid_linear_attention())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_supports_mla_graph_kv_bucketing : std::false_type {};

template <typename T>
struct has_supports_mla_graph_kv_bucketing<
    T,
    std::void_t<decltype(std::declval<T>()->supports_mla_graph_kv_bucketing())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_create_graph_forward_metadata_state : std::false_type {};

template <typename T>
struct has_create_graph_forward_metadata_state<
    T,
    std::void_t<
        decltype(std::declval<T>()->create_graph_forward_metadata_state())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_prepare_graph_forward_metadata : std::false_type {};

template <typename T>
struct has_prepare_graph_forward_metadata<
    T,
    std::void_t<decltype(std::declval<T>()->prepare_graph_forward_metadata(
        std::declval<ModelGraphMetadataState*>(),
        std::declval<const torch::Tensor&>(),
        std::declval<ModelParamsView&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_pooler : std::false_type {};

template <typename T>
struct has_pooler<T,
                  std::void_t<decltype(std::declval<T>()->pooler(
                      std::declval<const torch::Tensor&>(),
                      std::declval<const torch::Tensor&>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_write_context_kv : std::false_type {};

template <typename T>
struct has_write_context_kv<
    T,
    std::void_t<decltype(std::declval<T>()->write_context_kv(
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>(),
        std::declval<std::vector<KVCache>&>(),
        std::declval<const ModelParamsView&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_dflash2_candidates : std::false_type {};

template <typename T>
struct has_dflash2_candidates<
    T,
    std::void_t<decltype(std::declval<T>()->dflash2_candidates(
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>(),
        std::declval<const torch::Tensor&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_dspark_markov_bias : std::false_type {};

template <typename T>
struct has_dspark_markov_bias<
    T,
    std::void_t<decltype(std::declval<T>()->dspark_markov_bias(
        std::declval<const torch::Tensor&>()))>> : std::true_type {};

// An empty holder counts as not loaded, so its owner may adopt the target's.
template <typename Holder>
bool weight_loaded(const Holder& holder) {
  return !holder.is_empty() && holder->is_weight_loaded();
}

template <typename T, typename = void>
struct has_get_multimodal_embeddings : std::false_type {};

template <typename T>
struct has_get_multimodal_embeddings<
    T,
    std::void_t<decltype(std::declval<T>()->get_multimodal_embeddings(
        std::declval<const ModelParamsView&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_vlm_get_input_embeddings : std::false_type {};

template <typename T>
struct has_vlm_get_input_embeddings<
    T,
    std::void_t<decltype(std::declval<T>()->get_input_embeddings(
        std::declval<const torch::Tensor&>(),
        std::declval<const ModelParamsView&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_tie_lm_head_weights : std::false_type {};

template <typename T>
struct has_tie_lm_head_weights<
    T,
    std::void_t<decltype(std::declval<T>()->tie_lm_head_weights(
        std::declval<layer::WordEmbedding>()))>> : std::true_type {};

}  // namespace detail
}  // namespace xllm
