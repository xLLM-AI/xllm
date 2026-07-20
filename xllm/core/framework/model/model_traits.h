/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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
struct ModelInputParams;
struct ModelGraphMetadataState;
class KVCache;

namespace layer {
class LmHead;
class WordEmbedding;
#if defined(USE_NPU)
class NpuLmHead;
class NpuWordEmbedding;
#endif
}  // namespace layer

namespace detail {
template <typename T, typename = void>
struct has_get_lm_head : std::false_type {};

template <typename T>
struct has_get_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->get_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_lm_head : std::false_type {};

template <typename T>
struct has_set_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->set_lm_head(
                           std::declval<layer::LmHead&>()))>> : std::true_type {
};

template <typename T, typename = void>
struct has_get_word_embedding : std::false_type {};

template <typename T>
struct has_get_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_word_embedding : std::false_type {};

template <typename T>
struct has_set_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_word_embedding(
        std::declval<layer::WordEmbedding&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_logits_with_hidden : std::false_type {};

template <typename T>
struct has_logits_with_hidden<T,
                              std::void_t<decltype(std::declval<T>()->logits(
                                  std::declval<const torch::Tensor&>(),
                                  std::declval<const torch::Tensor&>(),
                                  std::declval<torch::Tensor&>()))>>
    : std::true_type {};

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
struct has_is_hybrid_linear_attention : std::false_type {};

template <typename T>
struct has_is_hybrid_linear_attention<
    T,
    std::void_t<decltype(std::declval<T>()->is_hybrid_linear_attention())>>
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
        std::declval<ModelInputParams&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_pooler : std::false_type {};

template <typename T>
struct has_pooler<T,
                  std::void_t<decltype(std::declval<T>()->pooler(
                      std::declval<const torch::Tensor&>(),
                      std::declval<const torch::Tensor&>()))>>
    : std::true_type {};

#if defined(USE_NPU)
template <typename T, typename = void>
struct has_get_npu_lm_head : std::false_type {};

template <typename T>
struct has_get_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_lm_head : std::false_type {};

template <typename T>
struct has_set_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_lm_head(
        std::declval<layer::NpuLmHead&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_get_npu_word_embedding : std::false_type {};

template <typename T>
struct has_get_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_word_embedding : std::false_type {};

template <typename T>
struct has_set_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_word_embedding(
        std::declval<layer::NpuWordEmbedding&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_init_or_refresh_rolling_runtime : std::false_type {};

template <typename T>
struct has_init_or_refresh_rolling_runtime<
    T,
    std::void_t<decltype(std::declval<T>()->init_or_refresh_rolling_runtime(
        std::declval<::xllm::Stream*>(),
        std::declval<::xllm::Stream*>(),
        std::declval<int32_t>(),
        std::declval<int32_t>(),
        std::declval<const std::string&>()))>> : std::true_type {};

#endif

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
        std::declval<const ModelInputParams&>()))>> : std::true_type {};

}  // namespace detail
}  // namespace xllm
