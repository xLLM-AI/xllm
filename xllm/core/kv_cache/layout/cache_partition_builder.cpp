/* Copyright 2026 The xLLM Authors.

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

#include "core/kv_cache/layout/cache_partition_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/kv_cache/layout/deepseek_v4_cache_policy.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_layout_policy.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/util/tensor_options.h"

namespace xllm {
namespace {

bool is_glm5_next_cache_model(const ModelArgs& model_args) {
  return is_glm5_next_target_model_type(model_args.model_type()) ||
         model_args.model_type() == "glm5_next_mtp";
}

bool supports_recurrent_partition(const ModelArgs& model_args) {
  return model_args.model_type() == "qwen3_next" ||
         is_qwen3_5_target_model_type(model_args.model_type()) ||
         is_glm5_next_cache_model(model_args);
}

std::vector<int32_t> cache_layers(const KVCacheCapacity& capacity,
                                  const ModelArgs& model_args,
                                  bool linear) {
  const int64_t num_layers =
      capacity.n_layers() > 0 ? capacity.n_layers() : model_args.n_layers();
  CHECK_GT(num_layers, 0);
  CHECK_LE(num_layers, model_args.n_layers());
  CHECK_LE(num_layers, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> layers;
  layers.reserve(static_cast<size_t>(num_layers));
  for (int32_t layer = 0; layer < num_layers; ++layer) {
    if (is_linear_attention_layer(layer,
                                  model_args.full_attention_interval(),
                                  model_args.layer_types()) != linear) {
      continue;
    }
    layers.emplace_back(layer);
  }
  return layers;
}

std::vector<int32_t> indexer_cache_layers(const KVCacheCapacity& capacity,
                                          const ModelArgs& model_args) {
  const auto full_layers = cache_layers(capacity, model_args, /*linear=*/false);
  const auto mask = resolve_indexer_cache_enabled_layers(model_args);
  std::vector<int32_t> layers;
  layers.reserve(full_layers.size());
  for (const int32_t layer : full_layers) {
    if (!mask.empty() && !mask[static_cast<size_t>(layer)]) {
      continue;
    }
    layers.emplace_back(layer);
  }
  return layers;
}

CacheScalarType cache_scalar_type(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return CacheScalarType::FLOAT16;
    case torch::kBFloat16:
      return CacheScalarType::BFLOAT16;
    case torch::kFloat32:
      return CacheScalarType::FLOAT32;
    case torch::kChar:
      return CacheScalarType::INT8;
    case torch::kByte:
      return CacheScalarType::UINT8;
    case torch::kFloat8_e4m3fn:
      return CacheScalarType::FLOAT8_E4M3FN;
    case torch::kFloat8_e5m2:
      return CacheScalarType::FLOAT8_E5M2;
    default:
      LOG(FATAL) << "Unsupported paged cache dtype: " << dtype;
  }
  return CacheScalarType::FLOAT16;
}

}  // namespace

std::optional<PagedCachePartition> build_paged_cache_partition(
    CachePartitionId id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    const KVCacheShape& shape,
    torch::ScalarType dtype,
    int32_t token_stride,
    bool enable_prefix_cache) {
  if (shape.has_grouped_cache_layout() ||
      (model_args.index_kpool_compress() &&
       !is_glm5_next_cache_model(model_args)) ||
      ((shape.has_conv_cache_shape() || shape.has_ssm_cache_shape()) &&
       !supports_recurrent_partition(model_args))) {
    return std::nullopt;
  }
  auto layers = cache_layers(capacity, model_args, /*linear=*/false);
  if (layers.empty()) {
    return std::nullopt;
  }

  const CacheScalarType value_dtype = cache_scalar_type(dtype);
  const bool packed_mla = capacity.enable_mla_kv_cache_quant();
  const bool quantized_kv = capacity.scale_slot_size() > 0;
  const CacheScalarType kv_dtype =
      packed_mla || quantized_kv ? CacheScalarType::INT8 : value_dtype;
  std::vector<CacheComponent> components;
  components.reserve(8);
  if (model_args.enable_mla()) {
    components.emplace_back(
        CacheComponent{CacheComponentRole::MLA_LATENT, kv_dtype, layers});
    if (model_args.qk_rope_head_dim() > 0) {
      components.emplace_back(
          CacheComponent{CacheComponentRole::MLA_ROPE,
                         packed_mla ? CacheScalarType::BFLOAT16 : kv_dtype,
                         layers});
    }
    if (packed_mla) {
      components.emplace_back(CacheComponent{
          CacheComponentRole::MLA_SCALE, CacheScalarType::FLOAT32, layers});
    }
  } else {
    components.emplace_back(
        CacheComponent{CacheComponentRole::KEY, kv_dtype, layers});
    components.emplace_back(
        CacheComponent{CacheComponentRole::VALUE, kv_dtype, layers});
  }
  if (quantized_kv && !packed_mla) {
    components.emplace_back(CacheComponent{
        CacheComponentRole::KEY_SCALE, CacheScalarType::FLOAT32, layers});
    if (shape.has_value_cache_shape()) {
      components.emplace_back(CacheComponent{
          CacheComponentRole::VALUE_SCALE, CacheScalarType::FLOAT32, layers});
    }
  }
  if (shape.has_index_cache_shape()) {
    auto index_layers = indexer_cache_layers(capacity, model_args);
    if (!index_layers.empty()) {
      components.emplace_back(CacheComponent{
          CacheComponentRole::INDEX,
          capacity.enable_indexer_cache_quant() ? CacheScalarType::INT8
                                                : value_dtype,
          index_layers});
      // Packed KPool rows retain gate scores and validity with the keys.
      // Compressed history keeps only pooled keys; its unfinished pool lives
      // in the recurrent tail. Physical row geometry stays in KVCacheShape.
      if (model_args.index_kpool_compress() &&
          shape.kpool_layout() == KPoolCacheLayout::PACKED) {
        components.emplace_back(CacheComponent{
            CacheComponentRole::INDEX_GATE, value_dtype, index_layers});
        components.emplace_back(CacheComponent{
            CacheComponentRole::INDEX_VALID, value_dtype, index_layers});
      }
      if (capacity.enable_indexer_cache_quant()) {
#if defined(USE_NPU)
        const CacheScalarType scale_dtype = CacheScalarType::FLOAT16;
#else
        const CacheScalarType scale_dtype = CacheScalarType::FLOAT32;
#endif
        components.emplace_back(CacheComponent{CacheComponentRole::INDEX_SCALE,
                                               scale_dtype,
                                               std::move(index_layers)});
      }
    }
  }
  return PagedCachePartition(
      std::move(id), token_stride, enable_prefix_cache, std::move(components));
}

std::vector<PagedCachePartition> build_grouped_cache_partitions(
    const std::string& model_id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    torch::ScalarType dtype,
    int32_t token_stride) {
  CHECK(is_deepseek_v4_family(model_args.model_type()));
  const bool is_v41 = model_args.model_type() == "deepseek_v41";
  const auto layers = cache_layers(capacity, model_args, /*linear=*/false);
  const CacheScalarType value_dtype = cache_scalar_type(dtype);
  const auto cache_policy = get_dsv4_cache_policy(dtype);
  const auto& ratios = model_args.compress_ratios();
  const auto& source_layers = model_args.kv_source_layer_ids();
  auto history_layers = [&](int32_t ratio) {
    std::vector<int32_t> owned;
    owned.reserve(layers.size());
    for (const int32_t layer : layers) {
      const int32_t layer_ratio = static_cast<size_t>(layer) < ratios.size()
                                      ? ratios[static_cast<size_t>(layer)]
                                      : 1;
      if (layer_ratio != ratio ||
          (is_v41 &&
           std::find(source_layers.begin(), source_layers.end(), layer) ==
               source_layers.end())) {
        continue;
      }
      owned.emplace_back(layer);
    }
    return owned;
  };

  std::vector<CacheComponent> window_components;
  window_components.reserve(5);
  window_components.emplace_back(
      CacheComponent{CacheComponentRole::KEY, value_dtype, layers});
  // Compression accumulators use SWA IDs, not compressed-history IDs.
  std::vector<int32_t> state_layers;
  state_layers.reserve(layers.size());
  const auto c2_layers = history_layers(2);
  const auto c4_layers = history_layers(4);
  const auto c128_layers = history_layers(128);
  for (const int32_t layer : layers) {
    const auto& owned = is_v41 ? c2_layers : c4_layers;
    if (!std::binary_search(owned.begin(), owned.end(), layer) &&
        (is_v41 ||
         !std::binary_search(c128_layers.begin(), c128_layers.end(), layer))) {
      continue;
    }
    state_layers.emplace_back(layer);
  }
  if (!state_layers.empty()) {
    window_components.emplace_back(
        CacheComponent{CacheComponentRole::COMPRESS_KV_STATE,
                       CacheScalarType::FLOAT32,
                       state_layers});
    window_components.emplace_back(
        CacheComponent{CacheComponentRole::COMPRESS_SCORE_STATE,
                       CacheScalarType::FLOAT32,
                       std::move(state_layers)});
  }
  if (!is_v41 && !c4_layers.empty()) {
    window_components.emplace_back(
        CacheComponent{CacheComponentRole::COMPRESS_INDEX_KV_STATE,
                       CacheScalarType::FLOAT32,
                       c4_layers});
    window_components.emplace_back(
        CacheComponent{CacheComponentRole::COMPRESS_INDEX_SCORE_STATE,
                       CacheScalarType::FLOAT32,
                       c4_layers});
  }
  const int64_t semantic_window = std::max(model_args.window_size(), 1);
  const int64_t effective_window =
      model_args.max_seq_len() > 0
          ? std::min<int64_t>(semantic_window, model_args.max_seq_len())
          : semantic_window;
  CHECK_LE(effective_window, std::numeric_limits<int32_t>::max());
  std::vector<PagedCachePartition> partitions;
  partitions.reserve(5);
  partitions.emplace_back(CachePartitionId{model_id, 0},
                          token_stride,
                          /*enable_prefix_cache=*/true,
                          std::move(window_components),
                          /*compression_ratio=*/1,
                          static_cast<int32_t>(effective_window));
  const std::pair<int32_t, int64_t> pools[] = {{1, capacity.c1_count()},
                                               {2, capacity.c2_count()},
                                               {4, capacity.c4_count()},
                                               {128, capacity.c128_count()}};
  for (const auto& [ratio, count] : pools) {
    if (count == 0) {
      continue;
    }
    auto owned = history_layers(ratio);
    std::vector<CacheComponent> components;
    components.reserve(3);
    components.emplace_back(
        CacheComponent{CacheComponentRole::KEY, value_dtype, owned});
    if (ratio != 128) {
      components.emplace_back(CacheComponent{
          CacheComponentRole::INDEX,
          is_v41 ? value_dtype : cache_scalar_type(cache_policy.index_dtype),
          owned});
    }
    if (!is_v41 && ratio == 4 && cache_policy.has_indexer_cache_scale) {
      components.emplace_back(
          CacheComponent{CacheComponentRole::INDEX_SCALE,
                         cache_scalar_type(cache_policy.scale_dtype),
                         std::move(owned)});
    }
    const int64_t stride = static_cast<int64_t>(token_stride) * ratio;
    CHECK_LE(stride, std::numeric_limits<int32_t>::max());
    partitions.emplace_back(
        CachePartitionId{model_id, static_cast<uint32_t>(partitions.size())},
        static_cast<int32_t>(stride),
        /*enable_prefix_cache=*/ratio == 4 || ratio == 128,
        std::move(components),
        ratio);
  }
  return partitions;
}

std::optional<RecurrentCachePartition> build_recurrent_cache_partition(
    CachePartitionId id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    const KVCacheShape& shape,
    torch::ScalarType dtype,
    int32_t checkpoint_stride,
    bool enable_prefix_cache) {
  if (!supports_recurrent_partition(model_args) ||
      shape.has_grouped_cache_layout()) {
    return std::nullopt;
  }
  std::vector<CacheComponent> components;
  components.reserve(3);
  int32_t snapshots = 0;
  if (shape.has_conv_cache_shape() && shape.has_ssm_cache_shape()) {
    auto layers = cache_layers(capacity, model_args, /*linear=*/true);
    if (!layers.empty()) {
      const int64_t physical_rows = shape.linear_ssm_checkpoint_stride();
      CHECK_LE(physical_rows, std::numeric_limits<int32_t>::max());
      snapshots = static_cast<int32_t>(physical_rows);
      components.emplace_back(CacheComponent{
          CacheComponentRole::CONV, cache_scalar_type(dtype), layers});
      components.emplace_back(
          CacheComponent{CacheComponentRole::SSM,
                         cache_scalar_type(resolve_ssm_dtype(
                             model_args.mamba_ssm_dtype(), dtype)),
                         std::move(layers)});
    }
  }
  int32_t tail_tokens = 0;
  if (shape.has_kpool_tail_shape()) {
    auto layers = indexer_cache_layers(capacity, model_args);
    if (!layers.empty()) {
      const int64_t tokens = shape.kpool_tail_shape()[2];
      CHECK_LE(tokens, std::numeric_limits<int32_t>::max());
      tail_tokens = static_cast<int32_t>(tokens);
      components.emplace_back(CacheComponent{CacheComponentRole::KPOOL_TAIL,
                                             cache_scalar_type(dtype),
                                             std::move(layers)});
    }
  }
  if (components.empty()) {
    return std::nullopt;
  }
  // Decode or disabled prefix caching may leave this unused stride unset.
  const int32_t effective_stride =
      enable_prefix_cache ? checkpoint_stride : std::max(checkpoint_stride, 1);
  return RecurrentCachePartition(std::move(id),
                                 effective_stride,
                                 snapshots,
                                 enable_prefix_cache,
                                 std::move(components),
                                 tail_tokens);
}

}  // namespace xllm
