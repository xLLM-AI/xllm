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

#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_layout_policy.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/util/tensor_options.h"

namespace xllm {
namespace {

bool is_qwen_hybrid_model(const ModelArgs& model_args) {
  return model_args.model_type() == "qwen3_next" ||
         is_qwen3_5_target_model_type(model_args.model_type());
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
  if (shape.has_grouped_cache_layout() || model_args.index_kpool_compress() ||
      ((shape.has_conv_cache_shape() || shape.has_ssm_cache_shape()) &&
       !is_qwen_hybrid_model(model_args))) {
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
  components.reserve(6);
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
    const auto mask = resolve_indexer_cache_enabled_layers(model_args);
    std::vector<int32_t> index_layers;
    index_layers.reserve(layers.size());
    for (const int32_t layer : layers) {
      if (!mask.empty() && !mask[static_cast<size_t>(layer)]) {
        continue;
      }
      index_layers.emplace_back(layer);
    }
    if (!index_layers.empty()) {
      components.emplace_back(CacheComponent{
          CacheComponentRole::INDEX,
          capacity.enable_indexer_cache_quant() ? CacheScalarType::INT8
                                                : value_dtype,
          index_layers});
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

std::optional<RecurrentCachePartition> build_recurrent_cache_partition(
    CachePartitionId id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    const KVCacheShape& shape,
    torch::ScalarType dtype,
    int32_t checkpoint_stride,
    bool enable_prefix_cache) {
  if (!is_qwen_hybrid_model(model_args) || shape.has_grouped_cache_layout() ||
      model_args.index_kpool_compress() || !shape.has_conv_cache_shape() ||
      !shape.has_ssm_cache_shape()) {
    return std::nullopt;
  }
  auto layers = cache_layers(capacity, model_args, /*linear=*/true);
  if (layers.empty()) {
    return std::nullopt;
  }
  const int64_t snapshots = shape.linear_ssm_checkpoint_stride();
  CHECK_LE(snapshots, std::numeric_limits<int32_t>::max());
  // Decode or disabled prefix caching may leave this unused stride unset.
  const int32_t effective_stride =
      enable_prefix_cache ? checkpoint_stride : std::max(checkpoint_stride, 1);
  return RecurrentCachePartition(std::move(id),
                                 effective_stride,
                                 static_cast<int32_t>(snapshots),
                                 enable_prefix_cache,
                                 std::move(layers),
                                 cache_scalar_type(dtype),
                                 cache_scalar_type(resolve_ssm_dtype(
                                     model_args.mamba_ssm_dtype(), dtype)));
}

}  // namespace xllm
