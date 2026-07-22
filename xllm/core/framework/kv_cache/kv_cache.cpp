/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache/kv_cache.h"

#include <glog/logging.h>

#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/kv_cache/deepseek_v4_kv_cache_impl.h"
#include "framework/kv_cache/indexed_kv_cache_impl.h"
#include "framework/kv_cache/linear_attention_kv_cache_impl.h"
#include "framework/kv_cache/quantized_kv_cache_impl.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "platform/sleepable_allocator.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace {

std::unique_ptr<KVCacheImpl> create_kv_cache_impl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_id) {
  CHECK_GE(layer_id, 0) << "KV cache layer_id must be non-negative.";

#if !defined(USE_MLU)
  CHECK(!create_options.enable_kv_cache_quant())
      << "KV cache quantization is only supported on MLU backend.";
#endif

  const bool is_linear_layer =
      create_options.enable_linear_attention() &&
      is_linear_attention_layer(layer_id,
                                create_options.full_attention_interval());
  if (is_linear_layer) {
    return std::make_unique<LinearAttentionKVCacheImpl>(kv_cache_shape,
                                                        create_options);
  }

  if (create_options.enable_kv_cache_quant() &&
      !create_options.enable_lighting_indexer()) {
    return std::make_unique<QuantizedKVCacheImpl>(kv_cache_shape,
                                                  create_options);
  }

  bool enable_indexer_cache = create_options.enable_lighting_indexer();
  const std::vector<bool>& indexer_cache_enabled_layers =
      create_options.indexer_cache_enabled_layers();
  if (!indexer_cache_enabled_layers.empty()) {
    CHECK_EQ(indexer_cache_enabled_layers.size(),
             static_cast<size_t>(create_options.num_layers()))
        << "Indexer cache layer mask must match num_layers.";
    enable_indexer_cache =
        enable_indexer_cache &&
        indexer_cache_enabled_layers[static_cast<size_t>(layer_id)];
  }

  if (enable_indexer_cache) {
    return std::make_unique<IndexedKVCacheImpl>(kv_cache_shape, create_options);
  }

  if (create_options.enable_kv_cache_quant()) {
    return std::make_unique<QuantizedKVCacheImpl>(kv_cache_shape,
                                                  create_options);
  }

  return std::make_unique<KVCacheImpl>(kv_cache_shape, create_options);
}

std::unique_ptr<KVCacheImpl> create_host_kv_cache_impl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count) {
  if (util::is_deepseek_v4_model_type(create_options.model_type())) {
    return std::make_unique<DeepSeekV4KVCacheImpl>(
        kv_cache_shape, create_options, type, layer_count);
  }

  switch (type) {
    case BlockType::SINGLE:
      return std::make_unique<LinearAttentionKVCacheImpl>(
          kv_cache_shape, create_options, type, layer_count);
    case BlockType::KV:
      if (create_options.enable_lighting_indexer()) {
        return std::make_unique<IndexedKVCacheImpl>(
            kv_cache_shape, create_options, type, layer_count);
      }
      return std::make_unique<KVCacheImpl>(
          kv_cache_shape, create_options, type, layer_count);
    default:
      LOG(FATAL) << "Unsupported non-DSV4 host block type: "
                 << static_cast<int32_t>(type);
  }
}

std::string int32_vector_string(const std::vector<int32_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

// Allocate standard key/value KV cache for all layers on a single VMM-backed
// SleepableAllocator region (RL deep-sleep mode). Each layer's K/V tensor is a
// view into the region at aligned offsets, so sleep()/wake_up() release and
// re-acquire the whole region's physical HBM at once.
void allocate_sleepable_kv_caches(std::vector<KVCache>& kv_caches,
                                  const KVCacheShape& kv_cache_shape,
                                  const KVCacheCreateOptions& create_options) {
  CHECK(!create_options.enable_xtensor())
      << "enable_sleep_mode is incompatible with xtensor mode.";
  CHECK(kv_cache_shape.has_key_cache_shape() &&
        kv_cache_shape.has_value_cache_shape())
      << "Sleep mode requires key and value cache shapes.";
  CHECK(!kv_cache_shape.has_index_cache_shape() &&
        !kv_cache_shape.has_conv_cache_shape() &&
        !kv_cache_shape.has_ssm_cache_shape())
      << "Sleep mode only supports standard key/value KV cache.";
  CHECK(!create_options.enable_linear_attention() &&
        !create_options.enable_lighting_indexer() &&
        !create_options.enable_kv_cache_quant())
      << "Sleep mode does not support linear/indexer/quantized KV cache.";

  const int64_t num_layers = create_options.num_layers();
  const std::vector<int64_t>& k_shape = kv_cache_shape.key_cache_shape();
  const std::vector<int64_t>& v_shape = kv_cache_shape.value_cache_shape();
  const size_t elt_size = torch::elementSize(create_options.dtype());

  auto numel_of = [](const std::vector<int64_t>& shape) {
    return std::accumulate(
        shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  };
  constexpr size_t kTensorAlign = 512;
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
  const size_t k_stride = align_up(numel_of(k_shape) * elt_size, kTensorAlign);
  const size_t v_stride = align_up(numel_of(v_shape) * elt_size, kTensorAlign);
  const size_t total_bytes =
      (k_stride + v_stride) * static_cast<size_t>(num_layers);

  // Map the (large) KV region in 1 GiB physical chunks rather than a single
  // huge handle: a single multi-GiB aclrtMallocPhysical is more failure-prone
  // under HBM fragmentation than several smaller chunks.
  constexpr size_t kKvChunkBytes = 1ULL << 30;  // 1 GiB
  void* base = SleepableAllocator::get_instance().reserve_and_map(
      MemTag::KV_CACHE, create_options.device(), total_bytes, kKvChunkBytes);

  uintptr_t addr = reinterpret_cast<uintptr_t>(base);
  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    torch::Tensor k_tensor = get_tensor_from_blob(
        k_shape, create_options.dtype(), reinterpret_cast<void*>(addr));
    addr += k_stride;
    torch::Tensor v_tensor = get_tensor_from_blob(
        v_shape, create_options.dtype(), reinterpret_cast<void*>(addr));
    addr += v_stride;
#if defined(USE_NPU)
    k_tensor = at_npu::native::npu_format_cast(k_tensor, ACL_FORMAT_ND);
    v_tensor = at_npu::native::npu_format_cast(v_tensor, ACL_FORMAT_ND);
#endif
    kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
  }

  LOG(INFO) << "Allocated sleepable KV cache: num_layers=" << num_layers
            << ", total_bytes=" << total_bytes << ", base=" << base;
}

}  // namespace

KVCache::KVCache() : impl_(std::make_unique<KVCacheImpl>()) {}

KVCache::KVCache(const KVCacheTensors& tensors)
    : impl_(std::make_unique<KVCacheImpl>(tensors)) {}

KVCache::KVCache(const IndexedKVCacheTensors& tensors)
    : impl_(std::make_unique<IndexedKVCacheImpl>(tensors)) {}

KVCache::KVCache(const LinearAttentionKVCacheTensors& tensors)
    : impl_(std::make_unique<LinearAttentionKVCacheImpl>(tensors)) {}

KVCache::KVCache(const QuantizedKVCacheTensors& tensors)
    : impl_(std::make_unique<QuantizedKVCacheImpl>(tensors)) {}

KVCache::KVCache(const DeepSeekV4KVCacheTensors& tensors)
    : impl_(std::make_unique<DeepSeekV4KVCacheImpl>(tensors)) {}

KVCache::KVCache(const KVCacheShape& kv_cache_shape,
                 const KVCacheCreateOptions& create_options,
                 int64_t layer_id)
    : impl_(create_kv_cache_impl(kv_cache_shape, create_options, layer_id)) {}

KVCache::KVCache(const KVCacheShape& kv_cache_shape,
                 const KVCacheCreateOptions& create_options,
                 BlockType type,
                 int64_t layer_count)
    : impl_(create_host_kv_cache_impl(kv_cache_shape,
                                      create_options,
                                      type,
                                      layer_count)) {}

torch::Tensor KVCache::get_k_cache() const { return impl_->get_k_cache(); }

torch::Tensor KVCache::get_v_cache() const { return impl_->get_v_cache(); }

torch::Tensor KVCache::get_index_cache() const {
  return impl_->get_index_cache();
}

std::vector<KVCacheTensor> KVCache::get_cache_tensors() const {
  return impl_->get_cache_tensors();
}

std::optional<torch::Tensor> KVCache::get_k_cache_scale() const {
  return impl_->get_k_cache_scale();
}

std::optional<torch::Tensor> KVCache::get_v_cache_scale() const {
  return impl_->get_v_cache_scale();
}

std::optional<torch::Tensor> KVCache::get_indexer_cache_scale() const {
  return impl_->get_indexer_cache_scale();
}

torch::Tensor KVCache::get_conv_cache() const {
  return impl_->get_conv_cache();
}

torch::Tensor KVCache::get_ssm_cache() const { return impl_->get_ssm_cache(); }

torch::Tensor KVCache::get_swa_cache() const { return impl_->get_swa_cache(); }

BlockTypeTensorMap KVCache::get_block_type_tensors(BlockType type) const {
  return impl_->get_block_type_tensors(type);
}

torch::Tensor KVCache::get_compress_kv_state() const {
  return impl_->get_compress_kv_state();
}

torch::Tensor KVCache::get_compress_score_state() const {
  return impl_->get_compress_score_state();
}

torch::Tensor KVCache::get_compress_index_kv_state() const {
  return impl_->get_compress_index_kv_state();
}

torch::Tensor KVCache::get_compress_index_score_state() const {
  return impl_->get_compress_index_score_state();
}

torch::Tensor KVCache::get_compress_state() const {
  return impl_->get_compress_state();
}

torch::Tensor KVCache::get_compress_index_state() const {
  return impl_->get_compress_index_state();
}

std::vector<std::vector<int64_t>> KVCache::get_shapes() {
  return impl_->get_shapes();
}

bool KVCache::empty() const { return impl_->empty(); }

void KVCache::swap_blocks(torch::Tensor& src_tensor,
                          torch::Tensor& dst_tensor) {
  impl_->swap_blocks(src_tensor, dst_tensor);
}

void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options) {
  CHECK(kv_caches.empty()) << "KV caches are already initialized.";

  const int64_t num_layers = create_options.num_layers();
  kv_caches.reserve(num_layers);

  if (util::is_target_model_type(create_options.model_type(),
                                 /*target_type=*/"deepseek_v4",
                                 /*match_mtp=*/true)) {
    std::vector<int32_t> layer_compress_ratios;
    layer_compress_ratios.reserve(static_cast<size_t>(num_layers));
    std::map<int32_t, std::string> ratio_shape_summaries;
    const std::vector<int32_t>& compress_ratios =
        create_options.compress_ratios();

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      const int32_t compress_ratio =
          layer_idx < static_cast<int64_t>(compress_ratios.size())
              ? compress_ratios[static_cast<size_t>(layer_idx)]
              : 1;
      DeepSeekV4KVCacheTensors tensors =
          create_dsv4_cache_tensors(kv_cache_shape, create_options, layer_idx);
      layer_compress_ratios.emplace_back(compress_ratio);
      if (ratio_shape_summaries.find(compress_ratio) ==
          ratio_shape_summaries.end()) {
        ratio_shape_summaries.emplace(
            compress_ratio, dsv4_shape_summary(tensors, compress_ratio));
      }
      kv_caches.emplace_back(tensors);
    }

    LOG(INFO) << "[DSV4][KVCacheInit] layer_crs: "
              << int32_vector_string(layer_compress_ratios);
    for (const std::pair<const int32_t, std::string>& summary :
         ratio_shape_summaries) {
      LOG(INFO) << "[DSV4][KVCacheInit] cr_" << summary.first
                << " shapes: " << summary.second;
    }
    return;
  }

  if (create_options.enable_sleep_mode()) {
    allocate_sleepable_kv_caches(kv_caches, kv_cache_shape, create_options);
    return;
  }

  if (create_options.enable_xtensor()) {
    CHECK(kv_cache_shape.has_key_cache_shape())
        << "key_cache_shape must be initialized for XTensor mode.";
    CHECK(kv_cache_shape.has_value_cache_shape())
        << "value_cache_shape must be initialized for XTensor mode.";
    CHECK(!kv_cache_shape.has_index_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!kv_cache_shape.has_conv_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!kv_cache_shape.has_ssm_cache_shape())
        << "Only support key and value cache for XTensor mode.";
    CHECK(!create_options.model_id().empty())
        << "model_id must not be empty for XTensor mode.";
    CHECK(!create_options.enable_linear_attention())
        << "Linear attention is not supported for XTensor mode.";

    XTensorAllocator& allocator = XTensorAllocator::get_instance();
    std::vector<torch::Tensor> k_tensors =
        allocator.create_k_tensors(create_options.model_id(),
                                   kv_cache_shape.key_cache_shape(),
                                   create_options.dtype(),
                                   num_layers);
    std::vector<torch::Tensor> v_tensors =
        allocator.create_v_tensors(create_options.model_id(),
                                   kv_cache_shape.value_cache_shape(),
                                   create_options.dtype(),
                                   num_layers);

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      torch::Tensor k_tensor = k_tensors[layer_idx];
      torch::Tensor v_tensor = v_tensors[layer_idx];
#if defined(USE_NPU)
      k_tensor = at_npu::native::npu_format_cast(k_tensor, ACL_FORMAT_ND);
      v_tensor = at_npu::native::npu_format_cast(v_tensor, ACL_FORMAT_ND);
#endif
      kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
    }
    return;
  }

  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    kv_caches.emplace_back(kv_cache_shape, create_options, layer_idx);
  }
}

}  // namespace xllm
