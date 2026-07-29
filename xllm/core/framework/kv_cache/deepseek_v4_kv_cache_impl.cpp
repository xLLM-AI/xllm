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

#include "framework/kv_cache/deepseek_v4_kv_cache_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <utility>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"

namespace xllm {
namespace {

torch::Tensor swap_tensor_blocks(const torch::Tensor& tensor,
                                 const torch::Tensor& src_tensor,
                                 const torch::Tensor& dst_tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor selected = torch::index_select(tensor, 0, src_tensor);
  torch::Tensor result = tensor.clone();
  result.index_copy_(0, dst_tensor, selected);
  return result;
}

torch::Tensor cast_to_nd_format(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
#if defined(USE_NPU)
  if (!tensor.device().is_privateuseone()) {
    return tensor;
  }
  return at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND);
#else
  return tensor;
#endif
}

std::vector<int64_t> dsv4_block_shape(int64_t block_count,
                                      int64_t block_size,
                                      int64_t n_heads,
                                      int64_t head_dim) {
#if defined(USE_MLU)
  return {block_count, n_heads, block_size, head_dim};
#else
  return {block_count, block_size, n_heads, head_dim};
#endif
}

std::string tensor_shape_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}

}  // namespace

Dsv4StateCache Dsv4StateCache::from_split(torch::Tensor kv,
                                          torch::Tensor score) {
  Dsv4StateCache state;
  state.kv_ = std::move(kv);
  state.score_ = std::move(score);
  return state;
}

Dsv4StateCache Dsv4StateCache::from_packed(torch::Tensor packed,
                                           torch::Tensor fallback_kv,
                                           torch::Tensor fallback_score) {
  Dsv4StateCache state;
  state.packed_layout_ = true;
  if (packed.defined()) {
    state.packed_ = std::move(packed);
  } else {
    state.kv_ = std::move(fallback_kv);
    state.score_ = std::move(fallback_score);
  }
  return state;
}

torch::Tensor Dsv4StateCache::kv() const {
  if (!packed_.defined()) {
    return kv_;
  }
  const int64_t state_width = packed_.size(2) / 2;
  return packed_.narrow(/*dim=*/2, /*start=*/0, /*length=*/state_width);
}

torch::Tensor Dsv4StateCache::score() const {
  if (!packed_.defined()) {
    return score_;
  }
  const int64_t state_width = packed_.size(2) / 2;
  return packed_.narrow(
      /*dim=*/2, /*start=*/state_width, /*length=*/state_width);
}

torch::Tensor Dsv4StateCache::packed() const { return packed_; }

void Dsv4StateCache::swap_blocks(const torch::Tensor& src,
                                 const torch::Tensor& dst) {
  if (packed_layout_) {
    packed_ = swap_tensor_blocks(packed_, src, dst);
    return;
  }
  kv_ = swap_tensor_blocks(kv_, src, dst);
  score_ = swap_tensor_blocks(score_, src, dst);
}

DeepSeekV4KVCacheImpl::DeepSeekV4KVCacheImpl(
    const DeepSeekV4KVCacheTensors& tensors)
    : key_cache_(tensors.key_cache),
      index_cache_(tensors.index_cache),
      indexer_cache_scale_(tensors.indexer_cache_scale),
      swa_cache_(tensors.swa_cache),
#if defined(USE_MLU)
      compress_state_(
          Dsv4StateCache::from_packed(tensors.compress_state,
                                      tensors.compress_kv_state,
                                      tensors.compress_score_state)),
      index_state_(
          Dsv4StateCache::from_packed(tensors.compress_index_state,
                                      tensors.compress_index_kv_state,
                                      tensors.compress_index_score_state)),
#else
      compress_state_(Dsv4StateCache::from_split(tensors.compress_kv_state,
                                                 tensors.compress_score_state)),
      index_state_(
          Dsv4StateCache::from_split(tensors.compress_index_kv_state,
                                     tensors.compress_index_score_state)),
#endif
      compressed_block_type_(tensors.compressed_block_type) {
}

DeepSeekV4KVCacheImpl::DeepSeekV4KVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count) {
  CHECK(kv_cache_shape.has_key_cache_shape())
      << "DeepSeek V4 host kv cache shape must contain pool counts.";
  const std::vector<int64_t>& pool_counts = kv_cache_shape.key_cache_shape();
  CHECK_GE(pool_counts.size(), 3) << "DeepSeek V4 host kv cache shape must be "
                                  << "[swa_count, c4_count, c128_count].";
  CHECK_GT(create_options.block_size(), 0)
      << "DeepSeek V4 block_size must be positive.";
  CHECK_GT(create_options.head_dim(), 0)
      << "DeepSeek V4 head_dim must be positive.";
  CHECK_GT(layer_count, 0)
      << "DeepSeek V4 host prefix cache layer count must be positive.";

  const double factor = create_options.host_blocks_factor();
  const int64_t host_swa_count = scale_host_block_count(pool_counts[0], factor);
  const int64_t host_c4_count = scale_host_block_count(pool_counts[1], factor);
  const int64_t host_c128_count =
      scale_host_block_count(pool_counts[2], factor);
  const int64_t block_size = create_options.block_size();
  const int64_t head_dim = create_options.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(create_options.index_head_dim(), 1);
  const int64_t n_heads = 1;
  const int64_t index_n_heads = 1;
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(create_options.dtype());

  // Host tensor shape: device per-block dims with a layer dimension inserted at
  // index 1, i.e. [host_block_count, layer_count, ...per_block_dims].
  auto host_group_shape = [&](int64_t block_count, int64_t heads, int64_t dim) {
    std::vector<int64_t> shape =
        dsv4_block_shape(block_count, block_size, heads, dim);
    shape.insert(shape.begin() + 1, layer_count);
    return shape;
  };

  switch (type) {
    case BlockType::SWA:
      host_page_aligned_regions_.reserve(1);
      create_host_tensor(host_group_shape(host_swa_count, n_heads, head_dim),
                         create_options.dtype(),
                         &swa_cache_,
                         nullptr);
      break;
    case BlockType::C4:
      host_page_aligned_regions_.reserve(2);
      create_host_tensor(host_group_shape(host_c4_count, n_heads, head_dim),
                         create_options.dtype(),
                         &key_cache_,
                         nullptr);
      create_host_tensor(
          host_group_shape(host_c4_count, index_n_heads, index_head_dim),
          cache_policy.index_dtype,
          &index_cache_,
          nullptr);
      break;
    case BlockType::C128:
      host_page_aligned_regions_.reserve(1);
      create_host_tensor(host_group_shape(host_c128_count, n_heads, head_dim),
                         create_options.dtype(),
                         &key_cache_,
                         nullptr);
      break;
    default:
      LOG(FATAL) << "Unsupported DeepSeek V4 host block type: "
                 << static_cast<int32_t>(type);
  }
}

torch::Tensor DeepSeekV4KVCacheImpl::get_k_cache() const { return key_cache_; }

torch::Tensor DeepSeekV4KVCacheImpl::get_index_cache() const {
  return index_cache_;
}

std::optional<torch::Tensor> DeepSeekV4KVCacheImpl::get_indexer_cache_scale()
    const {
  if (indexer_cache_scale_.defined() && indexer_cache_scale_.numel() > 0) {
    return indexer_cache_scale_;
  }
  return std::nullopt;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_swa_cache() const {
  return swa_cache_;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_kv_state() const {
  return compress_state_.kv();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_score_state() const {
  return compress_state_.score();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_kv_state() const {
  return index_state_.kv();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_score_state() const {
  return index_state_.score();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_state() const {
  return compress_state_.packed();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_state() const {
  return index_state_.packed();
}

std::vector<KVCacheTensor> DeepSeekV4KVCacheImpl::get_cache_tensors() const {
  std::vector<KVCacheTensor> tensors;
  tensors.reserve(8);
  auto add_tensor = [&tensors](KVCacheTensorRole role,
                               const torch::Tensor& tensor,
                               BlockType block_type) {
    if (tensor.defined() && tensor.numel() > 0) {
      tensors.emplace_back(KVCacheTensor{
          role, tensor, cache_group_id(block_type), /*sequence_scoped=*/false});
    }
  };

  add_tensor(KVCacheTensorRole::WINDOW, swa_cache_, BlockType::SWA);
  add_tensor(KVCacheTensorRole::KEY, key_cache_, compressed_block_type_);
  add_tensor(KVCacheTensorRole::INDEX, index_cache_, compressed_block_type_);
  add_tensor(KVCacheTensorRole::INDEX_SCALE,
             indexer_cache_scale_,
             compressed_block_type_);
  add_tensor(KVCacheTensorRole::KV_STATE, compress_state_.kv(), BlockType::SWA);
  add_tensor(
      KVCacheTensorRole::SCORE_STATE, compress_state_.score(), BlockType::SWA);
  add_tensor(
      KVCacheTensorRole::INDEX_KV_STATE, index_state_.kv(), BlockType::SWA);
  add_tensor(KVCacheTensorRole::INDEX_SCORE_STATE,
             index_state_.score(),
             BlockType::SWA);
  return tensors;
}

BlockTypeTensorMap DeepSeekV4KVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map;
  switch (type) {
    case BlockType::SWA:
      if (swa_cache_.defined() && swa_cache_.numel() > 0) {
        tensor_map.emplace(KVCacheTensorRole::SWA, swa_cache_);
      }
      break;
    case BlockType::C4:
      if (key_cache_.defined() && key_cache_.numel() > 0 &&
          index_cache_.defined() && index_cache_.numel() > 0) {
        tensor_map.emplace(KVCacheTensorRole::KEY, key_cache_);
        tensor_map.emplace(KVCacheTensorRole::INDEX, index_cache_);
      }
      break;
    case BlockType::C128:
      if (key_cache_.defined() && key_cache_.numel() > 0 &&
          (!index_cache_.defined() || index_cache_.numel() == 0)) {
        tensor_map.emplace(KVCacheTensorRole::KEY, key_cache_);
      }
      break;
    default:
      break;
  }
  return tensor_map;
}

bool DeepSeekV4KVCacheImpl::empty() const { return !swa_cache_.defined(); }

std::vector<std::vector<int64_t>> DeepSeekV4KVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(3);
  if (key_cache_.defined()) {
    shapes.emplace_back(key_cache_.sizes().vec());
  }
  if (index_cache_.defined()) {
    shapes.emplace_back(index_cache_.sizes().vec());
  }
  if (swa_cache_.defined()) {
    shapes.emplace_back(swa_cache_.sizes().vec());
  }
  return shapes;
}

void DeepSeekV4KVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                                        torch::Tensor& dst_tensor) {
  key_cache_ = swap_tensor_blocks(key_cache_, src_tensor, dst_tensor);
  index_cache_ = swap_tensor_blocks(index_cache_, src_tensor, dst_tensor);
  indexer_cache_scale_ =
      swap_tensor_blocks(indexer_cache_scale_, src_tensor, dst_tensor);
  swa_cache_ = swap_tensor_blocks(swa_cache_, src_tensor, dst_tensor);
  compress_state_.swap_blocks(src_tensor, dst_tensor);
  index_state_.swap_blocks(src_tensor, dst_tensor);
}

DeepSeekV4KVCacheTensors create_dsv4_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_idx) {
  CHECK(kv_cache_shape.has_key_cache_shape())
      << "DeepSeek V4 cache shape must contain cache pool counts.";
  const std::vector<int64_t>& pool_counts = kv_cache_shape.key_cache_shape();
  CHECK_GE(pool_counts.size(), 3)
      << "DeepSeek V4 cache shape must be [swa_count, c4_count, c128_count].";
  CHECK_GT(create_options.block_size(), 0)
      << "DeepSeek V4 block_size must be positive.";
  CHECK_GT(create_options.head_dim(), 0)
      << "DeepSeek V4 head_dim must be positive.";
  CHECK_GT(create_options.window_size(), 0)
      << "DeepSeek V4 window_size must be positive.";

  const int64_t swa_count = pool_counts[0];
  const int64_t c4_count = pool_counts[1];
  const int64_t c128_count = pool_counts[2];
  const int64_t block_size = create_options.block_size();
  const int64_t head_dim = create_options.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(create_options.index_head_dim(), 1);
  const int64_t n_heads = 1;
  const int64_t index_n_heads = 1;
  const std::vector<int32_t>& compress_ratios =
      create_options.compress_ratios();
  const int32_t compress_ratio =
      layer_idx < static_cast<int64_t>(compress_ratios.size())
          ? compress_ratios[static_cast<size_t>(layer_idx)]
          : 1;

  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(create_options.dtype());

#if defined(USE_NPU)
  const bool use_huge_page_allocator =
      create_options.enable_kv_cache_huge_page_allocator();
#endif
  auto allocate_tensor = [&](const std::vector<int64_t>& dims,
                             torch::ScalarType dtype) {
#if defined(USE_NPU)
    if (use_huge_page_allocator) {
      return alloc_npu_huge_page_tensor(dims, dtype, ACL_FORMAT_ND);
    }
#endif
    return cast_to_nd_format(torch::empty(
        dims, torch::dtype(dtype).device(create_options.device())));
  };

  DeepSeekV4KVCacheTensors tensors;
  if (compress_ratio == 1) {
    tensors.swa_cache = allocate_tensor(
        dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
        create_options.dtype());
  } else if (compress_ratio == 4) {
    tensors.compressed_block_type = BlockType::C4;
    tensors.key_cache = allocate_tensor(
        dsv4_block_shape(c4_count, block_size, n_heads, head_dim),
        create_options.dtype());
    tensors.index_cache = allocate_tensor(
        dsv4_block_shape(c4_count, block_size, index_n_heads, index_head_dim),
        cache_policy.index_dtype);
    if (cache_policy.has_indexer_cache_scale) {
      tensors.indexer_cache_scale =
          allocate_tensor({c4_count, block_size, 1}, cache_policy.scale_dtype);
    }
    tensors.swa_cache = allocate_tensor(
        dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
        create_options.dtype());
#if defined(USE_MLU)
    // coff_dim = 2 * head_dim for ratio==4; merged dim = 2 * coff_dim. The
    // owning compress_state backs the narrow-view getters and is passed
    // directly to fused_compress_*_kv (which requires a contiguous
    // state_cache).
    const int64_t cmp_coff_dim = 2 * head_dim;
    tensors.compress_state = allocate_tensor(
        {swa_count, block_size, 2 * cmp_coff_dim}, torch::kFloat32);
    tensors.compress_kv_state = tensors.compress_state.narrow(
        /*dim=*/2, /*start=*/0, /*length=*/cmp_coff_dim);
    tensors.compress_score_state = tensors.compress_state.narrow(
        /*dim=*/2, /*start=*/cmp_coff_dim, /*length=*/cmp_coff_dim);
    const int64_t idx_coff_dim = 2 * index_head_dim;
    tensors.compress_index_state = allocate_tensor(
        {swa_count, block_size, 2 * idx_coff_dim}, torch::kFloat32);
    tensors.compress_index_kv_state = tensors.compress_index_state.narrow(
        /*dim=*/2, /*start=*/0, /*length=*/idx_coff_dim);
    tensors.compress_index_score_state = tensors.compress_index_state.narrow(
        /*dim=*/2, /*start=*/idx_coff_dim, /*length=*/idx_coff_dim);
#else
    tensors.compress_kv_state =
        allocate_tensor({swa_count, block_size, 2 * head_dim}, torch::kFloat32);
    tensors.compress_score_state =
        allocate_tensor({swa_count, block_size, 2 * head_dim}, torch::kFloat32);
    tensors.compress_index_kv_state = allocate_tensor(
        {swa_count, block_size, 2 * index_head_dim}, torch::kFloat32);
    tensors.compress_index_score_state = allocate_tensor(
        {swa_count, block_size, 2 * index_head_dim}, torch::kFloat32);
#endif
  } else if (compress_ratio == 128) {
    tensors.compressed_block_type = BlockType::C128;
    tensors.key_cache = allocate_tensor(
        dsv4_block_shape(c128_count, block_size, n_heads, head_dim),
        create_options.dtype());
    tensors.swa_cache = allocate_tensor(
        dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
        create_options.dtype());
#if defined(USE_MLU)
    // coff_dim = head_dim for ratio==128; merged dim = 2 * coff_dim.
    const int64_t cmp_coff_dim = head_dim;
    tensors.compress_state = allocate_tensor(
        {swa_count, block_size, 2 * cmp_coff_dim}, torch::kFloat32);
    tensors.compress_kv_state = tensors.compress_state.narrow(
        /*dim=*/2, /*start=*/0, /*length=*/cmp_coff_dim);
    tensors.compress_score_state = tensors.compress_state.narrow(
        /*dim=*/2, /*start=*/cmp_coff_dim, /*length=*/cmp_coff_dim);
#else
    tensors.compress_kv_state =
        allocate_tensor({swa_count, block_size, head_dim}, torch::kFloat32);
    tensors.compress_score_state =
        allocate_tensor({swa_count, block_size, head_dim}, torch::kFloat32);
#endif
  } else {
    tensors.swa_cache = allocate_tensor(
        dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
        create_options.dtype());
  }

  return tensors;
}

std::string dsv4_shape_summary(const DeepSeekV4KVCacheTensors& tensors,
                               int32_t compress_ratio) {
  std::ostringstream summary;
  if (compress_ratio == 1) {
    summary << "swa_cache=" << tensor_shape_string(tensors.swa_cache);
  } else if (compress_ratio == 4) {
    summary << "key_cache=" << tensor_shape_string(tensors.key_cache)
            << ", index_cache=" << tensor_shape_string(tensors.index_cache)
            << ", indexer_cache_scale="
            << tensor_shape_string(tensors.indexer_cache_scale)
            << ", swa_cache=" << tensor_shape_string(tensors.swa_cache)
            << ", compress_kv_state="
            << tensor_shape_string(tensors.compress_kv_state)
            << ", compress_score_state="
            << tensor_shape_string(tensors.compress_score_state)
            << ", compress_index_kv_state="
            << tensor_shape_string(tensors.compress_index_kv_state)
            << ", compress_index_score_state="
            << tensor_shape_string(tensors.compress_index_score_state);
  } else if (compress_ratio == 128) {
    summary << "key_cache=" << tensor_shape_string(tensors.key_cache)
            << ", swa_cache=" << tensor_shape_string(tensors.swa_cache)
            << ", compress_kv_state="
            << tensor_shape_string(tensors.compress_kv_state)
            << ", compress_score_state="
            << tensor_shape_string(tensors.compress_score_state);
  } else {
    summary << "swa_cache=" << tensor_shape_string(tensors.swa_cache);
  }
  return summary.str();
}

}  // namespace xllm
