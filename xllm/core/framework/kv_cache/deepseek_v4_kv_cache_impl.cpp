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

DeepSeekV4KVCacheImpl::DeepSeekV4KVCacheImpl(
    const DeepSeekV4KVCacheTensors& tensors)
    : key_cache_(tensors.key_cache),
      index_cache_(tensors.index_cache),
      indexer_cache_scale_(tensors.indexer_cache_scale),
      swa_cache_(tensors.swa_cache),
      compress_kv_state_(tensors.compress_kv_state),
      compress_score_state_(tensors.compress_score_state),
      compress_index_kv_state_(tensors.compress_index_kv_state),
      compress_index_score_state_(tensors.compress_index_score_state) {}

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
  return compress_kv_state_;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_score_state() const {
  return compress_score_state_;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_kv_state() const {
  return compress_index_kv_state_;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_score_state() const {
  return compress_index_score_state_;
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
  compress_kv_state_ =
      swap_tensor_blocks(compress_kv_state_, src_tensor, dst_tensor);
  compress_score_state_ =
      swap_tensor_blocks(compress_score_state_, src_tensor, dst_tensor);
  compress_index_kv_state_ =
      swap_tensor_blocks(compress_index_kv_state_, src_tensor, dst_tensor);
  compress_index_score_state_ =
      swap_tensor_blocks(compress_index_score_state_, src_tensor, dst_tensor);
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

  const torch::TensorOptions cache_options =
      torch::dtype(create_options.dtype()).device(create_options.device());
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(create_options.dtype());
  const torch::TensorOptions index_options =
      torch::dtype(cache_policy.index_dtype).device(create_options.device());
  const torch::TensorOptions state_options =
      torch::dtype(torch::kFloat32).device(create_options.device());
  const torch::TensorOptions scale_options =
      torch::dtype(cache_policy.scale_dtype).device(create_options.device());

  DeepSeekV4KVCacheTensors tensors;
  if (compress_ratio == 1) {
    tensors.swa_cache =
        torch::empty(dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
                     cache_options);
  } else if (compress_ratio == 4) {
    tensors.key_cache =
        torch::empty(dsv4_block_shape(c4_count, block_size, n_heads, head_dim),
                     cache_options);
    tensors.index_cache = torch::empty(
        dsv4_block_shape(c4_count, block_size, index_n_heads, index_head_dim),
        index_options);
    if (cache_policy.has_indexer_cache_scale) {
      tensors.indexer_cache_scale =
          torch::empty({c4_count, block_size, 1}, scale_options);
    }
    tensors.swa_cache =
        torch::empty(dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
                     cache_options);
    tensors.compress_kv_state =
        torch::empty({swa_count, block_size, 2 * head_dim}, state_options);
    tensors.compress_score_state =
        torch::empty({swa_count, block_size, 2 * head_dim}, state_options);
    tensors.compress_index_kv_state = torch::empty(
        {swa_count, block_size, 2 * index_head_dim}, state_options);
    tensors.compress_index_score_state = torch::empty(
        {swa_count, block_size, 2 * index_head_dim}, state_options);
  } else if (compress_ratio == 128) {
    tensors.key_cache = torch::empty(
        dsv4_block_shape(c128_count, block_size, n_heads, head_dim),
        cache_options);
    tensors.swa_cache =
        torch::empty(dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
                     cache_options);
    tensors.compress_kv_state =
        torch::empty({swa_count, block_size, head_dim}, state_options);
    tensors.compress_score_state =
        torch::empty({swa_count, block_size, head_dim}, state_options);
  } else {
    tensors.swa_cache =
        torch::empty(dsv4_block_shape(swa_count, block_size, n_heads, head_dim),
                     cache_options);
  }

  tensors.key_cache = cast_to_nd_format(tensors.key_cache);
  tensors.index_cache = cast_to_nd_format(tensors.index_cache);
  tensors.indexer_cache_scale = cast_to_nd_format(tensors.indexer_cache_scale);
  tensors.swa_cache = cast_to_nd_format(tensors.swa_cache);
  tensors.compress_kv_state = cast_to_nd_format(tensors.compress_kv_state);
  tensors.compress_score_state =
      cast_to_nd_format(tensors.compress_score_state);
  tensors.compress_index_kv_state =
      cast_to_nd_format(tensors.compress_index_kv_state);
  tensors.compress_index_score_state =
      cast_to_nd_format(tensors.compress_index_score_state);
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
