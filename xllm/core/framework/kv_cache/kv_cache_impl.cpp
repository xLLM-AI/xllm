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

#include "framework/kv_cache/kv_cache_impl.h"

#include <utility>

#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "util/tensor_helper.h"

namespace xllm {

void KVCacheImpl::create_host_tensor(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     torch::Tensor* tensor,
                                     std::vector<int64_t>* shape) {
  CHECK(tensor != nullptr) << "tensor must not be null.";
  HostPageAlignedRegion region;
  create_host_page_aligned_tensor(dims, dtype, tensor, &region);
  host_page_aligned_regions_.emplace_back(std::move(region));
  if (shape != nullptr) {
    *shape = dims;
  }
}

KVCacheImpl::KVCacheImpl(const KVCacheTensors& tensors)
    : key_cache_(tensors.key_cache),
      value_cache_(tensors.value_cache),
      key_cache_shape_(get_tensor_shape(tensors.key_cache)),
      value_cache_shape_(get_tensor_shape(tensors.value_cache)) {}

KVCacheImpl::KVCacheImpl(const KVCacheShape& kv_cache_shape,
                         const KVCacheCreateOptions& create_options)
    : KVCacheImpl(create_kv_cache_tensors(kv_cache_shape, create_options)) {
  key_cache_shape_ = kv_cache_shape.key_cache_shape();
  if (kv_cache_shape.has_value_cache_shape()) {
    value_cache_shape_ = kv_cache_shape.value_cache_shape();
  }
}

KVCacheImpl::KVCacheImpl(const KVCacheShape& kv_cache_shape,
                         const KVCacheCreateOptions& create_options,
                         BlockType type,
                         int64_t layer_count) {
  CHECK(type == BlockType::KV)
      << "Base KVCacheImpl host cache only supports BlockType::KV.";
  host_page_aligned_regions_.reserve(2);
  if (kv_cache_shape.has_key_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.key_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.dtype(),
        &key_cache_,
        &key_cache_shape_);
  }
  if (kv_cache_shape.has_value_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.value_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.dtype(),
        &value_cache_,
        &value_cache_shape_);
  }
}

torch::Tensor KVCacheImpl::get_k_cache() const { return key_cache_; }

torch::Tensor KVCacheImpl::get_v_cache() const { return value_cache_; }

std::optional<torch::Tensor> KVCacheImpl::get_k_cache_scale() const {
  return std::nullopt;
}

std::optional<torch::Tensor> KVCacheImpl::get_v_cache_scale() const {
  return std::nullopt;
}

std::optional<torch::Tensor> KVCacheImpl::get_indexer_cache_scale() const {
  return std::nullopt;
}

torch::Tensor KVCacheImpl::get_index_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_conv_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_ssm_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_swa_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_compress_kv_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_score_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_kv_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_score_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_state() const {
  return torch::Tensor();
}

std::vector<KVCacheTensor> KVCacheImpl::get_cache_tensors() const {
  std::vector<KVCacheTensor> tensors;
  tensors.reserve(8);
  auto add_tensor = [&tensors](KVCacheTensorRole role,
                               const torch::Tensor& tensor,
                               BlockType block_type) {
    if (tensor.defined() && tensor.numel() > 0) {
      tensors.emplace_back(KVCacheTensor{role,
                                         tensor,
                                         cache_group_id(block_type),
                                         block_type == BlockType::SINGLE});
    }
  };

  add_tensor(KVCacheTensorRole::KEY, get_k_cache(), BlockType::KV);
  add_tensor(KVCacheTensorRole::VALUE, get_v_cache(), BlockType::KV);
  add_tensor(KVCacheTensorRole::INDEX, get_index_cache(), BlockType::KV);
  const auto index_scale = get_indexer_cache_scale();
  if (index_scale.has_value()) {
    add_tensor(
        KVCacheTensorRole::INDEX_SCALE, index_scale.value(), BlockType::KV);
  }
  add_tensor(KVCacheTensorRole::CONV, get_conv_cache(), BlockType::SINGLE);
  add_tensor(KVCacheTensorRole::SSM, get_ssm_cache(), BlockType::SINGLE);
  const auto key_scale = get_k_cache_scale();
  if (key_scale.has_value()) {
    add_tensor(KVCacheTensorRole::KEY_SCALE, key_scale.value(), BlockType::KV);
  }
  const auto value_scale = get_v_cache_scale();
  if (value_scale.has_value()) {
    add_tensor(
        KVCacheTensorRole::VALUE_SCALE, value_scale.value(), BlockType::KV);
  }
  return tensors;
}

BlockTypeTensorMap KVCacheImpl::get_block_type_tensors(BlockType type) const {
  BlockTypeTensorMap tensor_map;
  if (type != BlockType::KV) {
    return tensor_map;
  }
  if (key_cache_.defined() && key_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::KEY, key_cache_);
  }
  if (value_cache_.defined() && value_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::VALUE, value_cache_);
  }
  return tensor_map;
}

bool KVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined();
}

std::vector<std::vector<int64_t>> KVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(2);
  shapes.emplace_back(key_cache_shape_);
  shapes.emplace_back(value_cache_shape_);
  return shapes;
}

void KVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                              torch::Tensor& dst_tensor) {
  // batch select keys and values
  auto selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  auto selected_values = torch::index_select(value_cache_, 0, src_tensor);

  // batch copy keys and values to dst indices
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  value_cache_.index_copy_(0, dst_tensor, selected_values);
}

}  // namespace xllm
