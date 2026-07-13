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

#include "framework/kv_cache/indexed_kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_shape.h"
#include "util/tensor_helper.h"

namespace xllm {

namespace {

std::vector<int64_t> get_index_cache_shape(
    const IndexedKVCacheTensors& tensors) {
  return get_tensor_shape(tensors.index_cache);
}

void swap_tensor_blocks(torch::Tensor& tensor,
                        const torch::Tensor& src_tensor,
                        const torch::Tensor& dst_tensor) {
  if (!tensor.defined()) {
    return;
  }

  torch::Tensor selected = torch::index_select(tensor, 0, src_tensor);
  tensor.index_copy_(0, dst_tensor, selected);
}

}  // namespace

IndexedKVCacheImpl::IndexedKVCacheImpl(const IndexedKVCacheTensors& tensors)
    : KVCacheImpl(tensors.kv_cache_tensors),
      index_cache_(tensors.index_cache),
      index_cache_shape_(get_index_cache_shape(tensors)) {}

IndexedKVCacheImpl::IndexedKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : IndexedKVCacheImpl(
          create_indexed_kv_cache_tensors(kv_cache_shape, create_options)) {
  key_cache_shape_ = kv_cache_shape.key_cache_shape();
  if (kv_cache_shape.has_value_cache_shape()) {
    value_cache_shape_ = kv_cache_shape.value_cache_shape();
  }
  index_cache_shape_ = kv_cache_shape.index_cache_shape();
}

torch::Tensor IndexedKVCacheImpl::get_index_cache() const {
  return index_cache_;
}

bool IndexedKVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined() ||
         !index_cache_.defined();
}

std::vector<std::vector<int64_t>> IndexedKVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(3);
  shapes.emplace_back(key_cache_shape_);
  shapes.emplace_back(value_cache_shape_);
  shapes.emplace_back(index_cache_shape_);
  return shapes;
}

void IndexedKVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                                     torch::Tensor& dst_tensor) {
  swap_tensor_blocks(key_cache_, src_tensor, dst_tensor);
  swap_tensor_blocks(value_cache_, src_tensor, dst_tensor);
  swap_tensor_blocks(index_cache_, src_tensor, dst_tensor);
}

}  // namespace xllm
