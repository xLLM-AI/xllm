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

#include "core/kv_cache/storage/kpool_kv_cache_impl.h"

#include <glog/logging.h>

#include "core/kv_cache/layout/kv_cache_shape.h"

namespace xllm {

KPoolKVCacheImpl::KPoolKVCacheImpl(const IndexedKVCacheTensors& tensors)
    : IndexedKVCacheImpl(tensors), kpool_tail_(tensors.kpool_tail) {
  CHECK(kpool_tail_.defined()) << "KPool cache requires tail state.";
}

KPoolKVCacheImpl::KPoolKVCacheImpl(const KVCacheShape& kv_cache_shape,
                                   const KVCacheCreateOptions& create_options)
    : KPoolKVCacheImpl(
          kv_cache_shape,
          create_indexed_kv_cache_tensors(kv_cache_shape, create_options)) {}

KPoolKVCacheImpl::KPoolKVCacheImpl(const KVCacheShape& kv_cache_shape,
                                   const IndexedKVCacheTensors& tensors)
    : IndexedKVCacheImpl(kv_cache_shape, tensors),
      kpool_tail_(tensors.kpool_tail) {
  CHECK(kpool_tail_.defined()) << "KPool cache requires tail state.";
}

torch::Tensor KPoolKVCacheImpl::get_kpool_tail() const { return kpool_tail_; }

BlockTypeTensorMap KPoolKVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map =
      IndexedKVCacheImpl::get_block_type_tensors(type);
  if (type == BlockType::LINEAR && kpool_tail_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::KPOOL_TAIL, kpool_tail_);
  }
  return tensor_map;
}

std::vector<std::vector<int64_t>> KPoolKVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes = IndexedKVCacheImpl::get_shapes();
  shapes.emplace_back(kpool_tail_.sizes().vec());
  return shapes;
}

}  // namespace xllm
