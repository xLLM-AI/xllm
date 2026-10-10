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

#pragma once

#include "core/kv_cache/storage/indexed_kv_cache_impl.h"

namespace xllm {

class KPoolKVCacheImpl final : public IndexedKVCacheImpl {
 public:
  explicit KPoolKVCacheImpl(const IndexedKVCacheTensors& tensors);
  KPoolKVCacheImpl(const KVCacheShape& kv_cache_shape,
                   const KVCacheCreateOptions& create_options);

  torch::Tensor get_kpool_tail() const override;
  BlockTypeTensorMap get_block_type_tensors(BlockType type) const override;
  std::vector<std::vector<int64_t>> get_shapes() const override;

 private:
  KPoolKVCacheImpl(const KVCacheShape& kv_cache_shape,
                   const IndexedKVCacheTensors& tensors);

  torch::Tensor kpool_tail_;
};

}  // namespace xllm
