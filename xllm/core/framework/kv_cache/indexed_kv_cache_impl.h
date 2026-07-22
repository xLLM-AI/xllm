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

#pragma once

#include "framework/kv_cache/kv_cache_impl.h"

namespace xllm {

class IndexedKVCacheImpl final : public KVCacheImpl {
 public:
  explicit IndexedKVCacheImpl(const IndexedKVCacheTensors& tensors);
  IndexedKVCacheImpl(const KVCacheShape& kv_cache_shape,
                     const KVCacheCreateOptions& create_options);
  IndexedKVCacheImpl(const KVCacheShape& kv_cache_shape,
                     const KVCacheCreateOptions& create_options,
                     BlockType type,
                     int64_t layer_count);

  torch::Tensor get_index_cache() const override;
  std::optional<torch::Tensor> get_k_cache_scale() const override;
  std::optional<torch::Tensor> get_v_cache_scale() const override;
  std::optional<torch::Tensor> get_indexer_cache_scale() const override;

  BlockTypeTensorMap get_block_type_tensors(BlockType type) const override;

  bool empty() const override;

  std::vector<std::vector<int64_t>> get_shapes() const override;

  void swap_blocks(torch::Tensor& src_tensor,
                   torch::Tensor& dst_tensor) override;

 private:
  torch::Tensor index_cache_;
  std::optional<torch::Tensor> index_cache_scale_;
  std::optional<torch::Tensor> key_cache_scale_;
  std::optional<torch::Tensor> value_cache_scale_;
  std::vector<int64_t> index_cache_shape_;
  std::vector<int64_t> index_cache_scale_shape_;
};

}  // namespace xllm
