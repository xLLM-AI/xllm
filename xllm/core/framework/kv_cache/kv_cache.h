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

#include <memory>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "framework/kv_cache/kv_cache_impl.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"

namespace xllm {

class KVCache final {
 public:
  KVCache();
  explicit KVCache(const KVCacheTensors& tensors);
  explicit KVCache(const IndexedKVCacheTensors& tensors);
  explicit KVCache(const LinearAttentionKVCacheTensors& tensors);
  explicit KVCache(const QuantizedKVCacheTensors& tensors);
  explicit KVCache(const DeepSeekV4KVCacheTensors& tensors);
  KVCache(const KVCacheShape& kv_cache_shape,
          const KVCacheCreateOptions& create_options,
          int64_t layer_id);
  KVCache(const KVCacheShape& kv_cache_shape,
          const KVCacheCreateOptions& create_options,
          BlockType type,
          int64_t layer_count);
  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;
  KVCache(KVCache&&) noexcept = default;
  KVCache& operator=(KVCache&&) noexcept = default;
  ~KVCache() = default;

  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;
  torch::Tensor get_index_cache() const;

  std::optional<torch::Tensor> get_k_cache_scale() const;
  std::optional<torch::Tensor> get_v_cache_scale() const;
  std::optional<torch::Tensor> get_indexer_cache_scale() const;

  torch::Tensor get_conv_cache() const;
  torch::Tensor get_ssm_cache() const;
  std::vector<KVCacheTensor> get_cache_tensors() const;
  torch::Tensor get_swa_cache() const;
  torch::Tensor get_compress_kv_state() const;
  torch::Tensor get_compress_score_state() const;
  torch::Tensor get_compress_index_kv_state() const;
  torch::Tensor get_compress_index_score_state() const;
  torch::Tensor get_compress_state() const;
  torch::Tensor get_compress_index_state() const;
  BlockTypeTensorMap get_block_type_tensors(BlockType type) const;
  std::vector<std::vector<int64_t>> get_shapes();

  bool empty() const;

  void swap_blocks(torch::Tensor& src_tensor, torch::Tensor& dst_tensor);

 private:
  std::unique_ptr<KVCacheImpl> impl_;
};

void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options);

}  // namespace xllm
