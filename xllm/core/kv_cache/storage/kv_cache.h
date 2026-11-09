/* Copyright 2025-2026 The xLLM Authors.

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

#include <memory>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_impl.h"
#include "core/kv_cache/storage/kv_cache_utils.h"

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
  torch::Tensor get_kpool_tail() const;

  bool has_request_state() const;

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

  bool is_static_shared() const noexcept { return static_layout_ != nullptr; }
  int64_t cache_row_stride(KVCacheTensorRole role) const;
  int64_t linear_state_slot_count() const;
  int64_t linear_checkpoint_stride() const;
  // Resolve a logical block/state ID to this layer and role's physical row.
  int64_t physical_row_index(int64_t slot_id, KVCacheTensorRole role) const;
  torch::Tensor physical_row_indices(const torch::Tensor& slot_ids,
                                     KVCacheTensorRole role) const;
  // Rows are [state id, group slots...]; a no-op outside shared storage.
  void update_linear_state_slots(const std::vector<int32_t>& rows) const;
  // This layer's [Conv, SSM] physical rows, [2, num_slots]. Undefined
  // outside shared storage and for layers without recurrent state.
  torch::Tensor linear_state_slot_row() const;
  torch::Tensor full_block_rows() const;
  void update_full_block_rows(const std::vector<int32_t>& rows) const;
  int64_t shared_checkpoint_stride() const;
  torch::Tensor physical_token_slots(const torch::Tensor& token_slots,
                                     int64_t block_size) const;

 private:
  [[nodiscard]] KVCache create_shared_view() const;

  std::shared_ptr<KVCacheImpl> impl_;
  std::shared_ptr<const StaticKVCacheLayout> static_layout_;
  std::shared_ptr<LinearStateSlotTable> linear_state_slots_;
  std::shared_ptr<LinearStateSlotTable> full_block_slots_;
  size_t linear_group_ = 0;
  size_t full_layer_index_ = 0;

  friend void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                                 const KVCacheShape& kv_cache_shape,
                                 const KVCacheCreateOptions& create_options);
};

void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options);

}  // namespace xllm
