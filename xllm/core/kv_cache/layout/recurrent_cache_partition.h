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

#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/cache_partition.h"

namespace xllm {

// Components share one live slot and are checkpointed together. A KPool tail
// belongs to indexed Full layers, independently of the Conv/SSM layer
// ownership, and can also exist without Conv/SSM. SSM snapshots and tail tokens
// within a slot are distinct from the token interval between prefix
// checkpoints.
class RecurrentCachePartition final {
 public:
  RecurrentCachePartition(CachePartitionId id,
                          int32_t checkpoint_stride,
                          int32_t ssm_snapshots_per_slot,
                          bool enable_prefix_cache,
                          std::vector<CacheComponent> components,
                          int32_t tail_tokens_per_slot = 0);

  const CachePartitionId& id() const { return id_; }
  int32_t checkpoint_stride() const { return checkpoint_stride_; }
  int32_t ssm_snapshots_per_slot() const { return ssm_snapshots_per_slot_; }
  int32_t tail_tokens_per_slot() const { return tail_tokens_per_slot_; }
  bool enable_prefix_cache() const { return enable_prefix_cache_; }
  const std::vector<CacheComponent>& components() const { return components_; }
  const CacheComponent* find_component(CacheComponentRole role) const;

 private:
  CachePartitionId id_;
  int32_t checkpoint_stride_;
  int32_t ssm_snapshots_per_slot_;
  bool enable_prefix_cache_;
  std::vector<CacheComponent> components_;
  int32_t tail_tokens_per_slot_;
};

}  // namespace xllm
