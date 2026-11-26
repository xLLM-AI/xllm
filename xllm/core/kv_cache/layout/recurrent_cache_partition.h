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

// Conv and SSM share one live slot and are checkpointed together. SSM snapshots
// within that slot serve speculative verification, independently of the token
// interval between prefix checkpoints.
class RecurrentCachePartition final {
 public:
  RecurrentCachePartition(CachePartitionId id,
                          int32_t checkpoint_stride,
                          int32_t ssm_snapshots_per_slot,
                          bool enable_prefix_cache,
                          std::vector<int32_t> layers,
                          CacheScalarType conv_dtype,
                          CacheScalarType ssm_dtype);

  const CachePartitionId& id() const { return id_; }
  int32_t checkpoint_stride() const { return checkpoint_stride_; }
  int32_t ssm_snapshots_per_slot() const { return ssm_snapshots_per_slot_; }
  bool enable_prefix_cache() const { return enable_prefix_cache_; }
  const CacheComponent& conv() const { return conv_; }
  const CacheComponent& ssm() const { return ssm_; }

 private:
  CachePartitionId id_;
  int32_t checkpoint_stride_;
  int32_t ssm_snapshots_per_slot_;
  bool enable_prefix_cache_;
  CacheComponent conv_;
  CacheComponent ssm_;
};

}  // namespace xllm
