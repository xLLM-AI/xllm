/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_shard_layout.h"

#include <glog/logging.h>

namespace xllm {

KVShardLayout::KVShardLayout(int32_t physical_block_size,
                             int32_t dcp_size,
                             int32_t dcp_rank)
    : physical_block_size_(physical_block_size),
      dcp_size_(dcp_size),
      dcp_rank_(dcp_rank) {
  CHECK_GT(physical_block_size_, 0) << "physical_block_size must be positive";
  CHECK_GT(dcp_size_, 0) << "dcp_size must be positive";
  CHECK_GE(dcp_rank_, 0) << "dcp_rank must be non-negative";
  CHECK_LT(dcp_rank_, dcp_size_) << "dcp_rank must be smaller than dcp_size";
}

int32_t KVShardLayout::owner_of(int64_t global_slot) const {
  CHECK_GE(global_slot, 0) << "global_slot must be non-negative";
  const int64_t logical_offset = global_slot % logical_block_size();
  return static_cast<int32_t>(logical_offset / physical_block_size_);
}

bool KVShardLayout::owns(int64_t global_slot) const {
  return global_slot >= 0 && owner_of(global_slot) == dcp_rank_;
}

int64_t KVShardLayout::localize(int64_t global_slot) const {
  if (!owns(global_slot)) {
    return kInvalidSlot;
  }
  const int64_t logical_block_id = global_slot / logical_block_size();
  const int64_t local_offset = global_slot % physical_block_size_;
  return logical_block_id * physical_block_size_ + local_offset;
}

int64_t KVShardLayout::globalize(int64_t local_slot) const {
  CHECK_GE(local_slot, 0) << "local_slot must be non-negative";
  const int64_t local_block_id = local_slot / physical_block_size_;
  const int64_t local_offset = local_slot % physical_block_size_;
  return (local_block_id * dcp_size_ + dcp_rank_) * physical_block_size_ +
         local_offset;
}

}  // namespace xllm
