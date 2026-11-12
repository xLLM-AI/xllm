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

#include "core/framework/batch/shared_kv_cache_mapping_tracker.h"

#include <glog/logging.h>

#include <utility>

#include "core/kv_cache/block/static_kv_cache_block_pool.h"

namespace xllm {

void SharedKVCacheMappingTracker::begin() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto update = pending_.rbegin(); update != pending_.rend(); ++update) {
    known_epochs_[update->type_index][update->id] = update->previous_epoch;
  }
  pending_.clear();
}

void SharedKVCacheMappingTracker::acknowledge() {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.clear();
}

void SharedKVCacheMappingTracker::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.clear();
  pool_.reset();
  for (auto& epochs : known_epochs_) {
    epochs.clear();
  }
}

void SharedKVCacheMappingTracker::append_rows(
    std::shared_ptr<StaticKVCacheBlockPool> pool,
    BlockType type,
    Slice<Block> blocks,
    std::vector<int32_t>* rows,
    std::vector<uint64_t>* lease_epochs) {
  CHECK(pool != nullptr);
  CHECK(pool->uses_slabs());
  CHECK(type == BlockType::KV || type == BlockType::LINEAR);
  std::lock_guard<std::mutex> lock(mutex_);
  if (pool_ != pool) {
    CHECK(pending_.empty()) << "A worker group must use one shared pool";
    pool_ = std::move(pool);
    for (auto& epochs : known_epochs_) {
      epochs.assign(pool_->num_slots(), 0);
    }
  }
  const size_t type_index = type == BlockType::KV ? 0 : 1;
  auto& known_epochs = known_epochs_[type_index];
  const size_t row_start = rows->size();
  std::vector<uint64_t> epochs;
  pool_->append_changed_rows(type, blocks, known_epochs, rows, &epochs);
  if (epochs.empty()) {
    return;
  }
  const size_t row_width = (rows->size() - row_start) / epochs.size();
  pending_.reserve(pending_.size() + epochs.size());
  for (size_t row = 0; row < epochs.size(); ++row) {
    const int32_t id = (*rows)[row_start + row * row_width];
    pending_.emplace_back(PendingUpdate{type_index, id, known_epochs[id]});
    known_epochs[id] = epochs[row];
  }
  if (lease_epochs != nullptr) {
    lease_epochs->insert(lease_epochs->end(), epochs.begin(), epochs.end());
  }
}

}  // namespace xllm
