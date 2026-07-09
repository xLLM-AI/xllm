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

#include "concurrent_block_manager_impl.h"

#include <utility>

namespace xllm {

ConcurrentBlockManagerImpl::ConcurrentBlockManagerImpl(
    std::unique_ptr<BlockManager> inner)
    : BlockManager(inner->options()), inner_(std::move(inner)) {
  CHECK(inner_ != nullptr) << "ConcurrentBlockManagerImpl needs an inner leaf";
}

std::vector<Block> ConcurrentBlockManagerImpl::allocate(size_t num_blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Block> blocks = inner_->allocate(num_blocks);
  // Route Block dtor -> free through this wrapper so the wrapper's lock covers
  // the free path. inner_ remains the physical owner; free(id) re-acquires the
  // wrapper lock and forwards.
  for (Block& block : blocks) {
    block.set_manager(this);
  }
  return blocks;
}

Block ConcurrentBlockManagerImpl::allocate() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  Block block = inner_->allocate();
  block.set_manager(this);
  return block;
}

void ConcurrentBlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->deallocate(blocks);
}

void ConcurrentBlockManagerImpl::free(int32_t block_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->free(block_id);
}

std::vector<Block> ConcurrentBlockManagerImpl::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Block> blocks = inner_->allocate_shared(
      token_ids, existed_shared_blocks, mm_data, block_hashes);
  for (Block& block : blocks) {
    block.set_manager(this);
  }
  return blocks;
}

void ConcurrentBlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                                       std::vector<Block>& blocks,
                                       size_t existed_shared_blocks_num,
                                       const MMData& mm_data,
                                       const Slice<XXH3Key>& block_hashes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->cache(
      token_ids, blocks, existed_shared_blocks_num, mm_data, block_hashes);
}

void ConcurrentBlockManagerImpl::cache(const std::vector<Block>& blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->cache(blocks);
}

std::optional<std::vector<Block>>
ConcurrentBlockManagerImpl::allocate_for_sequence(Sequence* seq,
                                                  size_t num_tokens) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto blocks = inner_->allocate_for_sequence(seq, num_tokens);
  if (blocks.has_value()) {
    // Route Block dtor -> free through this wrapper so the wrapper's lock
    // covers the free path. inner_ stays the physical owner; free(id)
    // re-acquires the wrapper lock and forwards.
    for (Block& block : *blocks) {
      block.set_manager(this);
    }
  }
  return blocks;
}

void ConcurrentBlockManagerImpl::release_out_of_window(Sequence* seq) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->release_out_of_window(seq);
}

void ConcurrentBlockManagerImpl::reset_prefix_cache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  inner_->reset_prefix_cache();
}

size_t ConcurrentBlockManagerImpl::num_blocks_in_prefix_cache() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return inner_->num_blocks_in_prefix_cache();
}

size_t ConcurrentBlockManagerImpl::num_free_blocks() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return inner_->num_free_blocks();
}

size_t ConcurrentBlockManagerImpl::num_used_blocks() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return inner_->num_used_blocks();
}

size_t ConcurrentBlockManagerImpl::num_total_blocks() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return inner_->num_total_blocks();
}

double ConcurrentBlockManagerImpl::kv_cache_utilization() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return inner_->kv_cache_utilization();
}

}  // namespace xllm
