/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

namespace xllm {

ConcurrentBlockManagerImpl::ConcurrentBlockManagerImpl(const Options& options)
    : BlockManagerImpl(options) {}

std::vector<Block> ConcurrentBlockManagerImpl::allocate(size_t num_blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::allocate(num_blocks);
}

void ConcurrentBlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  BlockManagerImpl::deallocate(blocks);
}

std::vector<Block> ConcurrentBlockManagerImpl::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::allocate_shared(
      token_ids, existed_shared_blocks, mm_data);
}

void ConcurrentBlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                                       std::vector<Block>& blocks,
                                       size_t existed_shared_blocks_num,
                                       const MMData& mm_data) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  BlockManagerImpl::cache(
      token_ids, blocks, existed_shared_blocks_num, mm_data);
}

void ConcurrentBlockManagerImpl::cache(const std::vector<Block>& blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  BlockManagerImpl::cache(blocks);
}

size_t ConcurrentBlockManagerImpl::num_blocks_in_prefix_cache() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::num_blocks_in_prefix_cache();
}

size_t ConcurrentBlockManagerImpl::num_free_blocks() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::num_free_blocks();
}

double ConcurrentBlockManagerImpl::kv_cache_utilization() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::kv_cache_utilization();
}

size_t ConcurrentBlockManagerImpl::num_used_blocks() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::num_used_blocks();
}

void ConcurrentBlockManagerImpl::free(int32_t block_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  BlockManagerImpl::free(block_id);
}

Block ConcurrentBlockManagerImpl::allocate() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return BlockManagerImpl::allocate();
}

}  // namespace xllm
