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

#include "single_block_manager.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {
namespace {

// One physical id == one block; no prefix cache; blocks live under the SINGLE
// slot of the sequence's KVCacheState. id 0 stays the reserved padding slot
// (BlockManagerImpl reserves it in its ctor), so usable ids are [1, n-1].
BlockManager::Options make_single_block_options(uint32_t num_blocks) {
  BlockManager::Options options;
  options.num_blocks(num_blocks);
  options.block_size(/*unused, one id == one block=*/1);
  options.enable_prefix_cache(false);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::SINGLE);
  return options;
}

}  // namespace

SingleBlockManager::SingleBlockManager(uint32_t num_blocks,
                                       std::string resource_name,
                                       std::string exhaustion_message)
    : BlockManagerImpl(make_single_block_options(num_blocks)),
      resource_name_(std::move(resource_name)),
      exhaustion_message_(std::move(exhaustion_message)) {}

std::optional<std::vector<Block>> SingleBlockManager::allocate_for_sequence(
    Sequence* seq,
    size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  // One block per sequence, reused for its lifetime.
  if (seq->kv_state().num_blocks(block_type()) > 0) {
    return std::vector<Block>{};
  }
  std::vector<Block> blocks = allocate(1);
  if (blocks.empty()) {
    LOG(ERROR) << "Failed to allocate " << resource_name_
               << "! free=" << num_free_blocks()
               << ", used=" << num_used_blocks()
               << ", total=" << num_total_blocks()
               << (exhaustion_message_.empty() ? "" : ". ")
               << exhaustion_message_;
    return std::nullopt;
  }
  return blocks;
}

Block SingleBlockManager::allocate() {
  // Surface exhaustion with this resource's identity instead of the generic
  // base message, then delegate to the counting allocate(1) path (the base
  // zero-arg allocate() skips num_used accounting -- it exists for padding).
  // The base ctor's padding reservation runs before this vtable entry is
  // active, so it still uses BlockManagerImpl::allocate().
  CHECK_GT(num_free_blocks(), 0u)
      << (exhaustion_message_.empty()
              ? "No more " + resource_name_ + " blocks available"
              : exhaustion_message_);
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  CHECK_EQ(blocks.size(), 1u);
  return std::move(blocks[0]);
}

std::vector<Block> SingleBlockManager::allocate_shared(
    const Slice<int32_t>& /*token_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/,
    size_t* /*matched_tokens*/) {
  NOT_IMPLEMENTED();
  return {};
}

void SingleBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                               std::vector<Block>& /*blocks*/,
                               size_t /*existed_shared_blocks_num*/,
                               const MMData& /*mm_data*/,
                               const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

void SingleBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  NOT_IMPLEMENTED();
}

}  // namespace xllm
