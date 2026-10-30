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

#include "core/kv_cache/block/linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"

namespace xllm {

LinearStateBlockManager::LinearStateBlockManager(uint32_t num_slots,
                                                 int32_t chunk_stride,
                                                 bool enable_prefix_cache,
                                                 bool instance_is_decode)
    : BlockManagerImpl(
          BlockManager::Options{.num_blocks_ = num_slots,
                                .block_size_ = chunk_stride,
                                .enable_prefix_cache_ = enable_prefix_cache,
                                .block_type_ = BlockType::LINEAR,
                                .instance_is_decode_ = instance_is_decode}) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  KVCacheState& kv_state = seq->kv_state();
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK(blocks->empty() || blocks->back().is_valid());
  if (blocks->size() > 1) {
    const bool shared_source =
        kv_state.shared_blocks_num(BlockType::LINEAR) == blocks->size();
    const size_t keep_begin = blocks->size() - 1;
    deallocate(Slice<Block>(*blocks).slice(0, keep_begin));
    blocks->erase(blocks->begin(), blocks->begin() + keep_begin);
    kv_state.set_shared_blocks_num(BlockType::LINEAR, shared_source ? 1 : 0);
  }
  const bool decode_step =
      options_.instance_is_decode() || !seq->is_prefill_stage();
  size_t checkpoint_index = 0;
  std::optional<XXH3Key> checkpoint_hash;
  if (!decode_step && !seq->is_graph_warmup() && prefix_cache_ != nullptr &&
      !blocks->empty()) {
    const size_t chunk_stride = block_size();
    const size_t cached_tokens = seq->kv_cache_tokens_num();
    checkpoint_index = cached_tokens / chunk_stride;
    if (cached_tokens % chunk_stride == 0 &&
        checkpoint_index > kv_state.num_cached_blocks(BlockType::LINEAR)) {
      seq->update_linear_state_hashes(static_cast<uint32_t>(chunk_stride));
      const Slice<XXH3Key> hashes = seq->linear_state_hashes();
      CHECK_GE(hashes.size(), checkpoint_index);
      if (prefix_cache_->contains(hashes[checkpoint_index - 1])) {
        kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
      } else {
        checkpoint_hash = hashes[checkpoint_index - 1];
      }
    }
  }
  if (!blocks->empty() &&
      (decode_step || prefix_cache_ == nullptr ||
       (!checkpoint_hash.has_value() && !blocks->back().is_shared()))) {
    return std::vector<Block>{};
  }
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    // Checkpoint retention must not stall a private live slot.
    if (!blocks->empty() && !blocks->back().is_shared()) {
      return std::vector<Block>{};
    }
    return std::nullopt;
  }
  if (checkpoint_hash.has_value()) {
    blocks->back().set_hash_value(checkpoint_hash->data);
    prefix_cache_->insert(Slice<Block>(*blocks));
    kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
  }
  return allocated;
}

Block LinearStateBlockManager::allocate() {
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes) {
  // Leave a token for forward execution; KV block boundaries cannot constrain
  // sparse linear checkpoints without losing the matching recurrent state.
  if (token_ids.size() == 0 || prefix_cache_ == nullptr) {
    return {};
  }
  std::vector<Block> blocks = prefix_cache_->match(
      token_ids.slice(0, token_ids.size() - 1), {}, MMData(), block_hashes);
  for (const Block& block : blocks) {
    if (block.is_valid() && mark_used(&usage_accounted_ids_, block.id())) {
      num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return blocks;
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || prefix_cache_ == nullptr) {
    return;
  }
  const size_t publish_end =
      std::min(token_ids.size() / block_size(), blocks.size() - 1);
  if (publish_end <= existed_shared_blocks_num) {
    return;
  }
  CHECK_GE(block_hashes.size(), publish_end);
  BlockManagerImpl::cache(token_ids.slice(0, publish_end * block_size()),
                          blocks,
                          existed_shared_blocks_num,
                          mm_data,
                          block_hashes);
}

}  // namespace xllm
