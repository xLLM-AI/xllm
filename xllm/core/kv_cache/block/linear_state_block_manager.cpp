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

LinearStateBlockManager::LinearStateBlockManager(BlockManager::Options options)
    : BlockManagerImpl(options.block_type(BlockType::LINEAR)
                           .hasher_type(BlockHasherType::TEXT)) {
  CHECK_GT(options.num_blocks(), 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
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
  if (!decode_step && !seq->is_graph_warmup() &&
      options_.enable_prefix_cache() && !blocks->empty()) {
    const size_t chunk_stride = block_size();
    const size_t cached_tokens = kv_state.kv_cache_tokens_num();
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
      (decode_step || !options_.enable_prefix_cache() ||
       (!checkpoint_hash.has_value() && !blocks->back().is_shared()))) {
    return std::vector<Block>{};
  }
  const auto& pool = options_.shared_block_pool();
  std::vector<Block> allocated = pool && pool->uses_slabs()
                                     ? allocate_with_placement(1, seq)
                                     : BlockManagerImpl::allocate(1);
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
  if (token_ids.size() == 0) {
    return {};
  }
  return BlockManagerImpl::allocate_shared(
      token_ids.slice(0, token_ids.size() - 1), {}, MMData(), block_hashes);
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || !options_.enable_prefix_cache()) {
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

HostLinearStateBlockManager::HostLinearStateBlockManager(
    BlockManager::Options options)
    : BlockManagerImpl(options.block_type(BlockType::LINEAR)
                           .hasher_type(BlockHasherType::TEXT)) {}

bool HostLinearStateBlockManager::allocate_for_prefetch(Sequence* seq,
                                                        size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size();
  const size_t cached_blocks = host_state.num_cached_blocks(BlockType::LINEAR);
  const size_t shared_blocks = host_state.shared_blocks_num(BlockType::LINEAR);
  const size_t cursor = std::min(cached_blocks, target_blocks);
  const bool source_in_target =
      cached_blocks > 0 && cached_blocks <= target_blocks;

  std::vector<Block> old_blocks = host_state.take_blocks(BlockType::LINEAR);
  // The vector is indexed by logical checkpoint, while only the deepest
  // matched checkpoint has a physical source after prefix-cache admission.
  // Keep the earlier positions as invalid placeholders so Store transfers and
  // restore selection continue to use absolute checkpoint indices.
  std::vector<Block> blocks(target_blocks);
  Block source;
  if (source_in_target) {
    for (size_t index = std::min(cached_blocks, old_blocks.size()); index > 0;
         --index) {
      Block& candidate = old_blocks[index - 1];
      if (candidate.is_valid()) {
        source = std::move(candidate);
        break;
      }
    }
  }
  std::vector<Block> dropped;
  dropped.reserve(old_blocks.size());
  for (size_t index = 0; index < old_blocks.size(); ++index) {
    Block& block = old_blocks[index];
    if (!block.is_valid()) {
      continue;
    }
    if (index >= cursor && index < target_blocks) {
      blocks[index] = std::move(block);
    } else {
      dropped.emplace_back(std::move(block));
    }
  }
  if (!dropped.empty()) {
    deallocate(dropped);
  }

  const bool has_source = source_in_target && source.is_valid();
  if (has_source) {
    blocks[cached_blocks - 1] = std::move(source);
  }

  seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), target_blocks);

  std::vector<size_t> missing;
  missing.reserve(target_blocks > cursor ? target_blocks - cursor : 0);
  for (size_t index = cursor; index < target_blocks; ++index) {
    if (!blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  const bool complete = allocate_missing_blocks(missing, hashes, blocks);
  if (!blocks.empty()) {
    host_state.replace_composite_blocks(
        BlockType::LINEAR,
        std::move(blocks),
        has_source ? std::min(shared_blocks, target_blocks) : 0,
        cursor);
  }
  return complete;
}

void HostLinearStateBlockManager::trim_prefetch_blocks(Sequence& seq,
                                                       size_t max_hit_tokens) {
  KVCacheState& host_state = seq.host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::LINEAR);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());

  // A Linear checkpoint vector is a logical index space. Keep only the
  // deepest valid checkpoint in the accepted gate range; all other physical
  // aliases, including valid blocks after the range, are temporary Store
  // state and must be released by this leaf.
  size_t source_index = keep;
  for (size_t index = keep; index > 0; --index) {
    if (blocks[index - 1].is_valid()) {
      source_index = index - 1;
      break;
    }
  }

  Block source;
  if (source_index < keep) {
    source = std::move(blocks[source_index]);
  }
  deallocate(blocks);
  blocks.clear();
  if (!source.is_valid()) {
    host_state.erase_blocks(BlockType::LINEAR);
    return;
  }
  blocks.resize(source_index + 1);
  blocks[source_index] = std::move(source);

  // Host restore selection addresses checkpoints by their absolute index.
  seq.update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq.linear_state_hashes();
  CHECK_GT(hashes.size(), source_index);
  BlockManagerImpl::cache(seq.hash_tokens(options_.hasher_type()),
                          blocks,
                          /*existed_shared_blocks_num=*/0,
                          seq.mm_data(),
                          hashes);

  host_state.replace_composite_blocks(
      BlockType::LINEAR, std::move(blocks), source_index + 1, source_index + 1);
}

}  // namespace xllm
