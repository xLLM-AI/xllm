/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/kv_cache/block/block_manager_impl.h"

#include <unordered_set>

#include "core/framework/request/sequence.h"
#include "core/kv_cache/prefix/prefix_cache_factory.h"
namespace xllm {
namespace {

bool clear_used(std::vector<uint8_t>& usage_ids, int32_t block_id) {
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids.size());
  if (usage_ids[block_id] == 0) {
    return false;
  }
  usage_ids[block_id] = 0;
  return true;
}

}  // namespace

bool BlockManagerImpl::mark_used(std::vector<uint8_t>& usage_ids,
                                 int32_t block_id) {
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids.size());
  if (usage_ids[block_id] != 0) {
    return false;
  }
  usage_ids[block_id] = 1;
  return true;
}

BlockManagerImpl::BlockManagerImpl(const Options& options)
    : BlockManager(options) {
  CHECK_GT(options.num_blocks(), 0) << "No blocks to allocate";
  CHECK_GT(options.block_size(), 0) << "Block size must be positive";
  if (options_.enable_prefix_cache()) {
    PrefixCache::Options prefix_cache_options;
    prefix_cache_options.block_size(options.block_size())
        .hasher_type(options.hasher_type())
        .block_type(options.block_type());
    prefix_cache_ = create_prefix_cache(prefix_cache_options);
    CHECK(prefix_cache_) << "Failed to create prefix cache!";
    if (options.event_journal()) {
      prefix_cache_->set_event_sink(options.event_journal(),
                                    options.event_scope());
    }
  }

  size_t total_blocks = options_.num_blocks();
  block_size_ = options_.block_size();
  if (options_.shared_block_pool()) {
    CHECK_EQ(total_blocks, options_.shared_block_pool()->num_slots());
    usage_accounted_ids_.assign(total_blocks, 0);
    return;
  }
  num_free_blocks_.store(total_blocks, std::memory_order_relaxed);
  usage_accounted_ids_.assign(total_blocks, 0);
  free_blocks_.reserve(total_blocks);
  for (int32_t i = 0; i < total_blocks; ++i) {
    // push smaller block ids to the back of the vector
    free_blocks_.push_back(total_blocks - i - 1);
  }

  // reserve block 0 for padding
  padding_block_ = allocate();
  CHECK_EQ(padding_block_.id(), 0) << "Padding block id should be 0";
}

std::vector<Block> BlockManagerImpl::allocate(size_t num_blocks) {
  return allocate_with_placement(num_blocks, nullptr);
}

std::vector<Block> BlockManagerImpl::allocate_with_placement(
    size_t num_blocks,
    const void* placement_hint) {
  if (options_.shared_block_pool()) {
    const auto& pool = options_.shared_block_pool();
    std::vector<int32_t> ids =
        pool->allocate(num_blocks, block_type(), placement_hint);
    // Composite coordinates typed-slab victims across leaves; coarse legacy
    // pools retain their leaf-local cache reclamation.
    while (ids.size() != num_blocks && !pool->uses_slabs() &&
           evict_prefix_cache(1) != 0) {
      ids = pool->allocate(num_blocks, block_type());
    }
    std::vector<Block> blocks;
    blocks.reserve(ids.size());
    for (int32_t block_id : ids) {
      CHECK(mark_used(usage_accounted_ids_, block_id));
      blocks.emplace_back(block_id, this);
    }
    num_used_blocks_.fetch_add(ids.size(), std::memory_order_relaxed);
    return blocks;
  }
  if (!has_enough_blocks(num_blocks)) {
    return {};
  }
  CHECK(num_blocks <= num_free_blocks_) << "Not enough blocks available";
  std::vector<Block> blocks;
  blocks.reserve(num_blocks);
  for (uint32_t i = 0; i < num_blocks; ++i) {
    size_t prev_count =
        num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
    const int32_t block_id = free_blocks_[prev_count - 1];
    CHECK(mark_used(usage_accounted_ids_, block_id))
        << "block " << block_id << " usage accounted repeatedly";
    blocks.emplace_back(block_id, this);
  }

  // const auto block_ids = allocate(num_blocks);
  num_used_blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
  return blocks;
}

void BlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  for (const auto& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    // Prefix-cache blocks may be shared by cache aliases, so only drop
    // effective usage when no sequence owner remains. Without prefix cache,
    // deallocate() keeps shared physical usage pinned by any remaining alias;
    // standalone pools retain their existing logical release behavior. The
    // physical id returns only when the last Block alias is dropped.
    const bool logically_released =
        options_.enable_prefix_cache()
            ? block.ref_count() <= 2u
            : (!options_.shared_block_pool() || block.ref_count() == 1u);
    if (logically_released && clear_used(usage_accounted_ids_, block.id())) {
      if (num_used_blocks_ == 0) {
        LOG(ERROR) << "num_used_blocks_==0 cannot fetch_sub for id:"
                   << block.id()
                   << ", total block size: " << num_total_blocks();
        std::unordered_set<int32_t> block_id_set;
        block_id_set.insert(block.id());
        std::string error_msg = "Block already released: ";
        for (auto& id : free_blocks_) {
          if (block_id_set.count(id) != 0) {
            error_msg.append(std::to_string(id)).append(" ");
          }
        }
        LOG(FATAL) << error_msg;
      }
      num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
      if (options_.shared_block_pool()) {
        options_.shared_block_pool()->mark_inactive(block.id(), block_type());
      }
    }
  }
}

void BlockManagerImpl::release(std::span<Block> blocks) {
  for (Block& block : blocks) {
    if (block.is_valid() && prefix_cache_ && block.ref_count() == 2u &&
        prefix_cache_->owns(block)) {
      deallocate(Slice<Block>(&block, /*size=*/1));
    }
    // Uncached blocks remain active until free() retires their final alias.
    block = Block();
  }
}

size_t BlockManagerImpl::evict_prefix_cache(size_t num_blocks) {
  if (!prefix_cache_) {
    return 0;
  }
  AUTO_COUNTER(prefix_cache_latency_seconds_evict);
  return prefix_cache_->evict(num_blocks);
}

std::vector<PrefixCacheEntry> BlockManagerImpl::prefix_cache_entries() const {
  return prefix_cache_ ? prefix_cache_->entries()
                       : std::vector<PrefixCacheEntry>{};
}

bool BlockManagerImpl::evict_cached_block(const XXH3Key& hash) {
  return prefix_cache_ && prefix_cache_->evict_block(hash);
}

void BlockManagerImpl::set_prefix_dependency(const XXH3Key& hash,
                                             const XXH3Key& dependency) {
  if (prefix_cache_) {
    prefix_cache_->set_dependency(hash, dependency);
  }
}

bool BlockManagerImpl::has_enough_blocks(uint32_t num_blocks) {
  const size_t free_blocks = num_free_blocks();
  if (num_blocks <= free_blocks) {
    return true;
  }
  const size_t required = num_blocks - free_blocks;
  if (evict_prefix_cache(required) < required) {
    return false;
  }
  if (num_blocks <= num_free_blocks()) {
    return true;
  }
  if (!options_.shared_block_pool()) {
    LOG(WARNING) << "Potential block leak, free blocks in allocator: "
                 << num_free_blocks_
                 << " blocks in prefix cache: " << num_blocks_in_prefix_cache();
  }
  return false;
}

std::vector<Block> BlockManagerImpl::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  // only allocate shared blocks for prefix caching prefill sequences
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_match);

    std::vector<Block> shared_blocks = prefix_cache_->match(
        token_ids, existed_shared_blocks, mm_data, block_hashes);

    const size_t prefix_length =
        shared_blocks.size() * static_cast<size_t>(block_size_);
    VLOG(1) << "Prefix cache matched " << shared_blocks.size()
            << " blocks, prefix_length=" << prefix_length;

    // update effective block usage
    for (const auto& block : shared_blocks) {
      if (block.is_valid() && mark_used(usage_accounted_ids_, block.id())) {
        num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
        if (options_.shared_block_pool()) {
          options_.shared_block_pool()->mark_active(block.id(), block_type());
        }
      }
    }
    return shared_blocks;
  }
  return {};
}

void BlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                             std::vector<Block>& blocks,
                             size_t existed_shared_blocks_num,
                             const MMData& mm_data,
                             const Slice<XXH3Key>& block_hashes) {
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(
        token_ids, blocks, existed_shared_blocks_num, mm_data, block_hashes);
  }
}

void BlockManagerImpl::cache(const std::vector<Block>& blocks) {
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(blocks);
  }
}

// allocate a block id
Block BlockManagerImpl::allocate() {
  if (options_.shared_block_pool()) {
    std::vector<Block> blocks = allocate(1);
    return blocks.empty() ? Block() : std::move(blocks[0]);
  }
  CHECK(num_free_blocks_ > 0) << "No more blocks available";
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_blocks_[prev_count - 1];
  return {block_id, this};
}

// caller should make sure the block_id is valid
void BlockManagerImpl::free(int32_t block_id) {
  if (options_.shared_block_pool()) {
    if (clear_used(usage_accounted_ids_, block_id)) {
      num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
    }
    options_.shared_block_pool()->release(block_id, block_type());
    return;
  }
  // do nothing for reserved block 0
  if (block_id != 0) {
    if (clear_used(usage_accounted_ids_, block_id)) {
      CHECK_GT(num_used_blocks_.load(std::memory_order_relaxed), 0u);
      num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
    }
    size_t prev_count =
        num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
    CHECK(prev_count < free_blocks_.size());
    free_blocks_[prev_count] = block_id;
  }
}

// Flat incremental growth (the default sequence-level policy). Reads how many
// blocks the sequence already holds for this manager's block_type(), allocates
// the delta to cover num_tokens, and returns it (does not insert into the
// sequence -- the CompositeBlockManager commits the returned blocks). Returns
// std::nullopt on post-eviction shortage. SlidingWindow / Single override.
std::optional<std::vector<Block>> BlockManagerImpl::allocate_for_sequence(
    Sequence* seq,
    KVCacheState& kv_state,
    size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  const size_t held = kv_state.num_blocks(block_type());
  const size_t num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
  if (num_blocks_needed <= held) {
    return std::vector<Block>{};
  }
  const size_t num_additional = num_blocks_needed - held;
  const auto& pool = options_.shared_block_pool();
  std::vector<Block> blocks = pool && pool->uses_slabs()
                                  ? allocate_with_placement(num_additional, seq)
                                  : allocate(num_additional);
  if (blocks.size() != num_additional) {
    return std::nullopt;
  }
  return blocks;
}

bool BlockManagerImpl::allocate_for_prefetch(Sequence* seq, size_t num_tokens) {
  return allocate_prefetch_range(seq, num_tokens, /*required_begin_block=*/0);
}

bool BlockManagerImpl::allocate_prefetch_range(Sequence* seq,
                                               size_t num_tokens,
                                               size_t required_begin_block) {
  if (seq == nullptr) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size_;
  required_begin_block = std::min(required_begin_block, target_blocks);
  const size_t shared_blocks = host_state.shared_blocks_num(block_type());
  const size_t cached_blocks = host_state.num_cached_blocks(block_type());
  std::vector<Block> blocks = host_state.take_blocks(block_type());
  blocks.resize(target_blocks);

  std::vector<size_t> missing;
  missing.reserve(target_blocks - required_begin_block);
  for (size_t index = required_begin_block; index < target_blocks; ++index) {
    if (!blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  seq->update_block_hashes(static_cast<uint32_t>(block_size_),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes = seq->block_hashes();
  CHECK_GE(hashes.size(), target_blocks);
  size_t allocatable = std::min(
      missing.size(), num_free_blocks() + num_blocks_in_prefix_cache());
  const auto& pool = options_.shared_block_pool();
  const auto allocate_prefetch = [this, seq, &pool](size_t count) {
    return pool && pool->uses_slabs() ? allocate_with_placement(count, seq)
                                      : allocate(count);
  };
  std::vector<Block> allocated = allocate_prefetch(allocatable);
  if (allocated.empty() && allocatable > 0) {
    allocatable = std::min(missing.size(), num_free_blocks());
    allocated = allocate_prefetch(allocatable);
  }
  for (size_t index = 0; index < allocated.size(); ++index) {
    const size_t block_index = missing[index];
    allocated[index].set_hash_value(hashes[block_index].data);
    blocks[block_index] = std::move(allocated[index]);
  }
  const bool complete = allocated.size() == missing.size();

  if (!blocks.empty()) {
    host_state.replace_composite_blocks(block_type(),
                                        std::move(blocks),
                                        std::min(shared_blocks, target_blocks),
                                        std::min(cached_blocks, target_blocks));
  }
  return complete;
}

bool BlockManagerImpl::allocate_missing_blocks(
    const std::vector<size_t>& missing,
    const Slice<XXH3Key>& hashes,
    std::vector<Block>& blocks) {
  size_t allocatable = std::min(
      missing.size(), num_free_blocks() + num_blocks_in_prefix_cache());
  std::vector<Block> allocated = allocate(allocatable);
  if (allocated.empty() && allocatable > 0) {
    allocatable = std::min(missing.size(), num_free_blocks());
    allocated = allocate(allocatable);
  }
  for (size_t index = 0; index < allocated.size(); ++index) {
    const size_t block_index = missing[index];
    allocated[index].set_hash_value(hashes[block_index].data);
    blocks[block_index] = std::move(allocated[index]);
  }
  return allocated.size() == missing.size();
}

void BlockManagerImpl::trim_prefetch_blocks(Sequence& seq,
                                            size_t max_hit_tokens) {
  KVCacheState& host_state = seq.host_kv_state();
  const size_t keep = std::min(max_hit_tokens / block_size_,
                               host_state.num_blocks(block_type()));
  const size_t cached =
      std::min(host_state.num_cached_blocks(block_type()), keep);
  std::vector<Block> blocks = host_state.take_blocks(block_type());
  if (blocks.size() > keep) {
    deallocate(Slice<Block>(blocks).slice(keep, blocks.size()));
    blocks.resize(keep);
  }

  if (blocks.empty()) {
    host_state.erase_blocks(block_type());
    return;
  }

  // Store prefetch owns newly materialized Host blocks until this point. Once
  // the gate boundary is known, publish exactly the retained range and make
  // both state cursors describe that published logical range.
  seq.update_block_hashes(static_cast<uint32_t>(block_size_),
                          options_.hasher_type());
  cache(seq.hash_tokens(options_.hasher_type()),
        blocks,
        cached,
        seq.mm_data(),
        seq.block_hashes());
  host_state.replace_composite_blocks(
      block_type(), std::move(blocks), keep, keep);
}

}  // namespace xllm
