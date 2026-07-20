/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager_impl.h"

#include <unordered_set>

#include "framework/prefix_cache/prefix_cache_factory.h"
namespace xllm {
namespace {

bool mark_used(std::vector<uint8_t>* usage_ids, int32_t block_id) {
  CHECK(usage_ids != nullptr);
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids->size());
  if ((*usage_ids)[block_id] != 0) {
    return false;
  }
  (*usage_ids)[block_id] = 1;
  return true;
}

bool clear_used(std::vector<uint8_t>* usage_ids, int32_t block_id) {
  CHECK(usage_ids != nullptr);
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids->size());
  if ((*usage_ids)[block_id] == 0) {
    return false;
  }
  (*usage_ids)[block_id] = 0;
  return true;
}

}  // namespace

BlockManagerImpl::BlockManagerImpl(const Options& options)
    : BlockManager(options) {
  CHECK_GT(options.num_blocks(), 0) << "No blocks to allocate";
  CHECK_GT(options.block_size(), 0) << "Block size must be positive";
  if (options_.enable_prefix_cache()) {
    PrefixCache::Options prefix_cache_options;
    // A prefix cache indexes on a single block-boundary stride. For KV that is
    // the KV block size; for the linear-state checkpoint index it is the
    // prefill chunk stride. Feed the right one into the cache's single
    // block_size slot so the LINEAR cache reuses the base by-block machinery
    // verbatim.
    const int32_t prefix_cache_block_size =
        options.block_type() == BlockType::LINEAR
            ? options.linear_chunk_stride()
            : options.block_size();
    prefix_cache_options.block_size(prefix_cache_block_size)
        .hasher_type(options.hasher_type())
        .block_type(options.block_type());
    prefix_cache_ = create_prefix_cache(prefix_cache_options);
    CHECK(prefix_cache_) << "Failed to create prefix cache!";
  }

  size_t total_blocks = options_.num_blocks();
  block_size_ = options_.block_size();
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
    CHECK(mark_used(&usage_accounted_ids_, block_id))
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
    // deallocate() marks the passed ids logically released immediately; their
    // physical ids return to the free list when the last Block alias is
    // dropped.
    if ((!options_.enable_prefix_cache() || block.ref_count() <= 2u) &&
        clear_used(&usage_accounted_ids_, block.id())) {
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
    }
  }
}

bool BlockManagerImpl::has_enough_blocks(uint32_t num_blocks) {
  if (num_blocks <= num_free_blocks_) {
    return true;
  }

  // prefix cache is disabled, no way to evict blocks
  if (!options_.enable_prefix_cache()) {
    return false;
  }

  // try to evict some blocks from the prefix cache
  const uint32_t n_blocks_to_evict = num_blocks - num_free_blocks_;

  AUTO_COUNTER(prefix_cache_latency_seconds_evict);
  const uint32_t n_blocks_evicted = prefix_cache_->evict(n_blocks_to_evict);
  if (n_blocks_evicted < n_blocks_to_evict) {
    return false;
  }

  if (num_free_blocks_ >= num_blocks) {
    return true;
  }

  LOG(WARNING) << "Potential block leak, free blocks in allocator: "
               << num_free_blocks_
               << " blocks in prefix cache: " << prefix_cache_->num_blocks();
  return false;
}

std::vector<Block> BlockManagerImpl::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes,
    size_t* matched_tokens) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_match);

    size_t prefix_length = 0;
    std::vector<Block> shared_blocks =
        prefix_cache_->match(token_ids,
                             existed_shared_blocks,
                             mm_data,
                             block_hashes,
                             &prefix_length);

    COUNTER_ADD(prefix_cache_match_length_total, prefix_length);
    VLOG(1) << "Prefix cache matched " << shared_blocks.size()
            << " blocks, prefix_length=" << prefix_length;

    // update effective block usage
    for (const auto& block : shared_blocks) {
      if (mark_used(&usage_accounted_ids_, block.id())) {
        num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (matched_tokens != nullptr) {
      *matched_tokens = prefix_length;
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
  CHECK(num_free_blocks_ > 0) << "No more blocks available";
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_blocks_[prev_count - 1];
  return {block_id, this};
}

// caller should make sure the block_id is valid
void BlockManagerImpl::free(int32_t block_id) {
  // do nothing for reserved block 0
  if (block_id != 0) {
    if (clear_used(&usage_accounted_ids_, block_id)) {
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
    size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (block_size_ == 0) {
    return std::vector<Block>{};
  }
  const size_t held = seq->kv_state().num_blocks(block_type());
  const size_t num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
  if (num_blocks_needed <= held) {
    return std::vector<Block>{};
  }
  const size_t num_additional = num_blocks_needed - held;
  std::vector<Block> blocks = allocate(num_additional);
  if (blocks.size() != num_additional) {
    return std::nullopt;
  }
  return blocks;
}

}  // namespace xllm
