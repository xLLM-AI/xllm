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

#include "core/kv_cache/prefix/prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <utility>

#include "core/framework/multimodal/mm_data.h"

namespace xllm {

std::vector<Block> PrefixCache::match(const Slice<int32_t>& token_ids,
                                      const Slice<Block>& existed_shared_blocks,
                                      const MMData& mm_data,
                                      const Slice<XXH3Key>& block_hashes) {
  const size_t n_blocks =
      num_hash_blocks(hasher_type_, token_ids.size(), block_size_);
  if (n_blocks == 0) {
    return std::vector<Block>();
  }

  const int64_t now = absl::ToUnixMicros(absl::Now());

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  const size_t start_block = std::min(existed_shared_blocks.size(), n_blocks);
  blocks.insert(blocks.end(),
                existed_shared_blocks.begin(),
                existed_shared_blocks.begin() + start_block);

  auto batch_begin = lru_lst_.end();

  auto match_block = [&](const XXH3Key& token_hash_key) -> bool {
    auto iter = cached_blocks_.find(token_hash_key);
    if (iter == cached_blocks_.end()) {
      return false;
    }
    blocks.emplace_back(iter->second.block_);
    batch_begin = touch_node(iter->second, batch_begin, now);
    return true;
  };

  // Fast path: precomputed chained hash covers every matchable block, so we
  // only do hash-table lookups and never recompute a hash.
  if (block_hashes.size() >= n_blocks) {
    for (size_t b = start_block; b < n_blocks; ++b) {
      if (!match_block(block_hashes[b])) {
        break;
      }
    }
  } else {
    // Fallback: compute the chained hash on the fly.
    XXH3Key token_hash_key =
        existed_shared_blocks.empty()
            ? XXH3Key{}
            : XXH3Key{existed_shared_blocks.back().get_immutable_hash_value()};
    auto hasher =
        BlockHasher::create(hasher_type_, mm_data, start_block * block_size_);
    for (size_t b = start_block; b < n_blocks; ++b) {
      const size_t i = b * block_size_;
      const uint8_t* pre_hash_value = (b == 0) ? nullptr : token_hash_key.data;
      hasher->compute(
          token_ids, i, i + block_size_, pre_hash_value, token_hash_key);
      if (!match_block(token_hash_key)) {
        break;
      }
    }
  }

  return blocks;
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           size_t existed_shared_blocks_num,
                           const MMData& mm_data,
                           const Slice<XXH3Key>& block_hashes) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // align tokens to block boundary
  const size_t n_blocks =
      std::min(num_hash_blocks(hasher_type_, token_ids.size(), block_size_),
               blocks.size());

  if (n_blocks <= existed_shared_blocks_num) {
    return n_blocks * block_size_;
  }

  auto batch_begin = lru_lst_.end();

  // Fill `token_hash_key` with the chained hash of block `block_idx`, reusing
  // the precomputed hash when it covers all blocks, otherwise computing it.
  // The KV / C4 / C128 prefix is solid (no placeholders), so the compute path
  // seeds the chain in O(1) from the previous block's stamped hash. SWA's
  // slid-out placeholders are handled by LinearStatePrefixCache::insert, which
  // overrides this.
  const bool use_precomputed = block_hashes.size() >= n_blocks;
  XXH3Key token_hash_key = existed_shared_blocks_num == 0
                               ? XXH3Key{}
                               : XXH3Key{blocks[existed_shared_blocks_num - 1]
                                             .get_immutable_hash_value()};
  std::unique_ptr<BlockHasher> hasher;
  if (!use_precomputed) {
    hasher = BlockHasher::create(
        hasher_type_, mm_data, existed_shared_blocks_num * block_size_);
  }
  auto fill_block_hash = [&](size_t block_idx) {
    if (use_precomputed) {
      token_hash_key = block_hashes[block_idx];
      return;
    }
    const size_t i = block_idx * block_size_;
    const uint8_t* pre_hash_value =
        (block_idx == 0) ? nullptr : token_hash_key.data;
    hasher->compute(
        token_ids, i, i + block_size_, pre_hash_value, token_hash_key);
  };

  for (size_t block_idx = existed_shared_blocks_num; block_idx < n_blocks;
       ++block_idx) {
    fill_block_hash(block_idx);
    blocks[block_idx].set_hash_value(token_hash_key.data);

    const bool inserted =
        cache_node(batch_begin, token_hash_key, blocks[block_idx], now);
    batch_begin->chain_known_ = true;
    batch_begin->block_index_ = block_idx;
    auto& parent = batch_begin->parent_;
    parent = block_idx == 0
                 ? std::nullopt
                 : std::make_optional<XXH3Key>(
                       blocks[block_idx - 1].get_immutable_hash_value());

    if (inserted && event_journal_ && event_journal_->enabled()) {
      const size_t begin = block_idx * block_size_;
      std::vector<int32_t> tokens(token_ids.begin() + begin,
                                  token_ids.begin() + begin + block_size_);
      event_journal_->store(
          event_scope_,
          KVCacheEventJournal::encode_hash(token_hash_key.data),
          parent ? KVCacheEventJournal::encode_hash(parent->data)
                 : std::string(),
          std::move(tokens),
          static_cast<uint32_t>(block_idx));
    }
  }

  return n_blocks * block_size_;
}

size_t PrefixCache::insert(Slice<Block> blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  auto batch_begin = lru_lst_.end();
  XXH3Key token_hash_key;

  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i].is_valid()) {
      continue;
    }
    token_hash_key.set(blocks[i].get_immutable_hash_value());

    if (cache_node(batch_begin, token_hash_key, blocks[i], now) &&
        event_journal_ && event_journal_->enabled()) {
      event_journal_->store_copy(
          event_scope_, KVCacheEventJournal::encode_hash(token_hash_key.data));
    }
  }

  return blocks.size() * block_size_;
}

bool PrefixCache::contains(const XXH3Key& hash) const {
  return cached_blocks_.find(hash) != cached_blocks_.end();
}

bool PrefixCache::owns(const Block& block) const {
  return cached_block_ids_.contains(block.id());
}

std::vector<PrefixCacheEntry> PrefixCache::entries() const {
  std::vector<PrefixCacheEntry> result;
  result.reserve(cached_blocks_.size());
  for (const Node& node : lru_lst_) {
    result.emplace_back(PrefixCacheEntry{
        .hash = XXH3Key(node.block_.get_immutable_hash_value()),
        .parent = node.parent_,
        .dependency = node.dependency_,
        .block_id = node.block_.id(),
        .block_index = node.block_index_,
        .last_access_time = node.last_access_time_,
        .chain_known = node.chain_known_,
        .evictable = !node.block_.is_shared(),
    });
  }
  return result;
}

void PrefixCache::set_dependency(const XXH3Key& hash, XXH3Key dependency) {
  const auto it = cached_blocks_.find(hash);
  if (it != cached_blocks_.end()) {
    it->second.dependency_ = std::move(dependency);
  }
}

PrefixCache::DNodeList::iterator PrefixCache::evict_node(
    decltype(cached_blocks_)::iterator it) {
  auto& node = it->second;
  cached_block_ids_.erase(node.block_.id());
  auto next = lru_lst_.erase(lru_lst_.iterator_to(node));
  auto removed = cached_blocks_.extract(it);
  if (event_journal_ && event_journal_->enabled()) {
    event_journal_->remove(
        event_scope_, KVCacheEventJournal::encode_hash(removed.key().data));
  }
  return next;
}

bool PrefixCache::evict_block(const XXH3Key& hash) {
  const auto it = cached_blocks_.find(hash);
  if (it == cached_blocks_.end() || it->second.block_.is_shared()) {
    return false;
  }
  evict_node(it);
  return true;
}

size_t PrefixCache::evict(size_t n_blocks) {
  size_t evict_count = 0;
  auto iter_node = lru_lst_.begin();
  const auto end = lru_lst_.end();
  while (evict_count < n_blocks && iter_node != end) {
    if (iter_node->block_.is_shared()) {
      ++iter_node;
      continue;
    }

    const XXH3Key hash(iter_node->block_.get_immutable_hash_value());
    iter_node = evict_node(cached_blocks_.find(hash));
    ++evict_count;
  }

  return evict_count;
}

}  // namespace xllm
