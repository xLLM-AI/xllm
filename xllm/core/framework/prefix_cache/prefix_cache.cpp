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

#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <iostream>
#include <thread>

#include "common/metrics.h"
#include "core/framework/multimodal/mm_data.h"

namespace xllm {

std::vector<Block> PrefixCache::match(const Slice<int32_t>& token_ids,
                                      const Slice<Block>& existed_shared_blocks,
                                      const MMData& mm_data,
                                      const Slice<XXH3Key>& block_hashes,
                                      size_t* matched_tokens) {
  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  if (n_tokens == 0) {
    if (matched_tokens != nullptr) {
      *matched_tokens = 0;
    }
    return std::vector<Block>();
  }

  const size_t n_blocks = n_tokens / block_size_;
  total_blocks_.fetch_add(n_blocks);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  blocks.insert(
      blocks.end(), existed_shared_blocks.begin(), existed_shared_blocks.end());

  DNodeList node_list;
  const size_t start_block = existed_shared_blocks.size();

  // Look up one block by its chained hash; on hit, record the block and move
  // its LRU node to the front of the working list. Returns false on miss.
  auto match_block = [&](const XXH3Key& token_hash_key) -> bool {
    auto iter = cached_blocks_.find(token_hash_key);
    if (iter == cached_blocks_.end()) {
      return false;
    }
    blocks.emplace_back(iter->second->block);
    lru_lst_.remove_node(iter->second);
    node_list.push_front(iter->second);
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

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  matched_blocks_.fetch_add(blocks.size());

  if (matched_tokens != nullptr) {
    *matched_tokens = blocks.size() * block_size_;
  }

  int64_t int_rate_percent = static_cast<int64_t>(
      static_cast<double>(blocks.size()) * 100.0 / n_blocks);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_rate, int_rate_percent);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_num, blocks.size());

  return blocks;
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           size_t existed_shared_blocks_num,
                           const MMData& mm_data,
                           const Slice<XXH3Key>& block_hashes) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());

  if (n_blocks == 0) {
    return 0;
  }
  CHECK_GE(n_blocks, existed_shared_blocks_num);
  // truncate the token ids and blocks to boundary

  DNodeList node_list;

  // Fill `token_hash_key` with the chained hash of block `block_idx`, reusing
  // the precomputed hash when it covers all blocks, otherwise computing it.
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

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;
    }
  }

  const size_t n_tokens = n_blocks * block_size_;
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return n_tokens;
}

size_t PrefixCache::insert(const std::vector<Block>& blocks) {
  Slice<Block> slice(blocks);
  return insert(slice);
}

size_t PrefixCache::insert(Slice<Block>& blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  DNodeList node_list;
  XXH3Key token_hash_key;

  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i].is_valid()) {
      continue;
    }
    token_hash_key.set(blocks[i].get_immutable_hash_value());

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[i];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;
    }
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return blocks.size() * block_size_;
}

Block PrefixCache::find(const XXH3Key& hash) {
  auto iter = cached_blocks_.find(hash);
  if (iter == cached_blocks_.end()) {
    return Block();
  }
  lru_lst_.move_back(iter->second);
  return iter->second->block;
}

bool PrefixCache::contains(const XXH3Key& hash) const {
  return cached_blocks_.find(hash) != cached_blocks_.end();
}

size_t PrefixCache::evict(size_t n_blocks) {
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  while (evict_count < n_blocks) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    XXH3Key token_hash_key(del_node->block.get_immutable_hash_value());

    cached_blocks_.erase(token_hash_key);

    delete del_node;
    ++evict_count;
    --num_blocks_;
  }

  return evict_count;
}

uint32_t PrefixCache::compute_hash_keys(const Slice<int32_t>& token_ids,
                                        std::vector<Block>& blocks,
                                        const size_t cached_blocks) {
  if (blocks.size() == 0) {
    return 0;
  }
  int32_t block_size = blocks[0].size();
  const size_t n_blocks = (token_ids.size() + block_size - 1) / block_size;
  if (blocks.size() > n_blocks) {
    LOG(ERROR) << "token ids do not cover the allocate block.";
    return 0;
  }
  size_t full_block_size =
      std::min(token_ids.size() / block_size, blocks.size());

  for (size_t i = cached_blocks; i < full_block_size; i++) {
    if (i == 0) {
      xxh3_128bits_hash(nullptr,
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    } else {
      xxh3_128bits_hash(blocks[i - 1].get_mutable_hash_value(),
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    }
  }

  return full_block_size;
}

}  // namespace xllm
