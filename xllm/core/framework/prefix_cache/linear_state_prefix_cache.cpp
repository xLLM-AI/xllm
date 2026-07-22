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

#include "linear_state_prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <memory>

#include "common/metrics.h"

namespace xllm {

std::vector<Block> LinearStatePrefixCache::match(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  if (n_tokens == 0) {
    return {};
  }
  const size_t n_blocks = n_tokens / block_size_;
  total_blocks_.fetch_add(n_blocks);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  blocks.insert(
      blocks.end(), existed_shared_blocks.begin(), existed_shared_blocks.end());

  DNodeList node_list;
  const size_t start_block = existed_shared_blocks.size();

  // Hit: emplace + LRU-touch. Miss: emplace invalid placeholder.
  auto probe = [&](const XXH3Key& key) -> bool {
    auto iter = cached_blocks_.find(key);
    if (iter == cached_blocks_.end()) {
      blocks.emplace_back();
      return false;
    }
    blocks.emplace_back(iter->second->block);
    lru_lst_.remove_node(iter->second);
    node_list.push_front(iter->second);
    return true;
  };

  size_t last_hit_idx = 0;
  bool has_any_hit = false;
  if (block_hashes.size() >= n_blocks) {
    for (size_t b = start_block; b < n_blocks; ++b) {
      if (probe(block_hashes[b])) {
        last_hit_idx = b;
        has_any_hit = true;
      }
    }
  } else {
    // Fallback: compute chain on the fly, keeping it advancing across misses.
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
      if (probe(token_hash_key)) {
        last_hit_idx = b;
        has_any_hit = true;
      }
    }
  }

  // Trim trailing placeholders; end at deepest hit.
  const size_t keep =
      has_any_hit ? (last_hit_idx + 1) : existed_shared_blocks.size();
  blocks.resize(keep);

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  size_t valid_hits = 0;
  for (const auto& b : blocks) {
    if (b.is_valid()) {
      ++valid_hits;
    }
  }
  matched_blocks_.fetch_add(valid_hits);

  const int64_t int_rate_percent =
      static_cast<int64_t>(static_cast<double>(valid_hits) * 100.0 / n_blocks);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_rate, int_rate_percent);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_num, valid_hits);

  return blocks;
}

size_t LinearStatePrefixCache::insert(const Slice<int32_t>& token_ids,
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

  DNodeList node_list;

  // Fill `token_hash_key` with the chained hash of block `block_idx`, reusing
  // the precomputed hash when it covers all blocks, otherwise computing it.
  // The block vector may contain invalid placeholders (SWA slid-out slots), so
  // the compute path walks the chain from token 0 rather than seeding from
  // blocks[existed_shared_blocks_num - 1] whose stamped hash is undefined when
  // it is a placeholder. The pre-existed walk hashes tokens in order to catch
  // the parent-hash cursor up.
  const bool use_precomputed = block_hashes.size() >= n_blocks;
  XXH3Key token_hash_key{};
  std::unique_ptr<BlockHasher> hasher;
  if (!use_precomputed) {
    hasher = BlockHasher::create(hasher_type_, mm_data, /*start=*/0);
    const uint8_t* prev_hash = nullptr;
    for (size_t b = 0; b < existed_shared_blocks_num; ++b) {
      const size_t i = b * block_size_;
      hasher->compute(token_ids, i, i + block_size_, prev_hash, token_hash_key);
      prev_hash = token_hash_key.data;
    }
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
    // Skip invalid placeholders (e.g. SWA slid-out slots): the chain-hash
    // cursor has already advanced above, but there is no physical block to
    // stamp or emplace here.
    if (!blocks[block_idx].is_valid()) {
      continue;
    }
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

}  // namespace xllm
