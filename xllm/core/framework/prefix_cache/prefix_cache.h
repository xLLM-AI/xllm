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

#pragma once

#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "block_hasher.h"
#include "common/macros.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/block/block.h"
#include "util/hash_util.h"
#include "util/slice.h"
#include "util/threadpool.h"

namespace xllm {

inline size_t round_down(size_t n, size_t multiple) {
  return (n / multiple) * multiple;
}

// LRU-evicted prefix cache keyed by chained per-block hash. Solid-prefix
// match by default; LinearStatePrefixCache overrides match with a
// gap-tolerant walk for SWA / LINEAR leaves.
class PrefixCache {
 public:
  struct Options {
    PROPERTY(int32_t, block_size) = 128;
    PROPERTY(BlockHasherType, hasher_type) = BlockHasherType::TEXT;
    // Picks the concrete cache in create_prefix_cache: LINEAR / SWA build
    // LinearStatePrefixCache (gap-tolerant), everything else this base class.
    // A prefix cache indexes on a single stride (block_size above): KV block
    // size for KV, prefill chunk stride for LINEAR, SWA base block for SWA.
    PROPERTY(BlockType, block_type) = BlockType::KV;
  };

  PrefixCache(const PrefixCache&) = delete;
  PrefixCache(PrefixCache&&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  PrefixCache& operator=(PrefixCache&&) = delete;

  explicit PrefixCache(uint32_t block_size,
                       BlockHasherType hasher_type = BlockHasherType::TEXT)
      : block_size_(block_size), hasher_type_(hasher_type), num_blocks_(0) {}

  virtual ~PrefixCache() {
    exited_.store(true);
    sleep(2);
  };

  // Solid-prefix probe: walks the chain per block position, stops on the
  // first miss. Reach in tokens is `returned.size() * block_size_`.
  // When `block_hashes` covers all matchable blocks it is consumed as-is;
  // otherwise the chain is computed on the fly from `token_ids` / `mm_data`.
  virtual std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {});

  // Insert blocks[existed_shared_blocks_num, n_blocks) into the cache and
  // stamp their chain hash. Assumes a solid prefix (no placeholders): the
  // compute path seeds the chain in O(1) from blocks[existed_shared_blocks_num
  // - 1]'s stamped hash. When `block_hashes` covers all inserted blocks it is
  // consumed as-is; otherwise the chain is computed on the fly. Returns the
  // token span of the walked range. LinearStatePrefixCache overrides this to
  // tolerate SWA slid-out placeholders.
  virtual size_t insert(const Slice<int32_t>& token_ids,
                        std::vector<Block>& blocks,
                        size_t existed_shared_blocks_num = 0,
                        const MMData& mm_data = MMData(),
                        const Slice<XXH3Key>& block_hashes = {});

  // Insert already-hashed blocks (hash stamped by the caller).
  virtual size_t insert(Slice<Block>& blocks);
  virtual size_t insert(const std::vector<Block>& blocks);

  // Point-lookup by chained hash key. `find` hits refresh LRU recency and
  // return the block; `contains` is LRU-neutral. Both return an invalid
  // Block / false on miss.
  Block find(const XXH3Key& hash);
  bool contains(const XXH3Key& hash) const;

  // Evict up to `n_blocks` LRU-oldest entries. Returns the number evicted.
  virtual size_t evict(size_t n_blocks);

  virtual size_t num_blocks() const {
    CHECK(num_blocks_ == cached_blocks_.size()) << "check block num failed";
    return num_blocks_;
  }

  float block_match_rate() {
    if (total_blocks_.load() == 0) {
      return 0;
    } else {
      return static_cast<float>(matched_blocks_.load()) / total_blocks_.load();
    }
  }

  static uint32_t compute_hash_keys(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    const size_t cached_blocks = 0);

 protected:
  struct Node {
    Block block;
    int64_t last_access_time = 0;
    Node* prev = nullptr;
    Node* next = nullptr;
  };

  struct DNodeList {
    DNodeList() {
      lst_front.next = &lst_back;
      lst_back.prev = &lst_front;
    }

    ~DNodeList() {
      Node* node = lst_front.next;
      while (node != &lst_back) {
        Node* next = node->next;
        delete node;
        node = next;
      }
    }

    bool is_empty() { return lst_front.next == &lst_back; }

    Node* remove_node(Node* node) {
      Node* next_node = node->next;
      node->prev->next = next_node;
      next_node->prev = node->prev;
      return next_node;
    }

    bool is_last(Node* node) { return node == &lst_back; }

    void push_front(Node* node) {
      node->next = lst_front.next;
      lst_front.next->prev = node;
      node->prev = &lst_front;
      lst_front.next = node;
    }

    Node* get_first() { return lst_front.next; }

    Node* pop_front() {
      if (lst_front.next == &lst_back) {
        return nullptr;
      }
      Node* node = lst_front.next;
      lst_front.next = node->next;
      node->next->prev = &lst_front;
      return node;
    }

    void push_back(Node* node) {
      node->prev = lst_back.prev;
      node->next = &lst_back;
      lst_back.prev->next = node;
      lst_back.prev = node;
    }

    void move_back(Node* node) {
      remove_node(node);
      push_back(node);
    }

    Node lst_front;
    Node lst_back;
  };

  DNodeList lru_lst_;
  uint32_t block_size_;
  BlockHasherType hasher_type_;
  size_t num_blocks_ = 0;
  std::atomic_bool exited_{false};

  std::unordered_map<XXH3Key, Node*, FixedStringKeyHash, FixedStringKeyEqual>
      cached_blocks_;

  std::atomic<uint64_t> total_blocks_{0}, matched_blocks_{0};
};

}  // namespace xllm
