/* Copyright 2025 The xLLM Authors. All Rights Reserved.
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
#include "framework/kv_cache/kv_cache_event.h"
#include "util/hash_util.h"
#include "util/slice.h"
#include "util/threadpool.h"

namespace xllm {

inline size_t round_down(size_t n, size_t multiple) {
  return (n / multiple) * multiple;
}

class PrefixCache {
 public:
  struct Options {
    PROPERTY(int32_t, block_size) = 128;
    PROPERTY(bool, enable_cache_upload) = false;
    PROPERTY(BlockHasherType, hasher_type) = BlockHasherType::TEXT;
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

  virtual std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData());

  // insert the token ids and blocks into the prefix tree
  // and set hash key to the corresponding block
  // return the length of new inserted tokens
  virtual size_t insert(const Slice<int32_t>& token_ids,
                        std::vector<Block>& blocks,
                        size_t existed_shared_blocks_num = 0,
                        const MMData& mm_data = MMData());

  // insert the blocks with hash key into the prefix tree
  virtual size_t insert(Slice<Block>& blocks);
  virtual size_t insert(const std::vector<Block>& blocks);

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  virtual size_t evict(size_t n_blocks);

  // get the number of blocks in the prefix cache
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

  virtual KvCacheEvent* get_upload_kvcache_events() { return nullptr; }

  static uint32_t compute_hash_keys(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    const size_t cached_blocks = 0);

 protected:
  virtual size_t insert(const Slice<int32_t>& token_ids,
                        std::vector<Block>& blocks,
                        size_t existed_shared_blocks_num,
                        const MMData& mm_data,
                        std::vector<XXH3Key>* insert_keys);

  size_t insert(Slice<Block>& blocks, std::vector<XXH3Key>* insert_keys);

  size_t evict(size_t n_blocks, std::vector<XXH3Key>* evict_keys);

  struct Node {
    Block block;
    // the last access time of the node, used to evict blocks
    int64_t last_access_time = 0;

    // the previous and next nodes, used to maintain the LRU list
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

    // remove the node from the LRU list, and return next node
    Node* remove_node(Node* node) {
      Node* next_node = node->next;

      node->prev->next = next_node;
      next_node->prev = node->prev;

      return next_node;
    }

    bool is_last(Node* node) { return node == &lst_back; }

    // add a new node to the front of the LRU list
    void push_front(Node* node) {
      node->next = lst_front.next;
      lst_front.next->prev = node;

      node->prev = &lst_front;
      lst_front.next = node;
    }

    Node* get_first() { return lst_front.next; }

    // pop out node to the back of the LRU list
    Node* pop_front() {
      if (lst_front.next == &lst_back) {
        return nullptr;
      }

      Node* node = lst_front.next;

      lst_front.next = node->next;
      node->next->prev = &lst_front;

      return node;
    }

    // add a new node to the back of the LRU list
    void push_back(Node* node) {
      node->prev = lst_back.prev;
      node->next = &lst_back;
      lst_back.prev->next = node;
      lst_back.prev = node;
    }

    // move the node to the back of the LRU list
    void move_back(Node* node) {
      remove_node(node);
      push_back(node);
    }

    // Node lst_front;
    Node lst_front;
    Node lst_back;
  };

  DNodeList lru_lst_;

  // the block size of the memory blocks
  uint32_t block_size_;

  // hasher type used to construct BlockHasher in match/insert.
  BlockHasherType hasher_type_;

  // the total number of blocks in the prefix cache
  size_t num_blocks_ = 0;

  std::atomic_bool exited_{false};

  std::unordered_map<XXH3Key, Node*, FixedStringKeyHash, FixedStringKeyEqual>
      cached_blocks_;

  std::atomic<uint64_t> total_blocks_{0}, matched_blocks_{0};
};

}  // namespace xllm
