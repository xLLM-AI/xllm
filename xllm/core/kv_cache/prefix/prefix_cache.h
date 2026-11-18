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

#pragma once

#include <glog/logging.h>

#include <boost/intrusive/list.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/kv_cache/block/block.h"
#include "core/kv_cache/events/kv_cache_event_journal.h"
#include "core/kv_cache/prefix/block_hasher.h"
#include "core/kv_cache/prefix/prefix_cache_entry.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

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
      : block_size_(block_size), hasher_type_(hasher_type) {}

  virtual ~PrefixCache() {
    if (event_journal_) {
      for (const auto& [hash, node] : cached_blocks_) {
        event_journal_->remove(event_scope_,
                               KVCacheEventJournal::encode_hash(hash.data));
      }
    }
  };

  void set_event_sink(std::shared_ptr<KVCacheEventJournal> journal,
                      KVCacheEventScope scope) {
    CHECK(cached_blocks_.empty());
    CHECK(hasher_type_ == BlockHasherType::TEXT);
    event_journal_ = std::move(journal);
    event_scope_ = std::move(scope);
  }

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
  size_t insert(Slice<Block> blocks);

  bool contains(const XXH3Key& hash) const;

  // Physical ownership is independent of a handle's hash: transfer aliases
  // may have been copied before publication stamped the sequence's handles.
  bool owns(const Block& block) const;

  // Evict up to `n_blocks` LRU-oldest entries. Returns the number evicted.
  virtual size_t evict(size_t n_blocks);

  std::vector<PrefixCacheEntry> entries() const;
  bool evict_block(const XXH3Key& hash);
  void set_dependency(const XXH3Key& hash, XXH3Key dependency);

  size_t num_blocks() const { return cached_blocks_.size(); }

 protected:
  std::shared_ptr<KVCacheEventJournal> event_journal_;
  KVCacheEventScope event_scope_;
  using NodeHook = boost::intrusive::list_member_hook<
      boost::intrusive::link_mode<boost::intrusive::normal_link>>;

  class Node final {
   public:
    Node(const Block& block, int64_t now)
        : block_(block), last_access_time_(now) {}

    Block block_;
    std::optional<XXH3Key> parent_;
    std::optional<XXH3Key> dependency_;
    bool chain_known_ = false;
    size_t block_index_ = 0;
    int64_t last_access_time_;
    NodeHook lru_hook_;
  };

  using DNodeList = boost::intrusive::list<
      Node,
      boost::intrusive::member_hook<Node, NodeHook, &Node::lru_hook_>,
      boost::intrusive::constant_time_size<false>>;

  // Keep touched blocks in reverse order for suffix-first eviction.
  DNodeList::iterator touch_node(Node& node,
                                 DNodeList::iterator before,
                                 int64_t now) {
    node.last_access_time_ = now;
    auto pos = lru_lst_.iterator_to(node);
    lru_lst_.splice(before, lru_lst_, pos);
    return pos;
  }

  bool cache_node(DNodeList::iterator& before,
                  const XXH3Key& hash,
                  const Block& block,
                  int64_t now) {
    auto [it, inserted] = cached_blocks_.try_emplace(hash, block, now);
    auto& node = it->second;
    if (inserted) {
      cached_block_ids_.insert(node.block_.id());
      before = lru_lst_.insert(before, node);
    } else {
      before = touch_node(node, before, now);
    }
    return inserted;
  }

  uint32_t block_size_;
  BlockHasherType hasher_type_;
  std::unordered_map<XXH3Key, Node, FixedStringKeyHash, FixedStringKeyEqual>
      cached_blocks_;
  DNodeList lru_lst_;
  std::unordered_set<int32_t> cached_block_ids_;

 private:
  DNodeList::iterator evict_node(decltype(cached_blocks_)::iterator it);
};

}  // namespace xllm
