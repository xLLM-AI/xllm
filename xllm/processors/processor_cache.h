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

#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "core/framework/multimodal/mm_data_item.h"
#include "core/util/hash_util.h"

namespace xllm {

class ProcessorCache final {
 public:
  explicit ProcessorCache(int64_t max_items);
  ~ProcessorCache() = default;

  std::optional<MMDataItem> lookup(const XXH3Key& key);
  void insert(const XXH3Key& key, const MMDataItem& item);
  void clear();

 private:
  using LruList = std::list<XXH3Key>;

  struct Entry {
    MMDataItem item;
    LruList::iterator lru_it;
  };

  using EntryMap = std::
      unordered_map<XXH3Key, Entry, FixedStringKeyHash, FixedStringKeyEqual>;

  void touch(EntryMap::iterator it);
  void erase(EntryMap::iterator it);
  void evict_if_full();

  std::mutex mutex_;
  int64_t max_items_ = 0;
  LruList lru_keys_;
  EntryMap entries_;
};

}  // namespace xllm
