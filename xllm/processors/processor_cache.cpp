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

#include "processors/processor_cache.h"

#include <glog/logging.h>

#include <iterator>
#include <utility>

namespace xllm {

ProcessorCache::ProcessorCache(int64_t max_items) : max_items_(max_items) {
  CHECK_GE(max_items, 0) << "ProcessorCache max_items must be non-negative";
}

std::optional<MMDataItem> ProcessorCache::lookup(const XXH3Key& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  EntryMap::iterator it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }

  touch(it);
  return it->second.item;
}

void ProcessorCache::insert(const XXH3Key& key, const MMDataItem& item) {
  if (max_items_ == 0 || !item.valid()) {
    return;
  }

  MMDataItem cached_item = item;
  std::lock_guard<std::mutex> lock(mutex_);
  EntryMap::iterator it = entries_.find(key);
  if (it != entries_.end()) {
    touch(it);
    return;
  }

  evict_if_full();
  lru_keys_.push_back(key);
  entries_.emplace(key,
                   Entry{std::move(cached_item), std::prev(lru_keys_.end())});
}

void ProcessorCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  lru_keys_.clear();
}

void ProcessorCache::touch(EntryMap::iterator it) {
  lru_keys_.splice(lru_keys_.end(), lru_keys_, it->second.lru_it);
}

void ProcessorCache::erase(EntryMap::iterator it) {
  lru_keys_.erase(it->second.lru_it);
  entries_.erase(it);
}

void ProcessorCache::evict_if_full() {
  while (!lru_keys_.empty() &&
         static_cast<int64_t>(entries_.size()) >= max_items_) {
    const XXH3Key& evict_key = lru_keys_.front();
    EntryMap::iterator it = entries_.find(evict_key);
    CHECK(it != entries_.end());
    erase(it);
  }
}

}  // namespace xllm
