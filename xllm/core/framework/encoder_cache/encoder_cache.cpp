/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/encoder_cache/encoder_cache.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {

EncoderCache::EncoderCache(int64_t max_size) : max_size_(max_size) {
  CHECK_GE(max_size, 0) << "EncoderCache max_size must be non-negative";
}

std::optional<torch::Tensor> EncoderCache::lookup(const XXH3Key& key) {
  EntryMap::iterator it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }

  touch(it);
  return it->second.tensor;
}

void EncoderCache::insert(const XXH3Key& key, torch::Tensor embedding) {
  if (!embedding.defined() || embedding.numel() == 0) {
    return;
  }
  EntryMap::iterator it = entries_.find(key);
  if (it != entries_.end()) {
    touch(it);
    return;
  }

  torch::Tensor owned = embedding.clone();
  const int64_t size = static_cast<int64_t>(owned.nbytes());
  if (size > max_size_) {
    return;
  }

  evict_until_fit(size);

  lru_keys_.push_back(key);
  entries_.emplace(key,
                   Entry{std::move(owned), size, std::prev(lru_keys_.end())});
  current_size_ += size;
}

void EncoderCache::clear() {
  entries_.clear();
  lru_keys_.clear();
  current_size_ = 0;
}

void EncoderCache::touch(EntryMap::iterator it) {
  lru_keys_.splice(lru_keys_.end(), lru_keys_, it->second.lru_it);
}

void EncoderCache::erase(EntryMap::iterator it) {
  current_size_ -= it->second.size;
  lru_keys_.erase(it->second.lru_it);
  entries_.erase(it);
}

void EncoderCache::evict_until_fit(int64_t size) {
  while (!lru_keys_.empty() && current_size_ + size > max_size_) {
    const XXH3Key& evict_key = lru_keys_.front();
    EntryMap::iterator it = entries_.find(evict_key);
    CHECK(it != entries_.end());
    erase(it);
  }
}

}  // namespace xllm
