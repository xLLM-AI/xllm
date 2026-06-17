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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

class EncoderCache {
 public:
  explicit EncoderCache(int64_t max_size);
  ~EncoderCache() = default;

  std::optional<torch::Tensor> lookup(const XXH3Key& key);
  void insert(const XXH3Key& key, torch::Tensor embedding);
  void clear();

 private:
  using LruList = std::list<XXH3Key>;

  struct Entry {
    torch::Tensor tensor;
    int64_t size = 0;
    LruList::iterator lru_it;
  };

  using EntryMap = std::
      unordered_map<XXH3Key, Entry, FixedStringKeyHash, FixedStringKeyEqual>;

  void touch(EntryMap::iterator it);
  void erase(EntryMap::iterator it);
  void evict_until_fit(int64_t size);

  int64_t max_size_ = 0;
  int64_t current_size_ = 0;
  LruList lru_keys_;
  EntryMap entries_;
};

}  // namespace xllm
