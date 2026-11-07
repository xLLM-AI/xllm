/* Copyright 2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "util/hash_util.h"

namespace xllm {

// Non-owning LRU snapshot. Inspecting a cache must not pin its physical blocks.
struct PrefixCacheEntry {
  XXH3Key hash;
  std::optional<XXH3Key> parent;
  std::optional<XXH3Key> dependency;
  int32_t block_id = -1;
  size_t block_index = 0;
  int64_t last_access_time = 0;
  bool chain_known = false;
  bool evictable = false;
};

}  // namespace xllm
