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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/kv_cache/block/block.h"

namespace xllm {

class StaticKVCacheBlockPool;

// Each ordered worker group tracks its acknowledged mapping versions. A
// prepared input remains tentative until every target worker acknowledges it;
// abandoned preparation is rolled back so the next input republishes its
// mappings.
class SharedKVCacheMappingTracker final {
 public:
  void begin();
  void acknowledge();
  void reset();

  void append_rows(std::shared_ptr<StaticKVCacheBlockPool> pool,
                   BlockType type,
                   Slice<Block> blocks,
                   std::vector<int32_t>* rows,
                   std::vector<uint64_t>* lease_epochs = nullptr);

 private:
  struct PendingUpdate {
    size_t type_index;
    int32_t id;
    uint64_t previous_epoch;
  };

  // Builders can prepare independent sequences on several threads. Their
  // deduplication and undo records belong to the same tentative input.
  std::mutex mutex_;
  std::shared_ptr<StaticKVCacheBlockPool> pool_;
  std::array<std::vector<uint64_t>, 2> known_epochs_;
  std::vector<PendingUpdate> pending_;
};

}  // namespace xllm
