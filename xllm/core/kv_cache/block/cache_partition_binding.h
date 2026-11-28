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
#include <cstdint>

#include "core/kv_cache/block/block.h"
#include "core/kv_cache/layout/sequence_cache_view.h"

namespace xllm {

// Reserved identities for callers that have not migrated from worker slots.
// Model-declared identities are supplied explicitly by the layout builder.
inline const CachePartitionId& legacy_cache_partition_id(BlockType type) {
  static const std::array<CachePartitionId, 8> kIds = {{{"", 0xfffffff0u},
                                                        {"", 0xfffffff1u},
                                                        {"", 0xfffffff2u},
                                                        {"", 0xfffffff3u},
                                                        {"", 0xfffffff4u},
                                                        {"", 0xfffffff5u},
                                                        {"", 0xfffffff6u},
                                                        {"", 0xfffffff7u}}};
  return kIds.at(static_cast<size_t>(type));
}

inline CacheRetention legacy_cache_retention(BlockType type) {
  switch (type) {
    case BlockType::SWA:
      return CacheRetention::WINDOW;
    case BlockType::LINEAR:
      return CacheRetention::RECURRENT;
    case BlockType::EMBEDDING:
      return CacheRetention::SEQUENCE;
    default:
      return CacheRetention::HISTORY;
  }
}

}  // namespace xllm
