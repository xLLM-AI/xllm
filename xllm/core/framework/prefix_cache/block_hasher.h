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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

void xxh3_128bits_hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value);

void mm_xxh3_128bits_hash(const std::vector<const uint8_t*>& mm_hash_values,
                          const uint8_t* pre_hash_value,
                          const Slice<int32_t>& token_ids,
                          uint8_t* hash_value);

// Type tag bound to the engine: LLM uses TEXT, VLM uses MM.
enum class BlockHasherType {
  TEXT,
  MM,
};

class BlockHasher {
 public:
  virtual ~BlockHasher() = default;

  virtual void compute(const Slice<int32_t>& token_ids,
                       int32_t start_token_idx,
                       int32_t end_token_idx,
                       const uint8_t* pre_hash_value,
                       XXH3Key& hash_key) = 0;

  // Factory: pick the implementation by engine-bound type.
  // mm_data is consumed only by MM hasher and ignored by TEXT hasher.
  static std::unique_ptr<BlockHasher> create(BlockHasherType type,
                                             const MMData& mm_data,
                                             int32_t start_token_idx = 0);
};

class TextBlockHasher : public BlockHasher {
 public:
  TextBlockHasher() = default;

  void compute(const Slice<int32_t>& token_ids,
               int32_t start_token_idx,
               int32_t end_token_idx,
               const uint8_t* pre_hash_value,
               XXH3Key& hash_key) override;
};

class MMBlockHasher : public BlockHasher {
 public:
  explicit MMBlockHasher(const MMData& mm_data, int32_t start_token_idx);

  void compute(const Slice<int32_t>& token_ids,
               int32_t start_token_idx,
               int32_t end_token_idx,
               const uint8_t* pre_hash_value,
               XXH3Key& hash_key) override;

 private:
  int32_t find_overlapping_mm_idx(int32_t start_token_idx);

  std::vector<const uint8_t*> get_block_mm_hash_values(int32_t start_token_idx,
                                                       int32_t end_token_idx);

  const MMData& mm_data_;
  int32_t next_item_idx_;
};

// Incrementally extend a chained prefix hash shared by both hash domains: KV
// hashes one entry per KV block, linear-state one entry per prefill chunk. The
// only real difference is the digest, so the chaining loop lives here once and
// the per-block digest is delegated to the engine-bound hasher (TEXT vs MM)
// built from |type| and |mm_data|. Hashes already present in |hashes| are kept;
// only blocks from hashes.size() up to |boundary_blocks| are appended, resuming
// the chain from the last present hash. A no-op when nothing new is covered, so
// it is safe to call before every match()/save. |boundary_blocks| *
// |block_size| must be within |token_ids|.
void extend_prefix_hashes(BlockHasherType type,
                          const MMData& mm_data,
                          const Slice<int32_t>& token_ids,
                          size_t block_size,
                          size_t boundary_blocks,
                          std::vector<XXH3Key>& hashes);

}  // namespace xllm
