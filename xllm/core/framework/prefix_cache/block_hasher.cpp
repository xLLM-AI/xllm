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

#include "block_hasher.h"

#include <string.h>
#include <xxHash/xxhash.h>

#include "core/framework/config/kv_cache_config.h"

namespace xllm {

void xxh3_128bits_hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    XXH128_hash_t xxh3_128bits_hash_value = XXH3_128bits_withSeed(
        reinterpret_cast<const void*>(token_ids.data()),
        sizeof(int32_t) * token_ids.size(),
        ::xllm::KVCacheConfig::get_instance().xxh3_128bits_seed());
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
  } else {
    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + XXH3_128BITS_HASH_VALUE_LEN;
    uint8_t* key = new uint8_t[data_len];
    memcpy(key, pre_hash_value, XXH3_128BITS_HASH_VALUE_LEN);
    memcpy(key + XXH3_128BITS_HASH_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    XXH128_hash_t xxh3_128bits_hash_value = XXH3_128bits_withSeed(
        reinterpret_cast<const void*>(key),
        data_len,
        ::xllm::KVCacheConfig::get_instance().xxh3_128bits_seed());
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
    delete[] key;
  }
}

void mm_xxh3_128bits_hash(const std::vector<const uint8_t*>& mm_hash_values,
                          const uint8_t* pre_hash_value,
                          const Slice<int32_t>& token_ids,
                          uint8_t* hash_value) {
  xxh3_128bits_hash(pre_hash_value, token_ids, hash_value);
  if (mm_hash_values.empty()) {
    return;
  }

  const size_t data_len =
      XXH3_128BITS_HASH_VALUE_LEN * (mm_hash_values.size() + 1);
  std::vector<uint8_t> key_buffer;
  key_buffer.reserve(data_len);
  key_buffer.insert(
      key_buffer.end(), hash_value, hash_value + XXH3_128BITS_HASH_VALUE_LEN);
  for (const uint8_t* mm_hash_value : mm_hash_values) {
    key_buffer.insert(key_buffer.end(),
                      mm_hash_value,
                      mm_hash_value + XXH3_128BITS_HASH_VALUE_LEN);
  }

  XXH128_hash_t mm_hash =
      XXH3_128bits_withSeed(reinterpret_cast<const void*>(key_buffer.data()),
                            key_buffer.size(),
                            KVCacheConfig::get_instance().xxh3_128bits_seed());
  std::memcpy(hash_value, &mm_hash, sizeof(mm_hash));
}

void extend_prefix_hashes(BlockHasherType type,
                          const MMData& mm_data,
                          const Slice<int32_t>& token_ids,
                          size_t block_size,
                          size_t boundary_blocks,
                          std::vector<XXH3Key>& hashes) {
  if (block_size == 0 || boundary_blocks <= hashes.size()) {
    return;
  }
  if (boundary_blocks * block_size > token_ids.size()) {
    return;
  }
  const size_t start_block = hashes.size();
  std::unique_ptr<BlockHasher> hasher = BlockHasher::create(
      type, mm_data, static_cast<int32_t>(start_block * block_size));
  hashes.reserve(boundary_blocks);
  // Keep the parent hash in a stable local rather than a pointer into `hashes`:
  // the emplace_back below may reallocate the vector and invalidate any
  // interior pointer. A null pointer (not a zero hash) marks the first block,
  // since an all-zero hash is itself a valid digest.
  XXH3Key previous_hash{};
  const uint8_t* previous_hash_ptr = nullptr;
  if (start_block > 0) {
    previous_hash = hashes.back();
    previous_hash_ptr = previous_hash.data;
  }
  for (size_t block_idx = start_block; block_idx < boundary_blocks;
       ++block_idx) {
    XXH3Key key;
    hasher->compute(token_ids,
                    static_cast<int32_t>(block_idx * block_size),
                    static_cast<int32_t>((block_idx + 1) * block_size),
                    previous_hash_ptr,
                    key);
    hashes.emplace_back(key);
    previous_hash = key;
    previous_hash_ptr = previous_hash.data;
  }
}

std::unique_ptr<BlockHasher> BlockHasher::create(BlockHasherType type,
                                                 const MMData& mm_data,
                                                 int32_t start_token_idx) {
  if (type == BlockHasherType::MM) {
    return std::make_unique<MMBlockHasher>(mm_data, start_token_idx);
  }
  return std::make_unique<TextBlockHasher>();
}

void TextBlockHasher::compute(const Slice<int32_t>& token_ids,
                              int32_t start_token_idx,
                              int32_t end_token_idx,
                              const uint8_t* pre_hash_value,
                              XXH3Key& hash_key) {
  const Slice<int32_t> block_token_ids =
      token_ids.slice(start_token_idx, end_token_idx);
  xxh3_128bits_hash(pre_hash_value, block_token_ids, hash_key.data);
}

MMBlockHasher::MMBlockHasher(const MMData& mm_data, int32_t start_token_idx)
    : mm_data_(mm_data),
      next_item_idx_(find_overlapping_mm_idx(start_token_idx)) {}

void MMBlockHasher::compute(const Slice<int32_t>& token_ids,
                            int32_t start_token_idx,
                            int32_t end_token_idx,
                            const uint8_t* pre_hash_value,
                            XXH3Key& hash_key) {
  const Slice<int32_t> block_token_ids =
      token_ids.slice(start_token_idx, end_token_idx);
  std::vector<const uint8_t*> mm_hash_values =
      get_block_mm_hash_values(start_token_idx, end_token_idx);
  mm_xxh3_128bits_hash(
      mm_hash_values, pre_hash_value, block_token_ids, hash_key.data);
}

int32_t MMBlockHasher::find_overlapping_mm_idx(int32_t start_token_idx) {
  const auto& mm_items = mm_data_.items<MMItemVec>();
  const int32_t num_mm_items = mm_data_.size();
  int32_t next_item_idx = 0;
  while (next_item_idx < num_mm_items) {
    const auto& pos = mm_items[next_item_idx].state().token_pos();
    const int32_t item_end_token_idx = pos.offset + pos.length;
    if (start_token_idx < item_end_token_idx) {
      break;
    }
    ++next_item_idx;
  }
  return next_item_idx;
}

std::vector<const uint8_t*> MMBlockHasher::get_block_mm_hash_values(
    int32_t start_token_idx,
    int32_t end_token_idx) {
  const auto& mm_items = mm_data_.items<MMItemVec>();
  std::vector<const uint8_t*> mm_hash_values;
  const int32_t num_mm_items = mm_data_.size();
  while (next_item_idx_ < num_mm_items) {
    const auto& pos = mm_items[next_item_idx_].state().token_pos();
    const int32_t item_start_token_idx = pos.offset;
    const int32_t item_end_token_idx = item_start_token_idx + pos.length;
    if (end_token_idx <= item_start_token_idx) {
      break;
    }
    if (start_token_idx >= item_end_token_idx) {
      ++next_item_idx_;
      continue;
    }
    const auto& schedule_data =
        mm_items[next_item_idx_].state().schedule_data();
    mm_hash_values.push_back(schedule_data.key.data);
    if (item_end_token_idx > end_token_idx) {
      break;
    }
    ++next_item_idx_;
  }
  return mm_hash_values;
}

}  // namespace xllm
