/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <openssl/sha.h>
#include <string.h>

#include <cstdint>
#include <vector>

#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

class BlockManager;

class Block final {
 public:
  ~Block();

  Block() = default;
  Block(int32_t id, BlockManager* allocator);

  Block(const Block& other);
  Block& operator=(const Block& other);

  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the block id
  constexpr int32_t id() const { return id_; }

  // get the block size
  constexpr uint32_t size() const { return size_; }

  // get the reference count, 0 if the block is invalid after move
  uint32_t ref_count() const { return ref_count_ == nullptr ? 0 : *ref_count_; }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

  // check if the block is valid
  bool is_valid() const { return id_ >= 0 && ref_count_ != nullptr; }

  // owner manager that allocated this block.
  BlockManager* manager() const { return manager_; }

  // NOTE: Below block `hash_value_` is used for prefix cache and
  // for recording the hash value of the current block and previous blocks.
  // hash_value_ = Hash(current_block, Hash(pre_block)).
  const uint8_t* get_immutable_hash_value() const { return hash_value_; }
  uint8_t* get_mutable_hash_value() { return hash_value_; }

  void set_hash_value(const uint8_t* hash_value) {
    memcpy(hash_value_, hash_value, XXH3_128BITS_HASH_VALUE_LEN);
  }

  constexpr uint32_t get_hash_value_len() const {
    return XXH3_128BITS_HASH_VALUE_LEN;
  }

 private:
  // increase reference count
  void inc_ref_count();

  // decrease reference count
  void dec_ref_count();

  // block id
  int32_t id_ = -1;

  // block size
  uint32_t size_ = 0;

  // reference count
  uint32_t* ref_count_ = nullptr;

  // manager that manages this block
  BlockManager* manager_ = nullptr;

  uint8_t hash_value_[XXH3_128BITS_HASH_VALUE_LEN];
};

// equeal operator, mainly used for testing
inline constexpr bool operator==(const Block& lhs, const Block& rhs) {
  return lhs.id() == rhs.id();
}

}  // namespace xllm
