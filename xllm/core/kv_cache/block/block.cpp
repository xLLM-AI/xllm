/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/kv_cache/block/block.h"

#include <glog/logging.h>
#include <string.h>

#include <cstdint>
#include <utility>

#include "core/kv_cache/block/block_manager.h"

namespace xllm {
KVCacheBlockLease::KVCacheBlockLease(std::vector<Block> blocks)
    : blocks_(std::move(blocks)) {}

KVCacheBlockLease::~KVCacheBlockLease() { Block::release(std::move(blocks_)); }

void Block::release(std::vector<Block> blocks) {
  size_t begin = 0;
  while (begin < blocks.size()) {
    BlockManager* manager = blocks[begin].manager();
    size_t end = begin + 1;
    while (end < blocks.size() && blocks[end].manager() == manager) {
      ++end;
    }
    if (manager != nullptr) {
      manager->release(std::span<Block>(blocks).subspan(begin, end - begin));
    }
    begin = end;
  }
}

Block::Block(int32_t id, BlockManager* manager)
    : id_(id),
      size_(manager == nullptr ? 0
                               : static_cast<uint32_t>(manager->block_size())),
      pin_block_(false),
      ref_count_(new std::atomic<uint64_t>{/*desired=*/1}),
      manager_(manager) {}

Block::~Block() { dec_ref_count(); }

Block::Block(const Block& other, bool pin_block)
    : id_(other.id_),
      size_(other.size_),
      pin_block_(pin_block || other.pin_block_),
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  inc_ref_count();
}

Block& Block::operator=(const Block& other) {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    size_ = other.size_;
    pin_block_ = other.pin_block_;
    manager_ = other.manager_;
    ref_count_ = other.ref_count_;

    memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

    inc_ref_count();
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : id_(other.id_),
      size_(other.size_),
      pin_block_(other.pin_block_),
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  other.id_ = -1;
  other.size_ = 0;
  other.ref_count_ = nullptr;
  other.manager_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    size_ = other.size_;
    pin_block_ = other.pin_block_;
    manager_ = other.manager_;
    ref_count_ = other.ref_count_;

    memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

    other.id_ = -1;
    other.size_ = 0;
    other.ref_count_ = nullptr;
    other.manager_ = nullptr;
  }

  return *this;
}

void Block::inc_ref_count() {
  if (ref_count_ != nullptr) {
    // Another alias appearing only needs atomicity, not ordering, so relaxed.
    ref_count_->fetch_add(ref_count_delta(), std::memory_order_relaxed);
  }
}

void Block::dec_ref_count() {
  if (ref_count_ == nullptr) {
    return;
  }
  // acq_rel pairs the releasing decrements with the acquiring final one, so the
  // thread that drives the count to zero observes all prior alias writes before
  // it frees.
  const uint64_t prior =
      ref_count_->fetch_sub(ref_count_delta(), std::memory_order_acq_rel);
  if (static_cast<uint32_t>(prior) == 1) {
    delete ref_count_;
    if (manager_ != nullptr) {
      manager_->free(id_);
    }
  }
}

}  // namespace xllm
