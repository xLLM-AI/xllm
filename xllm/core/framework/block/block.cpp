/* Copyright 2025-2026 The xLLM Authors.
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

#include "block.h"

#include <glog/logging.h>
#include <string.h>

#include <cstdint>

#include "block_manager.h"

namespace xllm {
Block::Block(int32_t id, BlockManager* manager)
    : id_(id), ref_count_(new std::atomic<uint32_t>(1)), manager_(manager) {
  // get the block size from the manager
  size_ = manager_ == nullptr ? 0 : manager_->block_size();
}

Block::~Block() {
  // decrease reference count
  dec_ref_count();
}

// copy constructor
Block::Block(const Block& other)
    : id_(other.id_),
      size_(other.size_),
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  // increase reference count
  inc_ref_count();
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    size_ = other.size_;
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
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, XXH3_128BITS_HASH_VALUE_LEN);

  // reset other without adjusting the reference count
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
    ref_count_->fetch_add(1, std::memory_order_relaxed);
  }
}

void Block::dec_ref_count() {
  if (ref_count_ == nullptr) {
    return;
  }
  // acq_rel pairs the releasing decrements with the acquiring final one, so the
  // thread that drives the count to zero observes all prior alias writes before
  // it frees. fetch_sub returns the previous value; == 1 means we just hit
  // zero.
  if (ref_count_->fetch_sub(1, std::memory_order_acq_rel) == 1) {
    // release the reference count memory
    delete ref_count_;
    // return the block id to the manager
    if (manager_ != nullptr) {
      manager_->free(id_);
    }
  }
}

}  // namespace xllm
