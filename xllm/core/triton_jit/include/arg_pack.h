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

#include <cstddef>
#include <cstring>
#include <vector>

namespace xllm::triton_jit {

class ArgPack final {
 public:
  ArgPack() { storage_.resize(kInitialBytes); }

  template <typename T>
  void push(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ArgPack: trivially copyable only");
    size_t align = alignof(T);
    size_t off = (cursor_ + align - 1) & ~(align - 1);
    if (off + sizeof(T) > storage_.size()) {
      grow_and_repack(off + sizeof(T));
      // off is unchanged: cursor_ was already aligned, and grow preserves the
      // packed layout, so the existing entries' offsets stay valid.
    }
    std::byte* p = storage_.data() + off;
    std::memcpy(p, &v, sizeof(T));
    ptrs_.push_back(p);
    cursor_ = off + sizeof(T);
  }

  void* const* data() const { return ptrs_.data(); }
  size_t size() const { return ptrs_.size(); }

 private:
  static constexpr size_t kInitialBytes = 4096;

  void grow_and_repack(size_t need) {
    const std::byte* old_base = storage_.data();
    size_t new_cap = storage_.size();
    while (new_cap < need) {
      new_cap *= 2;
    }
    storage_.resize(new_cap);
    const std::byte* new_base = storage_.data();
    if (new_base != old_base) {
      std::ptrdiff_t shift = new_base - old_base;
      for (void*& p : ptrs_) {
        p = static_cast<std::byte*>(p) + shift;
      }
    }
  }

  std::vector<std::byte> storage_;
  std::vector<void*> ptrs_;
  size_t cursor_ = 0;
};

}  // namespace xllm::triton_jit
