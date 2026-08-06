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
#include <memory>

namespace xllm::mlu {

// Owns a page-aligned MLU pinned-host allocation. The implementation records
// the allocating context across moves and restores the releasing thread's
// current context after freeing the allocation.
class MLUHostMemoryRegion final {
 public:
  MLUHostMemoryRegion();
  explicit MLUHostMemoryRegion(size_t bytes);
  MLUHostMemoryRegion(const MLUHostMemoryRegion&) = delete;
  MLUHostMemoryRegion& operator=(const MLUHostMemoryRegion&) = delete;
  MLUHostMemoryRegion(MLUHostMemoryRegion&& other) noexcept;
  MLUHostMemoryRegion& operator=(MLUHostMemoryRegion&& other) noexcept;
  ~MLUHostMemoryRegion();

  void* data() const;
  size_t size() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::mlu
