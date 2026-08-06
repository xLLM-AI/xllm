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

#include "platform/mlu/mlu_host_memory.h"

#include <cn_api.h>
#include <glog/logging.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <string>

namespace xllm::mlu {
namespace {

std::string cn_error_text(CNresult result) {
  const char* text = nullptr;
  const CNresult text_result = cnGetErrorString(result, &text);
  if (text_result != CN_SUCCESS || text == nullptr) {
    return "unknown CNDrv error";
  }
  return text;
}

size_t page_aligned_bytes(size_t bytes) {
  const int64_t system_page_size = sysconf(_SC_PAGESIZE);
  CHECK_GT(system_page_size, 0) << "Failed to query system page size.";
  const size_t page_size = static_cast<size_t>(system_page_size);
  CHECK_LE(bytes, std::numeric_limits<size_t>::max() - (page_size - 1))
      << "MLU host region byte count overflows page rounding: bytes=" << bytes
      << ", page_size=" << page_size;
  return ((bytes + page_size - 1) / page_size) * page_size;
}

}  // namespace

class MLUHostMemoryRegion::Impl final {
 public:
  explicit Impl(size_t bytes) : size_(page_aligned_bytes(bytes)) {
    CNresult result = cnCtxGetCurrent(&owner_context_);
    CHECK_EQ(result, CN_SUCCESS)
        << "cnCtxGetCurrent before cnHostMemAlloc failed: result="
        << static_cast<int32_t>(result) << ", error=" << cn_error_text(result)
        << ", bytes=" << size_;
    CHECK(owner_context_ != nullptr)
        << "cnHostMemAlloc requires a current MLU context, bytes=" << size_;

    result = cnHostMemAlloc(
        &data_, static_cast<uint64_t>(size_), CN_MEMHOSTALLOC_PORTABLE);
    CHECK_EQ(result, CN_SUCCESS)
        << "cnHostMemAlloc failed: result=" << static_cast<int32_t>(result)
        << ", error=" << cn_error_text(result) << ", bytes=" << size_;

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (reinterpret_cast<uintptr_t>(data_) % page_size != 0) {
      void* unaligned_ptr = data_;
      const CNresult free_result = cnFreeHost(data_);
      data_ = nullptr;
      size_ = 0;
      owner_context_ = nullptr;
      if (free_result != CN_SUCCESS) {
        LOG(ERROR) << "cnFreeHost for unaligned allocation failed: result="
                   << static_cast<int32_t>(free_result)
                   << ", error=" << cn_error_text(free_result);
      }
      LOG(FATAL) << "cnHostMemAlloc returned an unaligned pointer: page_size="
                 << page_size << ", requested_bytes=" << bytes
                 << ", returned_ptr=" << unaligned_ptr;
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  ~Impl() { release(); }

  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void release() {
    if (data_ == nullptr || size_ == 0) {
      data_ = nullptr;
      size_ = 0;
      owner_context_ = nullptr;
      return;
    }

    CNcontext previous_context = nullptr;
    const CNresult get_result = cnCtxGetCurrent(&previous_context);
    if (get_result != CN_SUCCESS) {
      LOG(ERROR) << "cnCtxGetCurrent before cnFreeHost failed: result="
                 << static_cast<int32_t>(get_result)
                 << ", error=" << cn_error_text(get_result);
    }

    const CNresult set_result = cnCtxSetCurrent(owner_context_);
    if (set_result != CN_SUCCESS) {
      LOG(ERROR) << "cnCtxSetCurrent before cnFreeHost failed: result="
                 << static_cast<int32_t>(set_result)
                 << ", error=" << cn_error_text(set_result);
    } else {
      const CNresult free_result = cnFreeHost(data_);
      if (free_result != CN_SUCCESS) {
        LOG(ERROR) << "cnFreeHost failed: result="
                   << static_cast<int32_t>(free_result)
                   << ", error=" << cn_error_text(free_result)
                   << ", bytes=" << size_ << ", ptr=" << data_;
      }
    }

    if (get_result == CN_SUCCESS) {
      const CNresult restore_result = cnCtxSetCurrent(previous_context);
      if (restore_result != CN_SUCCESS) {
        LOG(ERROR) << "cnCtxSetCurrent restore failed: result="
                   << static_cast<int32_t>(restore_result)
                   << ", error=" << cn_error_text(restore_result);
      }
    }

    data_ = nullptr;
    size_ = 0;
    owner_context_ = nullptr;
  }

  void* data_ = nullptr;
  size_t size_ = 0;
  CNcontext owner_context_ = nullptr;
};

MLUHostMemoryRegion::MLUHostMemoryRegion() = default;

MLUHostMemoryRegion::MLUHostMemoryRegion(size_t bytes) {
  if (bytes > 0) {
    impl_ = std::make_unique<Impl>(bytes);
  }
}

MLUHostMemoryRegion::MLUHostMemoryRegion(MLUHostMemoryRegion&& other) noexcept =
    default;

MLUHostMemoryRegion& MLUHostMemoryRegion::operator=(
    MLUHostMemoryRegion&& other) noexcept = default;

MLUHostMemoryRegion::~MLUHostMemoryRegion() = default;

void* MLUHostMemoryRegion::data() const {
  return impl_ == nullptr ? nullptr : impl_->data();
}

size_t MLUHostMemoryRegion::size() const {
  return impl_ == nullptr ? 0 : impl_->size();
}

}  // namespace xllm::mlu
