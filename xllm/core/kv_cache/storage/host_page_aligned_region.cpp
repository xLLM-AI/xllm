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

#include "core/kv_cache/storage/host_page_aligned_region.h"

#include <glog/logging.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#include "core/kv_cache/storage/tensor_byte_size.h"
#if defined(USE_MLU)
#include "core/platform/mlu/mlu_host_memory.h"
#endif
#if defined(USE_NPU)
#include "acl/acl_rt.h"

extern "C" aclError aclrtHostRegister(void* ptr,
                                      uint64_t size,
                                      aclrtHostRegisterType type,
                                      void** dev_ptr);
extern "C" aclError aclrtHostUnregister(void* ptr);
#endif

namespace xllm {

namespace {

#if !defined(USE_MLU)
void release_host_region(void*& base_ptr, size_t& total_bytes) {
  if (base_ptr == nullptr || total_bytes == 0) {
    base_ptr = nullptr;
    total_bytes = 0;
    return;
  }
#if defined(USE_NPU)
  aclrtHostUnregister(base_ptr);
#endif
  munlock(base_ptr, total_bytes);
  munmap(base_ptr, total_bytes);
  base_ptr = nullptr;
  total_bytes = 0;
}
#endif

}  // namespace

HostPageAlignedRegion::HostPageAlignedRegion() = default;

HostPageAlignedRegion::HostPageAlignedRegion(size_t bytes) {
  if (bytes == 0) {
    return;
  }
#if defined(USE_MLU)
  mlu_region_ = std::make_unique<mlu::MLUHostMemoryRegion>(bytes);
  base_ptr = mlu_region_->data();
  total_bytes = mlu_region_->size();
#else
  const int64_t system_page_size = sysconf(_SC_PAGESIZE);
  CHECK_GT(system_page_size, 0) << "Failed to query system page size.";
  const size_t page_size = static_cast<size_t>(system_page_size);
  CHECK_LE(bytes, std::numeric_limits<size_t>::max() - (page_size - 1))
      << "Host region byte count overflows page rounding: bytes=" << bytes
      << ", page_size=" << page_size;
  total_bytes = ((bytes + page_size - 1) / page_size) * page_size;

  base_ptr = mmap(nullptr,
                  total_bytes,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS,
                  -1,
                  0);
  CHECK(base_ptr != MAP_FAILED)
      << "Failed to mmap host memory, size=" << total_bytes;
  if (mlock(base_ptr, total_bytes) != 0) {
    const int32_t err = errno;
    munmap(base_ptr, total_bytes);
    base_ptr = nullptr;
    total_bytes = 0;
    LOG(FATAL) << "mlock failed, errno=" << err << " (" << strerror(err)
               << "). Run: ulimit -l unlimited";
  }

#if defined(USE_NPU)
  void* mapped_ptr = nullptr;
  const aclError ret = aclrtHostRegister(
      base_ptr, total_bytes, ACL_HOST_REGISTER_MAPPED, &mapped_ptr);
  if (ret != ACL_SUCCESS) {
    munlock(base_ptr, total_bytes);
    munmap(base_ptr, total_bytes);
    base_ptr = nullptr;
    total_bytes = 0;
  }
  CHECK_EQ(ret, ACL_SUCCESS) << "aclrtHostRegister fail: " << ret;
#endif
#endif
}

HostPageAlignedRegion::HostPageAlignedRegion(
    HostPageAlignedRegion&& other) noexcept
    : base_ptr(other.base_ptr),
      total_bytes(other.total_bytes)
#if defined(USE_MLU)
      ,
      mlu_region_(std::move(other.mlu_region_))
#endif
{
  other.base_ptr = nullptr;
  other.total_bytes = 0;
}

HostPageAlignedRegion& HostPageAlignedRegion::operator=(
    HostPageAlignedRegion&& other) noexcept {
  if (this == &other) {
    return *this;
  }
#if !defined(USE_MLU)
  release_host_region(base_ptr, total_bytes);
#else
  mlu_region_ = std::move(other.mlu_region_);
#endif
  base_ptr = other.base_ptr;
  total_bytes = other.total_bytes;
  other.base_ptr = nullptr;
  other.total_bytes = 0;
  return *this;
}

HostPageAlignedRegion::~HostPageAlignedRegion() {
#if !defined(USE_MLU)
  release_host_region(base_ptr, total_bytes);
#endif
}

void create_host_page_aligned_tensor(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     torch::Tensor* tensor,
                                     HostPageAlignedRegion* region) {
  CHECK(tensor != nullptr) << "tensor must not be null.";
  CHECK(region != nullptr) << "region must not be null.";
  const size_t tensor_bytes = detail::get_tensor_nbytes(dims, dtype);
  CHECK_GT(tensor_bytes, static_cast<size_t>(0))
      << "host cache tensor bytes must be positive.";

  *region = HostPageAlignedRegion(tensor_bytes);
  std::memset(region->base_ptr, 0, region->total_bytes);
  *tensor = torch::from_blob(
      region->base_ptr,
      dims,
      torch::TensorOptions().dtype(dtype).device(torch::Device(torch::kCPU)));
  CHECK(tensor->is_contiguous()) << "host cache tensor must be contiguous.";
}

}  // namespace xllm
