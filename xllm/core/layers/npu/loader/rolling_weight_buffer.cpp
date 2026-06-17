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

#include "rolling_weight_buffer.h"

#include <acl/acl_rt.h>
#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/xtensor/xtensor_allocator.h"

namespace xllm {
namespace layer {

RollingWeightBuffer::RollingWeightBuffer(int32_t num_slots,
                                         size_t storage_size,
                                         const std::string& model_id)
    : num_slots_(num_slots),
      storage_size_(storage_size),
      model_id_(model_id),
      use_xtensor_(::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
  CHECK_GT(num_slots_, 0) << "num_slots must be > 0";
  CHECK_GT(storage_size_, 0u) << "storage_size must be > 0";

  refresh_address();
}

RollingWeightBuffer::~RollingWeightBuffer() {
  if (base_ptr_ != nullptr && !use_xtensor_) {
    // Only free if we allocated via aclrtMalloc; XTensor manages its own
    // memory.
    auto ret = aclrtFree(base_ptr_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "aclrtFree failed for RollingWeightBuffer, ret=" << ret;
    }
    base_ptr_ = nullptr;
  }
}

void* RollingWeightBuffer::get_slot_ptr(int32_t layer_index) const {
  CHECK(base_ptr_ != nullptr) << "RollingWeightBuffer not allocated";
  int32_t slot = layer_index % num_slots_;
  return static_cast<char*>(base_ptr_) +
         static_cast<size_t>(slot) * storage_size_;
}

void RollingWeightBuffer::refresh_address() {
  size_t total = static_cast<size_t>(num_slots_) * storage_size_;

  if (use_xtensor_) {
    void* new_base_ptr = nullptr;
    auto& allocator = XTensorAllocator::get_instance();
    bool ok = allocator.allocate_weight(model_id_, new_base_ptr, total);
    CHECK(ok)
        << "XTensorAllocator::allocate_weight failed for RollingWeightBuffer"
        << ", total=" << total;
    base_ptr_ = new_base_ptr;
    LOG(INFO) << "RollingWeightBuffer: refreshed " << total
              << " bytes via XTensor (" << num_slots_ << " slots x "
              << storage_size_ << " bytes/slot), base_ptr=" << base_ptr_;
    return;
  }

  // Non-XTensor mode: allocate once via aclrtMalloc and reuse.
  if (base_ptr_ == nullptr) {
    auto ret = aclrtMalloc(&base_ptr_, total, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_EQ(ret, ACL_SUCCESS)
        << "aclrtMalloc failed for RollingWeightBuffer, total=" << total
        << ", ret=" << ret;
    LOG(INFO) << "RollingWeightBuffer: allocated " << total
              << " bytes via aclrtMalloc (" << num_slots_ << " slots x "
              << storage_size_ << " bytes/slot)";
  }
}

}  // namespace layer
}  // namespace xllm
