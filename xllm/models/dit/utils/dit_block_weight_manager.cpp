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

#include "models/dit/utils/dit_block_weight_manager.h"

#include <glog/logging.h>

#include <cstring>

#include "core/layers/npu/loader/rolling_weight_buffer.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace dit {

namespace {

constexpr size_t kAlignment = 64;

inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

void BlockWeightLoader::build_from_module(torch::nn::Module& block) {
  free_host_weights();
  params_.clear();
  slot_tensors_.clear();
  rolling_buffer_ = nullptr;
  slot_index_ = -1;

  size_t offset = 0;
  auto named_params = block.named_parameters(/*recurse=*/true);
  for (auto& kv : named_params) {
    auto& param = kv.value();
    ParamInfo info;
    info.name = kv.key();
    info.sizes = param.sizes().vec();
    info.dtype = param.scalar_type();
    info.owner = &block;

    offset = align_up(offset, kAlignment);
    info.host_offset = offset;
    offset += param.nbytes();

    params_.push_back(info);
  }

  storage_size_ = align_up(offset, kAlignment);

  if (storage_size_ == 0) {
    return;
  }

  aclError ret = aclrtMallocHost(&host_pinned_storage_, storage_size_);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "BlockWeightLoader: aclrtMallocHost failed, size=" << storage_size_;

  size_t idx = 0;
  for (auto& kv : block.named_parameters(/*recurse=*/true)) {
    if (idx >= params_.size()) {
      break;
    }
    auto& param = kv.value();
    auto& info = params_[idx];

    void* dst = static_cast<char*>(host_pinned_storage_) +
                static_cast<ptrdiff_t>(info.host_offset);

    auto cpu_tensor = param.to(torch::kCPU).contiguous();
    std::memcpy(dst, cpu_tensor.data_ptr(), cpu_tensor.nbytes());

    ++idx;
  }
}

void BlockWeightLoader::set_rolling_buffer(
    std::shared_ptr<RollingWeightBuffer> buf,
    int32_t slot_index) {
  rolling_buffer_ = std::move(buf);
  slot_index_ = slot_index;
  refresh_slot_views();
}

void BlockWeightLoader::refresh_slot_views() {
  if (params_.empty()) {
    return;
  }

  CHECK(rolling_buffer_ != nullptr)
      << "BlockWeightLoader: rolling_buffer_ is null in refresh_slot_views";
  CHECK_GE(slot_index_, 0)
      << "BlockWeightLoader: invalid slot_index_ in refresh_slot_views";

  void* slot_ptr = rolling_buffer_->get_slot_ptr(slot_index_);

  slot_tensors_.clear();
  slot_tensors_.reserve(params_.size());
  for (const auto& info : params_) {
    void* tensor_ptr =
        static_cast<char*>(slot_ptr) + static_cast<ptrdiff_t>(info.host_offset);
    auto slot_tensor = get_tensor_from_blob(info.sizes, info.dtype, tensor_ptr);
    slot_tensors_.push_back(slot_tensor);
  }

  size_t idx = 0;
  for (auto& kv : params_[0].owner->named_parameters(/*recurse=*/true)) {
    if (idx >= slot_tensors_.size()) {
      break;
    }
    kv.value().set_data(slot_tensors_[idx]);
    ++idx;
  }
}

void BlockWeightLoader::copy_to_device_async(aclrtStream stream) {
  CHECK(host_pinned_storage_ != nullptr)
      << "BlockWeightLoader: host_pinned_storage_ is null";
  CHECK(rolling_buffer_ != nullptr) << "BlockWeightLoader: rolling_buffer_ "
                                       "is null in copy_to_device_async";
  CHECK_GE(slot_index_, 0)
      << "BlockWeightLoader: invalid slot_index_ in copy_to_device_async";

  void* slot_ptr = rolling_buffer_->get_slot_ptr(slot_index_);

  aclError ret = aclrtMemcpyAsync(slot_ptr,
                                  storage_size_,
                                  host_pinned_storage_,
                                  storage_size_,
                                  ACL_MEMCPY_HOST_TO_DEVICE,
                                  stream);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "BlockWeightLoader: aclrtMemcpyAsync H2D failed, size="
      << storage_size_;
}

void BlockWeightLoader::free_host_weights() {
  if (host_pinned_storage_ != nullptr) {
    aclError ret = aclrtFreeHost(host_pinned_storage_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "BlockWeightLoader: aclrtFreeHost failed, ret=" << ret;
    }
    host_pinned_storage_ = nullptr;
  }
  storage_size_ = 0;
}

}  // namespace dit
}  // namespace xllm
