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

#include "platform/mlu/mlu_batch_memcpy.h"

#include <cn_api.h>
#include <glog/logging.h>

#include <algorithm>
#include <exception>
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

const char* direction_name(bool h2d) { return h2d ? "H2D" : "D2H"; }

}  // namespace

void MLUBatchMemcpy::init(int32_t device_id) {
  if (initialized_) {
    CHECK_EQ(device_id_, device_id)
        << "MLUBatchMemcpy cannot be initialized for another device.";
    return;
  }
  CHECK_GE(device_id, 0) << "MLUBatchMemcpy device id must be non-negative.";
  device_id_ = device_id;
  initialized_ = true;
}

bool MLUBatchMemcpy::copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(src_tensors, dst_tensors, stream, Direction::H2D);
}

bool MLUBatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(src_tensors, dst_tensors, stream, Direction::D2H);
}

bool MLUBatchMemcpy::valid_inputs(const std::vector<torch::Tensor>& src_tensors,
                                  const std::vector<torch::Tensor>& dst_tensors,
                                  const Stream* stream,
                                  Direction direction) const {
  const bool h2d = direction == Direction::H2D;
  const char* operation = direction_name(h2d);
  if (!initialized_) {
    LOG(ERROR) << "MLU batch memcpy " << operation << " is not initialized.";
    return false;
  }
  if (stream == nullptr) {
    LOG(ERROR) << "MLU batch memcpy " << operation << " stream is null.";
    return false;
  }
  if (stream->get_stream()->device_index() != device_id_) {
    LOG(ERROR) << "MLU batch memcpy " << operation
               << " stream device mismatch: expected=" << device_id_
               << ", actual=" << stream->get_stream()->device_index();
    return false;
  }
  if (src_tensors.size() != dst_tensors.size()) {
    LOG(ERROR) << "MLU batch memcpy " << operation
               << " tensor count mismatch: src=" << src_tensors.size()
               << ", dst=" << dst_tensors.size();
    return false;
  }

  for (size_t index = 0; index < src_tensors.size(); ++index) {
    const torch::Tensor& src = src_tensors[index];
    const torch::Tensor& dst = dst_tensors[index];
    if (!src.defined() || !dst.defined()) {
      LOG(ERROR) << "MLU batch memcpy " << operation
                 << " has undefined tensor at index=" << index;
      return false;
    }
    if (!src.is_contiguous() || !dst.is_contiguous()) {
      LOG(ERROR) << "MLU batch memcpy " << operation
                 << " requires contiguous tensors at index=" << index;
      return false;
    }
    if (src.nbytes() != dst.nbytes()) {
      LOG(ERROR) << "MLU batch memcpy " << operation
                 << " byte count mismatch at index=" << index
                 << ", src=" << src.nbytes() << ", dst=" << dst.nbytes();
      return false;
    }
    const torch::Tensor& host = h2d ? src : dst;
    const torch::Tensor& mlu = h2d ? dst : src;
    if (!host.device().is_cpu()) {
      LOG(ERROR) << "MLU batch memcpy " << operation
                 << " host tensor is not CPU at index=" << index;
      return false;
    }
    if (mlu.device().type() != c10::DeviceType::PrivateUse1 ||
        !mlu.device().has_index() || mlu.device().index() != device_id_) {
      LOG(ERROR) << "MLU batch memcpy " << operation
                 << " device tensor mismatch at index=" << index
                 << ", expected_device=" << device_id_
                 << ", actual_device=" << mlu.device();
      return false;
    }
  }
  return true;
}

bool MLUBatchMemcpy::copy(const std::vector<torch::Tensor>& src_tensors,
                          const std::vector<torch::Tensor>& dst_tensors,
                          Stream* stream,
                          Direction direction) {
  try {
    if (!valid_inputs(src_tensors, dst_tensors, stream, direction)) {
      return false;
    }
    if (src_tensors.empty()) {
      return true;
    }

    const size_t count = src_tensors.size();
    std::vector<CNaddr> src_addrs;
    std::vector<CNaddr> dst_addrs;
    std::vector<size_t> byte_counts;
    src_addrs.reserve(count);
    dst_addrs.reserve(count);
    byte_counts.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      const uintptr_t src_ptr =
          reinterpret_cast<uintptr_t>(src_tensors[index].data_ptr());
      const uintptr_t dst_ptr =
          reinterpret_cast<uintptr_t>(dst_tensors[index].data_ptr());
      src_addrs.emplace_back(static_cast<CNaddr>(src_ptr));
      dst_addrs.emplace_back(static_cast<CNaddr>(dst_ptr));
      byte_counts.emplace_back(
          static_cast<size_t>(src_tensors[index].nbytes()));
    }

    CNmemcpyBatchAsyncAttributes attr{};
    attr.srcAccessOrder = CN_MEMCPY_SRC_ACCESS_ORDER_QUEUE;
    attr.flags = CN_MEMCPY_FLAG_DEFAULT;
    size_t attr_index = 0;
    const c10::StreamGuard guard = stream->set_stream_guard();
    const cnrtQueue_t runtime_queue = stream->get_stream()->stream();
    const CNqueue queue = reinterpret_cast<CNqueue>(runtime_queue);
    const bool h2d = direction == Direction::H2D;
    bool submitted = true;
    size_t failed_offset = 0;
    size_t failed_count = 0;
    CNresult submit_result = CN_SUCCESS;
    for (size_t offset = 0; offset < count; offset += kMaxBatchCopyCount) {
      const size_t chunk = std::min(kMaxBatchCopyCount, count - offset);
      submit_result = cnMemcpyBatchAsync(dst_addrs.data() + offset,
                                         src_addrs.data() + offset,
                                         byte_counts.data() + offset,
                                         chunk,
                                         &attr,
                                         &attr_index,
                                         /*numAttrs=*/1,
                                         queue);
      if (submit_result != CN_SUCCESS) {
        submitted = false;
        failed_offset = offset;
        failed_count = chunk;
        break;
      }
    }

    const CNresult sync_result = cnQueueSync(queue);
    if (!submitted) {
      LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
                 << " submission failed: chunk_offset=" << failed_offset
                 << ", chunk_count=" << failed_count
                 << ", result=" << static_cast<int32_t>(submit_result)
                 << ", error=" << cn_error_text(submit_result);
    }
    if (sync_result != CN_SUCCESS) {
      LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
                 << " queue sync failed: chunk_offset=0, chunk_count=" << count
                 << ", result=" << static_cast<int32_t>(sync_result)
                 << ", error=" << cn_error_text(sync_result);
    }
    return submitted && sync_result == CN_SUCCESS;
  } catch (const std::exception& error) {
    LOG(ERROR) << "MLU batch memcpy "
               << direction_name(direction == Direction::H2D)
               << " raised an exception: " << error.what();
  } catch (...) {
    LOG(ERROR) << "MLU batch memcpy "
               << direction_name(direction == Direction::H2D)
               << " raised an unknown exception.";
  }
  return false;
}

}  // namespace xllm::mlu
