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

void drain_queue_after_failure(CNqueue queue,
                               const char* operation,
                               size_t descriptor_count) {
  const CNresult drain_result = cnQueueSync(queue);
  LOG(ERROR) << "MLU batch memcpy " << operation
             << " failure recovery queue sync: descriptor_count="
             << descriptor_count
             << ", result=" << static_cast<int32_t>(drain_result)
             << ", error=" << cn_error_text(drain_result);
  if (drain_result != CN_SUCCESS) {
    LOG(FATAL) << "Failed to drain MLU batch memcpy queue after submission "
                  "failure: operation="
               << operation << ", descriptor_count=" << descriptor_count
               << ", result=" << static_cast<int32_t>(drain_result)
               << ", error=" << cn_error_text(drain_result);
  }
}

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

bool MLUBatchMemcpy::submit_h2d(const std::vector<torch::Tensor>& src_tensors,
                                const std::vector<torch::Tensor>& dst_tensors,
                                Stream* stream) {
  return copy(src_tensors,
              dst_tensors,
              stream,
              Direction::H2D,
              CompletionMode::SUBMIT_ONLY);
}

bool MLUBatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(src_tensors,
              dst_tensors,
              stream,
              Direction::D2H,
              CompletionMode::SYNCHRONIZE);
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
                          Direction direction,
                          CompletionMode completion_mode) {
  CNqueue queue = nullptr;
  bool queue_has_submitted_work = false;
  const bool h2d = direction == Direction::H2D;
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
    queue = reinterpret_cast<CNqueue>(runtime_queue);
    for (size_t offset = 0; offset < count; offset += kMaxBatchCopyCount) {
      const size_t chunk = std::min(kMaxBatchCopyCount, count - offset);
      const CNresult submit_result =
          cnMemcpyBatchAsync(dst_addrs.data() + offset,
                             src_addrs.data() + offset,
                             byte_counts.data() + offset,
                             chunk,
                             &attr,
                             &attr_index,
                             /*numAttrs=*/1,
                             queue);
      if (submit_result != CN_SUCCESS) {
        LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
                   << " submission failed: chunk_offset=" << offset
                   << ", chunk_count=" << chunk
                   << ", result=" << static_cast<int32_t>(submit_result)
                   << ", error=" << cn_error_text(submit_result);
        drain_queue_after_failure(queue, direction_name(h2d), count);
        return false;
      }
      queue_has_submitted_work = true;
    }

    if (completion_mode == CompletionMode::SUBMIT_ONLY) {
      return true;
    }

    const CNresult sync_result = cnQueueSync(queue);
    if (sync_result != CN_SUCCESS) {
      LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
                 << " queue sync failed: chunk_offset=0, chunk_count=" << count
                 << ", result=" << static_cast<int32_t>(sync_result)
                 << ", error=" << cn_error_text(sync_result);
    }
    return sync_result == CN_SUCCESS;
  } catch (const std::exception& error) {
    LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
               << " raised an exception: " << error.what();
  } catch (...) {
    LOG(ERROR) << "MLU batch memcpy " << direction_name(h2d)
               << " raised an unknown exception.";
  }
  if (queue_has_submitted_work) {
    drain_queue_after_failure(queue, direction_name(h2d), src_tensors.size());
  }
  return false;
}

}  // namespace xllm::mlu
