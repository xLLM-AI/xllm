/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/platform/stream.h"

#include <glog/logging.h>

#include <exception>
#include <memory>
#include <ostream>

namespace xllm {

namespace {

c10::Stream to_c10_stream(const PlatformStream& stream) {
  return stream.unwrap();
}

#if defined(USE_NPU)
PlatformStream get_stream_from_pool() {
  return c10_npu::getNPUStreamFromPool();
}
#elif defined(USE_MLU)
PlatformStream get_stream_from_pool() { return torch_mlu::getStreamFromPool(); }
#elif defined(USE_CUDA) || defined(USE_ILU)
PlatformStream get_stream_from_pool() { return c10::cuda::getStreamFromPool(); }
#elif defined(USE_MUSA)
PlatformStream get_stream_from_pool() { return c10::musa::getStreamFromPool(); }
#elif defined(USE_DCU)
PlatformStream get_stream_from_pool() { return c10::hip::getStreamFromPool(); }
#endif

}  // namespace

Stream::Stream(const int32_t timeout)
    : stream_(get_stream_from_pool()), timeout_(timeout) {}

Stream::Stream(PlatformStream stream, const int32_t timeout)
    : stream_(stream), timeout_(timeout) {}

int Stream::synchronize() const {
#if defined(USE_NPU)
  return aclrtSynchronizeStreamWithTimeout(stream_.stream(), timeout_);
#else
  stream_.synchronize();
  return 0;
#endif
}

c10::StreamGuard Stream::set_stream_guard() const {
  return c10::StreamGuard(to_c10_stream(stream_));
}

void Stream::wait_event(const c10::Event& event) {
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
  const c10::Stream& current_c10_stream = stream_;
#else
  c10::Stream current_c10_stream = stream_.unwrap();
#endif
  event.block(current_c10_stream);
}

void Stream::wait_stream(const Stream& other_stream) {
  // get the c10::Stream objects for the current stream and the other stream
  c10::Stream current_c10_stream = to_c10_stream(stream_);
  c10::Stream target_c10_stream = to_c10_stream(other_stream.stream_);

  c10::Event event(current_c10_stream.device_type());
  event.record(target_c10_stream);
  event.block(current_c10_stream);
}

StreamEventPtr Stream::record_event() const {
#if defined(USE_NPU)
  aclrtEvent event = nullptr;
  aclError ret = aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
  if (ret != ACL_SUCCESS) {
    ret = aclrtCreateEvent(&event);
  }
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to create NPU stream event: " << ret;
    return nullptr;
  }

  ret = aclrtRecordEvent(event, stream_.stream());
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to record NPU stream event: " << ret;
    aclrtDestroyEvent(event);
    return nullptr;
  }
  return std::make_shared<StreamEvent>(event);
#else
  try {
    c10::Stream current_c10_stream = to_c10_stream(stream_);
    StreamEventPtr event =
        std::make_shared<StreamEvent>(current_c10_stream.device_type());
    event->c10_event().record(current_c10_stream);
    return event;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to record stream event: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Failed to record stream event: unknown exception";
  }
  return nullptr;
#endif
}

bool Stream::wait_event(const StreamEventPtr& event) const {
  if (event == nullptr) {
    return true;
  }
#if defined(USE_NPU)
  aclError ret = aclrtStreamWaitEvent(stream_.stream(), event->npu_event());
  if (ret == ACL_SUCCESS) {
    return true;
  }
  LOG(ERROR) << "Failed to wait NPU stream event: " << ret
             << "; falling back to event synchronize.";
  ret = aclrtSynchronizeEvent(event->npu_event());
  if (ret == ACL_SUCCESS) {
    return true;
  }
  LOG(ERROR) << "Failed to synchronize NPU stream event: " << ret;
  return false;
#else
  try {
    event->c10_event().block(to_c10_stream(stream_));
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to wait stream event: " << e.what()
               << "; falling back to event synchronize.";
  } catch (...) {
    LOG(ERROR) << "Failed to wait stream event: unknown exception; falling "
                  "back to event synchronize.";
  }

  try {
    event->c10_event().synchronize();
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to synchronize stream event: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Failed to synchronize stream event: unknown exception";
  }
  return false;
#endif
}

std::ostream& operator<<(std::ostream& os, const Stream& stream) {
#if defined(USE_NPU)
  // NPUStream output: device index and stream id
  os << "NPUStream[device=" << stream.stream_.device_index()
     << ", stream_id=" << stream.stream_.id() << "]";
#elif defined(USE_MLU)
  // MLUStream output: device index and stream id
  os << "MLUStream[device=" << stream.stream_.device_index()
     << ", stream_id=" << stream.stream_.id() << "]";
#elif defined(USE_CUDA) || defined(USE_ILU) || defined(USE_DCU)
  // For CUDA, use the existing operator<< from c10::cuda::CUDAStream
  os << stream.stream_;
#else
  os << "UnknownStream";
#endif
  return os;
}

}  // namespace xllm
