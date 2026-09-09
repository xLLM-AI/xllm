/* Copyright 2025-2026 The xLLM Authors.

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

#include "common/mspti_helper.h"

#include <torch/torch.h>
#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <acl/acl.h>
#include <torch_npu/csrc/libs/init_npu.h>

#include <nlohmann/json.hpp>

namespace xllm {

MstxRange::MstxRange(const char* name) : name_(name) {
  int32_t dev_id = 0;
  aclrtGetDevice(&dev_id);
  stream_ = c10_npu::getCurrentNPUStream(dev_id);
  mstx_id_ = mstxRangeStartA(name, stream_);
}

MstxRange::~MstxRange() {
  aclrtSynchronizeStream(stream_);
  mstxRangeEnd(mstx_id_);
}

#ifdef USE_MSPTI
#define ALIGN_BUFFER(ptr, align) \
  (reinterpret_cast<uint8_t*>(   \
      (reinterpret_cast<uintptr_t>(ptr) + (align) - 1) & ~((align) - 1)))

static size_t ALIGN_SIZE = 64;
msptiSubscriberHandle MsptiMetrics::subscriber_ = nullptr;
uint8_t* MsptiMetrics::pBuffer_ = nullptr;

void MsptiMetrics::register_subscriber() {
  if (subscriber_) {
    return;
  }
  msptiSubscribe(&subscriber_, nullptr, nullptr);
  // regitster functions
  msptiActivityRegisterCallbacks(user_buffer_request, user_buffer_complete);

  msptiActivityEnable(MSPTI_ACTIVITY_KIND_MARKER);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_MEMORY);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_HCCL);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
}

void MsptiMetrics::release_subscriber() {
  msptiActivityDisable(MSPTI_ACTIVITY_KIND_MARKER);
  msptiActivityDisable(MSPTI_ACTIVITY_KIND_MEMORY);
  msptiActivityDisable(MSPTI_ACTIVITY_KIND_HCCL);
  msptiActivityDisable(MSPTI_ACTIVITY_KIND_KERNEL);

  msptiActivityFlushAll(1);
  msptiUnsubscribe(subscriber_);
  subscriber_ = nullptr;
}

void MsptiMetrics::user_buffer_request(uint8_t** buffer,
                                       size_t* size,
                                       size_t* maxNumRecords) {
  constexpr uint32_t SIZE = 5 * 1024 * 1024;
  pBuffer_ = (uint8_t*)malloc(SIZE + ALIGN_SIZE);
  *buffer = ALIGN_BUFFER(pBuffer_, ALIGN_SIZE);
  *size = 5 * 1024 * 1024;
  *maxNumRecords = 0;
}

void MsptiMetrics::user_buffer_complete(uint8_t* buffer,
                                        size_t size,
                                        size_t validSize) {
  if (validSize > 0) {
    msptiActivity* pRecord = nullptr;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, validSize, &pRecord);
      if (status == MSPTI_SUCCESS) {
        switch (pRecord->kind) {
          case MSPTI_ACTIVITY_KIND_MARKER:
            LOG(INFO) << handle_marker_event(pRecord);
            break;
          case MSPTI_ACTIVITY_KIND_MEMORY:
            LOG(INFO) << handle_memory_event(pRecord);
            break;
          case MSPTI_ACTIVITY_KIND_HCCL:
            LOG(INFO) << handle_hccl_event(pRecord);
            break;
          case MSPTI_ACTIVITY_KIND_KERNEL:
            LOG(INFO) << handle_kernel_event(pRecord);
            break;
          default:
            LOG(INFO) << " unknown record kind: " << pRecord->kind;
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    } while (1);
  }
  free(pBuffer_);
}

std::string MsptiMetrics::handle_marker_event(msptiActivity* pRecord) {
  msptiActivityMarker* pM = (msptiActivityMarker*)pRecord;
  nlohmann::json data;
  uint64_t id = pM->id;
  std::string name(pM->name);
  uint64_t flag = pM->flag;
  uint32_t sourceKind = pM->sourceKind;
  uint64_t timestamp = pM->timestamp;
  uint32_t deviceId = pM->objectId.ds.deviceId;
  uint32_t streamId = pM->objectId.ds.streamId;
  data["AscendKind"] = "MARKER";
  data["id"] = id;
  data["name"] = name;
  data["flag"] = flag;
  data["sourceKind"] = sourceKind;
  data["timestamp"] = timestamp;
  data["deviceId"] = deviceId;
  data["streamId"] = streamId;
  return data.dump();
}

std::string MsptiMetrics::handle_memory_event(msptiActivity* pRecord) {
  msptiActivityMemory* pM = (msptiActivityMemory*)pRecord;
  nlohmann::json data;
  uint64_t id = pM->correlationId;
  uint64_t start = pM->start;
  uint64_t end = pM->end;
  uint64_t address = pM->address;
  uint64_t bytes = pM->bytes;
  uint32_t memoryKind = pM->memoryKind;
  uint32_t deviceId = pM->deviceId;
  uint32_t streamId = pM->streamId;
  data["AscendKind"] = "MEMORY";
  data["id"] = id;
  data["start"] = start;
  data["end"] = end;
  data["address"] = address;
  data["memoryKind"] = memoryKind;
  data["bytes"] = bytes;
  data["deviceId"] = deviceId;
  data["streamId"] = streamId;
  return data.dump();
}

std::string MsptiMetrics::handle_hccl_event(msptiActivity* pRecord) {
  msptiActivityHccl* pM = (msptiActivityHccl*)pRecord;
  nlohmann::json data;
  std::string name(pM->name);
  std::string commName(pM->commName);
  uint64_t start = pM->start;
  uint64_t end = pM->end;
  uint64_t bandWidth = pM->bandWidth;
  uint32_t deviceId = pM->ds.deviceId;
  uint32_t streamId = pM->ds.streamId;
  data["AscendKind"] = "HCCL";
  data["name"] = name;
  data["commName"] = commName;
  data["start"] = start;
  data["end"] = end;
  data["bandWidth"] = bandWidth;
  data["deviceId"] = deviceId;
  data["streamId"] = streamId;
  return data.dump();
}

std::string MsptiMetrics::handle_kernel_event(msptiActivity* pRecord) {
  msptiActivityKernel* pM = (msptiActivityKernel*)pRecord;
  nlohmann::json data;
  uint64_t id = pM->correlationId;
  std::string name(pM->name);
  std::string type(pM->type);
  uint64_t start = pM->start;
  uint64_t end = pM->end;
  uint32_t deviceId = pM->ds.deviceId;
  uint32_t streamId = pM->ds.streamId;
  data["AscendKind"] = "KERNEL";
  data["id"] = id;
  data["name"] = name;
  data["type"] = type;
  data["start"] = start;
  data["end"] = end;
  data["deviceId"] = deviceId;
  data["streamId"] = streamId;
  return data.dump();
}
#endif
}  // namespace xllm
