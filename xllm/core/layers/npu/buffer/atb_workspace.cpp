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

#include "atb_workspace.h"

#include <acl/acl.h>

#include "atb_buffer.h"
#include "common/device_monitor.h"

namespace xllm {

AtbWorkspace::AtbWorkspace(at::Device device) {
  int32_t device_id = device.index();
  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    buffer_map_[device_id] = std::make_unique<AtbBuffer>(1, device);
  }
}

AtbWorkspace::~AtbWorkspace() = default;

void* AtbWorkspace::get_workspace_buffer(uint64_t bufferSize) {
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);

  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    LOG(FATAL) << " Fail to find device_id in buffer_map_ : " << device_id;
  }
  DeviceMonitor::get_instance().monitor_buffer(device_id, bufferSize);

  return it->second->get_buffer(bufferSize);
}

}  // namespace xllm
