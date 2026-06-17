/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/platform/device_name_utils.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/platform/device.h"

namespace xllm {

std::string DeviceNameUtils::to_device_string(int32_t device_id) {
  return Device::type_str() + ":" + std::to_string(device_id);
}

std::vector<torch::Device> DeviceNameUtils::parse_devices(
    const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto" || device_str.empty()) {
    // use all available devices if any
    const int32_t num_devices = static_cast<int32_t>(Device::device_count());
    if (num_devices == 0) {
      LOG(INFO) << "no devices found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_devices);
    for (int32_t i = 0; i < num_devices; ++i) {
      std::string device_name = to_device_string(i);
      devices.emplace_back(torch::Device(device_name));
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::unordered_set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    std::vector<std::string> parts = absl::StrSplit(device_str, ':');
    CHECK(parts.size() == 2) << "Invalid device string format: " << device_str;
    CHECK(parts[0] == Device::type_str())
        << "Unsupported device type: " << parts[0];

    int32_t device_index = 0;
    CHECK(absl::SimpleAtoi(parts[1], &device_index))
        << "Invalid device index: " << parts[1];

    devices.emplace_back(Device::type_torch(), device_index);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << device_str;
  return devices;
}

}  // namespace xllm
