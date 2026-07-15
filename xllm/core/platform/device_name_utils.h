/* Copyright 2025-2026 The xLLM Authors.
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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace xllm {

class DeviceNameUtils {
 public:
  static std::vector<torch::Device> parse_devices(
      const std::string& device_str);

  template <typename T>
  static std::string to_string(const std::vector<T>& items) {
    std::stringstream ss;
    for (size_t i = 0; i < items.size(); ++i) {
      const auto& item = items[i];
      if (i == 0) {
        ss << item;
      } else {
        ss << "," << item;
      }
    }
    return ss.str();
  }
};

}  // namespace xllm
