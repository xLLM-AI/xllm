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

#include "core/platform/device_name_utils.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/platform/device.h"

namespace xllm {
namespace {

TEST(DeviceNameUtilsTest, ToDeviceStringUsesCompiledDeviceType) {
  const std::string device_string = DeviceNameUtils::to_device_string(3);

  EXPECT_EQ(device_string, Device::type_str() + ":3");
}

TEST(DeviceNameUtilsTest, ParseGeneratedDeviceString) {
  const std::vector<torch::Device> devices =
      DeviceNameUtils::parse_devices(DeviceNameUtils::to_device_string(2));

  ASSERT_EQ(devices.size(), 1);
  EXPECT_EQ(devices[0].type(), Device::type_torch());
  EXPECT_EQ(devices[0].index(), 2);
}

}  // namespace
}  // namespace xllm
