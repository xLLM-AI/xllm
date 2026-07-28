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

#include "core/platform/shared_vmm_allocator.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "core/platform/device.h"
#include "core/platform/platform.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#include <torch_npu/torch_npu.h>

#include "tests/npu_test_environment.h"
#endif

namespace {

class SharedVMMAllocatorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
#if defined(USE_NPU)
    xllm::testing::init_npu_test_runtime();
#endif
    google::InitGoogleLogging("platform_vmm_test");
    google::SetStderrLogging(google::INFO);

#if defined(USE_NPU)
    int ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
#endif
  }

  void TearDown() override {
#if defined(USE_NPU)
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
#endif
    google::ShutdownGoogleLogging();
#if defined(USE_NPU)
    xllm::testing::finalize_npu_test_runtime();
#endif
  }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new SharedVMMAllocatorTestEnvironment);

bool HasDevice() { return xllm::Platform::device_count() > 0; }

void InitDevice() {
  xllm::Device device(0);
  device.set_device();
  device.init_device_context();
}

}  // namespace

TEST(SharedVMMAllocatorTest, BasicAllocateAndSwitch) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No accelerator device available";
  }

  InitDevice();
  xllm::SharedVMMAllocator allocator;
  const size_t reserve_size = 64 * 1024 * 1024;
  allocator.init(/*device_id=*/0, reserve_size);

  EXPECT_TRUE(allocator.is_initialized());
  EXPECT_GE(allocator.reserved_size(), reserve_size);
  EXPECT_EQ(allocator.current_offset(), 0u);

  void* first = allocator.allocate(1024 * 1024);
  EXPECT_NE(first, nullptr);
  allocator.deallocate(first);

  const size_t offset_after_first = allocator.current_offset();
  EXPECT_GT(offset_after_first, 0u);
  EXPECT_GE(allocator.mapped_size(), offset_after_first);

  void* second = allocator.allocate(1024 * 1024);
  EXPECT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_GE(allocator.current_offset(), offset_after_first);
  EXPECT_GE(allocator.high_water_mark(), allocator.current_offset());

  allocator.switch_to_new_virtual_space();
  EXPECT_EQ(allocator.current_offset(), 0u);

  void* third = allocator.allocate(1024 * 1024);
  EXPECT_NE(third, nullptr);
  EXPECT_NE(third, first);
}
