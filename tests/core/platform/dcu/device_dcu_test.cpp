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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "core/platform/device.h"

namespace {

class DeviceDcuTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("device_dcu_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new DeviceDcuTestEnvironment);

constexpr int kDeviceId = 0;

bool HasHipDevice() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

void PrepareDeviceOrSkip() {
  if (!HasHipDevice()) {
    GTEST_SKIP() << "No DCU/HIP device available";
  }

  ASSERT_GT(xllm::Device::device_count(), 0);
  ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
  ASSERT_EQ(hipFree(nullptr), hipSuccess);
}

// device_count()
TEST(DeviceDcuTest, DeviceCountMatchesHipRuntime) {
  PrepareDeviceOrSkip();

  int hip_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&hip_count), hipSuccess);

  int xllm_count = xllm::Device::device_count();

  EXPECT_EQ(xllm_count, hip_count);
  EXPECT_GT(xllm_count, 0);
}

// unwrap() index() type_torch()
TEST(DeviceDcuTest, DeviceConstructionAndAccessorsWork) {
  PrepareDeviceOrSkip();

  // Construct from an integer device index.
  xllm::Device dev_from_index(0);
  EXPECT_EQ(dev_from_index.index(), 0);
  EXPECT_EQ(dev_from_index.unwrap().type(), torch::kCUDA);
  EXPECT_EQ(dev_from_index.unwrap().index(), 0);

  torch::Device converted_from_index = dev_from_index;
  EXPECT_EQ(converted_from_index.type(), torch::kCUDA);
  EXPECT_EQ(converted_from_index.index(), 0);

  // Construct from torch::Device.
  torch::Device torch_dev0(torch::kCUDA, 0);
  xllm::Device dev_from_torch(torch_dev0);
  EXPECT_EQ(dev_from_torch.index(), 0);
  EXPECT_EQ(dev_from_torch.unwrap().type(), torch::kCUDA);
  EXPECT_EQ(dev_from_torch.unwrap().index(), 0);

  torch::Device converted_from_torch = dev_from_torch;
  EXPECT_EQ(converted_from_torch.type(), torch::kCUDA);
  EXPECT_EQ(converted_from_torch.index(), 0);

  // Check non-zero device IDs when multiple devices are available.
  int count = 0;
  ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
  if (count >= 2) {
    xllm::Device dev1(1);
    EXPECT_EQ(dev1.index(), 1);
    EXPECT_EQ(dev1.unwrap().type(), torch::kCUDA);
    EXPECT_EQ(dev1.unwrap().index(), 1);

    torch::Device converted1 = dev1;
    EXPECT_EQ(converted1.type(), torch::kCUDA);
    EXPECT_EQ(converted1.index(), 1);

    torch::Device torch_dev1(torch::kCUDA, 1);
    xllm::Device dev_from_torch1(torch_dev1);
    EXPECT_EQ(dev_from_torch1.index(), 1);
    EXPECT_EQ(dev_from_torch1.unwrap().type(), torch::kCUDA);
    EXPECT_EQ(dev_from_torch1.unwrap().index(), 1);
  }
}

// set_device()
TEST(DeviceDcuTest, SetDeviceChangesCurrentHipDevice) {
  PrepareDeviceOrSkip();

  int count = 0;
  ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
  ASSERT_GT(count, 0);

  int before = -1;
  ASSERT_EQ(hipGetDevice(&before), hipSuccess);

  int target = 0;
  if (count >= 2) {
    target = (before == 0) ? 1 : 0;
  } else {
    target = 0;
  }

  xllm::Device dev(target);
  dev.set_device();

  int after = -1;
  ASSERT_EQ(hipGetDevice(&after), hipSuccess);

  EXPECT_EQ(after, target);

  const int64_t total = dev.total_memory();
  const int64_t free = dev.free_memory();

  EXPECT_GT(total, 0);
  EXPECT_GT(free, 0);
  EXPECT_LE(free, total);
}

// total_memory()
TEST(DeviceDcuTest, MemoryInfoMatchesHipRuntimeExactly) {
  PrepareDeviceOrSkip();

  xllm::Device dev(0);
  dev.set_device();

  size_t hip_free = 0;
  size_t hip_total = 0;
  ASSERT_EQ(hipMemGetInfo(&hip_free, &hip_total), hipSuccess);

  const int64_t total = dev.total_memory();

  EXPECT_EQ(total, static_cast<int64_t>(hip_total));
}

// free_memory()
TEST(DeviceDcuTest, FreeMemoryMatchesHipRuntimeSnapshot) {
  PrepareDeviceOrSkip();

  xllm::Device dev(0);
  dev.set_device();

  // Reduce noise from previous allocator state.
  xllm::Device::empty_cache(0);
  ASSERT_EQ(dev.synchronize_default_stream(), 0);

  size_t hip_free_before = 0;
  size_t hip_total_before = 0;
  ASSERT_EQ(hipMemGetInfo(&hip_free_before, &hip_total_before), hipSuccess);

  const int64_t free = dev.free_memory();

  size_t hip_free_after = 0;
  size_t hip_total_after = 0;
  ASSERT_EQ(hipMemGetInfo(&hip_free_after, &hip_total_after), hipSuccess);

  // Ideally free == hip_free_before == hip_free_after, but free_memory()
  // calls hipMemGetInfo internally, so compare against the surrounding
  // runtime snapshots instead.
  EXPECT_LE(static_cast<int64_t>(hip_free_after), free);
  EXPECT_GE(static_cast<int64_t>(hip_free_before), free);
}

// empty_cache()
TEST(DeviceDcuTest, EmptyCacheReleasesCachedMemoryBackToRuntime) {
  PrepareDeviceOrSkip();

  xllm::Device dev(kDeviceId);
  dev.set_device();

  // Clear the cache first to reduce interference from previous tests.
  xllm::Device::empty_cache(kDeviceId);
  ASSERT_EQ(dev.synchronize_default_stream(), 0);

  const int64_t free_before = dev.free_memory();
  const int64_t total_before = dev.total_memory();

  auto opts = torch::TensorOptions()
                  .device(torch::Device(c10::DeviceType::CUDA, kDeviceId))
                  .dtype(torch::kFloat32);

  {
    // Use rand instead of empty to avoid HIP backend empty.memory_format
    // issues seen in some environments.
    torch::Tensor t = torch::rand({256, 1024, 1024}, opts);
    ASSERT_TRUE(t.defined());
    ASSERT_EQ(t.device().type(), c10::DeviceType::CUDA);
    ASSERT_EQ(t.device().index(), kDeviceId);

    ASSERT_EQ(dev.synchronize_default_stream(), 0);

    const int64_t free_after_alloc = dev.free_memory();

    EXPECT_LT(free_after_alloc, free_before);
  }

  // After tensor destruction, memory may still be cached by the allocator.
  ASSERT_EQ(dev.synchronize_default_stream(), 0);
  const int64_t free_after_release_before_empty = dev.free_memory();

  xllm::Device::empty_cache(kDeviceId);
  ASSERT_EQ(dev.synchronize_default_stream(), 0);

  const int64_t free_after_empty = dev.free_memory();

  // empty_cache should not reduce free memory, and usually recovers some.
  EXPECT_GE(free_after_empty, free_after_release_before_empty);
  EXPECT_LE(free_after_empty, total_before);
}

// set_seed()
TEST(DeviceDcuTest, SetSeedMakesRandomSequenceRepeatable) {
  PrepareDeviceOrSkip();

  xllm::Device dev(kDeviceId);
  dev.set_device();

  auto opts = torch::TensorOptions()
                  .device(torch::Device(c10::DeviceType::CUDA, kDeviceId))
                  .dtype(torch::kFloat32);

  // The same seed should produce the same sequence.
  dev.set_seed(12345);
  torch::Tensor a = torch::rand({16}, opts);

  dev.set_seed(12345);
  torch::Tensor b = torch::rand({16}, opts);

  // A different seed should produce a different sequence.
  dev.set_seed(54321);
  torch::Tensor c = torch::rand({16}, opts);

  EXPECT_TRUE(a.defined());
  EXPECT_TRUE(b.defined());
  EXPECT_TRUE(c.defined());

  EXPECT_EQ(a.device().type(), c10::DeviceType::CUDA);
  EXPECT_EQ(b.device().type(), c10::DeviceType::CUDA);
  EXPECT_EQ(c.device().type(), c10::DeviceType::CUDA);

  EXPECT_EQ(a.device().index(), kDeviceId);
  EXPECT_EQ(b.device().index(), kDeviceId);
  EXPECT_EQ(c.device().index(), kDeviceId);

  torch::Tensor a_cpu = a.cpu();
  torch::Tensor b_cpu = b.cpu();
  torch::Tensor c_cpu = c.cpu();

  EXPECT_TRUE(torch::allclose(a_cpu, b_cpu));
  EXPECT_FALSE(torch::allclose(a_cpu, c_cpu));
}

// synchronize_default_stream()
TEST(DeviceDcuTest, SynchronizeDefaultStreamWaitsForQueuedHipWork) {
  PrepareDeviceOrSkip();

  xllm::Device dev(kDeviceId);
  dev.set_device();

  c10::hip::HIPStream current_stream = c10::hip::getCurrentHIPStream(kDeviceId);
  hipStream_t stream = current_stream.stream();

  constexpr size_t kBytes = 256 << 20;
  constexpr unsigned char kPattern = 0x5A;

  void* dptr = nullptr;
  void* hptr = nullptr;

  ASSERT_EQ(hipMalloc(&dptr, kBytes), hipSuccess);
  ASSERT_EQ(hipHostMalloc(&hptr, kBytes, hipHostMallocDefault), hipSuccess);

  auto cleanup = [&]() {
    if (dptr) {
      EXPECT_EQ(hipFree(dptr), hipSuccess);
      dptr = nullptr;
    }
    if (hptr) {
      EXPECT_EQ(hipHostFree(hptr), hipSuccess);
      hptr = nullptr;
    }
  };

  std::memset(hptr, 0, kBytes);

  unsigned char* bytes = static_cast<unsigned char*>(hptr);

  ASSERT_EQ(hipMemsetAsync(dptr, kPattern, kBytes, stream), hipSuccess);

  ASSERT_EQ(hipMemcpyAsync(hptr, dptr, kBytes, hipMemcpyDeviceToHost, stream),
            hipSuccess);

  int rc = dev.synchronize_default_stream();
  EXPECT_EQ(rc, 0);

  for (size_t i = 0; i < kBytes; ++i) {
    ASSERT_EQ(bytes[i], kPattern) << "Mismatch at byte " << i;
  }

  cleanup();
}

// current_stream()
TEST(DeviceDcuTest, CurrentStreamTracksGuardedHipStream) {
  PrepareDeviceOrSkip();

  xllm::Device dev(kDeviceId);
  dev.set_device();

  c10::hip::HIPStream before = c10::hip::getCurrentHIPStream(kDeviceId);

  std::unique_ptr<xllm::Stream> pooled = dev.get_stream_from_pool();
  ASSERT_NE(pooled, nullptr);
  ASSERT_NE(pooled->get_stream(), nullptr);

  {
    [[maybe_unused]] auto guard = pooled->set_stream_guard();

    c10::hip::HIPStream current_after_guard =
        c10::hip::getCurrentHIPStream(kDeviceId);

    std::unique_ptr<xllm::Stream> current = dev.current_stream();
    ASSERT_NE(current, nullptr);
    ASSERT_NE(current->get_stream(), nullptr);

    EXPECT_EQ(current_after_guard, *pooled->get_stream());
    EXPECT_EQ(*current->get_stream(), current_after_guard);
    EXPECT_EQ(*current->get_stream(), *pooled->get_stream());
  }

  c10::hip::HIPStream after = c10::hip::getCurrentHIPStream(kDeviceId);

  EXPECT_EQ(after, before);
}

// get_stream_from_pool()
TEST(DeviceDcuTest, StreamFromPoolCanBeGuardedAndRestored) {
  PrepareDeviceOrSkip();

  xllm::Device dev(kDeviceId);
  dev.set_device();

  c10::hip::HIPStream before = c10::hip::getCurrentHIPStream(kDeviceId);

  std::unique_ptr<xllm::Stream> stream = dev.get_stream_from_pool();
  ASSERT_NE(stream, nullptr);
  ASSERT_NE(stream->get_stream(), nullptr);

  {
    auto guard = stream->set_stream_guard();
    c10::hip::HIPStream current = c10::hip::getCurrentHIPStream(kDeviceId);
    EXPECT_EQ(current, *stream->get_stream());
  }

  c10::hip::HIPStream after = c10::hip::getCurrentHIPStream(kDeviceId);
  EXPECT_EQ(after, before);
}

}  // namespace
