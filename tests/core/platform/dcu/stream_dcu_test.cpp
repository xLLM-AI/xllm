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

#include <c10/hip/HIPStream.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <torch/torch.h>

#include <cstring>
#include <sstream>

#include "core/platform/stream.h"

namespace {

// =============================================================================
// Test environment
// =============================================================================

class StreamDcuTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("stream_dcu_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new StreamDcuTestEnvironment);

// check dcu device
constexpr int kDeviceId = 0;

bool HasHipDevice() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

void PrepareDeviceOrSkip() {
  if (!HasHipDevice()) {
    GTEST_SKIP() << "No DCU/HIP device available";
  }
  ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
  ASSERT_EQ(hipFree(nullptr), hipSuccess);
}

// =============================================================================
// Tests
// =============================================================================

// xllm::Stream APIs under test: Stream(HIPStream), get_stream,
// set_stream_guard, synchronize
TEST(StreamDcuTest, ConstructFromExistingCurrentStreamWorks) {
  PrepareDeviceOrSkip();

  constexpr size_t kNumBytes = 1 << 16;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);

  c10::hip::HIPStream current = c10::hip::getCurrentHIPStream(kDeviceId);
  xllm::Stream wrapped(current);

  ASSERT_NE(wrapped.get_stream(), nullptr);
  EXPECT_EQ(*wrapped.get_stream(), current);

  auto guard = wrapped.set_stream_guard();
  EXPECT_EQ(*wrapped.get_stream(), c10::hip::getCurrentHIPStream(kDeviceId));

  ASSERT_EQ(
      hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, wrapped.get_stream()->stream()),
      hipSuccess);

  EXPECT_EQ(wrapped.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

// xllm::Stream APIs under test: Stream(), set_stream_guard, synchronize
TEST(StreamDcuTest, StreamGuardVerifiesOperationOnCorrectStream) {
  PrepareDeviceOrSkip();

  constexpr size_t kNumBytes = 1 << 16;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);

  xllm::Stream s1, s2;

  {
    auto guard = s1.set_stream_guard();
    ASSERT_EQ(
        hipMemsetAsync(dev_ptr, 0xAA, kNumBytes, s1.get_stream()->stream()),
        hipSuccess);
  }
  s1.synchronize();

  {
    auto guard = s2.set_stream_guard();
    ASSERT_EQ(
        hipMemsetAsync(dev_ptr, 0xBB, kNumBytes, s2.get_stream()->stream()),
        hipSuccess);
  }
  s2.synchronize();

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0xBB) << "Data mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

// xllm::Stream APIs under test: Stream(), set_stream_guard, synchronize
TEST(StreamDcuTest, StreamGuardSwitchesAndRestoresCurrentStream) {
  PrepareDeviceOrSkip();

  constexpr size_t kNumBytes = 1 << 16;
  unsigned char *ptr1 = nullptr, *ptr2 = nullptr, *ptr3 = nullptr;
  ASSERT_EQ(hipMalloc(&ptr1, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMalloc(&ptr2, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMalloc(&ptr3, kNumBytes), hipSuccess);

  ASSERT_EQ(hipMemset(ptr1, 0, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(ptr2, 0, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(ptr3, 0, kNumBytes), hipSuccess);

  xllm::Stream s1, s2, s3;

  ASSERT_NE(s1.get_stream()->stream(), s2.get_stream()->stream());
  ASSERT_NE(s2.get_stream()->stream(), s3.get_stream()->stream());
  ASSERT_NE(s1.get_stream()->stream(), s3.get_stream()->stream());

  {
    auto g1 = s1.set_stream_guard();
    ASSERT_EQ(hipMemsetAsync(ptr1, 0x11, kNumBytes, s1.get_stream()->stream()),
              hipSuccess);

    {
      auto g2 = s2.set_stream_guard();
      ASSERT_EQ(
          hipMemsetAsync(ptr2, 0x22, kNumBytes, s2.get_stream()->stream()),
          hipSuccess);

      {
        auto g3 = s3.set_stream_guard();
        ASSERT_EQ(
            hipMemsetAsync(ptr3, 0x33, kNumBytes, s3.get_stream()->stream()),
            hipSuccess);
      }
    }
  }

  // StreamGuard only restores the current-stream context on destruction; it
  // does not synchronize.  Explicitly sync all three streams before readback.
  s1.synchronize();
  s2.synchronize();
  s3.synchronize();

  std::vector<unsigned char> host(kNumBytes);

  ASSERT_EQ(hipMemcpy(host.data(), ptr1, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x11) << "ptr1 mismatch at byte " << i;
  }

  ASSERT_EQ(hipMemcpy(host.data(), ptr2, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x22) << "ptr2 mismatch at byte " << i;
  }

  ASSERT_EQ(hipMemcpy(host.data(), ptr3, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x33) << "ptr3 mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(ptr1), hipSuccess);
  ASSERT_EQ(hipFree(ptr2), hipSuccess);
  ASSERT_EQ(hipFree(ptr3), hipSuccess);
}

// xllm::Stream APIs under test: Stream(Stream&&), get_stream, synchronize
TEST(StreamDcuTest, MoveConstructPreservesPendingOperations) {
  PrepareDeviceOrSkip();

  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(dev_ptr, 0, kNumBytes), hipSuccess);

  xllm::Stream s1;
  ASSERT_NE(s1.get_stream(), nullptr);
  c10::hip::HIPStream s1_stream = *s1.get_stream();

  ASSERT_EQ(hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, s1_stream.stream()),
            hipSuccess);

  xllm::Stream s2(std::move(s1));
  ASSERT_NE(s2.get_stream(), nullptr);
  EXPECT_EQ(*s2.get_stream(), s1_stream);

  EXPECT_EQ(s2.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Data lost after move-construct at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);

  // Moved-from object remains in a valid state; destroying it must not crash.
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional use-after-move
  // verification
  s1.~Stream();
}

// xllm::Stream APIs under test: Stream(), get_stream, wait_stream, synchronize
TEST(StreamDcuTest, WaitStreamEstablishesOrdering) {
  PrepareDeviceOrSkip();

  xllm::Stream producer;
  xllm::Stream consumer;

  hipStream_t producer_stream = producer.get_stream()->stream();
  hipStream_t consumer_stream = consumer.get_stream()->stream();

  ASSERT_NE(producer_stream, nullptr);
  ASSERT_NE(consumer_stream, nullptr);
  ASSERT_NE(producer_stream, consumer_stream);

  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  unsigned char* host_ptr = nullptr;

  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipHostMalloc(reinterpret_cast<void**>(&host_ptr), kNumBytes, 0),
            hipSuccess);

  std::memset(host_ptr, 0, kNumBytes);

  ASSERT_EQ(hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, producer_stream),
            hipSuccess);

  consumer.wait_stream(producer);

  ASSERT_EQ(
      hipMemcpyAsync(
          host_ptr, dev_ptr, kNumBytes, hipMemcpyDeviceToHost, consumer_stream),
      hipSuccess);

  EXPECT_EQ(consumer.synchronize(), 0);

  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host_ptr[i], 0x5A) << "Mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
  ASSERT_EQ(hipHostFree(host_ptr), hipSuccess);
}

// xllm::Stream APIs under test: operator<<
TEST(StreamDcuTest, OutputStreamOperatorWorks) {
  PrepareDeviceOrSkip();

  xllm::Stream s;
  std::ostringstream oss;
  oss << s;
  std::string out = oss.str();
  EXPECT_FALSE(out.empty()) << "Output stream should not be empty";
}

// xllm::Stream APIs under test: set_stream_guard
TEST(StreamDcuTest, GuardRestoresOriginalStreamContext) {
  PrepareDeviceOrSkip();

  c10::hip::HIPStream original = c10::hip::getCurrentHIPStream(kDeviceId);
  xllm::Stream s;

  {
    auto guard = s.set_stream_guard();
    EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), *s.get_stream());
  }

  EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), original);
}

// xllm::Stream APIs under test: wait_stream, synchronize
TEST(StreamDcuTest, WaitStreamSelfIsSafe) {
  PrepareDeviceOrSkip();

  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(dev_ptr, 0, kNumBytes), hipSuccess);

  xllm::Stream s;
  hipStream_t stream = s.get_stream()->stream();

  ASSERT_EQ(hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, stream), hipSuccess);
  s.wait_stream(s);
  EXPECT_EQ(s.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Data mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

}  // namespace
