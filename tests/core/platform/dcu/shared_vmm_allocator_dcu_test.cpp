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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "core/platform/shared_vmm_allocator.h"
#include "core/platform/vmm_api.h"

namespace {

class SharedVMMAllocatorTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("shared_vmm_allocator_dcu_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new SharedVMMAllocatorTestEnvironment);

constexpr int kDeviceId = 0;
constexpr std::size_t kProbeBytes = 256;

bool HasDevice() { return xllm::Device::device_count() > 0; }

void InitDevice() {
  xllm::Device device(kDeviceId);
  device.set_device();
  device.init_device_context();

  ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
  ASSERT_EQ(hipFree(nullptr), hipSuccess);
}

std::size_t ReserveSizeForTest(std::size_t granularity) {
  return std::max<std::size_t>(granularity * 8, 8 * 1024 * 1024);
}

TEST(SharedVMMAllocatorDcuTest, InitAndBasicAllocate) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);
  const std::size_t reserve_size = ReserveSizeForTest(granularity);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, reserve_size);

  EXPECT_TRUE(allocator.is_initialized());
  EXPECT_GE(allocator.reserved_size(), reserve_size);
  EXPECT_EQ(allocator.current_offset(), 0u);
  EXPECT_EQ(allocator.mapped_size(), 0u);
  EXPECT_EQ(allocator.high_water_mark(), 0u);

  void* ptr = allocator.allocate(1);
  ASSERT_NE(ptr, nullptr);

  // The first 1-byte allocation is aligned to granularity.
  EXPECT_EQ(allocator.current_offset(), granularity);
  EXPECT_EQ(allocator.mapped_size(), granularity);
  EXPECT_EQ(allocator.high_water_mark(), granularity);

  // Verify the allocated address is actually accessible.
  std::vector<std::uint8_t> host(kProbeBytes, 0);
  ASSERT_EQ(hipMemset(ptr, 0x5A, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipMemcpy(host.data(), ptr, kProbeBytes, hipMemcpyDeviceToHost),
            hipSuccess);

  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0x5A) << "Mismatch at byte " << i;
  }
}

TEST(SharedVMMAllocatorDcuTest, ResetAllocationPointerReusesSameAddress) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  void* first = allocator.allocate(64);
  ASSERT_NE(first, nullptr);

  ASSERT_EQ(allocator.current_offset(), granularity);
  ASSERT_EQ(allocator.mapped_size(), granularity);
  ASSERT_EQ(allocator.high_water_mark(), granularity);

  void* second = allocator.allocate(64);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(first, second);

  ASSERT_EQ(allocator.current_offset(), granularity * 2);
  ASSERT_EQ(allocator.mapped_size(), granularity * 2);
  ASSERT_EQ(allocator.high_water_mark(), granularity * 2);

  const std::size_t mapped_before_reset = allocator.mapped_size();
  const std::size_t high_water_before_reset = allocator.high_water_mark();

  allocator.reset_allocation_pointer();

  EXPECT_EQ(allocator.current_offset(), 0u);
  EXPECT_EQ(allocator.mapped_size(), mapped_before_reset);
  EXPECT_EQ(allocator.high_water_mark(), high_water_before_reset);

  void* third = allocator.allocate(64);
  ASSERT_NE(third, nullptr);

  EXPECT_EQ(third, first);
  EXPECT_EQ(allocator.current_offset(), granularity);
  EXPECT_EQ(allocator.mapped_size(), mapped_before_reset);
  EXPECT_EQ(allocator.high_water_mark(), high_water_before_reset);

  void* fourth = allocator.allocate(64);
  ASSERT_NE(fourth, nullptr);

  EXPECT_EQ(fourth, second);
  EXPECT_EQ(allocator.current_offset(), granularity * 2);
  EXPECT_EQ(allocator.mapped_size(), mapped_before_reset);
  EXPECT_EQ(allocator.high_water_mark(), high_water_before_reset);
}

TEST(SharedVMMAllocatorDcuTest, SwitchToNewVirtualSpaceSharesPhysicalMemory) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  void* first = allocator.allocate(kProbeBytes);
  ASSERT_NE(first, nullptr);

  // Write data at offset 0 in the old virtual address space.
  ASSERT_EQ(hipMemset(first, 0x3C, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  allocator.switch_to_new_virtual_space();
  EXPECT_EQ(allocator.current_offset(), 0u);

  // Allocate the same offset in the new virtual address space; it should map
  // to the same physical memory.
  void* second = allocator.allocate(kProbeBytes);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(second, first);

  std::vector<std::uint8_t> host(kProbeBytes, 0);
  ASSERT_EQ(hipMemcpy(host.data(), second, kProbeBytes, hipMemcpyDeviceToHost),
            hipSuccess);

  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0x3C) << "Mismatch at byte " << i;
  }

  // Write through the new virtual address and verify the old virtual address
  // observes the same data, proving both spaces share the physical memory.
  ASSERT_EQ(hipMemset(second, 0xC7, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  std::fill(host.begin(), host.end(), 0);
  ASSERT_EQ(hipMemcpy(host.data(), first, kProbeBytes, hipMemcpyDeviceToHost),
            hipSuccess);

  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xC7) << "Alias mismatch at byte " << i;
  }
}

TEST(SharedVMMAllocatorDcuTest,
     SwitchToNewVirtualSpacePreservesNonZeroOffsetPages) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  // Old virtual address space: allocate two pages at offset 0 and
  // offset 1 * granularity.
  void* first_page = allocator.allocate(kProbeBytes);
  ASSERT_NE(first_page, nullptr);

  void* second_page = allocator.allocate(kProbeBytes);
  ASSERT_NE(second_page, nullptr);
  ASSERT_NE(first_page, second_page);

  // Write 0x3C to page 0 and 0x7E to page 1 in the old virtual address space.
  ASSERT_EQ(hipMemset(first_page, 0x3C, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipMemset(second_page, 0x7E, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  const std::size_t mapped_before = allocator.mapped_size();
  const std::size_t hwm_before = allocator.high_water_mark();

  allocator.switch_to_new_virtual_space();

  EXPECT_EQ(allocator.current_offset(), 0u);
  EXPECT_EQ(allocator.mapped_size(), mapped_before);
  EXPECT_EQ(allocator.high_water_mark(), hwm_before);

  // New virtual address space: allocate the same two page offsets again.
  void* first_page_new = allocator.allocate(kProbeBytes);
  ASSERT_NE(first_page_new, nullptr);

  void* second_page_new = allocator.allocate(kProbeBytes);
  ASSERT_NE(second_page_new, nullptr);

  EXPECT_NE(first_page_new, first_page);
  EXPECT_NE(second_page_new, second_page);

  std::vector<std::uint8_t> host0(kProbeBytes, 0);
  std::vector<std::uint8_t> host1(kProbeBytes, 0);

  ASSERT_EQ(
      hipMemcpy(
          host0.data(), first_page_new, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);
  ASSERT_EQ(
      hipMemcpy(
          host1.data(), second_page_new, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < host0.size(); ++i) {
    EXPECT_EQ(host0[i], 0x3C) << "Mismatch in first page at byte " << i;
  }
  for (std::size_t i = 0; i < host1.size(); ++i) {
    EXPECT_EQ(host1[i], 0x7E) << "Mismatch in second page at byte " << i;
  }

  // Write back through page 1 in the new virtual address space; page 1 in
  // the old virtual address space should observe the same data.
  ASSERT_EQ(hipMemset(second_page_new, 0xA4, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  std::fill(host1.begin(), host1.end(), 0);
  ASSERT_EQ(
      hipMemcpy(host1.data(), second_page, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < host1.size(); ++i) {
    EXPECT_EQ(host1[i], 0xA4) << "Alias mismatch in second page at byte " << i;
  }
}

TEST(SharedVMMAllocatorDcuTest, AllocateZeroReturnsNullptr) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  void* p1 = allocator.allocate(0);
  void* p2 = allocator.allocate(0);

  EXPECT_EQ(p1, nullptr);
  EXPECT_EQ(p2, nullptr);
  EXPECT_EQ(allocator.current_offset(), 0u);
  EXPECT_EQ(allocator.mapped_size(), 0u);
  EXPECT_EQ(allocator.high_water_mark(), 0u);
}

TEST(SharedVMMAllocatorDcuTest, DeallocateIsNoOp) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  void* ptr = allocator.allocate(123);
  ASSERT_NE(ptr, nullptr);

  const std::size_t offset_before = allocator.current_offset();
  const std::size_t mapped_before = allocator.mapped_size();
  const std::size_t high_water_before = allocator.high_water_mark();

  allocator.deallocate(ptr);

  EXPECT_EQ(allocator.current_offset(), offset_before);
  EXPECT_EQ(allocator.mapped_size(), mapped_before);
  EXPECT_EQ(allocator.high_water_mark(), high_water_before);

  void* next = allocator.allocate(123);
  ASSERT_NE(next, nullptr);

  // deallocate is a no-op, so the pointer should not be reused.
  EXPECT_NE(next, ptr);
  EXPECT_EQ(allocator.current_offset(), offset_before + granularity);
  EXPECT_EQ(allocator.high_water_mark(), allocator.current_offset());
}

TEST(SharedVMMAllocatorDcuTest, ExtendMappingCanGrowAcrossMultiplePages) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  // Request 3.5 pages and expect alignment to 4 pages.
  const std::size_t request = granularity * 3 + granularity / 2;
  const std::size_t aligned =
      ((request + granularity - 1) / granularity) * granularity;

  void* ptr = allocator.allocate(request);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(allocator.current_offset(), aligned);
  EXPECT_EQ(allocator.mapped_size(), aligned);
  EXPECT_EQ(allocator.high_water_mark(), aligned);
}

TEST(SharedVMMAllocatorDcuTest,
     ExtendMappingPreservesOldPagesAndMakesNewPagesAccessible) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  // Allocate the first page.
  void* first = allocator.allocate(kProbeBytes);
  ASSERT_NE(first, nullptr);

  ASSERT_EQ(hipMemset(first, 0x3C, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  // Allocate across the second page and trigger extend_mapping.
  void* second = allocator.allocate(kProbeBytes);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(first, second);

  ASSERT_EQ(hipMemset(second, 0xA4, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  std::vector<std::uint8_t> host_first(kProbeBytes, 0);
  std::vector<std::uint8_t> host_second(kProbeBytes, 0);

  ASSERT_EQ(
      hipMemcpy(host_first.data(), first, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);
  ASSERT_EQ(
      hipMemcpy(host_second.data(), second, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < kProbeBytes; ++i) {
    EXPECT_EQ(host_first[i], 0x3C) << "Mismatch in first page at byte " << i;
    EXPECT_EQ(host_second[i], 0xA4) << "Mismatch in second page at byte " << i;
  }
}

TEST(SharedVMMAllocatorDcuTest, ExtendMappingMapsNewPagesToAllVirtualSpaces) {
  if (!HasDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }
  InitDevice();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::SharedVMMAllocator allocator;
  allocator.init(kDeviceId, ReserveSizeForTest(granularity));

  // Old virtual address space: allocate two pages first.
  void* old_page0 = allocator.allocate(kProbeBytes);
  void* old_page1 = allocator.allocate(kProbeBytes);
  ASSERT_NE(old_page0, nullptr);
  ASSERT_NE(old_page1, nullptr);

  allocator.switch_to_new_virtual_space();
  EXPECT_EQ(allocator.current_offset(), 0u);

  // New virtual address space: allocate offsets 0 and 1 to match the old
  // virtual address space.
  void* new_page0 = allocator.allocate(kProbeBytes);
  void* new_page1 = allocator.allocate(kProbeBytes);
  ASSERT_NE(new_page0, nullptr);
  ASSERT_NE(new_page1, nullptr);

  // Allocate the third page, triggering extend_mapping for a new page.
  void* new_page2 = allocator.allocate(kProbeBytes);
  ASSERT_NE(new_page2, nullptr);

  // Write data to the newly mapped page in the new virtual address space.
  ASSERT_EQ(hipMemset(new_page2, 0x6B, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  void* old_page2 = reinterpret_cast<void*>(
      reinterpret_cast<std::uint8_t*>(old_page0) + 2 * granularity);

  std::vector<std::uint8_t> host(kProbeBytes, 0);
  ASSERT_EQ(
      hipMemcpy(host.data(), old_page2, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < kProbeBytes; ++i) {
    EXPECT_EQ(host[i], 0x6B) << "Mismatch at byte " << i;
  }
}

}  // namespace
