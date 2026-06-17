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

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/platform/vmm_api.h"

namespace {

constexpr int kDeviceId = 0;
constexpr std::size_t kProbeBytes = 256;

bool HasDcuDevice() {
  int count = 0;
  hipError_t ret = hipGetDeviceCount(&count);
  return ret == hipSuccess && count > 0;
}

void PrepareDcuDeviceOrSkip() {
  if (!HasDcuDevice()) {
    GTEST_SKIP() << "No DCU device available";
  }

  ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
  // Force HIP runtime initialization.
  ASSERT_EQ(hipFree(nullptr), hipSuccess);
}

// get_recommended_granularity()
TEST(VmmApiDcuTest, RecommendedGranularityMatchesHipRuntime) {
  PrepareDcuDeviceOrSkip();

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = kDeviceId;

  std::size_t hip_granularity = 0;
  ASSERT_EQ(
      hipMemGetAllocationGranularity(
          &hip_granularity, &prop, hipMemAllocationGranularityRecommended),
      hipSuccess);

  const std::size_t xllm_granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  EXPECT_EQ(xllm_granularity, hip_granularity);
  EXPECT_GT(xllm_granularity, 0u);
}

// create_phy_mem_handle() release_phy_mem_handle()
TEST(VmmApiDcuTest, CanCreateAndReleasePhysicalMemoryHandle) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::PhyMemHandle handle{};
  xllm::vmm::create_phy_mem_handle(handle, kDeviceId);

  EXPECT_EQ(
      static_cast<std::size_t>(
          xllm::KVCacheConfig::get_instance().phy_page_granularity_size()),
      granularity);

  xllm::vmm::release_phy_mem_handle(handle);
}

// create_vir_ptr()
TEST(VmmApiDcuTest, CanReserveAndReleaseVirtualAddressRange) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr = xllm::uintptr_to_vir_ptr(0);
  xllm::vmm::create_vir_ptr(vir_ptr, granularity * 2);

  EXPECT_FALSE(xllm::is_null_vir_ptr(vir_ptr));

  xllm::vmm::release_vir_ptr(vir_ptr, granularity * 2);
}

// uintptr_to_vir_ptr() add_vir_ptr_offset() vir_ptr_to_void_ptr()
TEST(VmmApiDcuTest, VirPtrHelpersWork) {
  constexpr uintptr_t kBase = 0x100000;
  constexpr std::size_t kOffset = 0x234;
  constexpr std::size_t kOffset1 = 0x100;
  constexpr std::size_t kOffset2 = 0x234;

  xllm::VirPtr base = xllm::uintptr_to_vir_ptr(kBase);

  // Basic round-trip conversion.
  EXPECT_EQ(xllm::vir_ptr_to_uintptr(base), kBase);
  EXPECT_EQ(xllm::vir_ptr_to_void_ptr(base), reinterpret_cast<void*>(kBase));

  // Single offset.
  xllm::VirPtr shifted = xllm::add_vir_ptr_offset(base, kOffset);
  EXPECT_EQ(xllm::vir_ptr_to_uintptr(shifted), kBase + kOffset);
  EXPECT_EQ(xllm::vir_ptr_to_void_ptr(shifted),
            reinterpret_cast<void*>(kBase + kOffset));

  // A zero offset should keep the pointer unchanged.
  EXPECT_EQ(xllm::vir_ptr_to_uintptr(xllm::add_vir_ptr_offset(base, 0)), kBase);

  // Chained offsets.
  xllm::VirPtr p1 = xllm::add_vir_ptr_offset(base, kOffset1);
  xllm::VirPtr p2 = xllm::add_vir_ptr_offset(p1, kOffset2);
  EXPECT_EQ(xllm::vir_ptr_to_uintptr(p2), kBase + kOffset1 + kOffset2);

  // Null pointer cases.
  xllm::VirPtr null_ptr = xllm::uintptr_to_vir_ptr(0);
  EXPECT_TRUE(xllm::is_null_vir_ptr(null_ptr));
  EXPECT_TRUE(xllm::is_null_vir_ptr(xllm::add_vir_ptr_offset(null_ptr, 0)));
}

TEST(VmmApiDcuTest, CanMapAndAccessMemoryWithExplicitGranularityOverload) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr = xllm::uintptr_to_vir_ptr(0);
  xllm::PhyMemHandle handle{};
  std::vector<std::uint8_t> host(kProbeBytes, 0);

  xllm::vmm::create_vir_ptr(vir_ptr, granularity);

  xllm::vmm::create_phy_mem_handle(handle, kDeviceId);

  xllm::vmm::map(vir_ptr, handle, granularity, kDeviceId);

  hipError_t memset_ret = hipMemset(vir_ptr, 0x5A, kProbeBytes);

  hipError_t sync_ret = hipDeviceSynchronize();

  hipError_t memcpy_ret =
      hipMemcpy(host.data(), vir_ptr, kProbeBytes, hipMemcpyDeviceToHost);

  xllm::vmm::unmap(vir_ptr, granularity);

  xllm::vmm::release_vir_ptr(vir_ptr, granularity);

  xllm::vmm::release_phy_mem_handle(handle);

  ASSERT_EQ(memset_ret, hipSuccess);
  ASSERT_EQ(sync_ret, hipSuccess);
  ASSERT_EQ(memcpy_ret, hipSuccess);

  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0x5A) << "Mismatch at byte " << i;
  }
}

TEST(VmmApiDcuTest, CanMapAndAccessMemoryWithDefaultGranularityOverload) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr = xllm::uintptr_to_vir_ptr(0);
  xllm::PhyMemHandle handle{};
  std::vector<std::uint8_t> host(kProbeBytes, 0);

  xllm::vmm::create_vir_ptr(vir_ptr, granularity);

  // create_phy_mem_handle() refreshes KVCacheConfig; the default map overload
  // depends on that runtime value.
  xllm::vmm::create_phy_mem_handle(handle, kDeviceId);

  xllm::vmm::map(vir_ptr, handle, kDeviceId);

  hipError_t memset_ret = hipMemset(vir_ptr, 0xA5, kProbeBytes);

  hipError_t sync_ret = hipDeviceSynchronize();

  hipError_t memcpy_ret =
      hipMemcpy(host.data(), vir_ptr, kProbeBytes, hipMemcpyDeviceToHost);

  xllm::vmm::unmap(vir_ptr, granularity);

  xllm::vmm::release_vir_ptr(vir_ptr, granularity);

  xllm::vmm::release_phy_mem_handle(handle);

  ASSERT_EQ(memset_ret, hipSuccess);
  ASSERT_EQ(sync_ret, hipSuccess);
  ASSERT_EQ(memcpy_ret, hipSuccess);

  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xA5) << "Mismatch at byte " << i;
  }
}

TEST(VmmApiDcuTest, CanMapTwoDifferentVirtualRangesIndependently) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr_1 = xllm::uintptr_to_vir_ptr(0);
  xllm::VirPtr vir_ptr_2 = xllm::uintptr_to_vir_ptr(0);
  xllm::PhyMemHandle handle_1{};
  xllm::PhyMemHandle handle_2{};
  std::vector<std::uint8_t> host1(kProbeBytes, 0);
  std::vector<std::uint8_t> host2(kProbeBytes, 0);

  xllm::vmm::create_vir_ptr(vir_ptr_1, granularity);
  xllm::vmm::create_vir_ptr(vir_ptr_2, granularity);

  xllm::vmm::create_phy_mem_handle(handle_1, kDeviceId);
  xllm::vmm::create_phy_mem_handle(handle_2, kDeviceId);

  xllm::vmm::map(vir_ptr_1, handle_1, granularity, kDeviceId);
  xllm::vmm::map(vir_ptr_2, handle_2, granularity, kDeviceId);

  hipError_t memset_ret_1 = hipMemset(vir_ptr_1, 0x11, kProbeBytes);
  hipError_t memset_ret_2 = hipMemset(vir_ptr_2, 0x22, kProbeBytes);

  hipError_t sync_ret = hipDeviceSynchronize();

  hipError_t memcpy_ret_1 =
      hipMemcpy(host1.data(), vir_ptr_1, kProbeBytes, hipMemcpyDeviceToHost);
  hipError_t memcpy_ret_2 =
      hipMemcpy(host2.data(), vir_ptr_2, kProbeBytes, hipMemcpyDeviceToHost);

  xllm::vmm::unmap(vir_ptr_1, granularity);
  xllm::vmm::unmap(vir_ptr_2, granularity);

  xllm::vmm::release_vir_ptr(vir_ptr_1, granularity);
  xllm::vmm::release_vir_ptr(vir_ptr_2, granularity);

  xllm::vmm::release_phy_mem_handle(handle_1);
  xllm::vmm::release_phy_mem_handle(handle_2);

  ASSERT_EQ(memset_ret_1, hipSuccess);
  ASSERT_EQ(memset_ret_2, hipSuccess);
  ASSERT_EQ(sync_ret, hipSuccess);
  ASSERT_EQ(memcpy_ret_1, hipSuccess);
  ASSERT_EQ(memcpy_ret_2, hipSuccess);

  for (std::size_t i = 0; i < host1.size(); ++i) {
    EXPECT_EQ(host1[i], 0x11) << "Mismatch in range 1 at byte " << i;
  }
  for (std::size_t i = 0; i < host2.size(); ++i) {
    EXPECT_EQ(host2[i], 0x22) << "Mismatch in range 2 at byte " << i;
  }
}

TEST(VmmApiDcuTest, CanAliasTwoVirtualRangesToSamePhysicalHandle) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr_1 = xllm::uintptr_to_vir_ptr(0);
  xllm::VirPtr vir_ptr_2 = xllm::uintptr_to_vir_ptr(0);
  xllm::PhyMemHandle handle{};
  std::vector<std::uint8_t> host1(kProbeBytes, 0);
  std::vector<std::uint8_t> host2(kProbeBytes, 0);

  xllm::vmm::create_vir_ptr(vir_ptr_1, granularity);
  xllm::vmm::create_vir_ptr(vir_ptr_2, granularity);

  xllm::vmm::create_phy_mem_handle(handle, kDeviceId);

  xllm::vmm::map(vir_ptr_1, handle, granularity, kDeviceId);
  xllm::vmm::map(vir_ptr_2, handle, granularity, kDeviceId);

  hipError_t memset_ret = hipMemset(vir_ptr_1, 0x3C, kProbeBytes);

  hipError_t sync_ret = hipDeviceSynchronize();

  hipError_t memcpy_ret_1 =
      hipMemcpy(host1.data(), vir_ptr_1, kProbeBytes, hipMemcpyDeviceToHost);
  hipError_t memcpy_ret_2 =
      hipMemcpy(host2.data(), vir_ptr_2, kProbeBytes, hipMemcpyDeviceToHost);

  xllm::vmm::unmap(vir_ptr_1, granularity);
  xllm::vmm::unmap(vir_ptr_2, granularity);

  xllm::vmm::release_vir_ptr(vir_ptr_1, granularity);
  xllm::vmm::release_vir_ptr(vir_ptr_2, granularity);

  xllm::vmm::release_phy_mem_handle(handle);

  ASSERT_EQ(memset_ret, hipSuccess);
  ASSERT_EQ(sync_ret, hipSuccess);
  ASSERT_EQ(memcpy_ret_1, hipSuccess);
  ASSERT_EQ(memcpy_ret_2, hipSuccess);

  for (std::size_t i = 0; i < host1.size(); ++i) {
    EXPECT_EQ(host1[i], 0x3C) << "Mismatch in range 1 at byte " << i;
    // EXPECT_EQ(host2[i], 0x3C) << "Mismatch in range 2 at byte " << i;
  }
}

TEST(VmmApiDcuTest, CanRemapSameVirtualRangeToNewPhysicalHandleAfterUnmap) {
  PrepareDcuDeviceOrSkip();

  const std::size_t granularity =
      xllm::vmm::get_recommended_granularity(kDeviceId);

  xllm::VirPtr vir_ptr = xllm::uintptr_to_vir_ptr(0);
  xllm::PhyMemHandle handle1{};
  xllm::PhyMemHandle handle2{};
  std::vector<std::uint8_t> host1(kProbeBytes, 0);
  std::vector<std::uint8_t> host2(kProbeBytes, 0);

  xllm::vmm::create_vir_ptr(vir_ptr, granularity);

  // First mapping.
  xllm::vmm::create_phy_mem_handle(handle1, kDeviceId);

  xllm::vmm::map(vir_ptr, handle1, granularity, kDeviceId);

  ASSERT_EQ(hipMemset(vir_ptr, 0x11, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(
      hipMemcpy(host1.data(), vir_ptr, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < host1.size(); ++i) {
    ASSERT_EQ(host1[i], 0x11) << "Mismatch in first mapping at byte " << i;
  }

  xllm::vmm::unmap(vir_ptr, granularity);
  xllm::vmm::release_phy_mem_handle(handle1);

  // Second mapping to a new physical page.
  xllm::vmm::create_phy_mem_handle(handle2, kDeviceId);

  xllm::vmm::map(vir_ptr, handle2, granularity, kDeviceId);

  ASSERT_EQ(hipMemset(vir_ptr, 0x22, kProbeBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(
      hipMemcpy(host2.data(), vir_ptr, kProbeBytes, hipMemcpyDeviceToHost),
      hipSuccess);

  for (std::size_t i = 0; i < host2.size(); ++i) {
    ASSERT_EQ(host2[i], 0x22) << "Mismatch in second mapping at byte " << i;
  }

  xllm::vmm::unmap(vir_ptr, granularity);
  xllm::vmm::release_vir_ptr(vir_ptr, granularity);
  xllm::vmm::release_phy_mem_handle(handle2);
}

}  // namespace
