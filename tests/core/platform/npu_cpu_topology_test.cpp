/* Copyright 2026 The xLLM Authors.

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

#include "core/platform/npu/npu_cpu_topology.h"

#include <gtest/gtest.h>

#include <numeric>

namespace xllm::npu {
namespace {

std::vector<NpuIdentity> inventory(int32_t count) {
  std::vector<NpuIdentity> result;
  result.reserve(count);
  for (int32_t id = 0; id < count; ++id) {
    result.emplace_back(NpuIdentity{id / 2, id % 2, id, id});
  }
  return result;
}

TEST(NpuCpuTopologyTest, ParsesA3InventoryAndIgnoresMcu) {
  const auto devices = parse_npu_inventory(
      "  NPU ID    Chip ID    Chip Logic ID    Chip Phy-ID    Chip Name\n"
      "  1         1          3                13             Ascend910\n"
      "  1         2          -                -              Mcu\n"
      "  0         0          0                10             Ascend910\n");
  ASSERT_EQ(devices.size(), 2U);
  EXPECT_EQ(devices[0].logical_id, 0);
  EXPECT_EQ(devices[1].logical_id, 3);
  EXPECT_EQ(devices[1].physical_id, 13);
  EXPECT_EQ(devices[1].card_id, 1);
  EXPECT_EQ(devices[1].chip_id, 1);
}

TEST(NpuCpuTopologyTest, RejectsUnknownAndDuplicateInventory) {
  EXPECT_TRUE(parse_npu_inventory("driver unavailable").empty());
  EXPECT_TRUE(
      parse_npu_inventory("NPU ID    Chip ID    Chip Logic ID\n0 0 1\n1 0 1\n")
          .empty());
}

TEST(NpuCpuTopologyTest, VisibleDeviceOrderIsNotSorted) {
  const auto devices = inventory(16);
  EXPECT_EQ(resolve_npu_logical_id(0, "12,3,7", devices), 12);
  EXPECT_EQ(resolve_npu_logical_id(1, "12,3,7", devices), 3);
  EXPECT_EQ(resolve_npu_logical_id(15, "", devices), 15);
  EXPECT_FALSE(resolve_npu_logical_id(3, "12,3,7", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,3", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,", devices));
  EXPECT_FALSE(resolve_npu_logical_id(-1, "", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "99", devices));
}

TEST(NpuCpuTopologyTest, HiddenNpusKeepTheirGlobalSlices) {
  const auto devices = inventory(16);
  const auto id = resolve_npu_logical_id(0, "15", devices);
  ASSERT_TRUE(id);
  CpuBindingTopology host;
  host.allowed_cpus.resize(640);
  std::iota(host.allowed_cpus.begin(), host.allowed_cpus.end(), 0);
  host.device_ids.resize(16);
  std::iota(host.device_ids.begin(), host.device_ids.end(), 0);
  const auto plan = make_cpu_binding_plan(
      host, *id, npu_cpu_binding_options(/*global_slice=*/true));
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus.front(), 602);
}

TEST(NpuCpuTopologyTest, ParsesTopologyByPhysicalId) {
  auto devices = inventory(2);
  devices[0].physical_id = 10;
  devices[1].physical_id = 11;
  const auto affinity = parse_npu_affinity(
      "  Phy-ID10  Phy-ID11  CPU Affinity\n"
      "Phy-ID10  X    SYS    0-19,40-59\n"
      "Phy-ID11  SYS  X      20-39,60-79\n",
      devices);
  ASSERT_EQ(affinity.size(), 2U);
  EXPECT_EQ(affinity.at(0).front(), 0);
  EXPECT_EQ(affinity.at(0).back(), 59);
  EXPECT_TRUE(parse_npu_affinity("Phy-ID10 X SIO\n", devices).empty());
}

TEST(NpuCpuTopologyTest, AscendRolesAndMemoryModeAreBackendOptions) {
  const auto a3 = npu_cpu_binding_options(/*global_slice=*/true);
  EXPECT_EQ(a3.mode, CpuBindingMode::GLOBAL_SLICE);
  EXPECT_EQ(a3.reserved_cpu_count, 2);
  EXPECT_EQ(a3.dedicated_threads,
            (std::vector<std::string>{"acl_thread", "release_thread"}));
  EXPECT_EQ(a3.memory_mode, CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP);
  const auto a2 = npu_cpu_binding_options(/*global_slice=*/false);
  EXPECT_EQ(a2.mode, CpuBindingMode::TOPO_AFFINITY);
  EXPECT_TRUE(a2.extend_numa_pool);
  EXPECT_EQ(a2.reserved_cpu_count, 2);
}

TEST(NpuCpuTopologyTest, A2SharedPoolAssignsAllRemainderToLastDevice) {
  CpuBindingTopology host;
  host.allowed_cpus = *parse_cpu_list("0-19");
  host.device_ids = {0, 1, 2};
  for (int32_t cpu = 0; cpu < 20; ++cpu) {
    host.cpu_nodes[cpu] = cpu / 10;
  }
  for (int32_t id : host.device_ids) {
    host.affinity[id] = *parse_cpu_list("0-9");
  }
  const auto options = npu_cpu_binding_options(/*global_slice=*/false);
  const auto first = make_cpu_binding_plan(host, 0, options);
  const auto middle = make_cpu_binding_plan(host, 1, options);
  const auto last = make_cpu_binding_plan(host, 2, options);
  ASSERT_TRUE(first);
  ASSERT_TRUE(middle);
  ASSERT_TRUE(last);
  EXPECT_EQ(first->worker_cpus, (std::vector<int32_t>{2, 3}));
  EXPECT_EQ(middle->worker_cpus, (std::vector<int32_t>{8, 9}));
  EXPECT_EQ(last->worker_cpus, (std::vector<int32_t>{14, 15, 16, 17}));
  EXPECT_EQ(last->reserved_cpus, (std::vector<int32_t>{12, 13}));
  EXPECT_EQ(last->thread_cpus.at("acl_thread"), (std::vector<int32_t>{18}));
  EXPECT_EQ(last->thread_cpus.at("release_thread"), (std::vector<int32_t>{19}));
}

TEST(NpuCpuTopologyTest, A3AlwaysReservesIrqCpusAndKeepsGlobalRemainderPolicy) {
  CpuBindingTopology host;
  host.allowed_cpus = *parse_cpu_list("0-639");
  host.device_ids.resize(16);
  std::iota(host.device_ids.begin(), host.device_ids.end(), 0);
  const auto options = npu_cpu_binding_options(/*global_slice=*/true);
  const auto first = make_cpu_binding_plan(host, 0, options);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->worker_cpus, *parse_cpu_list("2-37"));
  EXPECT_EQ(first->reserved_cpus, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(first->thread_cpus.at("acl_thread"), (std::vector<int32_t>{38}));
  EXPECT_EQ(first->thread_cpus.at("release_thread"),
            (std::vector<int32_t>{39}));
  host.allowed_cpus = *parse_cpu_list("0-19");
  host.device_ids = {0, 1, 2};
  const auto remainder_first = make_cpu_binding_plan(host, 0, options);
  const auto remainder_last = make_cpu_binding_plan(host, 2, options);
  ASSERT_TRUE(remainder_first);
  ASSERT_TRUE(remainder_last);
  EXPECT_EQ(remainder_first->worker_cpus.size(), 3U);
  EXPECT_EQ(remainder_last->worker_cpus.size(), 2U);
  // Missing A2 topology falls back to the same balanced global slicing.
  const auto fallback = make_cpu_binding_plan(
      host, 0, npu_cpu_binding_options(/*global_slice=*/false));
  ASSERT_TRUE(fallback);
  EXPECT_EQ(fallback->mode, CpuBindingMode::GLOBAL_SLICE);
  EXPECT_EQ(fallback->worker_cpus, (std::vector<int32_t>{2, 3, 4}));
  host.allowed_cpus = *parse_cpu_list("0-13");
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
}

}  // namespace
}  // namespace xllm::npu
