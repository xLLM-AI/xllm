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

#include "core/util/cpu_affinity.h"

#include <gtest/gtest.h>
#include <sched.h>

#include <algorithm>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

#include "core/framework/config/service_config.h"

#if defined(USE_NPU)
#include <acl/acl.h>

#include "core/platform/npu/startup.h"
#endif

namespace xllm {
namespace {

bool has_only_cpu(int32_t cpu) {
  cpu_set_t mask;
  return sched_getaffinity(0, sizeof(mask), &mask) == 0 &&
         CPU_COUNT(&mask) == 1 && CPU_ISSET(cpu, &mask);
}

TEST(ProcessCpuAffinityTest, SelectsGlobalRankAndHonorsDisable) {
  ServiceConfig config;
  config.cpu_affinity("[[0,1],[2,3]]");
  EXPECT_EQ(config.cpu_affinity_for_rank(1, 2), (std::vector<int32_t>{2, 3}));
  config.enable_cpu_affinity(false);
  EXPECT_TRUE(config.cpu_affinity_for_rank(1, 2).empty());
}

TEST(ProcessCpuAffinityTest, RejectsIncompleteRankMappingAndEmptyCpuLists) {
  ServiceConfig config;
  config.cpu_affinity("[[0,1]]");
  EXPECT_DEATH(config.cpu_affinity_for_rank(0, 2),
               "one CPU ID list per global node_rank");
  config.cpu_affinity("[[]]");
  EXPECT_DEATH(config.cpu_affinity_for_rank(0, 1), "nonempty CPU ID lists");
}

TEST(ProcessCpuAffinityTest, BindsExistingThreadsAndIsInheritedByNewThreads) {
  // Keep the test runner's own CPU placement unchanged.
  ASSERT_EXIT(
      {
        const int32_t cpu = CpuAffinity::get_available_cpu_ids().front();
        std::promise<void> ready;
        std::promise<void> inspect;
        auto inspect_signal = inspect.get_future();
        bool existing_bound = false;
        std::thread existing([&] {
          ready.set_value();
          inspect_signal.wait();
          existing_bound = has_only_cpu(cpu);
        });
        ready.get_future().wait();
        bind_process_to_cpus({cpu});
        inspect.set_value();
        existing.join();
        bool inherited = false;
        std::thread child([&] { inherited = has_only_cpu(cpu); });
        child.join();
        std::_Exit(existing_bound && inherited && has_only_cpu(cpu) ? 0 : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

#if defined(USE_NPU)
TEST(ProcessCpuAffinityTest,
     AutomaticPlacementUsesVisibleDevicesWithoutContext) {
  uint32_t count = 0;
  ASSERT_EQ(aclrtGetDeviceCount(&count), ACL_SUCCESS);
  ASSERT_GT(count, 0);
  const auto allowed = CpuAffinity::get_available_cpu_ids();
  const int32_t ranks = static_cast<int32_t>(count);
  for (int32_t rank = 0; rank < ranks; ++rank) {
    const auto cpus = default_npu_cpu_affinity(rank, ranks);
    ASSERT_FALSE(cpus.empty());
    for (int32_t cpu : cpus) {
      EXPECT_NE(std::find(allowed.begin(), allowed.end(), cpu), allowed.end());
    }
  }
  aclrtContext context = nullptr;
  EXPECT_NE(aclrtGetCurrentContext(&context), ACL_SUCCESS);
  EXPECT_EQ(context, nullptr);
}
#endif

}  // namespace
}  // namespace xllm
