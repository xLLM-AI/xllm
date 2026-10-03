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

#include "core/platform/numa_utils.h"

#include <gtest/gtest.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <climits>
#include <functional>
#include <memory>
#include <thread>

namespace xllm::numa {
namespace {

void expect_child_success(const std::function<bool()>& test) {
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    _exit(test() ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

bool memory_policy_is(int32_t expected_mode, int32_t node) {
  int mode = -1;
  using NodeMask =
      std::unique_ptr<struct bitmask, decltype(&numa_bitmask_free)>;
  NodeMask mask(numa_allocate_nodemask(), numa_bitmask_free);
  if (!mask) {
    return false;
  }
  const bool success =
      get_mempolicy(&mode, mask->maskp, mask->size + 1, nullptr, 0) == 0 &&
      mode == expected_mode && numa_bitmask_weight(mask.get()) == 1 &&
      numa_bitmask_isbitset(mask.get(), node);
  return success;
}

TEST(NumaUtilsTest, ThreadCpuQueryMatchesKernel) {
  cpu_set_t mask;
  ASSERT_EQ(sched_getaffinity(0, sizeof(mask), &mask), 0);
  const auto cpus = get_thread_cpus();
  EXPECT_EQ(cpus.size(), static_cast<size_t>(CPU_COUNT(&mask)));
  for (int32_t cpu : cpus) {
    EXPECT_TRUE(CPU_ISSET(cpu, &mask));
  }
  EXPECT_TRUE(get_thread_cpus(-1).empty());
}

TEST(NumaUtilsTest, InvalidCpuMasksDoNotChangeAffinity) {
  const auto before = get_thread_cpus();
  ASSERT_FALSE(before.empty());
  for (const auto& cpus :
       std::vector<std::vector<int32_t>>{{}, {-1}, {CPU_SETSIZE}}) {
    EXPECT_NE(bind_thread_to_cpus(cpus), 0);
    EXPECT_NE(bind_process_to_cpus(cpus), 0);
    EXPECT_EQ(get_thread_cpus(), before);
  }
  EXPECT_NE(bind_process_to_cpus({before[0]}, {{"special", {CPU_SETSIZE}}}), 0);
  EXPECT_EQ(get_thread_cpus(), before);
}

TEST(NumaUtilsTest, BindingCurrentThreadLeavesExistingThreadsUntouched) {
  const auto original = get_thread_cpus();
  if (original.size() < 2) {
    GTEST_SKIP() << "requires two allowed CPUs";
  }
  expect_child_success([&]() {
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    bool other_unchanged = false;
    std::thread other([&]() {
      ready = true;
      while (!done.load()) {
        std::this_thread::yield();
      }
      other_unchanged = get_thread_cpus() == original;
    });
    while (!ready.load()) {
      std::this_thread::yield();
    }
    const bool success = bind_thread_to_cpus({original[0]}) == 0 &&
                         get_thread_cpus() == std::vector<int32_t>{original[0]};
    done = true;
    other.join();
    return success && other_unchanged;
  });
  EXPECT_EQ(get_thread_cpus(), original);
}

TEST(NumaUtilsTest, ProcessBindingSupportsNamedRolesAndLateThreads) {
  const auto cpus = get_thread_cpus();
  if (cpus.size() < 3) {
    GTEST_SKIP() << "requires three allowed CPUs";
  }
  expect_child_success([&]() {
    const std::unordered_map<std::string, std::vector<int32_t>> roles{
        {"special", {cpus[1]}}, {"late", {cpus[2]}}};
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    bool special_bound = false;
    std::thread special([&]() {
      pthread_setname_np(pthread_self(), "special");
      ready = true;
      while (!done.load()) {
        std::this_thread::yield();
      }
      special_bound = get_thread_cpus() == std::vector<int32_t>{cpus[1]};
    });
    while (!ready.load()) {
      std::this_thread::yield();
    }
    bool success = bind_process_to_cpus({cpus[0]}, roles) == 0 &&
                   get_thread_cpus() == std::vector<int32_t>{cpus[0]};
    std::atomic<bool> late_ready{false};
    bool inherited = false;
    bool late_bound = false;
    std::thread late([&]() {
      pthread_setname_np(pthread_self(), "late");
      inherited = get_thread_cpus() == std::vector<int32_t>{cpus[0]};
      late_ready = true;
      while (!done.load()) {
        std::this_thread::yield();
      }
      late_bound = get_thread_cpus() == std::vector<int32_t>{cpus[2]};
    });
    while (!late_ready.load()) {
      std::this_thread::yield();
    }
    success &= bind_process_to_cpus({cpus[0]}, roles) == 0;
    done = true;
    special.join();
    late.join();
    return success && special_bound && inherited && late_bound;
  });
  EXPECT_EQ(get_thread_cpus(), cpus);
}

TEST(NumaUtilsTest, ProcessBindingRestoresAffinityAfterKernelRejectsMask) {
  const auto cpus = get_thread_cpus();
  ASSERT_FALSE(cpus.empty());
  int32_t absent_cpu = -1;
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    const std::string path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    if (access(path.c_str(), F_OK) != 0) {
      absent_cpu = cpu;
      break;
    }
  }
  if (absent_cpu < 0) {
    GTEST_SKIP() << "requires an absent CPU ID below CPU_SETSIZE";
  }
  expect_child_success([&]() {
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    bool restored = false;
    std::thread special([&]() {
      pthread_setname_np(pthread_self(), "special");
      ready = true;
      while (!done.load()) {
        std::this_thread::yield();
      }
      restored = get_thread_cpus() == cpus;
    });
    while (!ready.load()) {
      std::this_thread::yield();
    }
    const bool rejected =
        bind_process_to_cpus({cpus[0]}, {{"special", {absent_cpu}}}) != 0;
    const bool main_restored = get_thread_cpus() == cpus;
    done = true;
    special.join();
    return rejected && main_restored && restored;
  });
}

TEST(NumaUtilsTest, ExistingNodeInterfaceRespectsRestrictedAffinity) {
  const auto cpus = get_thread_cpus();
  const auto nodes = get_cpu_numa_nodes();
  if (cpus.empty() || !nodes.count(cpus[0])) {
    GTEST_SKIP() << "requires CPU NUMA topology";
  }
  const int32_t node = nodes.at(cpus[0]);
  expect_child_success([&]() {
    return bind_thread_to_cpus({cpus[0]}) == 0 &&
           get_numa_node_cpus(node) == std::vector<int32_t>{cpus[0]} &&
           bind_thread_to_numa_node(node) == 0 &&
           get_thread_cpus() == std::vector<int32_t>{cpus[0]};
  });
  EXPECT_EQ(get_thread_cpus(), cpus);
}

TEST(NumaUtilsTest, PreferredMemoryPolicyIsInheritedByNewThreads) {
  if (!is_numa_available()) {
    GTEST_SKIP() << "requires NUMA";
  }
  int initial_mode = -1;
  if (get_mempolicy(&initial_mode, nullptr, 0, nullptr, 0) != 0 &&
      errno == EPERM) {
    GTEST_SKIP() << "memory policy syscalls are blocked";
  }
  const int32_t node = get_current_numa_node();
  ASSERT_GE(node, 0);
  expect_child_success([&]() {
    if (bind_memory_to_numa_node(node, MemoryPolicy::PREFERRED) != 0 ||
        !memory_policy_is(MPOL_PREFERRED, node)) {
      return false;
    }
    bool inherited = false;
    std::thread child(
        [&]() { inherited = memory_policy_is(MPOL_PREFERRED, node); });
    child.join();
    return inherited;
  });
}

TEST(NumaUtilsTest, ExistingProcessInterfaceRetainsBindMemoryPolicy) {
  if (!is_numa_available()) {
    GTEST_SKIP() << "requires NUMA";
  }
  int initial_mode = -1;
  if (get_mempolicy(&initial_mode, nullptr, 0, nullptr, 0) != 0 &&
      errno == EPERM) {
    GTEST_SKIP() << "memory policy syscalls are blocked";
  }
  const auto cpus = get_thread_cpus();
  ASSERT_FALSE(cpus.empty());
  const auto nodes = get_cpu_numa_nodes();
  ASSERT_TRUE(nodes.count(cpus[0]));
  const int32_t node = nodes.at(cpus[0]);
  expect_child_success([&]() {
    return bind_thread_to_cpus({cpus[0]}) == 0 &&
           bind_process_to_numa_node(node) == 0 &&
           get_thread_cpus() == std::vector<int32_t>{cpus[0]} &&
           memory_policy_is(MPOL_BIND, node);
  });
}

TEST(NumaUtilsTest, InvalidMemoryNodesLeavePolicyUnchanged) {
  if (!is_numa_available()) {
    GTEST_SKIP() << "requires NUMA";
  }
  const int32_t node = get_current_numa_node();
  ASSERT_GE(node, 0);
  int mode = -1;
  if (get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0 && errno == EPERM) {
    GTEST_SKIP() << "memory policy syscalls are blocked";
  }
  expect_child_success([&]() {
    return bind_memory_to_numa_node(node, MemoryPolicy::PREFERRED) == 0 &&
           bind_memory_to_numa_node(-1) != 0 &&
           bind_memory_to_numa_node(INT_MAX) != 0 &&
           memory_policy_is(MPOL_PREFERRED, node);
  });
}

TEST(NumaUtilsTest, MigrationOnlyPreservesAllocationPolicy) {
  if (!is_numa_available()) {
    GTEST_SKIP() << "requires NUMA";
  }
  int mode = -1;
  if (get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0 && errno == EPERM) {
    GTEST_SKIP() << "memory policy syscalls are blocked";
  }
  const int32_t node = get_current_numa_node();
  ASSERT_GE(node, 0);
  expect_child_success([&]() {
    if (bind_memory_to_numa_node(node, MemoryPolicy::PREFERRED) != 0) {
      return false;
    }
    migrate_process_memory_to_numa_node(node);
    return memory_policy_is(MPOL_PREFERRED, node) &&
           migrate_process_memory_to_numa_node(-1) != 0 &&
           migrate_process_memory_to_numa_node(INT_MAX) != 0 &&
           memory_policy_is(MPOL_PREFERRED, node);
  });
}

}  // namespace
}  // namespace xllm::numa
