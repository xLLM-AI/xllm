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

#include "core/platform/cpu_binding.h"

#include <gtest/gtest.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <memory>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace {

CpuBindingTopology topology(int32_t cpu_count, int32_t device_count) {
  CpuBindingTopology result;
  result.allowed_cpus.resize(cpu_count);
  std::iota(result.allowed_cpus.begin(), result.allowed_cpus.end(), 0);
  result.device_ids.resize(device_count);
  std::iota(result.device_ids.begin(), result.device_ids.end(), 0);
  for (int32_t cpu = 0; cpu < cpu_count; ++cpu) {
    result.cpu_nodes.emplace(cpu, cpu / 40);
  }
  return result;
}

CpuBindingOptions role_options(bool global_slice, bool reserve_cpus) {
  CpuBindingOptions options;
  options.mode = global_slice ? CpuBindingMode::GLOBAL_SLICE
                              : CpuBindingMode::TOPO_AFFINITY;
  options.extend_numa_pool = true;
  options.reserved_cpu_count = reserve_cpus ? 2 : 0;
  options.dedicated_threads = {"control", "cleanup"};
  return options;
}

std::optional<CpuBindingPlan> role_plan(const CpuBindingTopology& host,
                                        int32_t device_id,
                                        bool global_slice,
                                        bool reserve_cpus,
                                        std::string* error) {
  return make_cpu_binding_plan(
      host, device_id, role_options(global_slice, reserve_cpus), error);
}

std::vector<int32_t> all_cpus(const CpuBindingPlan& plan) {
  auto result = plan.worker_cpus;
  result.insert(
      result.end(), plan.reserved_cpus.begin(), plan.reserved_cpus.end());
  for (const auto& [name, cpus] : plan.thread_cpus) {
    result.insert(result.end(), cpus.begin(), cpus.end());
  }
  std::sort(result.begin(), result.end());
  return result;
}

TEST(CpuBindingTest, ParsesSparseCpuSet) {
  EXPECT_EQ(parse_cpu_list("8-10,2,4-5,4"),
            (std::vector<int32_t>{2, 4, 5, 8, 9, 10}));
}

TEST(CpuBindingTest, RejectsMalformedCpuSets) {
  for (const auto& text : {"",
                           "-1",
                           "4-2",
                           "1,",
                           "1,,2",
                           "1x",
                           "1-2-3",
                           "2147483648",
                           "0-1048576"}) {
    EXPECT_FALSE(parse_cpu_list(text)) << text;
  }
}

TEST(CpuBindingTest, AllocatesDisjointPoolsWithDedicatedRoles) {
  const auto host = topology(640, 16);
  std::unordered_set<int32_t> used;
  std::string error;
  for (int32_t id = 0; id < 16; ++id) {
    const auto plan = role_plan(host, id, true, true, &error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->mode, CpuBindingMode::GLOBAL_SLICE);
    EXPECT_EQ(plan->worker_cpus.size(), 36U);
    EXPECT_EQ(plan->worker_cpus.front(), id * 40 + 2);
    EXPECT_EQ(plan->worker_cpus.back(), id * 40 + 37);
    EXPECT_EQ(plan->reserved_cpus,
              (std::vector<int32_t>{id * 40, id * 40 + 1}));
    EXPECT_EQ(plan->thread_cpus.at("control")[0], id * 40 + 38);
    EXPECT_EQ(plan->thread_cpus.at("cleanup")[0], id * 40 + 39);
    EXPECT_EQ(plan->memory_node, id);
    for (int32_t cpu : all_cpus(*plan)) {
      EXPECT_TRUE(used.insert(cpu).second);
    }
  }
  EXPECT_EQ(used.size(), 640U);
}

TEST(CpuBindingTest, DoesNotReserveCpusWhenDisabled) {
  const auto plan = role_plan(topology(640, 16), 7, true, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus.size(), 38U);
  EXPECT_EQ(plan->worker_cpus.front(), 280);
  EXPECT_TRUE(plan->reserved_cpus.empty());
}

TEST(CpuBindingTest, SplitsSparseRestrictedCpusetAndRemainder) {
  auto host = topology(0, 3);
  host.allowed_cpus = {1, 3, 7, 9, 11, 14, 19, 23, 25, 27, 33};
  const auto first = role_plan(host, 0, true, false, nullptr);
  const auto last = role_plan(host, 2, true, false, nullptr);
  ASSERT_TRUE(first);
  ASSERT_TRUE(last);
  EXPECT_EQ(all_cpus(*first), (std::vector<int32_t>{1, 3, 7, 9}));
  EXPECT_EQ(all_cpus(*last), (std::vector<int32_t>{25, 27, 33}));
}

TEST(CpuBindingTest, SupportsNonContiguousLogicalInventory) {
  auto host = topology(12, 2);
  host.device_ids[0] = 4;
  host.device_ids[1] = 8;
  const auto plan = role_plan(host, 8, true, true, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(all_cpus(*plan), (std::vector<int32_t>{6, 7, 8, 9, 10, 11}));
}

TEST(CpuBindingTest, RejectsInsufficientCpusetBeforeBinding) {
  std::string error;
  EXPECT_FALSE(role_plan(topology(79, 16), 0, true, true, &error));
  EXPECT_NE(error.find("insufficient"), std::string::npos);
  EXPECT_FALSE(role_plan(topology(47, 16), 15, true, false, nullptr));
  EXPECT_FALSE(role_plan(topology(640, 16), 16, true, false, nullptr));
}

TEST(CpuBindingTest, PartitionsSharedAffinityAcrossAllDevices) {
  auto host = topology(80, 2);
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("0-39");
  const auto first = role_plan(host, 0, false, true, nullptr);
  const auto second = role_plan(host, 1, false, true, nullptr);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->mode, CpuBindingMode::TOPO_AFFINITY);
  EXPECT_EQ(all_cpus(*first), *parse_cpu_list("0-39"));
  EXPECT_EQ(all_cpus(*second), *parse_cpu_list("40-79"));
}

TEST(CpuBindingTest, RespectsNonContiguousNumaIds) {
  auto host = topology(80, 2);
  for (auto& entry : host.cpu_nodes) {
    entry.second = entry.first < 40 ? 2 : 7;
  }
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("0-39");
  const auto plan = role_plan(host, 1, false, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->memory_node, 7);
}

TEST(CpuBindingTest, RejectsIncompleteAndPartiallyOverlappingTopology) {
  auto host = topology(120, 3);
  host.affinity[0] = *parse_cpu_list("0-39");
  EXPECT_FALSE(role_plan(host, 0, false, true, nullptr));
  host.affinity[1] = *parse_cpu_list("40-79");
  host.affinity[2] = *parse_cpu_list("80-119");
  EXPECT_FALSE(role_plan(host, 0, false, true, nullptr));
}

TEST(CpuBindingTest, EmptyTopologyUsesGlobalPolicy) {
  const auto plan = role_plan(topology(80, 2), 1, false, true, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->mode, CpuBindingMode::GLOBAL_SLICE);
  EXPECT_EQ(all_cpus(*plan), *parse_cpu_list("40-79"));
}

TEST(CpuBindingTest, EmptyAffinityIntersectionDoesNotWidenCpuset) {
  auto host = topology(80, 2);
  host.allowed_cpus = *parse_cpu_list("40-79");
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("40-79");
  EXPECT_FALSE(role_plan(host, 0, false, false, nullptr));
  const auto plan = role_plan(host, 1, false, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(all_cpus(*plan), host.allowed_cpus);
}

TEST(CpuBindingTest, AppliesWorkerAndRuntimeRolesOnlyInChildProcess) {
  cpu_set_t parent_mask;
  ASSERT_EQ(sched_getaffinity(0, sizeof(parent_mask), &parent_mask), 0);
  std::vector<int32_t> cpus;
  cpus.reserve(CPU_COUNT(&parent_mask));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &parent_mask)) {
      cpus.emplace_back(cpu);
    }
  }
  if (cpus.size() < 3) {
    GTEST_SKIP() << "requires three allowed CPUs";
  }
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    std::atomic<int32_t> ready{0};
    std::atomic<bool> done{false};
    const auto runtime_thread = [&](const char* name) {
      pthread_setname_np(pthread_self(), name);
      ++ready;
      while (!done.load()) {
        std::this_thread::yield();
      }
    };
    std::thread acl(runtime_thread, "control");
    std::thread release(runtime_thread, "cleanup");
    while (ready.load() != 2) {
      std::this_thread::yield();
    }
    CpuBindingPlan plan;
    plan.worker_cpus = {cpus[0]};
    plan.thread_cpus["control"] = {cpus[1]};
    plan.thread_cpus["cleanup"] = {cpus[2]};
    bool success = apply_cpu_binding_plan(plan);
    cpu_set_t mask;
    success &= sched_getaffinity(0, sizeof(mask), &mask) == 0 &&
               CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[0], &mask);
    success &=
        pthread_getaffinity_np(acl.native_handle(), sizeof(mask), &mask) == 0 &&
        CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[1], &mask);
    success &= pthread_getaffinity_np(
                   release.native_handle(), sizeof(mask), &mask) == 0 &&
               CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[2], &mask);
    done = true;
    acl.join();
    release.join();
    _exit(success ? 0 : 1);
  }
  int child_status = 0;
  ASSERT_EQ(waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&parent_mask, &after));
}

TEST(CpuBindingTest, InvalidPlansLeaveAffinityUnchanged) {
  cpu_set_t before;
  ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
  CpuBindingPlan plan;
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  plan.worker_cpus = {CPU_SETSIZE};
  plan.thread_cpus["control"] = {0};
  plan.thread_cpus["cleanup"] = {1};
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  plan.worker_cpus = {0};
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
}

TEST(CpuBindingTest, DefaultNumaPolicyUsesWholePoolWithoutDedicatedThreads) {
  auto host = topology(80, 4);
  host.affinity[0] = host.affinity[1] = *parse_cpu_list("0-39");
  host.affinity[2] = host.affinity[3] = *parse_cpu_list("40-79");
  std::unordered_set<int32_t> assigned;
  for (int32_t id = 0; id < 4; ++id) {
    const auto plan = make_cpu_binding_plan(host, id, CpuBindingOptions{});
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->mode, CpuBindingMode::TOPO_AFFINITY);
    EXPECT_EQ(plan->worker_cpus.size(), 20U);
    EXPECT_EQ(plan->worker_cpus.front(), id * 20);
    EXPECT_TRUE(plan->thread_cpus.empty());
    EXPECT_TRUE(plan->reserved_cpus.empty());
    EXPECT_EQ(plan->memory_node, id / 2);
    EXPECT_EQ(plan->memory_policy, numa::MemoryPolicy::BIND);
    for (int32_t cpu : plan->worker_cpus) {
      EXPECT_TRUE(assigned.insert(cpu).second);
    }
  }
  EXPECT_EQ(assigned.size(), 80U);
}

TEST(CpuBindingTest, DefaultPolicySupportsOneCpuPerWorker) {
  const auto host = topology(2, 2);
  const auto plan = make_cpu_binding_plan(host, 1, CpuBindingOptions{});
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus, (std::vector<int32_t>{1}));
  EXPECT_TRUE(plan->thread_cpus.empty());
}

TEST(CpuBindingTest, NumaPoolsRespectRestrictedCpusetWithoutExtension) {
  auto host = topology(80, 2);
  host.allowed_cpus = {1, 3, 7, 9, 41, 43};
  host.affinity[0] = host.affinity[1] = *parse_cpu_list("0-39");
  const auto first = make_cpu_binding_plan(host, 0, CpuBindingOptions{});
  const auto second = make_cpu_binding_plan(host, 1, CpuBindingOptions{});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->worker_cpus, (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(second->worker_cpus, (std::vector<int32_t>{7, 9}));
  EXPECT_EQ(second->memory_node, 0);
}

TEST(CpuBindingTest, RejectsInvalidRoleConfigurationAndCpuIds) {
  auto host = topology(16, 2);
  CpuBindingOptions options;
  options.dedicated_threads = {"same", "same"};
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
  options.dedicated_threads = {""};
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
  options.dedicated_threads = {"thread_name_too_long"};
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
  options.dedicated_threads.clear();
  options.reserved_cpu_count = -1;
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
  options.reserved_cpu_count = 16;
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
  options.reserved_cpu_count = 0;
  host.allowed_cpus.emplace_back(CPU_SETSIZE);
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, options));
}

TEST(CpuBindingTest, ProcessLifecycleRefreshesLateThreads) {
  const auto cpus = numa::get_thread_cpus();
  if (cpus.size() < 2) {
    GTEST_SKIP() << "requires two allowed CPUs";
  }
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    auto& binding = CpuBinding::get_instance();
    CpuBindingPlan invalid;
    bool success = !binding.initialize(invalid) && !binding.initialized();
    CpuBindingPlan plan;
    plan.device_id = 7;
    plan.worker_cpus = {cpus[0]};
    plan.thread_cpus["control"] = {cpus[1]};
    success &= binding.initialize(std::move(plan)) && binding.initialized();
    // Initialization is idempotent and must retain the original plan.
    CpuBindingPlan different;
    different.device_id = 7;
    different.worker_cpus = {cpus[1]};
    success &= binding.initialize(std::move(different));
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    bool inherited = false;
    bool rebound = false;
    std::thread late([&]() {
      pthread_setname_np(pthread_self(), "control");
      inherited = numa::get_thread_cpus() == std::vector<int32_t>{cpus[0]};
      ready = true;
      while (!done.load()) {
        std::this_thread::yield();
      }
      rebound = numa::get_thread_cpus() == std::vector<int32_t>{cpus[1]};
    });
    while (!ready.load()) {
      std::this_thread::yield();
    }
    binding.refresh_after_first_forward();
    success &= numa::get_thread_cpus() == std::vector<int32_t>{cpus[0]};
    done = true;
    late.join();
    _exit(success && inherited && rebound ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(numa::get_thread_cpus(), cpus);
}

TEST(CpuBindingTest, TopologyRemainderPolicyDoesNotChangeGlobalSlices) {
  auto host = topology(20, 3);
  for (int32_t id : host.device_ids) {
    host.affinity[id] = host.allowed_cpus;
  }
  CpuBindingOptions options;
  const auto balanced = make_cpu_binding_plan(host, 0, options);
  ASSERT_TRUE(balanced);
  EXPECT_EQ(balanced->worker_cpus.size(), 7U);
  options.topology_remainder_to_last = true;
  const auto first = make_cpu_binding_plan(host, 0, options);
  const auto last = make_cpu_binding_plan(host, 2, options);
  ASSERT_TRUE(first);
  ASSERT_TRUE(last);
  EXPECT_EQ(first->worker_cpus, *parse_cpu_list("0-5"));
  EXPECT_EQ(last->worker_cpus, *parse_cpu_list("12-19"));
  options.mode = CpuBindingMode::GLOBAL_SLICE;
  const auto global = make_cpu_binding_plan(host, 0, options);
  ASSERT_TRUE(global);
  EXPECT_EQ(global->worker_cpus, balanced->worker_cpus);
}

int32_t page_node(void* page) {
  int node = -1;
  if (get_mempolicy(&node, nullptr, 0, page, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
    return -1;
  }
  return node;
}

void* allocate_page_on_node(int32_t node) {
  using NodeMask =
      std::unique_ptr<struct bitmask, decltype(&numa_bitmask_free)>;
  NodeMask mask(numa_allocate_nodemask(), numa_bitmask_free);
  if (!mask) {
    return MAP_FAILED;
  }
  numa_bitmask_clearall(mask.get());
  numa_bitmask_setbit(mask.get(), node);
  if (set_mempolicy(MPOL_BIND, mask->maskp, mask->size + 1) != 0) {
    return MAP_FAILED;
  }
  void* page = mmap(nullptr,
                    sysconf(_SC_PAGESIZE),
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0);
  if (page != MAP_FAILED) {
    *static_cast<volatile char*>(page) = 1;
  }
  if (set_mempolicy(MPOL_DEFAULT, nullptr, 0) != 0) {
    return MAP_FAILED;
  }
  return page;
}

TEST(CpuBindingTest, MigratesExistingPagesOnlyOnceAfterWarmup) {
  const auto cpus = numa::get_thread_cpus();
  const auto nodes = numa::get_cpu_numa_nodes();
  std::vector<int32_t> memory_nodes;
  if (numa_all_nodes_ptr != nullptr) {
    memory_nodes.reserve(numa_all_nodes_ptr->size);
    for (unsigned long node = 0; node < numa_all_nodes_ptr->size; ++node) {
      if (numa_bitmask_isbitset(numa_all_nodes_ptr, node)) {
        memory_nodes.emplace_back(static_cast<int32_t>(node));
      }
    }
  }
  if (memory_nodes.size() < 2 || cpus.empty() || !nodes.count(cpus[0])) {
    GTEST_SKIP() << "requires two NUMA nodes";
  }
  int mode = -1;
  if (get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0 && errno == EPERM) {
    GTEST_SKIP() << "memory policy syscalls are blocked";
  }
  const int32_t source = memory_nodes.front();
  const int32_t target = memory_nodes.back();
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    auto& binding = CpuBinding::get_instance();
    binding.finish_warmup();  // An early no-op must not consume completion.
    CpuBindingPlan plan;
    plan.device_id = 0;
    plan.worker_cpus = {cpus[0]};
    plan.memory_node = target;
    plan.memory_mode = CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP;
    void* page = allocate_page_on_node(source);
    if (page == MAP_FAILED || page_node(page) != source ||
        !binding.initialize(std::move(plan))) {
      _exit(1);
    }
    binding.refresh_after_first_forward();
    if (page_node(page) != source ||
        get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0 ||
        mode != MPOL_DEFAULT) {
      _exit(2);
    }
    binding.finish_warmup();
    if (page_node(page) != target ||
        get_mempolicy(&mode, nullptr, 0, nullptr, 0) != 0 ||
        mode != MPOL_DEFAULT) {
      _exit(3);
    }
    void* later_page = allocate_page_on_node(source);
    if (later_page == MAP_FAILED || page_node(later_page) != source) {
      _exit(4);
    }
    binding.finish_warmup();
    _exit(page_node(later_page) == source ? 0 : 5);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(numa::get_thread_cpus(), cpus);
}

}  // namespace
}  // namespace xllm
