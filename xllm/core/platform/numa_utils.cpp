/* Copyright 2025-2026 The xLLM Authors.

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

#if defined(USE_MUSA)
#include <musa_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#elif defined(USE_MLU)
#include <cnrt.h>
#elif defined(USE_DCU)
#include <hip/hip_runtime_api.h>
#endif
#include <glog/logging.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace xllm {
namespace numa {
namespace {

bool read_numa_node(const std::string& numa_path, int32_t* numa_node) {
  if (numa_node == nullptr) {
    return false;
  }
  std::ifstream numa_file(numa_path);
  if (!numa_file.is_open()) {
    return false;
  }
  if (!(numa_file >> *numa_node)) {
    return false;
  }
  numa_file.close();
  return true;
}

int32_t get_numa_node_from_sysfs(const std::string& backend,
                                 int32_t device_index,
                                 const std::string& pci_bus_id) {
  std::string numa_path = "/sys/bus/pci/devices/" + pci_bus_id + "/numa_node";
  LOG(INFO) << backend << " device " << device_index
            << " NUMA sysfs path: " << numa_path;

  int32_t numa_node = -1;
  if (read_numa_node(numa_path, &numa_node)) {
    if (numa_node < 0) {
      LOG(WARNING) << backend << " device " << device_index << " PCI "
                   << pci_bus_id
                   << " has no valid NUMA node in sysfs, skipping NUMA binding";
      return -1;
    }

    LOG(INFO) << backend << " device " << device_index << " PCI " << pci_bus_id
              << " maps to NUMA node " << numa_node;
    return numa_node;
  }

  LOG(WARNING) << "Failed to read NUMA node for " << backend << " device "
               << device_index << " from " << numa_path;
  return -1;
}

bool build_cpu_set_for_numa_node(int32_t numa_node,
                                 cpu_set_t* cpu_set,
                                 int32_t* nr_cpus) {
  if (cpu_set == nullptr || nr_cpus == nullptr) {
    return false;
  }

  CPU_ZERO(cpu_set);
  *nr_cpus = 0;

  struct bitmask* node_cpu_mask = numa_allocate_cpumask();
  if (node_cpu_mask == nullptr) {
    LOG(ERROR) << "Failed to allocate CPU mask for NUMA node " << numa_node;
    return false;
  }

  if (numa_node_to_cpus(numa_node, node_cpu_mask) < 0) {
    LOG(ERROR) << "Failed to query CPUs for NUMA node " << numa_node;
    numa_free_cpumask(node_cpu_mask);
    return false;
  }

  cpu_set_t current_affinity;
  CPU_ZERO(&current_affinity);
  const bool has_affinity_constraint =
      (sched_getaffinity(0, sizeof(cpu_set_t), &current_affinity) == 0);
  if (!has_affinity_constraint) {
    LOG(WARNING) << "Failed to get current process affinity: "
                 << strerror(errno) << ". Falling back to NUMA node CPU list.";
  }

  const int32_t nr_possible_cpus = numa_num_possible_cpus();
  for (int32_t cpu = 0; cpu < nr_possible_cpus; ++cpu) {
    if (!numa_bitmask_isbitset(node_cpu_mask, cpu)) {
      continue;
    }
    if (cpu >= CPU_SETSIZE) {
      continue;
    }
    if (has_affinity_constraint && !CPU_ISSET(cpu, &current_affinity)) {
      continue;
    }

    CPU_SET(cpu, cpu_set);
    ++(*nr_cpus);
  }

  numa_free_cpumask(node_cpu_mask);
  return (*nr_cpus > 0);
}

bool make_cpu_mask(const std::vector<int32_t>& cpus, cpu_set_t* mask) {
  CPU_ZERO(mask);
  if (cpus.empty()) {
    errno = EINVAL;
    return false;
  }
  for (int32_t cpu : cpus) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
      errno = EINVAL;
      return false;
    }
    CPU_SET(cpu, mask);
  }
  return true;
}

int32_t set_thread_affinity(int32_t tid, const cpu_set_t& mask) {
  if (sched_setaffinity(tid, sizeof(mask), &mask) != 0) {
    return -1;
  }
  cpu_set_t actual;
  if (sched_getaffinity(tid, sizeof(actual), &actual) != 0) {
    return -1;
  }
  if (!CPU_EQUAL(&actual, &mask)) {
    errno = EINVAL;
    return -1;
  }
  return 0;
}

bool is_valid_numa_node(int32_t node) {
  return node >= 0 && numa_all_nodes_ptr != nullptr &&
         static_cast<unsigned long>(node) < numa_all_nodes_ptr->size &&
         numa_bitmask_isbitset(numa_all_nodes_ptr, node);
}

}  // namespace

std::vector<int32_t> get_thread_cpus(int32_t tid) {
  cpu_set_t mask;
  if (tid < 0 || sched_getaffinity(tid, sizeof(mask), &mask) != 0) {
    return {};
  }
  std::vector<int32_t> cpus;
  cpus.reserve(CPU_COUNT(&mask));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask)) {
      cpus.emplace_back(cpu);
    }
  }
  return cpus;
}

int32_t bind_thread_to_cpus(const std::vector<int32_t>& cpus, int32_t tid) {
  cpu_set_t mask;
  if (tid < 0 || !make_cpu_mask(cpus, &mask)) {
    return -1;
  }
  cpu_set_t previous;
  if (sched_getaffinity(tid, sizeof(previous), &previous) != 0) {
    return -1;
  }
  if (set_thread_affinity(tid, mask) == 0) {
    return 0;
  }
  const int32_t error = errno;
  if (sched_setaffinity(tid, sizeof(previous), &previous) != 0 &&
      errno != ESRCH) {
    LOG(WARNING) << "Failed to restore CPU affinity for tid=" << tid;
  }
  LOG(WARNING) << "Failed to bind CPU affinity for tid=" << tid << ": "
               << strerror(error);
  errno = error;
  return -1;
}

int32_t bind_process_to_cpus(
    const std::vector<int32_t>& cpus,
    const std::unordered_map<std::string, std::vector<int32_t>>& thread_cpus) {
  cpu_set_t default_mask;
  if (!make_cpu_mask(cpus, &default_mask)) {
    return -1;
  }
  std::unordered_map<std::string, cpu_set_t> overrides;
  for (const auto& [name, role_cpus] : thread_cpus) {
    cpu_set_t mask;
    if (!make_cpu_mask(role_cpus, &mask)) {
      return -1;
    }
    overrides.emplace(name, mask);
  }
  struct ThreadAffinity {
    int32_t tid;
    cpu_set_t previous;
    cpu_set_t requested;
  };
  std::vector<ThreadAffinity> threads;
  threads.reserve(256);
  std::error_code error;
  auto directory =
      std::filesystem::directory_iterator("/proc/self/task", error);
  if (error) {
    return -1;
  }
  const std::filesystem::directory_iterator end;
  for (; directory != end; directory.increment(error)) {
    if (error) {
      return -1;
    }
    const auto& entry = *directory;
    const std::string name = entry.path().filename().string();
    int32_t tid = -1;
    const auto parsed =
        std::from_chars(name.data(), name.data() + name.size(), tid);
    if (parsed.ec != std::errc() || tid <= 0) {
      continue;
    }
    cpu_set_t previous;
    if (sched_getaffinity(tid, sizeof(previous), &previous) != 0) {
      if (errno == ESRCH) {
        continue;
      }
      return -1;
    }
    std::string thread_name;
    std::ifstream comm(entry.path() / "comm");
    std::getline(comm, thread_name);
    const auto role = overrides.find(thread_name);
    threads.emplace_back(ThreadAffinity{
        tid, previous, role == overrides.end() ? default_mask : role->second});
  }
  if (error || threads.empty()) {
    return -1;
  }
  for (const auto& thread : threads) {
    if (set_thread_affinity(thread.tid, thread.requested) == 0 ||
        errno == ESRCH) {
      continue;
    }
    const int32_t bind_error = errno;
    for (const auto& restore : threads) {
      if (sched_setaffinity(
              restore.tid, sizeof(restore.previous), &restore.previous) != 0 &&
          errno != ESRCH) {
        LOG(WARNING) << "Failed to restore CPU affinity for tid="
                     << restore.tid;
      }
    }
    LOG(WARNING) << "Failed to bind process thread tid=" << thread.tid << ": "
                 << strerror(bind_error) << "; attempted to restore affinity";
    errno = bind_error;
    return -1;
  }
  return 0;
}

std::unordered_map<int32_t, int32_t> get_cpu_numa_nodes() {
  std::unordered_map<int32_t, int32_t> nodes;
  if (!is_numa_available() || numa_all_cpus_ptr == nullptr) {
    return nodes;
  }
  const int32_t possible_cpus = numa_num_possible_cpus();
  for (int32_t cpu = 0; cpu < possible_cpus; ++cpu) {
    if (!numa_bitmask_isbitset(numa_all_cpus_ptr, cpu)) {
      continue;
    }
    const int32_t node = numa_node_of_cpu(cpu);
    if (node >= 0) {
      nodes.emplace(cpu, node);
    }
  }
  return nodes;
}

int32_t migrate_process_memory_to_numa_node(int32_t numa_node) {
  if (!is_numa_available() || !is_valid_numa_node(numa_node)) {
    LOG(WARNING) << "Cannot migrate memory to NUMA node " << numa_node;
    return -1;
  }
  using NodeMask =
      std::unique_ptr<struct bitmask, decltype(&numa_bitmask_free)>;
  NodeMask target(numa_allocate_nodemask(), numa_bitmask_free);
  if (!target) {
    return -1;
  }
  numa_bitmask_clearall(target.get());
  numa_bitmask_setbit(target.get(), numa_node);
  const long remaining =
      numa_migrate_pages(getpid(), numa_all_nodes_ptr, target.get());
  if (remaining != 0) {
    LOG(WARNING) << "NUMA memory migration incomplete: result=" << remaining
                 << (remaining < 0 ? std::string(" error=") + strerror(errno)
                                   : "");
    return -1;
  }
  return 0;
}

int32_t bind_memory_to_numa_node(int32_t numa_node, MemoryPolicy policy) {
  if (!is_numa_available() || !is_valid_numa_node(numa_node)) {
    LOG(WARNING) << "Cannot set memory policy for NUMA node " << numa_node;
    return -1;
  }
  using NodeMask =
      std::unique_ptr<struct bitmask, decltype(&numa_bitmask_free)>;
  NodeMask target(numa_allocate_nodemask(), numa_bitmask_free);
  if (!target) {
    return -1;
  }
  numa_bitmask_clearall(target.get());
  numa_bitmask_setbit(target.get(), numa_node);
  NodeMask source(numa_get_membind(), numa_bitmask_free);
  const int32_t mode =
      policy == MemoryPolicy::BIND ? MPOL_BIND : MPOL_PREFERRED;
  // Match libnuma's bitmap convention, including the highest node bit.
  if (set_mempolicy(mode, target->maskp, target->size + 1) != 0) {
    LOG(WARNING) << "NUMA memory policy unavailable: " << strerror(errno);
    return -1;
  }
  if (policy == MemoryPolicy::BIND) {
    numa_set_strict(1);
  }
  if (source) {
    const long remaining =
        numa_migrate_pages(getpid(), source.get(), target.get());
    if (remaining != 0) {
      LOG(WARNING) << "NUMA memory migration incomplete: result=" << remaining
                   << (remaining < 0 ? std::string(" error=") + strerror(errno)
                                     : "");
    }
  }
  return 0;
}

bool is_numa_available() {
  // C++11 guarantees thread-safe initialization for function-local statics.
  // NUMA availability is probed only on the first call and stored in
  // `available`; subsequent calls directly reuse the cached result.
  static const bool available = []() {
    bool is_avail = (numa_available() >= 0);
    if (!is_avail) {
      LOG(WARNING) << "NUMA is not available on this system";
    }
    return is_avail;
  }();
  return available;
}

int32_t get_num_numa_nodes() {
  if (!is_numa_available()) {
    return -1;
  }
  return numa_num_configured_nodes();
}

int32_t get_device_numa_node(int32_t device_index) {
  if (!is_numa_available()) {
    return -1;
  }

#if defined(USE_MUSA)
  char pci_bus_id[32] = {0};
  musaError_t ret =
      musaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_index);
  if (ret == musaSuccess) {
    return get_numa_node_from_sysfs("MUSA", device_index, pci_bus_id);
  }

  LOG(WARNING) << "Failed to query PCI bus ID for MUSA device " << device_index
               << " via musaDeviceGetPCIBusId: " << musaGetErrorString(ret)
               << ", skipping NUMA binding";
#elif defined(USE_CUDA)
  char pci_bus_id[32] = {0};
  cudaError_t ret =
      cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_index);
  if (ret == cudaSuccess) {
    return get_numa_node_from_sysfs("CUDA", device_index, pci_bus_id);
  }

  LOG(WARNING) << "Failed to query PCI bus ID for CUDA device " << device_index
               << " via cudaDeviceGetPCIBusId: " << cudaGetErrorString(ret)
               << ", skipping NUMA binding";
#elif defined(USE_MLU)
  char pci_bus_id[32] = {0};
  cnrtRet_t ret =
      cnrtDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_index);
  if (ret == cnrtSuccess) {
    return get_numa_node_from_sysfs("MLU", device_index, pci_bus_id);
  }

  LOG(WARNING) << "Failed to query PCI bus ID for MLU device " << device_index
               << " via cnrtDeviceGetPCIBusId: " << cnrtGetErrorStr(ret)
               << ", error code " << ret << ", skipping NUMA binding";
#elif defined(USE_DCU)
  char pci_bus_id[32] = {0};
  hipError_t ret =
      hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_index);
  if (ret == hipSuccess) {
    return get_numa_node_from_sysfs("DCU", device_index, pci_bus_id);
  }

  LOG(WARNING) << "Failed to query PCI bus ID for DCU device " << device_index
               << " via hipDeviceGetPCIBusId: " << hipGetErrorString(ret)
               << ", skipping NUMA binding";
#else
  LOG(WARNING) << "Device NUMA detection is not supported for this backend, "
               << "device index " << device_index << ", skipping NUMA binding";
#endif

  return -1;
}

int32_t bind_process_to_numa_node(int32_t numa_node) {
  if (!is_numa_available()) {
    LOG(WARNING) << "NUMA not available, skipping process binding";
    return -1;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node << ", valid range is [0, "
               << num_nodes - 1 << "]";
    return -1;
  }

  const auto cpus = get_numa_node_cpus(numa_node);
  if (bind_thread_to_cpus(cpus, getpid()) != 0) {
    return -1;
  }
  // Preserve the existing API: CPU binding success is returned even when the
  // container disallows memory-policy syscalls. The memory helper logs failure.
  bind_memory_to_numa_node(numa_node, MemoryPolicy::BIND);
  LOG(INFO) << "Bound process main thread to NUMA node " << numa_node
            << " with " << cpus.size() << " CPUs";

  return 0;
}

int32_t bind_thread_to_numa_node(int32_t numa_node) {
  if (!is_numa_available()) {
    LOG(WARNING) << "NUMA not available, skipping thread binding";
    return -1;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node << ", valid range is [0, "
               << num_nodes - 1 << "]";
    return -1;
  }

  const auto cpus = get_numa_node_cpus(numa_node);
  if (bind_thread_to_cpus(cpus) != 0) {
    return -1;
  }
  LOG(INFO) << "Bound current thread to NUMA node " << numa_node << " with "
            << cpus.size() << " CPUs";

  return 0;
}

int32_t get_current_numa_node() {
  if (!is_numa_available()) {
    return -1;
  }

  int32_t cpu = sched_getcpu();
  if (cpu < 0) {
    LOG(WARNING) << "Failed to get current CPU";
    return -1;
  }

  return numa_node_of_cpu(cpu);
}

std::vector<int32_t> get_numa_node_cpus(int32_t numa_node) {
  std::vector<int32_t> cpus;

  if (!is_numa_available()) {
    return cpus;
  }

  int32_t num_nodes = get_num_numa_nodes();
  if (numa_node < 0 || numa_node >= num_nodes) {
    LOG(ERROR) << "Invalid NUMA node " << numa_node;
    return cpus;
  }

  cpu_set_t cpu_set;
  int32_t nr_cpus = 0;
  if (!build_cpu_set_for_numa_node(numa_node, &cpu_set, &nr_cpus)) {
    return cpus;
  }

  cpus.reserve(nr_cpus);
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &cpu_set)) {
      cpus.emplace_back(cpu);
    }
  }

  return cpus;
}

}  // namespace numa
}  // namespace xllm
