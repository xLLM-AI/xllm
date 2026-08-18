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

#include "numa_utils.h"

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
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
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

void apply_process_memory_policy(int32_t numa_node) {
  struct bitmask* node_mask = numa_allocate_nodemask();
  if (node_mask == nullptr) {
    LOG(WARNING) << "Failed to allocate NUMA node mask for memory policy";
    return;
  }

  numa_bitmask_clearall(node_mask);
  numa_bitmask_setbit(node_mask, numa_node);

  struct bitmask* old_mask = numa_get_membind();
  if (old_mask != nullptr) {
    long migrate_result = numa_migrate_pages(getpid(), old_mask, node_mask);
    if (migrate_result < 0) {
      LOG(WARNING) << "numa_migrate_pages failed: " << strerror(errno);
    }
    numa_free_nodemask(old_mask);
  }

  numa_set_membind(node_mask);
  numa_set_strict(1);
  numa_free_nodemask(node_mask);
}

}  // namespace

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

  cpu_set_t cpu_set;
  int32_t nr_cpus = 0;
  if (!build_cpu_set_for_numa_node(numa_node, &cpu_set, &nr_cpus)) {
    LOG(ERROR) << "No CPUs available on NUMA node " << numa_node
               << " after applying affinity constraints";
    return -1;
  }

  pid_t pid = getpid();
  if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpu_set) != 0) {
    LOG(ERROR) << "Failed to bind process to NUMA node " << numa_node << ": "
               << strerror(errno);
    return -1;
  }

  apply_process_memory_policy(numa_node);

  LOG(INFO) << "Successfully bound process " << pid << " to NUMA node "
            << numa_node << " with " << nr_cpus
            << " CPUs and strict NUMA memory policy";

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

  cpu_set_t cpu_set;
  int32_t nr_cpus = 0;
  if (!build_cpu_set_for_numa_node(numa_node, &cpu_set, &nr_cpus)) {
    LOG(ERROR) << "No CPUs available on NUMA node " << numa_node
               << " after applying affinity constraints";
    return -1;
  }

  pthread_t thread = pthread_self();
  int32_t ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpu_set);
  if (ret != 0) {
    LOG(ERROR) << "Failed to bind thread to NUMA node " << numa_node << ": "
               << strerror(ret);
    return -1;
  }

  LOG(INFO) << "Successfully bound thread to NUMA node " << numa_node
            << " with " << nr_cpus << " CPUs";

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

  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &cpu_set)) {
      cpus.push_back(cpu);
    }
  }

  return cpus;
}

}  // namespace numa
}  // namespace xllm
