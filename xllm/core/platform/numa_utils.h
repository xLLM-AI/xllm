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

#pragma once

#include <cstdint>
#include <vector>

namespace xllm {
namespace numa {

/**
 * @brief Check if NUMA is available on the current system
 * @return true if NUMA is available, false otherwise
 */
bool is_numa_available();

/**
 * @brief Get the number of NUMA nodes on the system
 * @return Number of NUMA nodes, or -1 if NUMA is not available
 */
int32_t get_num_numa_nodes();

/**
 * @brief Get the NUMA node ID for a given device index
 * @param device_index The backend-visible device index (e.g., CUDA/MLU ordinal)
 * @return The NUMA node ID, or -1 if unable to determine
 */
int32_t get_device_numa_node(int32_t device_index);

/**
 * @brief Bind current process to a specific NUMA node
 * @param numa_node The NUMA node ID to bind to
 * @return 0 on success, non-zero on failure
 */
int32_t bind_process_to_numa_node(int32_t numa_node);

/**
 * @brief Save current CPU affinity so spawned workers can restore it after
 * exec.
 * @return 0 on success, non-zero on failure
 */
int32_t preserve_current_cpu_affinity_for_spawn_worker();

/**
 * @brief Prepare spawn worker before binding it to a device-local NUMA node.
 * @return 0 on success, non-zero on failure
 */
int32_t prepare_spawn_worker_for_numa_binding();

/**
 * @brief Bind current thread to a specific NUMA node
 * @param numa_node The NUMA node ID to bind to
 * @return 0 on success, non-zero on failure
 */
int32_t bind_thread_to_numa_node(int32_t numa_node);

/**
 * @brief Get the NUMA node ID of the current process
 * @return The NUMA node ID, or -1 if unable to determine
 */
int32_t get_current_numa_node();

/**
 * @brief Get list of CPU cores for a given NUMA node
 * @param numa_node The NUMA node ID
 * @return Vector of CPU core IDs belonging to the NUMA node
 */
std::vector<int32_t> get_numa_node_cpus(int32_t numa_node);

}  // namespace numa
}  // namespace xllm
