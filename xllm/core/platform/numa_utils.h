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

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace numa {

// BIND restricts allocation to the selected node; PREFERRED permits fallback.
enum class MemoryPolicy { BIND, PREFERRED };

// CPU affinity is a per-thread property. A zero tid addresses the caller.
// These CPU operations do not require NUMA availability.
std::vector<int32_t> get_thread_cpus(int32_t tid = 0);
int32_t bind_thread_to_cpus(const std::vector<int32_t>& cpus, int32_t tid = 0);

// Bind all existing threads in this process, with optional overrides keyed by
// Linux thread name. Validate all masks first and restore prior masks on error.
// New threads inherit their creator's mask; callers can refresh after startup.
int32_t bind_process_to_cpus(
    const std::vector<int32_t>& cpus,
    const std::unordered_map<std::string, std::vector<int32_t>>& thread_cpus =
        {});

// Host CPU-to-NUMA topology, independent of any accelerator's device numbering.
std::unordered_map<int32_t, int32_t> get_cpu_numa_nodes();

// Migrate existing process pages from all NUMA nodes, without changing the
// calling thread's allocation policy. Returns zero only for complete migration;
// unavailable migration or remaining pages are logged and return nonzero.
int32_t migrate_process_memory_to_numa_node(int32_t numa_node);

// Set the calling thread's allocation policy and attempt to migrate existing
// process pages. Child threads inherit the policy. Migration is best effort;
// return nonzero if the policy itself cannot be installed.
int32_t bind_memory_to_numa_node(int32_t numa_node,
                                 MemoryPolicy policy = MemoryPolicy::BIND);

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
 * @brief Bind the process main thread to a NUMA node and request BIND memory
 * policy on the calling thread. Memory placement is best effort.
 * @param numa_node The NUMA node ID to bind to
 * @return 0 on success, non-zero on failure
 */
int32_t bind_process_to_numa_node(int32_t numa_node);

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
