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

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/platform/numa_utils.h"

namespace xllm {

enum class CpuBindingMode { GLOBAL_SLICE, TOPO_AFFINITY };

// Migration after warmup leaves the process allocation policy unchanged.
enum class CpuBindingMemoryMode { BIND_AT_STARTUP, MIGRATE_AFTER_WARMUP };

struct CpuBindingTopology {
  std::vector<int32_t> allowed_cpus;
  // Stable IDs within the device inventory supplied by the platform.
  std::vector<int32_t> device_ids;
  std::unordered_map<int32_t, std::vector<int32_t>> affinity;
  std::unordered_map<int32_t, int32_t> cpu_nodes;
};

struct CpuBindingOptions {
  CpuBindingMode mode = CpuBindingMode::TOPO_AFFINITY;
  bool extend_numa_pool = false;
  bool topology_remainder_to_last = false;
  int32_t reserved_cpu_count = 0;
  // Assign one CPU per named thread role, in this order, at the pool's end.
  std::vector<std::string> dedicated_threads;
  numa::MemoryPolicy memory_policy = numa::MemoryPolicy::BIND;
  CpuBindingMemoryMode memory_mode = CpuBindingMemoryMode::BIND_AT_STARTUP;
};

struct CpuBindingPlan {
  int32_t device_id = -1;
  CpuBindingMode mode = CpuBindingMode::GLOBAL_SLICE;
  std::vector<int32_t> worker_cpus;
  std::vector<int32_t> reserved_cpus;
  std::unordered_map<std::string, std::vector<int32_t>> thread_cpus;
  int32_t memory_node = -1;
  numa::MemoryPolicy memory_policy = numa::MemoryPolicy::BIND;
  CpuBindingMemoryMode memory_mode = CpuBindingMemoryMode::BIND_AT_STARTUP;
};

std::optional<std::vector<int32_t>> parse_cpu_list(const std::string& text);
std::optional<CpuBindingPlan> make_cpu_binding_plan(
    const CpuBindingTopology& topology,
    int32_t device_id,
    const CpuBindingOptions& options,
    std::string* error = nullptr);

// Apply CPU placement only. Memory policy and platform-specific reserved CPUs
// are handled during initialization. Failed affinity updates attempt rollback.
bool apply_cpu_binding_plan(const CpuBindingPlan& plan);

// One worker per process. The platform supplies a plan before model loading;
// the common lifecycle reapplies it after runtime initialization and loading.
class CpuBinding final {
 public:
  static CpuBinding& get_instance();
  bool initialize(CpuBindingPlan plan);
  bool initialized() const;
  void refresh_threads();
  void refresh_after_first_forward();
  void finish_warmup();

 private:
  mutable std::mutex mutex_;
  std::once_flag first_forward_;
  bool warmup_finished_ = false;
  std::optional<CpuBindingPlan> plan_;
};

}  // namespace xllm
