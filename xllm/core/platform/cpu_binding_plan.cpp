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

#include <sched.h>

#include <algorithm>
#include <charconv>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "core/platform/cpu_binding.h"

namespace xllm {
namespace {

std::optional<int32_t> parse_id(std::string_view text) {
  int32_t value = -1;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size() ||
      value < 0) {
    return std::nullopt;
  }
  return value;
}

std::vector<int32_t> sorted_unique(std::vector<int32_t> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<int32_t> intersect(const std::vector<int32_t>& left,
                               const std::vector<int32_t>& right) {
  const auto sorted = sorted_unique(left);
  std::vector<int32_t> result;
  result.reserve(std::min(left.size(), right.size()));
  std::set_intersection(sorted.begin(),
                        sorted.end(),
                        right.begin(),
                        right.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<int32_t> slice(const std::vector<int32_t>& cpus,
                           size_t index,
                           size_t count,
                           bool remainder_to_last = false) {
  const size_t base = cpus.size() / count;
  const size_t extra = cpus.size() % count;
  if (remainder_to_last) {
    const size_t start = index * base;
    const size_t end = index + 1 == count ? cpus.size() : start + base;
    return {cpus.begin() + start, cpus.begin() + end};
  }
  const size_t start = index * base + std::min(index, extra);
  const size_t length = base + (index < extra ? 1 : 0);
  return {cpus.begin() + start, cpus.begin() + start + length};
}

}  // namespace

std::optional<std::vector<int32_t>> parse_cpu_list(const std::string& text) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::vector<int32_t> result;
  result.reserve(text.size());
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const size_t dash = item.find('-');
    const auto first = parse_id(item.substr(0, dash));
    const auto last =
        dash == std::string::npos ? first : parse_id(item.substr(dash + 1));
    // Bound expansion of malformed topology output before allocating memory.
    if (!first || !last || *last < *first || *last >= 1024 * 1024) {
      return std::nullopt;
    }
    for (int32_t cpu = *first; cpu <= *last; ++cpu) {
      result.emplace_back(cpu);
    }
  }
  if (text.back() == ',' || result.empty()) {
    return std::nullopt;
  }
  return sorted_unique(std::move(result));
}

std::optional<CpuBindingPlan> make_cpu_binding_plan(
    const CpuBindingTopology& topology,
    int32_t device_id,
    const CpuBindingOptions& options,
    std::string* error) {
  const auto fail =
      [error](const std::string& reason) -> std::optional<CpuBindingPlan> {
    if (error != nullptr) {
      *error = reason;
    }
    return std::nullopt;
  };
  const auto allowed = sorted_unique(topology.allowed_cpus);
  if (allowed.empty() || allowed.front() < 0 || allowed.back() >= CPU_SETSIZE ||
      topology.device_ids.empty()) {
    return fail("empty or invalid CPU/device inventory");
  }
  const auto ids = sorted_unique(topology.device_ids);
  const auto device = std::find(ids.begin(), ids.end(), device_id);
  if (device == ids.end() || ids.front() < 0 ||
      ids.size() != topology.device_ids.size()) {
    return fail("invalid or duplicate device ID");
  }
  if (options.reserved_cpu_count < 0 ||
      static_cast<size_t>(options.reserved_cpu_count) >= allowed.size() ||
      options.dedicated_threads.size() >= allowed.size() ||
      (options.mode != CpuBindingMode::GLOBAL_SLICE &&
       options.mode != CpuBindingMode::TOPO_AFFINITY) ||
      (options.memory_mode != CpuBindingMemoryMode::BIND_AT_STARTUP &&
       options.memory_mode != CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP) ||
      (options.memory_policy != numa::MemoryPolicy::BIND &&
       options.memory_policy != numa::MemoryPolicy::PREFERRED)) {
    return fail("invalid CPU binding options");
  }
  std::unordered_set<std::string> roles;
  for (const auto& name : options.dedicated_threads) {
    if (name.empty() || name.size() > 15 || !roles.insert(name).second) {
      return fail("invalid or duplicate thread role");
    }
  }
  const size_t reserved = static_cast<size_t>(options.reserved_cpu_count);
  const size_t dedicated = options.dedicated_threads.size();
  const size_t minimum = reserved + dedicated + 1;
  std::vector<int32_t> pool;
  CpuBindingMode mode = CpuBindingMode::GLOBAL_SLICE;
  if (options.mode == CpuBindingMode::GLOBAL_SLICE ||
      topology.affinity.empty()) {
    if (allowed.size() / ids.size() < minimum) {
      return fail("insufficient allowed CPUs for all devices");
    }
    pool =
        slice(allowed, static_cast<size_t>(device - ids.begin()), ids.size());
  } else {
    mode = CpuBindingMode::TOPO_AFFINITY;
    // Ordered groups provide deterministic allocation, including hidden
    // devices.
    std::map<std::vector<int32_t>, std::vector<int32_t>> groups;
    std::set<int32_t> nodes;
    for (const auto& entry : topology.cpu_nodes) {
      if (entry.second >= 0) {
        nodes.insert(entry.second);
      }
    }
    for (int32_t id : ids) {
      const auto affinity = topology.affinity.find(id);
      if (affinity == topology.affinity.end()) {
        return fail("incomplete topology affinity; cannot ensure isolation");
      }
      auto cpus = intersect(affinity->second, allowed);
      if (cpus.empty()) {
        if (id == device_id) {
          return fail("device affinity does not intersect the allowed cpuset");
        }
        continue;
      }
      std::set<int32_t> affinity_nodes;
      for (int32_t cpu : cpus) {
        const auto node = topology.cpu_nodes.find(cpu);
        if (node == topology.cpu_nodes.end()) {
          return fail("incomplete CPU NUMA map");
        }
        affinity_nodes.insert(node->second);
      }
      if (options.extend_numa_pool && affinity_nodes.size() == 1 &&
          nodes.size() > 1) {
        auto next = nodes.upper_bound(*affinity_nodes.begin());
        if (next == nodes.end()) {
          next = nodes.begin();
        }
        for (int32_t cpu : allowed) {
          const auto node = topology.cpu_nodes.find(cpu);
          if (node != topology.cpu_nodes.end() && node->second == *next) {
            cpus.emplace_back(cpu);
          }
        }
        cpus = sorted_unique(std::move(cpus));
      }
      groups[cpus].emplace_back(id);
    }
    std::unordered_set<int32_t> claimed;
    for (const auto& [cpus, members] : groups) {
      for (int32_t cpu : cpus) {
        if (!claimed.insert(cpu).second) {
          return fail("partially overlapping topology groups");
        }
      }
      const auto member = std::find(members.begin(), members.end(), device_id);
      if (member == members.end()) {
        continue;
      }
      if (cpus.size() / members.size() < minimum) {
        return fail("insufficient CPUs in the shared topology affinity group");
      }
      pool = slice(cpus,
                   static_cast<size_t>(member - members.begin()),
                   members.size(),
                   options.topology_remainder_to_last);
    }
  }
  if (pool.size() < minimum) {
    return fail("CPU pool is too small for configured thread roles");
  }
  CpuBindingPlan plan;
  plan.device_id = device_id;
  plan.mode = mode;
  plan.memory_policy = options.memory_policy;
  plan.memory_mode = options.memory_mode;
  const size_t worker_end = pool.size() - dedicated;
  plan.worker_cpus.assign(pool.begin() + reserved, pool.begin() + worker_end);
  plan.reserved_cpus.assign(pool.begin(), pool.begin() + reserved);
  for (size_t index = 0; index < dedicated; ++index) {
    plan.thread_cpus.emplace(options.dedicated_threads[index],
                             std::vector<int32_t>{pool[worker_end + index]});
  }
  const auto node = topology.cpu_nodes.find(pool.front());
  if (node != topology.cpu_nodes.end()) {
    plan.memory_node = node->second;
  }
  return plan;
}

}  // namespace xllm
