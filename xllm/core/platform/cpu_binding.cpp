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

#include <glog/logging.h>
#include <sched.h>

#include <unordered_set>
#include <utility>

namespace xllm {
namespace {

std::string cpu_list(const std::vector<int32_t>& cpus) {
  std::string result;
  for (int32_t cpu : cpus) {
    if (!result.empty()) {
      result += ',';
    }
    result += std::to_string(cpu);
  }
  return result;
}

bool validate_plan(const CpuBindingPlan& plan) {
  if (plan.worker_cpus.empty() ||
      (plan.memory_mode != CpuBindingMemoryMode::BIND_AT_STARTUP &&
       plan.memory_mode != CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP)) {
    return false;
  }
  std::unordered_set<int32_t> assigned;
  const auto add_cpus = [&assigned](const std::vector<int32_t>& cpus) {
    for (int32_t cpu : cpus) {
      if (cpu < 0 || cpu >= CPU_SETSIZE || !assigned.insert(cpu).second) {
        return false;
      }
    }
    return true;
  };
  if (!add_cpus(plan.worker_cpus) || !add_cpus(plan.reserved_cpus)) {
    return false;
  }
  for (const auto& [name, cpus] : plan.thread_cpus) {
    if (name.empty() || name.size() > 15 || cpus.empty() || !add_cpus(cpus)) {
      return false;
    }
  }
  return plan.memory_policy == numa::MemoryPolicy::BIND ||
         plan.memory_policy == numa::MemoryPolicy::PREFERRED;
}

}  // namespace

bool apply_cpu_binding_plan(const CpuBindingPlan& plan) {
  if (!validate_plan(plan) ||
      numa::bind_process_to_cpus(plan.worker_cpus, plan.thread_cpus) != 0) {
    return false;
  }
  LOG(INFO) << "CPU binding applied: device=" << plan.device_id << " mode="
            << (plan.mode == CpuBindingMode::GLOBAL_SLICE ? "global_slice"
                                                          : "topo_affinity")
            << " worker_cpus=" << cpu_list(plan.worker_cpus);
  for (const auto& [name, cpus] : plan.thread_cpus) {
    LOG(INFO) << "CPU binding role: " << name << " cpus=" << cpu_list(cpus);
  }
  return true;
}

CpuBinding& CpuBinding::get_instance() {
  static CpuBinding instance;
  return instance;
}

bool CpuBinding::initialize(CpuBindingPlan plan) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (plan_) {
    if (plan_->device_id != plan.device_id) {
      LOG(WARNING) << "CPU binding already belongs to device "
                   << plan_->device_id;
      return false;
    }
    return true;
  }
  if (!apply_cpu_binding_plan(plan)) {
    LOG(WARNING) << "CPU binding: failed to apply the worker plan";
    return false;
  }
  if (plan.memory_mode == CpuBindingMemoryMode::BIND_AT_STARTUP &&
      plan.memory_node >= 0 &&
      numa::bind_memory_to_numa_node(plan.memory_node, plan.memory_policy) ==
          0) {
    LOG(INFO) << "CPU binding memory node=" << plan.memory_node << " policy="
              << (plan.memory_policy == numa::MemoryPolicy::BIND ? "bind"
                                                                 : "preferred");
  }
  plan_ = std::move(plan);
  return true;
}

bool CpuBinding::initialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return plan_.has_value();
}

void CpuBinding::refresh_threads() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (plan_ && !apply_cpu_binding_plan(*plan_)) {
    LOG(WARNING) << "CPU binding: could not refresh thread placement";
  }
}

void CpuBinding::refresh_after_first_forward() {
  std::call_once(first_forward_, [this]() { refresh_threads(); });
}

void CpuBinding::finish_warmup() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!plan_ || warmup_finished_) {
    return;
  }
  warmup_finished_ = true;
  if (!apply_cpu_binding_plan(*plan_)) {
    LOG(WARNING) << "CPU binding: could not refresh placement after warmup";
  }
  if (plan_->memory_mode == CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP &&
      plan_->memory_node >= 0) {
    if (numa::migrate_process_memory_to_numa_node(plan_->memory_node) == 0) {
      LOG(INFO) << "CPU binding migrated process memory after warmup: node="
                << plan_->memory_node;
    }
  }
}

}  // namespace xllm
