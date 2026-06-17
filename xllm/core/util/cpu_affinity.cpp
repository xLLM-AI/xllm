/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/util/cpu_affinity.h"

#include <glog/logging.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>

namespace xllm {

namespace {

// Capture the process-wide initial CPU affinity (i.e. the cpuset / numactl
// upper bound) exactly once. Linux CPU affinity is per-thread, so
// `sched_getaffinity(0, ...)` returns the calling thread's mask. Once a
// worker thread has been pinned to a single core, any thread it spawns
// inherits that narrow mask, which makes the naive
// `CPU_ISSET(cpu_core, sched_getaffinity(0))` check wrongly reject any core
// outside that single-core set. We therefore snapshot the mask once from
// the main thread (via `CpuAffinity::set_cpu_affinity`) and reuse it.
const cpu_set_t& initial_process_cpu_set() {
  static const cpu_set_t mask = []() {
    cpu_set_t m;
    CPU_ZERO(&m);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &m) != 0) {
      LOG(WARNING) << "Failed to capture initial process CPU affinity: "
                   << strerror(errno) << ". Assuming all CPUs are allowed.";
      for (int32_t i = 0; i < CPU_SETSIZE; ++i) {
        CPU_SET(i, &m);
      }
    }
    return m;
  }();
  return mask;
}

}  // namespace

int32_t bind_thread_to_cpu_core(int32_t cpu_core) {
  if (cpu_core < 0 || cpu_core >= CPU_SETSIZE) {
    LOG(ERROR) << "Invalid CPU core " << cpu_core << ", valid range is [0, "
               << CPU_SETSIZE - 1 << "]";
    return -1;
  }

  const cpu_set_t& allowed = initial_process_cpu_set();
  if (!CPU_ISSET(cpu_core, &allowed)) {
    LOG(ERROR) << "CPU core " << cpu_core
               << " is not in the initial process affinity set";
    return -1;
  }

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_core, &cpu_set);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) !=
      0) {
    LOG(ERROR) << "Failed to bind thread to CPU core " << cpu_core << ": "
               << strerror(errno);
    return -1;
  }

  LOG(INFO) << "=== Successfully bound thread to CPU core " << cpu_core;
  return 0;
}

bool CpuAffinity::parse_cpu_affinity_string(const std::string& affinity_str,
                                            std::vector<int32_t>& out_ids) {
  LOG(INFO) << "==== affinity_str " << affinity_str;
  out_ids.clear();
  if (affinity_str.empty()) {
    return true;
  }

  std::string str = affinity_str;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  if (str.empty()) {
    return true;
  }

  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      return false;
    }

    size_t dash_pos = token.find('-');
    if (dash_pos != std::string::npos) {
      std::string start_str = token.substr(0, dash_pos);
      std::string end_str = token.substr(dash_pos + 1);
      if (start_str.empty() || end_str.empty()) {
        return false;
      }

      int32_t start, end;
      try {
        start = std::stoi(start_str);
        end = std::stoi(end_str);
      } catch (...) {
        return false;
      }

      if (start > end || start < 0 || end < 0) {
        return false;
      }

      for (int32_t id = start; id <= end; ++id) {
        out_ids.push_back(id);
      }
    } else {
      int32_t id;
      try {
        id = std::stoi(token);
      } catch (...) {
        return false;
      }
      if (id < 0) return false;
      out_ids.push_back(id);
    }
  }

  return true;
}

void CpuAffinity::set_cpu_affinity(const std::string& cpu_affinity) {
  LOG(INFO) << "Setting CPU affinity to " << cpu_affinity;

  // Warm up the initial process affinity cache while we are still on the
  // main thread (and before any worker has narrowed its own affinity), so
  // that `bind_thread_to_cpu_core` later sees the full numactl/cpuset mask.
  (void)initial_process_cpu_set();

  std::lock_guard<std::mutex> lock(mu_);

  std::vector<int32_t> cpu_ids;
  if (parse_cpu_affinity_string(cpu_affinity, cpu_ids)) {
    std::swap(cpu_cores_, cpu_ids);
  } else {
    cpu_cores_.clear();
  }

  next_index_ = 0;
}

int32_t CpuAffinity::get_available_cpu_cores_count() {
  return CPU_COUNT(&initial_process_cpu_set());
}

std::vector<int32_t> CpuAffinity::get_available_cpu_ids() {
  std::vector<int32_t> cpu_ids;
  cpu_ids.reserve(CPU_SETSIZE);

  auto cpu_set = initial_process_cpu_set();
  for (int32_t i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpu_set)) {
      cpu_ids.push_back(i);
    }
  }

  return cpu_ids;
}

void CpuAffinity::clear_cpu_affinity() {
  std::lock_guard<std::mutex> lock(mu_);

  cpu_cores_.clear();
  next_index_ = 0;
}

int32_t CpuAffinity::next_cpu_core() {
  std::lock_guard<std::mutex> lock(mu_);

  if (cpu_cores_.empty()) {
    return -1;
  }

  int32_t cpu_core = cpu_cores_[next_index_];
  next_index_ = (next_index_ + 1) % cpu_cores_.size();

  return cpu_core;
}

CpuAffinityThreadFactory::CpuAffinityThreadFactory(folly::StringPiece prefix,
                                                   bool cpu_binding)
    : folly::NamedThreadFactory(prefix), cpu_binding_(cpu_binding) {}

std::thread CpuAffinityThreadFactory::newThread(folly::Func&& func) {
  int32_t cpu_core = CpuAffinity::get_instance().next_cpu_core();
  bool cpu_binding = cpu_binding_;
  LOG(INFO) << "Creating thread with CPU affinity " << cpu_core;

  return folly::NamedThreadFactory::newThread(
      [cpu_core, cpu_binding, func = std::move(func)]() mutable {
        if (cpu_binding && cpu_core >= 0 &&
            bind_thread_to_cpu_core(cpu_core) != 0) {
          LOG(WARNING) << "Failed to bind thread to CPU core " << cpu_core
                       << ", running unbound";
        }
        func();
      });
}

}  // namespace xllm
