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

#pragma once

#include <folly/Executor.h>
#include <folly/Range.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "utils.h"

namespace xllm {

// Singleton class to manage CPU affinity
class CpuAffinity final {
 public:
  static CpuAffinity& get_instance() {
    static CpuAffinity instance;

    return instance;
  }

  void set_cpu_affinity(const std::string& cpu_affinity);

  void clear_cpu_affinity();

  // Get the number of available CPU cores.
  static int32_t get_available_cpu_cores_count();

  // Get the list of available CPU IDs.
  static std::vector<int32_t> get_available_cpu_ids();

  int32_t next_cpu_core();

  size_t total_cpu_cores() { return cpu_cores_.size(); }

 private:
  std::mutex mu_;
  std::vector<int32_t> cpu_cores_;
  int32_t next_index_{0};

  bool parse_cpu_affinity_string(const std::string& affinity_str,
                                 std::vector<int32_t>& out_ids);
};

// Bind the calling thread to the given CPU core. The core must be in the
// current process affinity mask. Returns 0 on success and -1 on failure.
int32_t bind_thread_to_cpu_core(int32_t cpu_core);

// A folly::ThreadFactory that names worker threads (via NamedThreadFactory)
// and binds each newly created thread to a CPU core from `cpu_cores`.
// Cores are dispatched round-robin in the order they appear in `cpu_cores`.
// When `cpu_cores` is empty, no binding is performed.
class CpuAffinityThreadFactory final : public folly::NamedThreadFactory {
 public:
  CpuAffinityThreadFactory(folly::StringPiece prefix, bool cpu_binding = false);

  std::thread newThread(folly::Func&& func) override;

 private:
  bool cpu_binding_;
};

}  // namespace xllm
