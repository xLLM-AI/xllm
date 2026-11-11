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
#include <vector>

#include "framework/request/sequence.h"

namespace xllm {

class KVCacheManager;

class SchedulerMetrics final {
 public:
  SchedulerMetrics(KVCacheManager* kv_cache_manager,
                   int32_t dp_size,
                   int32_t num_speculative_tokens,
                   bool collect_recent_latency);

  SchedulerMetrics(const SchedulerMetrics&) = delete;
  SchedulerMetrics& operator=(const SchedulerMetrics&) = delete;

  void update(std::vector<Sequence*>& sequences);
  void update_token_latency_metrics(std::vector<Sequence*>& sequences);
  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt);

  // round(latency / num_tokens) via (latency + num_tokens / 2) / num_tokens.
  // num_tokens must be > 0.
  static int64_t amortized_token_latency(int64_t latency, size_t num_tokens);

 private:
  friend class SchedulerMetricsTestPeer;

  void update_token_latency_metrics_impl(
      const std::vector<Sequence*>& sequences);
  void update_memory_metrics(const std::vector<Sequence*>& sequences);
  std::vector<int64_t> get_num_occupied_slots(
      const std::vector<Sequence*>& sequences) const;

  KVCacheManager* kv_cache_manager_;
  int32_t dp_size_;
  int32_t num_speculative_tokens_;
  bool collect_recent_latency_;
  std::vector<int64_t> recent_ttft_;
  std::vector<int64_t> recent_tbt_;
  std::mutex latency_metrics_mutex_;
};

}  // namespace xllm
