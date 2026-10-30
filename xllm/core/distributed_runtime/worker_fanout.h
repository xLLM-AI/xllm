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

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <chrono>
#include <string>
#include <tuple>
#include <vector>

#include "common/types.h"
#include "core/framework/config/execution_config.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/util/env_var.h"

namespace xllm {

// Fans out one KV-cache capacity estimate per worker and collects their
// memory snapshots; a failed worker estimate is fatal.
template <typename WorkerRange>
std::vector<KVCacheMemorySnapshot> poll_worker_memory(
    const WorkerRange& workers) {
  CHECK(!workers.empty()) << "KV cache estimation requires workers";
  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(workers.size());
  for (auto& worker : workers) {
    futures.emplace_back(worker->estimate_kv_cache_capacity_async());
  }
  const auto results = folly::collectAll(futures).get();
  std::vector<KVCacheMemorySnapshot> worker_memory;
  worker_memory.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    CHECK(results[i].hasValue())
        << "Failed to estimate KV cache capacity for worker: " << i;
    const auto& [available_memory, total_memory] = results[i].value();
    worker_memory.emplace_back(
        KVCacheMemorySnapshot{available_memory, total_memory});
  }
  return worker_memory;
}

// Fans one model load out to every worker; true only when all succeed.
template <typename WorkerRange>
bool load_model(const WorkerRange& workers,
                const std::string& model_path,
                MasterStatus master_status) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers.size());
  for (auto& worker : workers) {
    futures.push_back(
        worker->init_model_async(model_path,
                                 ExecutionConfig::get_instance().random_seed(),
                                 master_status));
  }
  const auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

// Fans the process-group health check out to every worker and waits for all
// results with the configurable timeout (XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_
// SECONDS). Best-effort: exceptions propagate to the caller.
template <typename WorkerRange>
void fanout_process_group_test(const WorkerRange& workers) {
  if (workers.size() <= 1) {
    return;
  }
  std::vector<folly::SemiFuture<folly::Unit>> futures;
  futures.reserve(workers.size());
  for (auto& worker : workers) {
    futures.emplace_back(worker->process_group_test_async());
  }
  const auto timeout_seconds = util::get_process_group_test_timeout_seconds();
  folly::collectAll(futures)
      .within(std::chrono::seconds(timeout_seconds))
      .get();
}

}  // namespace xllm
