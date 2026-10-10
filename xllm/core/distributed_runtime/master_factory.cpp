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

#include "core/distributed_runtime/master_factory.h"

#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/kv_cache_config.h"

namespace xllm {

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  }
  if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  }
  if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  }
  if (backend == "rec") {
    LOG(INFO) << "creating rec master";
    return std::make_unique<RecMaster>(options);
  }

  LOG(FATAL) << "Failed to create master, backend is " << backend;
  return nullptr;
}

std::unique_ptr<LLMMaster> fork_llm_master(LLMMaster* master,
                                           const Options& options) {
  // sleep/wakeup/fork_master requires --enable_virtual_memory=true
  if (!KVCacheConfig::get_instance().enable_virtual_memory()) {
    LOG(WARNING) << "fork_master requires virtual memory to be enabled";
    return nullptr;
  }

  static std::atomic<int64_t> server_idx{1};
  CHECK(master != nullptr);

  Options new_options = master->options();

  if (!options.model_id().empty()) {
    new_options.model_id() = options.model_id();
  }
  if (!options.model_path().empty()) {
    new_options.model_path() = options.model_path();
  }
  new_options.master_node_addr() = options.master_node_addr();
  new_options.server_idx() = server_idx.fetch_add(1, std::memory_order_relaxed);
  new_options.master_status() = options.master_status();
  // Set nnodes and dp_size from fork request (tp_size * dp_size = nnodes)
  if (options.nnodes() > 0 && new_options.nnodes() >= options.nnodes()) {
    new_options.nnodes() = options.nnodes();
  }
  if (options.dp_size() > 0 && new_options.dp_size() >= options.dp_size()) {
    new_options.dp_size() = options.dp_size();
  }
  auto new_master = std::make_unique<LLMMaster>(new_options);
  new_master->run();
  return new_master;
}

}  // namespace xllm
