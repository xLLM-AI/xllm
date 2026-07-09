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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace xllm {

class WorkerServer;

class SpawnWorkerServer final {
 public:
  explicit SpawnWorkerServer(const std::string& master_node_addr,
                             int32_t local_rank,
                             int32_t global_rank,
                             int32_t world_size,
                             int32_t device_idx,
                             int32_t num_decoding_tokens,
                             int32_t block_size,
                             int32_t max_tokens_per_batch,
                             int32_t max_seqs_per_batch,
                             bool enable_shm,
                             uint64_t input_shm_size,
                             uint64_t output_shm_size,
                             bool is_local,
                             bool enable_prefill_sp,
                             const std::string& task_type,
                             const std::string& worker_type,
                             bool enable_speculative_decode,
                             int32_t num_speculative_tokens,
                             const std::string& speculative_algorithm,
                             const std::string& communication_backend,
                             const std::string& npu_kernel_backend,
                             const std::string& rank_tablefile,
                             bool enable_graph,
                             bool enable_graph_mode_decode_no_padding,
                             bool enable_prefill_piecewise_graph,
                             int32_t max_tokens_for_graph_mode,
                             int64_t max_encoder_cache_size,
                             int32_t dp_size,
                             int32_t tp_size,
                             int32_t sp_size,
                             int32_t cfg_size);

  ~SpawnWorkerServer();

  void run();

  static void handle_signal(int signum);

 private:
  std::atomic<bool> done_{false};
  std::unique_ptr<WorkerServer> worker_server_;
};

}  // namespace xllm
