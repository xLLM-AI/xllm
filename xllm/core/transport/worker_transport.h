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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/prefetch_result.h"
#include "runtime/forward_params.h"
#include "runtime/options.h"

namespace xllm {
class CommChannel;
class VlmForwardInput;
class ForwardInputEncoding;

// Owns worker communication resources and selects the model payload route.
class WorkerTransport final {
 public:
  WorkerTransport(const std::string& worker_address,
                  int32_t rank,
                  int32_t dp_local_tp_size,
                  const runtime::Options& options);
  ~WorkerTransport();
  WorkerTransport(const WorkerTransport&) = delete;
  WorkerTransport& operator=(const WorkerTransport&) = delete;

  bool hello();

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape);

  bool get_cache_info(uint64_t& cluster_id, std::string& addr, uint16_t& port);

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports);

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports);

  // P2P link for weight transfer
  bool link_p2p(const std::string& remote_addr);
  bool unlink_p2p(const std::string& remote_addr);

  bool init_model_async(const std::string& model_weights_path,
                        int32_t random_seed,
                        folly::Promise<bool>& promise,
                        MasterStatus master_status);

  bool estimate_kv_cache_capacity(int64_t& available_memory,
                                  int64_t& total_memory);

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings);

  void execute_llm_model_async(
      const LlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr);

  void execute_dit_model_async(
      const DiTForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  void execute_vlm_model_async(
      const VlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr);

  bool process_group_test();

  bool allocate_kv_cache_with_transfer(const KVCacheShape& kv_cache_shape);

  void transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      folly::Promise<uint32_t>& promise);

  void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  bool wait_for_kv_blocks_transfer(uint64_t batch_id);

  void prefetch_from_storage(const StoragePrefetchRequest& request,
                             std::shared_ptr<PrefetchResult> result,
                             size_t worker_index);

  bool get_last_llm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  bool get_last_vlm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  // Check if the connection to worker is healthy
  bool check_health();

  bool sleep(MasterStatus master_status, uint64_t transition_id = 0);

  bool wakeup(const WakeupOptions& options);

  bool start_profile();

  bool stop_profile();

 private:
  std::unique_ptr<CommChannel> channel_;
};
}  // namespace xllm
