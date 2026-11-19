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

#pragma once

#include <folly/futures/Future.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/prefetch_result.h"
#include "runtime/forward_params.h"

namespace xllm {
class VlmForwardInput;
class ForwardInputEncoding;

class CommChannel {
 public:
  virtual ~CommChannel() = default;

  virtual bool hello() = 0;

  virtual bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) = 0;

  virtual bool get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              uint16_t& port) = 0;

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports) = 0;

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports) = 0;

  virtual bool link_p2p(const std::string& remote_addr) = 0;

  virtual bool unlink_p2p(const std::string& remote_addr) = 0;

  virtual bool init_model_async(const std::string& model_weights_path,
                                int32_t random_seed,
                                folly::Promise<bool>& promise,
                                MasterStatus master_status) = 0;

  virtual bool estimate_kv_cache_capacity(int64_t& available_memory,
                                          int64_t& total_memory) = 0;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) = 0;

  virtual void execute_llm_model_async(
      const LlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr) = 0;

  virtual void execute_dit_model_async(
      const DiTForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise) = 0;

  virtual void execute_vlm_model_async(
      const VlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr) = 0;

  virtual bool process_group_test() = 0;

  virtual bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) = 0;

  virtual void transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      folly::Promise<uint32_t>& promise) = 0;

  virtual void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) = 0;

  virtual bool wait_for_kv_blocks_transfer(uint64_t batch_id) = 0;

  virtual void prefetch_from_storage(const StoragePrefetchRequest& request,
                                     std::shared_ptr<PrefetchResult> result,
                                     size_t worker_index) = 0;

  virtual bool get_last_llm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise) = 0;

  virtual bool get_last_vlm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise) = 0;

  virtual bool check_health() = 0;

  virtual bool sleep(MasterStatus master_status,
                     uint64_t transition_id = 0) = 0;

  virtual bool wakeup(const WakeupOptions& options) = 0;

  virtual bool start_profile() = 0;

  virtual bool stop_profile() = 0;
};
}  // namespace xllm
