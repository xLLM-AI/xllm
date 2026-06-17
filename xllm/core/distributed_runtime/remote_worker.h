/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <brpc/channel.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "comm_channel.h"
#include "common/macros.h"
#include "common/types.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/executor.h"
#include "runtime/forward_params.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/worker_client.h"
#include "util/threadpool.h"
#include "worker.pb.h"

namespace xllm {

class RemoteWorker : public WorkerClient {
 public:
  explicit RemoteWorker(int32_t global_rank,
                        const std::string& server_address,
                        const torch::Device& d,
                        std::unique_ptr<CommChannel> channel);
  virtual ~RemoteWorker() = default;

  bool wait_for_server_ready(const std::string& server_address);

  virtual bool init_model(const std::string& model_weights_path,
                          int32_t random_seed,
                          MasterStatus master_status) override;

  virtual std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  virtual bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              uint16_t& port) override;

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports) override;

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports) override;

  // P2P link for weight transfer
  virtual bool link_p2p(const std::string& remote_addr) override;
  virtual bool unlink_p2p(const std::string& remote_addr) override;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {}) override;

  // prepare input request
  virtual ForwardInput prepare_inputs(Batch& batch) override;

  virtual std::optional<ForwardOutput> step(
      const ForwardInput& inputs) override;

  virtual folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path,
      int32_t random_seed,
      MasterStatus master_status) override;

  virtual folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async() override;

  virtual folly::SemiFuture<bool> allocate_kv_cache_async(
      const KVCacheShape& kv_cache_shape) override;

  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      const KVCacheShape& kv_cache_shape) override;

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {}) override;

  virtual folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  virtual void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  virtual void prefetch_from_storage(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<int32_t>> flag,
      std::shared_ptr<std::atomic<uint32_t>> success_cnt) override;

  // Run the model and return the output.
  virtual folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs) override;

  virtual folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs) override;

  virtual folly::SemiFuture<folly::Unit> process_group_test_async() override;

  virtual const torch::Device& device() const override;

  folly::SemiFuture<std::optional<RawForwardOutput>>
  get_last_step_result_async();

  virtual int64_t get_active_activation_memory() override;

  virtual folly::SemiFuture<int64_t> get_active_activation_memory_async()
      override;

  // Check if the connection to worker is healthy
  bool check_health();

  // Get worker global rank
  int32_t global_rank() const { return global_rank_; }

  virtual folly::SemiFuture<bool> sleep_async(
      MasterStatus master_status) override;

  virtual folly::SemiFuture<bool> wakeup_async(
      const WakeupOptions& options) override;

  virtual folly::SemiFuture<bool> start_profile_async() override;

  virtual folly::SemiFuture<bool> stop_profile_async() override;

 private:
  DISALLOW_COPY_AND_ASSIGN(RemoteWorker);

 private:
  int32_t global_rank_;
  // connection resource
  std::unique_ptr<CommChannel> channel_;
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"RemoteWorker.request"};
  // copy working thread
  ThreadPool copy_threadpool_{/*num_threads=*/4,
                              /*cpu_binding=*/false,
                              /*pool_name=*/"RemoteWorker.copy"};
  const torch::Device device_;
};
}  // namespace xllm
