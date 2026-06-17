/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "common/macros.h"
#include "core/distributed_runtime/master.h"
#include "dist_manager.h"
#include "engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/eplb/eplb_manager.h"
#include "framework/eplb/eplb_policy.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/quant_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/worker.h"
#include "runtime/worker_client.h"
#include "runtime/xservice_client.h"
#include "util/threadpool.h"
namespace xllm {

class ModelLoader;

class LLMEngine : public Engine {
 public:
  // create an engine with the given devices
  LLMEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  virtual ~LLMEngine() = default;

  ForwardOutput step(std::vector<Batch>& batch) override;

  const runtime::Options& options() const { return options_; }

  bool init(MasterStatus master_status) override;

  void update_last_step_result(std::vector<Batch>& batch) override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const override;

  // P/D
  bool pull_kv_blocks(
      const int32_t src_dp_size,
      const int32_t src_dp_rank,
      const std::vector<uint64_t>& src_cluster_ids,
      const std::vector<std::string>& src_addrs,
      const std::vector<uint64_t>& src_blocks,
      const int32_t dst_dp_rank,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {}) override;

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  void transfer_kv_blocks(
      const uint32_t dp_rank,
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  void prefetch_from_storage(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<int32_t>> flag,
      std::vector<std::shared_ptr<std::atomic<uint32_t>>>* prefetch_results)
      override;

  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<std::string>& addrs,
                      std::vector<uint16_t>& ports) override;

  void get_xtensor_info(
      std::vector<size_t>& worker_free_phy_pages,
      std::unordered_map<std::string, std::vector<WeightSegment>>&
          model_weight_segments) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports,
                    const int32_t src_dp_size,
                    const int32_t src_kv_split_size = 1) override;

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports,
                      const int32_t src_dp_size,
                      const int32_t src_kv_split_size = 1) override;

  // P2P link for weight transfer - each worker links to one remote addr
  bool link_p2p(const std::vector<std::string>& remote_addrs) override;

  bool unlink_p2p(const std::vector<std::string>& remote_addrs) override;

  std::shared_ptr<DistManager> get_dist_manager() { return dist_manager_; };

  bool sleep(MasterStatus master_status) override;

  bool wakeup(const WakeupOptions& options) override;

  bool start_profile() override;

  bool stop_profile() override;

  // XTensor mode: get GlobalXTensor offsets for allocated blocks via RPC
  // Calls worker in the specified DP group to compute offsets
  bool get_xtensor_offsets_for_blocks(
      int32_t dp_rank,
      const std::vector<int32_t>& block_ids,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets) override;

 private:
  friend class SpeculativeEngine;
  // setup workers internal
  void setup_workers(const runtime::Options& options);
  bool init_model(MasterStatus master_status = MasterStatus::WAKEUP);
  int64_t get_effective_xtensor_weight_size(
      const ModelLoader& model_loader) const;
  KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(const KVCacheCapacity& kv_cache_cap);
  std::vector<ForwardInput> prepare_inputs(std::vector<Batch>& batch);
  void process_group_test();

 protected:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // quantization args
  QuantArgs quant_args_;

  // worker client which is used for call worker
  // The reason for adding a worker client is to unify the
  // access code for both local and remote workers, thereby
  // introducing an additional worker_client abstraction.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // config for kv cache
  int64_t n_local_kv_heads_ = 0;
  int64_t n_local_q_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t n_local_linear_v_heads_ = 0;
  int64_t n_local_linear_k_heads_ = 0;

  // common frequently used args
  uint32_t dp_size_ = 1;
  uint32_t cp_size_ = 1;
  uint32_t worker_clients_num_;
  uint32_t dp_local_tp_size_;
  uint32_t dp_local_size_;

  // For multi-node serving
  // engine brpc server, all workers connect to engine_server_,
  // engine_server_ will send a UniqueId for workers to
  // create process group. And workers send worker brpc server
  // address to engine, engine will create WorkerClient for each worker.
  // Engine call workers to step via these WorkerClients.
  std::shared_ptr<DistManager> dist_manager_ = nullptr;

  torch::Tensor expert_load_data_;
  std::unique_ptr<EplbManager> eplb_manager_ = nullptr;
  void process_eplb_data(
      const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results);

  // threadpool for handle forward_input in parallel.
  // Since the batch is created in every step,
  // the thread_pool is placed inside the engine so as to
  // avoid creating a new thread pool for each batch.
  // NOTE: Perhaps it can be optimized to create a global thread pool.
  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

  bool layer_forward_interrupted_ = false;

  // threadpool for link cluster
  std::unique_ptr<ThreadPool> link_threadpool_;
};

}  // namespace xllm
