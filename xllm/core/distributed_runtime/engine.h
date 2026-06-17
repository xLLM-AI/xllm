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

#include <folly/futures/Future.h>

#include <unordered_map>

#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/options.h"

namespace xllm {
class Engine {
 public:
  virtual ~Engine() = default;

  virtual bool init() { return true; };

  virtual bool init(MasterStatus master_status) { return true; };

  // execute model with batch input
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;

  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // return the tokenizer
  virtual const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  // return the block manager
  virtual BlockManagerPool* block_manager_pool() const {
    auto p = reinterpret_cast<BlockManagerPool*>(kv_cache_manager_.get());
    if (!p) {
      LOG(FATAL) << "kv_cache_manager_ is not BlockManagerPool type!";
    }
    return p;
  }

  // return the model args
  virtual const ModelArgs& model_args() const { return args_; }

  // return the tokenizer args
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }

  // return the active activation memory
  virtual std::vector<int64_t> get_active_activation_memory() const = 0;

  // P/D
  virtual bool pull_kv_blocks(
      const int32_t src_dp_size,
      const int32_t src_dp_rank,
      const std::vector<uint64_t>& src_cluster_ids,
      const std::vector<std::string>& src_addrs,
      const std::vector<uint64_t>& src_blocks,
      const int32_t dst_dp_rank,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {}) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
    return {};
  };

  virtual void transfer_kv_blocks(
      const uint32_t dp_rank,
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    NOT_IMPLEMENTED();
  };

  virtual void prefetch_from_storage(
      const uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<int32_t>> flag,
      std::vector<std::shared_ptr<std::atomic<uint32_t>>>* prefetch_results) {
    NOT_IMPLEMENTED();
  };

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<uint16_t>& ports) {
    NOT_IMPLEMENTED();
  };

  // Get XTensor info for etcd registration (from dp group 0)
  // worker_free_phy_pages: free pages per worker
  // model_weight_segments: weight segments in GlobalXTensor per model
  virtual void get_xtensor_info(
      std::vector<size_t>& worker_free_phy_pages,
      std::unordered_map<std::string, std::vector<WeightSegment>>&
          model_weight_segments) {
    NOT_IMPLEMENTED();
  };

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports,
                            const int32_t src_dp_size,
                            const int32_t src_kv_split_size = 1) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports,
                              const int32_t src_dp_size,
                              const int32_t src_kv_split_size = 1) {
    NOT_IMPLEMENTED();
    return false;
  };

  // P2P link for weight transfer - each worker links to one remote addr.
  virtual bool link_p2p(const std::vector<std::string>& remote_addrs) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool unlink_p2p(const std::vector<std::string>& remote_addrs) {
    NOT_IMPLEMENTED();
    return false;
  };

  virtual bool sleep(MasterStatus master_status) {
    LOG(FATAL) << " sleep is not implemented!";
    return false;
  };

  virtual bool wakeup(const WakeupOptions& options) {
    LOG(FATAL) << " wakeup is not implemented!";
    return false;
  };

  // Start/stop online timeline profiling on all workers. CUDA only for now.
  virtual bool start_profile() {
    LOG(ERROR) << "start_profile is not implemented for this engine!";
    return false;
  };

  virtual bool stop_profile() {
    LOG(ERROR) << "stop_profile is not implemented for this engine!";
    return false;
  };

  // XTensor mode: get GlobalXTensor offsets for allocated blocks
  // Returns per-layer K/V offsets for each block
  // Output: offsets[layer_id] = {k_offsets, v_offsets}
  // dp_rank: Target DP rank to query (offsets come from workers in that DP
  // group)
  virtual bool get_xtensor_offsets_for_blocks(
      int32_t dp_rank,
      const std::vector<int32_t>& block_ids,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets) {
    return false;
  };

 protected:
  // model args
  ModelArgs args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // kv cache manager
  // support `block manager` and `xtensor manager` currently.
  std::unique_ptr<KVCacheManager> kv_cache_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
};
}  // namespace xllm
