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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "common/types.h"
#include "forward_params.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/executor.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"
#include "util/threadpool.h"

namespace xllm {

class Worker {
 public:
  Worker(const ParallelArgs& parallel_args,
         const torch::Device& device,
         const runtime::Options& options,
         WorkerType worker_type);

  ~Worker();

  // initialize model, cache manager. blocking call
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status);

  bool sleep(MasterStatus master_status);

  bool wakeup(const WakeupOptions& options);

  folly::SemiFuture<bool> wakeup_async(const WakeupOptions& options);

  // Start/stop online timeline profiling on this worker's device.
  bool start_profile();

  bool stop_profile();

  folly::SemiFuture<bool> start_profile_async();

  folly::SemiFuture<bool> stop_profile_async();

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();

  // allocate kv cache. blocking call
  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape);

  void get_cache_info(uint64_t& cluster_id, std::string& addr, uint16_t& port);

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports);

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports);

  // P2P link for weight transfer
  bool link_p2p(const std::string& remote_addr);
  bool unlink_p2p(const std::string& remote_addr);

  const bool is_driver();

  // prepare input for execution
  ForwardInput prepare_inputs(Batch& batch);

  std::optional<ForwardOutput> step(const ForwardInput& inputs);

  // initialize model, cache manager. async call
  folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path,
      int32_t random_seed,
      MasterStatus master_status);

  folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async();

  // initialize kv cache. async call
  folly::SemiFuture<bool> allocate_kv_cache_async(
      const KVCacheShape& kv_cache_shape);

  // initialize kv cache with kv cache transfer. async call
  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      const KVCacheShape& kv_cache_shape);

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {});

  virtual uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  virtual uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs);

  folly::SemiFuture<folly::Unit> process_group_test_async();

  const torch::Device& device() const;

  folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result_async();

  int64_t get_active_activation_memory();

  folly::SemiFuture<int64_t> get_active_activation_memory_async();

 private:
  WorkerImpl* impl_ = nullptr;
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"Worker.async"};
};

}  // namespace xllm
