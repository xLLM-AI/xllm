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
#include "runtime/worker.h"
#include "util/threadpool.h"

namespace xllm {

// Light client for call local thread worker,
// which just a wrapper of worker instance.
class WorkerClient {
 public:
  WorkerClient() = default;
  WorkerClient(Worker* w, const runtime::Options& options);
  virtual ~WorkerClient() = default;

  // initialize model, cache manager. blocking call
  virtual bool init_model(const std::string& model_weights_path,
                          int32_t random_seed,
                          MasterStatus master_status);

  virtual folly::SemiFuture<bool> sleep_async(MasterStatus master_status);

  virtual folly::SemiFuture<bool> wakeup_async(const WakeupOptions& options);

  virtual folly::SemiFuture<bool> update_weights_async(
      const std::string& weights_path);

  // Start/stop online timeline profiling on the underlying worker.
  virtual folly::SemiFuture<bool> start_profile_async();

  virtual folly::SemiFuture<bool> stop_profile_async();

  virtual std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();

  // allocate kv cache. blocking call
  virtual bool allocate_kv_cache(const KVCacheShape& kv_cache_shape);

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              uint16_t& port);

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports);

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports);

  // P2P link for weight transfer
  virtual bool link_p2p(const std::string& remote_addr);
  virtual bool unlink_p2p(const std::string& remote_addr);

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {});

  // prepare input for execution
  virtual ForwardInput prepare_inputs(Batch& batch);

  virtual std::optional<ForwardOutput> step(const ForwardInput& inputs);

  // initialize model, cache manager. async call
  virtual folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path,
      int32_t random_seed,
      MasterStatus master_status);

  virtual folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async();

  // allocate kv cache. async call
  virtual folly::SemiFuture<bool> allocate_kv_cache_async(
      const KVCacheShape& kv_cache_shape);

  // allocate kv cache with kv cache transfer. async call
  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      const KVCacheShape& kv_cache_shape);

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {});

  virtual folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info);

  virtual void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  virtual void prefetch_from_storage(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      std::shared_ptr<std::atomic<int32_t>> flag,
      std::shared_ptr<std::atomic<uint32_t>> success_cnt);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  virtual folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs);

  // Remote execution path: send ForwardInput transport payload and decode
  // worker response as RawForwardOutput for distributed engine handling.
  virtual folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs);

  virtual folly::SemiFuture<folly::Unit> process_group_test_async();

  virtual const torch::Device& device() const;

  virtual folly::SemiFuture<std::optional<RawForwardOutput>>
  get_last_step_result_async();

  virtual folly::SemiFuture<std::optional<ForwardOutput>>
  get_last_step_result_single_process_async();

  virtual int64_t get_active_activation_memory();

  virtual folly::SemiFuture<int64_t> get_active_activation_memory_async();

 private:
  // Build the fake-token RawForwardOutput that schedule-overlap expects in
  // place of the real (still in-flight) forward result. Mirrors the
  // multi-process WorkerService::step overlap branch: one output per sampled
  // sequence carrying a single negative placeholder token (id = -(i+1)),
  // which WorkerImpl::update_input_by_last_step_output replaces with the real
  // token of the previous step on the next iteration.
  void build_fake_overlap_output(const ForwardInput& input,
                                 RawForwardOutput& raw_output) const;

  Worker* worker_ = nullptr;  // not owend

  // Runtime options for the in-process worker. Used to read
  // enable_schedule_overlap() from the SAME source WorkerImpl branches on
  // (options_.enable_schedule_overlap_), so the client and worker never
  // disagree about whether the producer-consumer overlap pipeline is active.
  runtime::Options options_;

  // Dedicated single-thread pool for dispatching the in-process worker step
  // pipeline. The engine fans out one step per worker; without a per-worker
  // dispatch thread, the engine thread would synchronously serialise each
  // worker's `prepare_work_before_execute` (which can issue blocking CUDA
  // calls like `cudaHostAlloc`). With multi-device single-process NCCL the
  // first worker's step starts its collectives and busy-waits for the
  // partner; if the partner is still queued behind the engine-thread
  // prepare, the GPUs deadlock. Running prepare+dispatch on this dedicated
  // thread per WorkerClient lets all workers begin their steps in parallel.
  std::unique_ptr<ThreadPool> dispatch_threadpool_;
};

}  // namespace xllm
