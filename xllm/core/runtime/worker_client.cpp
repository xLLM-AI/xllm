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

#include "worker_client.h"

#include <folly/Unit.h>
#include <folly/executors/GlobalExecutor.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/params_utils.h"
#include "util/timer.h"

#if defined(USE_CUDA) || defined(USE_DCU)
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace xllm {

WorkerClient::WorkerClient(Worker* w, const runtime::Options& options)
    : worker_(w),
      options_(options),
      dispatch_threadpool_(
          std::make_unique<ThreadPool>(/*num_threads=*/1,
                                       /*cpu_binding=*/false,
                                       /*pool_name=*/"WorkerClient.dispatch")) {
  // Pin the dispatch thread to the worker's device so any CUDA driver calls
  // issued during prepare (cudaHostAlloc, stream guards, etc.) operate on
  // the right context without competing with the engine thread.
  if (worker_ != nullptr) {
    const Device dev = Device(worker_->device());
    dev.set_device();
  }
}

bool WorkerClient::init_model(const std::string& model_weights_path,
                              int32_t random_seed,
                              MasterStatus master_status) {
  return worker_->init_model(model_weights_path, random_seed, master_status);
}

bool WorkerClient::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  return worker_->allocate_kv_cache(kv_cache_shape);
}

void WorkerClient::get_cache_info(uint64_t& cluster_id,
                                  std::string& addr,
                                  uint16_t& port) {
  worker_->get_cache_info(cluster_id, addr, port);
}

bool WorkerClient::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<uint16_t>& ports) {
  return worker_->link_cluster(cluster_ids, addrs, ports);
}

bool WorkerClient::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<uint16_t>& ports) {
  return worker_->unlink_cluster(cluster_ids, addrs, ports);
}

bool WorkerClient::link_p2p(const std::string& remote_addr) {
  return worker_->link_p2p(remote_addr);
}

bool WorkerClient::unlink_p2p(const std::string& remote_addr) {
  return worker_->unlink_p2p(remote_addr);
}

std::tuple<int64_t, int64_t> WorkerClient::estimate_kv_cache_capacity() {
  return worker_->estimate_kv_cache_capacity();
}

bool WorkerClient::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  auto future = worker_->pull_kv_blocks_async(src_cluster_id,
                                              src_addr,
                                              src_blocks,
                                              dst_blocks,
                                              src_linear_state_ids,
                                              dst_linear_state_ids);
  return std::move(future).get();
}

ForwardInput WorkerClient::prepare_inputs(Batch& batch) {
  return worker_->prepare_inputs(batch);
}

std::optional<ForwardOutput> WorkerClient::step(const ForwardInput& inputs) {
  return worker_->step(inputs);
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
WorkerClient::estimate_kv_cache_capacity_async() {
  return worker_->estimate_kv_cache_capacity_async();
}

folly::SemiFuture<std::optional<ForwardOutput>> WorkerClient::step_async(
    const ForwardInput& input) {
  return worker_->step_async(input);
}

void WorkerClient::build_fake_overlap_output(
    const ForwardInput& input,
    RawForwardOutput& raw_output) const {
  // Mirror WorkerService::step's overlap branch fake-token construction.
  // The number of placeholder tokens equals the number of sampled sequences
  // for this step (decode + any prefill-sampled seqs), read from
  // sampling_params.sample_idxes. Packed engine inputs carry the layout in
  // the host buffer, so unpack first when sample_idxes is not materialized.
  int32_t num_samples = 0;
  if (input.sampling_params.sample_idxes.defined()) {
    num_samples =
        static_cast<int32_t>(input.sampling_params.sample_idxes.size(0));
  } else if (input.input_host_buffer_has_layout) {
    ForwardInput unpacked_input;
    const bool unpacked = detail::unpack_from_input_host_buffer(
        input, torch::Device(torch::kCPU), unpacked_input);
    if (unpacked && unpacked_input.sampling_params.sample_idxes.defined()) {
      num_samples = static_cast<int32_t>(
          unpacked_input.sampling_params.sample_idxes.size(0));
    }
  }

  raw_output.outputs.clear();
  raw_output.outputs.reserve(num_samples);
  for (int32_t i = 0; i < num_samples; ++i) {
    RawSampleOutput sample_output;
    RawToken token;
    // Negative 1-based index placeholder: -(i+1). On the next step,
    // update_input_by_last_step_output replaces it with next_tokens[i].
    token.id = -(static_cast<int64_t>(i) + 1);
    sample_output.tokens.emplace_back(std::move(token));
    raw_output.outputs.emplace_back(std::move(sample_output));
  }
  raw_output.prepared_layer_id = -1;
}

folly::SemiFuture<std::optional<RawForwardOutput>>
WorkerClient::step_remote_async(const ForwardInput& input) {
  // Single-node single-process path: dispatch the step on the worker and
  // convert the resulting ForwardOutput into a RawForwardOutput so callers
  // (e.g. LLMEngine::step) get the same shape they would receive from a
  // RemoteWorker over brpc.
  if (worker_ == nullptr) {
    LOG(FATAL) << "WorkerClient Method step_remote_async with ForwardInput "
                  "param is UnImplemented.";
    return folly::makeSemiFuture(std::optional<RawForwardOutput>(std::nullopt));
  }

  // Schedule-overlap path: the engine pipelines step N+1 while step N's GPU
  // work is still in flight, and fetches step N's real result on the next
  // iteration via get_last_step_result_async. We MUST NOT block the returned
  // future on the worker's forward completion here: LLMEngine::step does
  // collectAll(futures).get(), and the consumer that unblocks the worker's
  // producer-consumer cv (update_last_step_result -> get_last_step_result) is
  // only dispatched by the scheduler AFTER engine_->step() returns. Waiting on
  // the forward here would deadlock the worker's cv against the engine thread.
  //
  // Instead, mirror multi-process WorkerService::step: kick the forward off
  // fire-and-forget on the dispatch thread (it runs, records its result, and
  // satisfies the cv handshake against the separate get_last_step path), and
  // immediately resolve with a fake-token RawForwardOutput. The real tokens
  // are picked up next iteration.
  if (options_.enable_schedule_overlap()) {
    RawForwardOutput fake_output;
    build_fake_overlap_output(input, fake_output);
    dispatch_threadpool_->schedule([this, input]() mutable {
      worker_->step_async(input)
          .via(folly::getGlobalCPUExecutor())
          .thenValue([](std::optional<ForwardOutput>&& /*unused*/) {});
    });
    return folly::makeSemiFuture(
        std::optional<RawForwardOutput>(std::move(fake_output)));
  }

  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  // Non-overlap path: run the entire prepare+step kickoff on the per-worker
  // dispatch thread. WorkerImpl::step_async runs prepare_work_before_execute
  // synchronously on the calling thread, which can issue blocking CUDA driver
  // calls. If we ran it on the engine thread, worker N+1's prepare would not
  // start until worker N's prepare finished — meanwhile worker N's step kernel
  // is already busy-waiting on NCCL collectives that need worker N+1, which
  // deadlocks both GPUs. Dispatching here lets every worker's prepare run in
  // parallel on its own thread.
  dispatch_threadpool_->schedule([this,
                                  input,
                                  promise = std::move(promise)]() mutable {
    worker_->step_async(input)
        .via(folly::getGlobalCPUExecutor())
        .thenValue([promise = std::move(promise)](
                       std::optional<ForwardOutput>&& forward_output) mutable {
          if (!forward_output.has_value()) {
            promise.setValue(std::nullopt);
            return;
          }
          RawForwardOutput raw;
          forward_output_to_raw(forward_output.value(), raw);
          promise.setValue(std::optional<RawForwardOutput>(std::move(raw)));
        });
  });
  return future;
}

folly::SemiFuture<folly::Unit> WorkerClient::process_group_test_async() {
  return worker_->process_group_test_async();
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> WorkerClient::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  return worker_->init_model_async(
      model_weights_path, random_seed, master_status);
}

folly::SemiFuture<bool> WorkerClient::allocate_kv_cache_async(
    const KVCacheShape& kv_cache_shape) {
  return worker_->allocate_kv_cache_async(kv_cache_shape);
}

folly::SemiFuture<bool> WorkerClient::allocate_kv_cache_with_transfer_async(
    const KVCacheShape& kv_cache_shape) {
  return worker_->allocate_kv_cache_with_transfer_async(kv_cache_shape);
}

folly::SemiFuture<bool> WorkerClient::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  return worker_->pull_kv_blocks_async(src_cluster_id,
                                       src_addr,
                                       src_blocks,
                                       dst_blocks,
                                       src_linear_state_ids,
                                       dst_linear_state_ids);
}

folly::SemiFuture<uint32_t> WorkerClient::transfer_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  LOG(FATAL) << "WorkerClient Method transfer_kv_blocks with return "
                "folly::SemiFuture<uint32_t> is "
                "UnImplemented.";
  return folly::makeSemiFuture(uint32_t(0));
}

void WorkerClient::prefetch_from_storage(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    std::shared_ptr<std::atomic<int32_t>> flag,
    std::shared_ptr<std::atomic<uint32_t>> success_cnt) {
  NOT_IMPLEMENTED();
}

void WorkerClient::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  NOT_IMPLEMENTED();
}

folly::SemiFuture<bool> WorkerClient::sleep_async(MasterStatus master_status) {
  LOG(FATAL) << "WorkerClient Method sleep is UnImplemented.";
}

folly::SemiFuture<bool> WorkerClient::wakeup_async(
    const WakeupOptions& options) {
  return worker_->wakeup_async(options);
}

folly::SemiFuture<bool> WorkerClient::update_weights_async(
    const std::string& /*weights_path*/) {
  LOG(FATAL) << "WorkerClient Method update_weights is UnImplemented.";
}

folly::SemiFuture<bool> WorkerClient::start_profile_async() {
  return worker_->start_profile_async();
}

folly::SemiFuture<bool> WorkerClient::stop_profile_async() {
  return worker_->stop_profile_async();
}

const torch::Device& WorkerClient::device() const { return worker_->device(); }

folly::SemiFuture<std::optional<RawForwardOutput>>
WorkerClient::get_last_step_result_async() {
  // Same single-node single-process pattern as step_remote_async: bridge the
  // worker's ForwardOutput into the engine-facing RawForwardOutput shape.
  if (worker_ == nullptr) {
    return folly::makeSemiFuture(std::optional<RawForwardOutput>(std::nullopt));
  }
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  dispatch_threadpool_->schedule([this,
                                  promise = std::move(promise)]() mutable {
    worker_->get_last_step_result_async()
        .via(folly::getGlobalCPUExecutor())
        .thenValue([promise = std::move(promise)](
                       std::optional<ForwardOutput>&& forward_output) mutable {
          if (!forward_output.has_value()) {
            promise.setValue(std::nullopt);
            return;
          }
          RawForwardOutput raw;
          forward_output_to_raw(forward_output.value(), raw);
          promise.setValue(std::optional<RawForwardOutput>(std::move(raw)));
        });
  });
  return future;
}

folly::SemiFuture<std::optional<ForwardOutput>>
WorkerClient::get_last_step_result_single_process_async() {
  return worker_->get_last_step_result_async();
}

int64_t WorkerClient::get_active_activation_memory() {
  return worker_->get_active_activation_memory();
}

folly::SemiFuture<int64_t> WorkerClient::get_active_activation_memory_async() {
  return worker_->get_active_activation_memory_async();
}

}  // namespace xllm
