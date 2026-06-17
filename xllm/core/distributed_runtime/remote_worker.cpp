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

#include "remote_worker.h"

#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <optional>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/service_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/params_utils.h"
#include "util/hash_util.h"

namespace xllm {

RemoteWorker::RemoteWorker(int32_t global_rank,
                           const std::string& server_address,
                           const torch::Device& d,
                           std::unique_ptr<CommChannel> channel)
    : global_rank_(global_rank), device_(d), channel_(std::move(channel)) {
  CHECK(wait_for_server_ready(server_address))
      << "Failed to wait for remote worker server ready: " << server_address
      << ", global_rank: " << global_rank_;
}

bool RemoteWorker::wait_for_server_ready(const std::string& server_address) {
  // Retry until server initialize ready
  int try_count = 0;
  const int sleep_time_second = 3;
  while (try_count <
         ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    if (channel_->hello()) {
      LOG(INFO) << "RemoteWorker Hello connected, server_address: "
                << server_address << ", global_rank_: " << global_rank_;
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    }

    try_count++;
  }

  if (try_count >=
      ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    LOG(ERROR) << "RemoteWorker Hello method failed, global_rank_ is "
               << global_rank_;
    return false;
  }

  return true;
}

bool RemoteWorker::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  return channel_->allocate_kv_cache(kv_cache_shape);
}

void RemoteWorker::get_cache_info(uint64_t& cluster_id,
                                  std::string& addr,
                                  uint16_t& port) {
  channel_->get_cache_info(cluster_id, addr, port);
}

bool RemoteWorker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<uint16_t>& ports) {
  return channel_->link_cluster(cluster_ids, addrs, ports);
}

bool RemoteWorker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<uint16_t>& ports) {
  return channel_->unlink_cluster(cluster_ids, addrs, ports);
}

bool RemoteWorker::link_p2p(const std::string& remote_addr) {
  return channel_->link_p2p(remote_addr);
}

bool RemoteWorker::unlink_p2p(const std::string& remote_addr) {
  return channel_->unlink_p2p(remote_addr);
}

bool RemoteWorker::init_model(const std::string& model_weights_path,
                              int32_t random_seed,
                              MasterStatus master_status) {
  return channel_->init_model(model_weights_path, random_seed, master_status);
}

std::tuple<int64_t, int64_t> RemoteWorker::estimate_kv_cache_capacity() {
  proto::Empty req;
  proto::DeviceMemory mem;
  brpc::Controller cntl;
  int64_t available_memory = 0;
  int64_t total_memory = 0;

  channel_->estimate_kv_cache_capacity(available_memory, total_memory);
  std::tuple<int64_t, int64_t> result(available_memory, total_memory);
  return result;
}

bool RemoteWorker::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  return channel_->pull_kv_blocks(src_cluster_id,
                                  src_addr,
                                  src_blocks,
                                  dst_blocks,
                                  src_linear_state_ids,
                                  dst_linear_state_ids);
}

ForwardInput RemoteWorker::prepare_inputs(Batch& batch) {
  NOT_IMPLEMENTED();
  return {};
}

std::optional<ForwardOutput> RemoteWorker::step(const ForwardInput& inputs) {
  NOT_IMPLEMENTED();
  return std::nullopt;
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
RemoteWorker::estimate_kv_cache_capacity_async() {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Empty req;
    proto::DeviceMemory mem;
    brpc::Controller cntl;
    int64_t available_memory = 0;
    int64_t total_memory = 0;

    channel_->estimate_kv_cache_capacity(available_memory, total_memory);
    std::tuple<int64_t, int64_t> result(available_memory, total_memory);
    promise.setValue(result);
  });
  return future;
}

folly::SemiFuture<std::optional<ForwardOutput>> RemoteWorker::step_async(
    const ForwardInput& input) {
  LOG(FATAL) << "RemoteWorker Method step_async with "
                "ForwardInput param is UnImplemented.";
  return folly::makeSemiFuture(std::optional<ForwardOutput>(std::nullopt));
}

folly::SemiFuture<std::optional<RawForwardOutput>>
RemoteWorker::step_remote_async(const ForwardInput& input) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, input = std::move(input), promise = std::move(promise)]() mutable {
        channel_->execute_model_async(input, promise);
      });
  return future;
}

folly::SemiFuture<folly::Unit> RemoteWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->process_group_test();
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_weights_path,
                        random_seed,
                        promise = std::move(promise),
                        master_status]() mutable {
    // call InitModel with callback
    channel_->init_model_async(
        model_weights_path, random_seed, promise, master_status);
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_kv_cache_async(
    const KVCacheShape& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        if (!channel_->allocate_kv_cache(kv_cache_shape)) {
          LOG(ERROR) << "allocate_kv_cache_async failed";
          promise.setValue(false);
        } else {
          promise.setValue(true);
        }
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::allocate_kv_cache_with_transfer_async(
    const KVCacheShape& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        if (!channel_->allocate_kv_cache_with_transfer(kv_cache_shape)) {
          LOG(ERROR) << "AllocateKVCacheWithTransfer failed";
          promise.setValue(false);
        } else {
          promise.setValue(true);
        }
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        src_blocks,
                        dst_blocks,
                        src_linear_state_ids,
                        dst_linear_state_ids,
                        promise = std::move(promise)]() mutable {
    if (!channel_->pull_kv_blocks(src_cluster_id,
                                  src_addr,
                                  src_blocks,
                                  dst_blocks,
                                  src_linear_state_ids,
                                  dst_linear_state_ids)) {
      LOG(ERROR) << "PullKVCache failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<uint32_t> RemoteWorker::transfer_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  folly::Promise<uint32_t> promise;
  auto future = promise.getSemiFuture();
  copy_threadpool_.schedule(
      [this,
       block_transfer_info = std::move(block_transfer_info),
       promise = std::move(promise)]() mutable {
        channel_->transfer_kv_blocks(block_transfer_info, promise);
      });
  return future;
}

void RemoteWorker::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  copy_threadpool_.schedule(
      [this,
       batch_id = batch_id,
       block_transfer_info = std::move(block_transfer_info)]() mutable {
        channel_->transfer_kv_blocks(batch_id, block_transfer_info);
      });
}

void RemoteWorker::prefetch_from_storage(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    std::shared_ptr<std::atomic<int32_t>> flag,
    std::shared_ptr<std::atomic<uint32_t>> success_cnt) {
  copy_threadpool_.schedule(
      [this,
       block_transfer_info = std::move(block_transfer_info),
       flag = flag,
       success_cnt = success_cnt]() mutable {
        channel_->prefetch_from_storage(block_transfer_info, flag, success_cnt);
      });
}

const torch::Device& RemoteWorker::device() const {
  NOT_IMPLEMENTED();
  return device_;
}

folly::SemiFuture<std::optional<RawForwardOutput>>
RemoteWorker::get_last_step_result_async() {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->get_last_step_result_async(promise);
  });
  return future;
}

int64_t RemoteWorker::get_active_activation_memory() {
  int64_t memory = 0;
  channel_->get_active_activation_memory(memory);
  return memory;
}

folly::SemiFuture<int64_t> RemoteWorker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    channel_->get_active_activation_memory_async(promise);
  });
  return future;
}

bool RemoteWorker::check_health() { return channel_->check_health(); }

folly::SemiFuture<bool> RemoteWorker::sleep_async(MasterStatus master_status) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, promise = std::move(promise), master_status]() mutable {
        if (!channel_->sleep(master_status)) {
          LOG(ERROR) << "Sleep failed";
          promise.setValue(false);
        } else {
          promise.setValue(true);
        }
      });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::wakeup_async(
    const WakeupOptions& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    if (!channel_->wakeup(options)) {
      LOG(ERROR) << "Wakeup failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::start_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    if (!channel_->start_profile()) {
      LOG(ERROR) << "StartProfile failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

folly::SemiFuture<bool> RemoteWorker::stop_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    if (!channel_->stop_profile()) {
      LOG(ERROR) << "StopProfile failed";
      promise.setValue(false);
    } else {
      promise.setValue(true);
    }
  });
  return future;
}

}  // namespace xllm
