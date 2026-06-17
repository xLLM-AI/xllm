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

#include "worker_client.h"

#include <folly/Unit.h>
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
#include "util/timer.h"

namespace xllm {

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

folly::SemiFuture<std::optional<RawForwardOutput>>
WorkerClient::step_remote_async(const ForwardInput& input) {
  LOG(FATAL) << "WorkerClient Method step_remote_async with ForwardInput "
                "param is UnImplemented.";
  return folly::makeSemiFuture(std::optional<RawForwardOutput>(std::nullopt));
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

folly::SemiFuture<bool> WorkerClient::start_profile_async() {
  return worker_->start_profile_async();
}

folly::SemiFuture<bool> WorkerClient::stop_profile_async() {
  return worker_->stop_profile_async();
}

const torch::Device& WorkerClient::device() const { return worker_->device(); }

folly::SemiFuture<std::optional<RawForwardOutput>>
WorkerClient::get_last_step_result_async() {
  return folly::makeSemiFuture(std::optional<RawForwardOutput>(std::nullopt));
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
