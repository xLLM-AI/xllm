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

#include "core/transport/worker_transport.h"

#include <glog/logging.h>

#include <utility>

#include "core/transport/rpc_channel.h"
#include "core/transport/shm_channel.h"
#include "core/util/net.h"

namespace xllm {
WorkerTransport::WorkerTransport(const std::string& worker_address,
                                 int32_t rank,
                                 int32_t dp_local_tp_size,
                                 const runtime::Options& options) {
  CHECK_GT(dp_local_tp_size, 0);
  auto rpc = std::make_unique<RpcChannel>();
  CHECK(rpc->init_brpc(worker_address))
      << "Failed to initialize worker transport";
  if (options.enable_shm() &&
      net::extract_ip(options.master_node_addr().value_or("")) ==
          net::extract_ip(worker_address)) {
    channel_ = std::make_unique<ShmChannel>(rank / dp_local_tp_size,
                                            rank,
                                            rank % dp_local_tp_size == 0,
                                            options,
                                            std::move(rpc));
  } else {
    channel_ = std::move(rpc);
  }
}

WorkerTransport::~WorkerTransport() = default;

bool WorkerTransport::hello() { return channel_->hello(); }

bool WorkerTransport::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  return channel_->allocate_kv_cache(kv_cache_shape);
}

bool WorkerTransport::get_cache_info(uint64_t& cluster_id,
                                     std::string& addr,
                                     uint16_t& port) {
  return channel_->get_cache_info(cluster_id, addr, port);
}

bool WorkerTransport::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                   const std::vector<std::string>& addrs,
                                   const std::vector<uint16_t>& ports) {
  return channel_->link_cluster(cluster_ids, addrs, ports);
}

bool WorkerTransport::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                     const std::vector<std::string>& addrs,
                                     const std::vector<uint16_t>& ports) {
  return channel_->unlink_cluster(cluster_ids, addrs, ports);
}

bool WorkerTransport::link_p2p(const std::string& remote_addr) {
  return channel_->link_p2p(remote_addr);
}

bool WorkerTransport::unlink_p2p(const std::string& remote_addr) {
  return channel_->unlink_p2p(remote_addr);
}

bool WorkerTransport::init_model_async(const std::string& model_weights_path,
                                       int32_t random_seed,
                                       folly::Promise<bool>& promise,
                                       MasterStatus master_status) {
  return channel_->init_model_async(
      model_weights_path, random_seed, promise, master_status);
}

bool WorkerTransport::estimate_kv_cache_capacity(int64_t& available_memory,
                                                 int64_t& total_memory) {
  return channel_->estimate_kv_cache_capacity(available_memory, total_memory);
}

bool WorkerTransport::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  return channel_->pull_kv_blocks(src_cluster_id, src_addr, mappings);
}

void WorkerTransport::execute_llm_model_async(
    const LlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise,
    ForwardInputEncoding* encoding) {
  channel_->execute_llm_model_async(input, promise, encoding);
}

void WorkerTransport::execute_dit_model_async(
    const DiTForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  channel_->execute_dit_model_async(input, promise);
}

void WorkerTransport::execute_vlm_model_async(
    const VlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise,
    ForwardInputEncoding* encoding) {
  channel_->execute_vlm_model_async(input, promise, encoding);
}

bool WorkerTransport::process_group_test() {
  return channel_->process_group_test();
}

bool WorkerTransport::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return channel_->allocate_kv_cache_with_transfer(kv_cache_shape);
}

void WorkerTransport::transfer_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    folly::Promise<uint32_t>& promise) {
  channel_->transfer_kv_blocks(block_transfer_info, promise);
}

void WorkerTransport::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  channel_->transfer_kv_blocks(batch_id, block_transfer_info);
}

bool WorkerTransport::wait_for_kv_blocks_transfer(uint64_t batch_id) {
  return channel_->wait_for_kv_blocks_transfer(batch_id);
}

void WorkerTransport::prefetch_from_storage(
    const StoragePrefetchRequest& request,
    std::shared_ptr<PrefetchResult> result,
    size_t worker_index) {
  channel_->prefetch_from_storage(request, result, worker_index);
}

bool WorkerTransport::get_last_llm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  return channel_->get_last_llm_step_result_async(promise);
}

bool WorkerTransport::get_last_vlm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  return channel_->get_last_vlm_step_result_async(promise);
}

bool WorkerTransport::check_health() { return channel_->check_health(); }

bool WorkerTransport::sleep(MasterStatus master_status,
                            uint64_t transition_id) {
  return channel_->sleep(master_status, transition_id);
}

bool WorkerTransport::wakeup(const WakeupOptions& options) {
  return channel_->wakeup(options);
}

bool WorkerTransport::start_profile() { return channel_->start_profile(); }

bool WorkerTransport::stop_profile() { return channel_->stop_profile(); }

}  // namespace xllm
