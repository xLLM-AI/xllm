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

#include "core/transport/shm_channel.h"

#include <glog/logging.h>

#include <utility>

#include "core/transport/rpc_channel.h"
#include "runtime/vlm_forward_params.h"
#include "util/net.h"

namespace xllm {

ShmChannel::ShmChannel(int32_t dp_group,
                       int32_t rank,
                       bool is_driver,
                       const runtime::Options& options,
                       std::unique_ptr<RpcChannel> rpc)
    : rpc_(std::move(rpc)),
      enable_shm_(options.enable_shm()),
      input_shm_size_(options.input_shm_size()) {
  bool is_creator;

  std::string name_prefix =
      "xllm_" + net::extract_port(options.master_node_addr().value_or(""));
  if (is_driver) {
    auto name = SharedMemoryTransport::create_unique_name(
        name_prefix, dp_group, ForwardType::RAW_INPUT, rank);
    input_shm_manager_ = std::make_unique<SharedMemoryTransport>(
        name, options.input_shm_size(), is_creator, ForwardType::RAW_INPUT);
    LOG(INFO) << "Create input shared memory manager with name: " << name
              << " and size: " << options.input_shm_size();
  }

  auto name = SharedMemoryTransport::create_unique_name(
      name_prefix, dp_group, ForwardType::RAW_OUTPUT, rank);
  output_shm_manager_ = std::make_unique<SharedMemoryTransport>(
      name, options.output_shm_size(), is_creator, ForwardType::RAW_OUTPUT);
  LOG(INFO) << "Create output shared memory manager with name: " << name
            << " and size: " << options.output_shm_size();
}

void ShmChannel::execute_llm_model_async(
    const LlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise,
    ForwardInputEncoding* encoding) {
  if (enable_shm_) {
    const bool input_fits =
        input_shm_manager_
            ? input_shm_manager_->input_write(input)
            : SharedMemoryTransport::can_write_input(input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "LLM shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      if (!output_shm_manager_->llm_output_read(output)) {
        LOG(ERROR) << "LLM shared-memory output read failed";
        // The model has already executed. Retrying with BRPC would execute it
        // twice and advance worker state again.
        promise.setValue(std::nullopt);
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  rpc_->execute_llm_model_async(input, promise, encoding);
}

void ShmChannel::execute_dit_model_async(
    const DiTForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  if (enable_shm_) {
    const bool input_fits =
        input_shm_manager_
            ? input_shm_manager_->input_write(input)
            : SharedMemoryTransport::can_write_input(input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "DIT shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      if (!output_shm_manager_->dit_output_read(output)) {
        LOG(ERROR) << "DIT shared-memory output read failed";
        // The model has already executed. Retrying with BRPC would execute it
        // twice and advance worker state again.
        promise.setValue(std::nullopt);
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  rpc_->execute_dit_model_async(input, promise);
}

void ShmChannel::execute_vlm_model_async(
    const VlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise,
    ForwardInputEncoding* encoding) {
  if (enable_shm_) {
    const bool input_fits =
        input_shm_manager_
            ? input_shm_manager_->input_write(input)
            : SharedMemoryTransport::can_write_input(input, input_shm_size_);
    if (!input_fits) {
      enable_shm_ = false;
      LOG(ERROR) << "VLM shared-memory input write failed, fallback to brpc";
    } else {
      RawForwardOutput output;
      if (!output_shm_manager_->vlm_output_read(output)) {
        LOG(ERROR) << "VLM shared-memory output read failed";
        // The model has already executed. Retrying with BRPC would execute it
        // twice and advance worker state again.
        promise.setValue(std::nullopt);
        return;
      }
      promise.setValue(std::move(output));
      return;
    }
  }
  rpc_->execute_vlm_model_async(input, promise, encoding);
}

ShmChannel::~ShmChannel() = default;

bool ShmChannel::hello() { return rpc_->hello(); }

bool ShmChannel::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  return rpc_->allocate_kv_cache(kv_cache_shape);
}

bool ShmChannel::get_cache_info(uint64_t& cluster_id,
                                std::string& addr,
                                uint16_t& port) {
  return rpc_->get_cache_info(cluster_id, addr, port);
}

bool ShmChannel::link_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports) {
  return rpc_->link_cluster(cluster_ids, addrs, ports);
}

bool ShmChannel::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<uint16_t>& ports) {
  return rpc_->unlink_cluster(cluster_ids, addrs, ports);
}

bool ShmChannel::link_p2p(const std::string& remote_addr) {
  return rpc_->link_p2p(remote_addr);
}

bool ShmChannel::unlink_p2p(const std::string& remote_addr) {
  return rpc_->unlink_p2p(remote_addr);
}

bool ShmChannel::init_model_async(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  folly::Promise<bool>& promise,
                                  MasterStatus master_status) {
  return rpc_->init_model_async(
      model_weights_path, random_seed, promise, master_status);
}

bool ShmChannel::estimate_kv_cache_capacity(int64_t& available_memory,
                                            int64_t& total_memory) {
  return rpc_->estimate_kv_cache_capacity(available_memory, total_memory);
}

bool ShmChannel::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  return rpc_->pull_kv_blocks(src_cluster_id, src_addr, mappings);
}

bool ShmChannel::process_group_test() { return rpc_->process_group_test(); }

bool ShmChannel::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return rpc_->allocate_kv_cache_with_transfer(kv_cache_shape);
}

void ShmChannel::transfer_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    folly::Promise<uint32_t>& promise) {
  rpc_->transfer_kv_blocks(block_transfer_info, promise);
}

void ShmChannel::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  rpc_->transfer_kv_blocks(batch_id, block_transfer_info);
}

bool ShmChannel::wait_for_kv_blocks_transfer(uint64_t batch_id) {
  return rpc_->wait_for_kv_blocks_transfer(batch_id);
}

void ShmChannel::prefetch_from_storage(const StoragePrefetchRequest& request,
                                       std::shared_ptr<PrefetchResult> result,
                                       size_t worker_index) {
  rpc_->prefetch_from_storage(request, result, worker_index);
}

bool ShmChannel::get_last_llm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  return rpc_->get_last_llm_step_result_async(promise);
}

bool ShmChannel::get_last_vlm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  return rpc_->get_last_vlm_step_result_async(promise);
}

bool ShmChannel::check_health() { return rpc_->check_health(); }

bool ShmChannel::sleep(MasterStatus master_status, uint64_t transition_id) {
  return rpc_->sleep(master_status, transition_id);
}

bool ShmChannel::wakeup(const WakeupOptions& options) {
  return rpc_->wakeup(options);
}

bool ShmChannel::start_profile() { return rpc_->start_profile(); }

bool ShmChannel::stop_profile() { return rpc_->stop_profile(); }

}  // namespace xllm
