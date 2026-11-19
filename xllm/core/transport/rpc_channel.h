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

#pragma once

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <folly/futures/Future.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/allocator/virtual_memory/mapped_virtual_memory_region.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/prefetch_result.h"
#include "core/transport/comm_channel.h"
#include "runtime/forward_params.h"
#include "worker.pb.h"

namespace xllm {
class VlmForwardInput;
class ForwardInputEncoding;

class RpcChannel final : public CommChannel {
 public:
  RpcChannel() = default;
  ~RpcChannel() override = default;

  bool init_brpc(const std::string& server_address);

  bool hello() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

  bool get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      uint16_t& port) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override;

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override;

  // P2P link for weight transfer
  bool link_p2p(const std::string& remote_addr) override;
  bool unlink_p2p(const std::string& remote_addr) override;

  bool init_model_async(const std::string& model_weights_path,
                        int32_t random_seed,
                        folly::Promise<bool>& promise,
                        MasterStatus master_status) override;

  bool estimate_kv_cache_capacity(int64_t& available_memory,
                                  int64_t& total_memory) override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override;

  void execute_llm_model_async(
      const LlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr) override;

  void execute_dit_model_async(
      const DiTForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise) override;

  void execute_vlm_model_async(
      const VlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr) override;

  bool process_group_test() override;

  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;

  void transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      folly::Promise<uint32_t>& promise) override;

  void transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  bool wait_for_kv_blocks_transfer(uint64_t batch_id) override;

  void prefetch_from_storage(const StoragePrefetchRequest& request,
                             std::shared_ptr<PrefetchResult> result,
                             size_t worker_index) override;

  bool get_last_llm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise) override;

  bool get_last_vlm_step_result_async(
      folly::Promise<std::optional<RawForwardOutput>>& promise) override;

  // Check if the connection to worker is healthy
  bool check_health() override;

  bool sleep(MasterStatus master_status, uint64_t transition_id = 0) override;

  bool wakeup(const WakeupOptions& options) override;

  bool start_profile() override;

  bool stop_profile() override;

 private:
  bool execute_llm_model_with_brpc(
      const LlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr);

  bool execute_dit_model_with_brpc(
      const DiTForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise);

  bool execute_vlm_model_with_brpc(
      const VlmForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise,
      ForwardInputEncoding* encoding = nullptr);

 private:
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::DistributeWorker_Stub> stub_;
};

class InitModelClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::Status response;
  brpc::Controller cntl;
  folly::Promise<bool> promise;
};

class ExecuteLLMModelClosure final : public google::protobuf::Closure {
 public:
  void Run() override;

  proto::LLMForwardResponse pb_output;
  brpc::Controller cntl;
  folly::Promise<std::optional<RawForwardOutput>> promise;
};

class ExecuteVLMModelClosure final : public google::protobuf::Closure {
 public:
  void Run() override;

  proto::VLMForwardResponse pb_output;
  brpc::Controller cntl;
  folly::Promise<std::optional<RawForwardOutput>> promise;
};

class ExecuteDITModelClosure final : public google::protobuf::Closure {
 public:
  void Run() override;

  proto::DITForwardResponse pb_output;
  brpc::Controller cntl;
  folly::Promise<std::optional<RawForwardOutput>> promise;
};

class TransferBlocksClosure : public google::protobuf::Closure {
 public:
  void Run();

  proto::TransferStatus response;
  brpc::Controller cntl;
  folly::Promise<uint32_t> promise;
};
}  // namespace xllm
