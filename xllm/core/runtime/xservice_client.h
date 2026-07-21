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

#include <brpc/channel.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/etcd_client.h"
#include "distributed_runtime/engine.h"
#include "forward_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request_output.h"
#include "scheduler/scheduler.h"
#include "xservice.pb.h"
namespace xllm {

class XServiceClient {
 public:
  static XServiceClient* get_instance() {
    static XServiceClient xservice_client;
    return &xservice_client;
  }

  ~XServiceClient();
  bool init(const std::string& etcd_addr,
            const std::string& instance_name = "",
            const BlockManagerPool* block_manager_pool = nullptr,
            const std::string& etcd_namespace = "");
  void set_scheduler(Scheduler* scheduler);
  void set_engine(Engine* engine);
  // Atomically replace the pool pointer that heartbeat/reconcile threads read.
  // Called from LLMEngine::rebuild_block_manager_pool after a runtime CP<->DP
  // flip. Without this, the heartbeat thread keeps the pointer captured at
  // init() time and dereferences the destroyed pool the moment
  // unique_ptr::reset swaps it in the engine (silent SIGSEGV, rank0 disappears
  // ~5s post-flip).
  void set_block_manager_pool(const BlockManagerPool* block_manager_pool);
  bool initialize_done() { return initialize_done_; }

  std::string get_instance_name();
  void register_instance(const InstanceInfo& instance_info);
  // Re-publish this instance's InstanceInfo to etcd after a runtime CP<->DP
  // flip changed dp_size. The peer scheduler pulls InstanceInfo lazily on
  // demand (DisaggPDScheduler::check_remote_instance_info) and its cache is
  // sticky; without this write, the peer keeps the pre-flip dp_size in
  // remote_instances_info_ and its own relink_after_flip will link to the
  // wrong topology. Rewrites the JSON we handed to etcd at register_instance
  // time with the new dp_size, keeping every other field intact.
  bool re_register_dp_size(int32_t new_dp_size);
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();
  std::vector<std::string> get_static_prefill_list();

  // get all xllm_service addrs
  std::vector<std::string> get_all_xservice_addrs();

  // response generation tokens to xllm service
  std::vector<bool> generations(const std::vector<RequestOutput>& outputs);

 private:
  bool register_instance_with_retry(const std::string& key,
                                    const std::string& value);
  bool reconcile_registration();
  void reconcile_registration_loop();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);
  void handle_xservices_watch(const etcd::Response& response,
                              const uint64_t& prefix_len);

  // connect to specific xllm_service
  bool connect_to_xservice(const std::string& xservice_addr);

  // remove the connection to specific xllm_service
  void disconnect_xservice(const std::string& xservice_addr);

  // call rpc with current master stub atomically.
  bool with_master_stub(
      const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
      std::string* master_addr);

  // find stub by address, caller should hold mutex_
  xllm_service::proto::XllmRpcService_Stub* find_stub_locked(
      const std::string& xservice_addr);

 private:
  XServiceClient() = default;

  std::atomic_bool exited_{false};
  std::atomic_bool register_done_{false};
  bool initialize_done_ = false;
  std::string instance_name_;
  std::string incarnation_id_;
  std::string registration_key_;
  std::string registration_value_;

  std::string master_xservice_addr_;
  std::unordered_map<std::string, std::unique_ptr<brpc::Channel>>
      xservice_channels_;
  std::unordered_map<std::string,
                     std::unique_ptr<xllm_service::proto::XllmRpcService_Stub>>
      xservice_stubs_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::unique_ptr<std::thread> reconcile_thread_;

  std::shared_mutex mutex_;
  std::mutex registration_mutex_;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  // Atomic so heartbeat_thread_/reconcile_thread_ can safely re-read after a
  // runtime CP<->DP flip's rebuild_block_manager_pool. Load with
  // memory_order_acquire on the reader side; store with release on the writer.
  std::atomic<const BlockManagerPool*> block_manager_pool_{nullptr};  // not own
  Scheduler* scheduler_ = nullptr;                                    // not own
  Engine* engine_ = nullptr;  // not own, for xtensor info
};

}  // namespace xllm
