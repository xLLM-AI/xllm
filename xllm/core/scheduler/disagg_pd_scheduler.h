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

#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

class DisaggPDScheduler : public ContinuousScheduler {
 public:
  DisaggPDScheduler(Engine* engine, const Options& options);

  virtual ~DisaggPDScheduler();

  virtual uint32_t get_waiting_requests_num() const override {
    return prefill_queue_->size();
  };

  void step(const absl::Duration& timeout) override;

  std::vector<Batch> prepare_batch() override;

  bool add_request(std::shared_ptr<Request>& request) override;

  // prefill-1: for prefill send new request to decode
  virtual void dispatch_requests();
  // prefill-2: for prefill send first token to decode
  virtual void prefill_send_first_generation();

  // decode-1: for decode recveive new request from prefill
  virtual bool decode_schedule(std::shared_ptr<Request>& request,
                               const std::string& prefill_instance_name);
  // decode-2: for decode receive first token from prefill
  virtual bool decode_recv_first_generation(
      const std::string& req_id,
      int64_t token_id,
      bool has_logprob,
      float logprob,
      double time_to_first_token_latency_seconds,
      std::vector<int64_t> top_tokens,
      std::vector<float> top_logprobs,
      const std::string& kv_cache_transfer_mode,
      std::vector<uint64_t> src_cluster_ids,
      std::vector<std::string> src_addrs,
      std::vector<uint64_t> src_block_ids,
      int32_t src_linear_state_id,
      int32_t src_dp_size,
      int32_t src_dp_rank,
      torch::Tensor mtp_bootstrap_embedding = torch::Tensor());

  // decode allocate blocks with prefix cache.
  bool try_allocate(Sequence* sequence);

  bool enable_schedule_overlap() { return options_.enable_schedule_overlap(); };

  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt);

  bool link_instance(const std::string& instance_name,
                     const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& addrs,
                     const std::vector<uint16_t>& ports,
                     const int32_t dp_size,
                     const int32_t src_kv_split_size);

  bool unlink_instance(const std::string& instance_name,
                       const std::vector<uint64_t>& cluster_ids,
                       const std::vector<std::string>& addrs,
                       const std::vector<uint16_t>& ports,
                       const int32_t dp_size,
                       const int32_t src_kv_split_size);

 protected:
  // Pre-execute prefill requests of different lengths at startup and obtain the
  // corresponding TTFT for calculating the estimated TTFT of requests.
  void profile_ttft();

  void profile_tpot();

  void cache_prefill_blocks(Request* request);

  // check remote instance info, if not exist, get from master service
  bool check_remote_instance_info(const std::string& instance_name);

  // create rpc channel to remote instance,
  // we can get remote instance info from master service.
  proto::DisaggPDService_Stub* create_rpc_channel(
      const std::string& instance_name);

  virtual void start_rpc_server();

  // Initialize RPC server and xservice client
  // This method waits for the RPC server to be initialized and sets up the
  // xservice client connection.
  void initialize_rpc_server(const std::string& server_name);

  // Register instance information including name, RPC address, type, and cache
  // info
  void register_instance_info(const std::string& server_name, Engine* engine);

  void update_token_latency_metrics(std::vector<Sequence*>& sequences) override;

  // remote instance name(ID) -> instance info
  std::unordered_map<std::string, InstanceInfo> remote_instances_info_;

  // rpc server for prefill/decode instance
  std::unique_ptr<std::thread> rpc_server_thread_;

  // request_id -> brpc channel
  // brpc channel is connected to remote instance rpc server
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      req_to_channel_map_;
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      instance_channel_map_;
  std::mutex req_to_channel_map_mutex_;
  std::mutex instance_channel_map_mutex_;

  // for prefill, dispatch request to Decode instance
  std::unique_ptr<std::thread> dispatch_thread_;

  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_offline_;

  // use threadpool to handle prefill-completed request
  ThreadPool prefill_threadpool_{/*num_threads=*/1,
                                 /*cpu_binding=*/false,
                                 /*pool_name=*/"DisaggPDScheduler.prefill"};

  // related decode instance name(ID) list
  std::vector<std::string> decode_inst_names_;
  // TODO later
  // std::vector<std::string> updated_decode_inst_names;
  int current_decode_idx_ = 0;

  // for decode
  // request_id -> Request object
  std::unordered_map<std::string, std::shared_ptr<Request>>
      received_request_map_;
  // prefill_instance_name -> set of request_ids.
  // Used for bulk cleanup when a prefill instance is unlinked.
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_received_requests_map_;
  // request_id -> prefill_instance_name.
  // Used to efficiently remove a request from
  // instance_to_received_requests_map_ when the request is processed.
  std::unordered_map<std::string, std::string> request_to_instance_map_;
  std::mutex received_request_map_mutex_;

  // Lock for multi-threaded read-write latency metrics
  std::vector<int64_t> recent_ttft_;
  std::vector<int64_t> recent_tbt_;
  std::mutex latency_metrics_mutex_;

  // Lock for multi-threaded read-write linked instances
  std::mutex linked_instances_mutex_;
  std::unordered_set<std::string> linked_instance_;

  std::string server_name_;
};

}  // namespace xllm
