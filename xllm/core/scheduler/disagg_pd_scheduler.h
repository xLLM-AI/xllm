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
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "runtime/xservice_client.h"
#include "scheduler/chunked_prefill_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

class DisaggPDScheduler : public ChunkedPrefillScheduler {
 public:
  DisaggPDScheduler(Engine* engine, const Options& options);

  virtual ~DisaggPDScheduler();

  virtual uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_->size();
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

  // Runtime CP<->DP switch orchestration hooks. Called by ModeSwitchService.
  //
  // The scheduler owns three async surfaces that must be quiesced before the
  // engine tears down and rebuilds the BlockManagerPool: the main scheduler
  // loop (already covered by ContinuousScheduler::pause), the dispatch_thread
  // for prefill dispatch, and brpc worker threads driving DisaggPDService
  // handlers (AddNewRequests / FirstGeneration / Link*). Any of these can hold
  // pointers into the pool or into per-request kv_state; letting them run
  // concurrent with rebuild is a use-after-free (silent SIGSEGV → rank0
  // zombie, observed on 82 CP=2 flip test).
  //
  // switch_gate_ is a shared_mutex: dispatch_thread and brpc handlers take
  // shared_lock at their entry points; rebuild takes unique_lock. begin_switch
  // hands the caller a unique_lock (blocking until every in-flight shared
  // holder releases). end_switch releases it. rebuild_after_flip performs
  // engine->rebuild_block_manager_pool + refreshes the scheduler-side
  // active_dp_size_/last_batch_/BatchFactory bookkeeping that step() would
  // otherwise do; after this ModeSwitchService can end_switch and resume.
  std::unique_lock<std::shared_mutex> begin_switch();
  void rebuild_after_flip(int32_t new_dp_size);

  // Rebuild every LlmDataDist P<->D link after this instance flipped.
  // The startup-time link topology encodes the peer's dp_size (see
  // LLMEngine::link_cluster: fan-out is dp_size wide per D-worker); once
  // either end flips, the (src_worker -> dst_worker) pairs shift and the
  // pre-flip links no longer cover the required pairs. The next
  // PushKvBlocks then returns LLM_NOT_YET_LINK (0x5010b007), which we
  // observed as "60 push errors, decode receives nothing, all DP requests
  // 90s timeout".
  //
  // The rebuild pattern for each linked peer:
  //   1. get_instance_info(peer)  -- pull peer's post-flip dp_size from
  //      etcd. The peer must have re-registered its dp_size before this
  //      call; ModeSwitchService orchestrates that ordering.
  //   2. engine_->unlink_cluster(peer_cluster_ids, peer_addrs, peer_ports,
  //                              old peer dp_size, kv_split_size)
  //   3. engine_->link_cluster(peer_cluster_ids, peer_addrs, peer_ports,
  //                            new peer dp_size, kv_split_size)
  //   4. remote_instances_info_[peer] = fresh InstanceInfo (so subsequent
  //      dispatch_requests / push_kv_blocks_async use the new dp_size).
  //
  // Callers must be holding switch_gate_ unique. Best-effort: a per-peer
  // failure logs and continues; on any relink failure returns false so
  // ModeSwitchService can surface it in the SwitchMode response.
  bool relink_after_flip();

 protected:
  // Reader-side accessor for the shared gate. The concurrent surfaces
  // (dispatch_thread, brpc-driven Disagg handlers) grab this at their entry
  // points so they cannot race a rebuild in flight.
  std::shared_mutex& switch_gate() { return switch_gate_; }
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

  // Serializes runtime CP<->DP flip against dispatch_thread and brpc handlers.
  // See begin_switch/rebuild_after_flip comments in the public section.
  // Held shared by dispatch_requests / try_allocate / decode_schedule /
  // decode_recv_first_generation / link_instance / unlink_instance.
  // Held unique by begin_switch during the flip+rebuild window.
  std::shared_mutex switch_gate_;

  std::string server_name_;
};

}  // namespace xllm
