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

#include <absl/hash/hash.h>
#include <brpc/channel.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/disaggregation/pd_service.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/config/speculative_config.h"
#include "disagg_pd.pb.h"
#include "distributed_runtime/xservice_client.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "scheduler/continuous_scheduler.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

class KVTransferCoordinator;

inline constexpr int32_t kDecodeAddNewInvalidRequestStatusCode = 400;
inline constexpr int32_t kDecodeAddNewPromptTooLongStatusCode = 413;

bool is_permanent_rejection(int32_t status_code);

// Flat KV managers reserve block 0 for padding, so only num_blocks - 1 blocks
// can belong to a request. Returns true only when the prompt can never fit,
// independent of current cache pressure.
bool exceeds_decode_capacity(size_t num_prompt_tokens,
                             size_t block_size,
                             size_t num_blocks);

bool has_rank_preserving_kv_groups(const proto::DisaggResponse& response);

class PDScheduler : public ContinuousScheduler {
 public:
  void start_runtime(PDServiceOptions response_options);

  ~PDScheduler() override;

  uint32_t get_waiting_requests_num() const override {
    return prefill_queue_->size() + num_prefetching_requests();
  };

  void step(const absl::Duration& timeout) override;

  // prefill-1: for prefill send new request to decode
  void dispatch_requests();
  // prefill-2: for prefill send first token to decode
  void prefill_send_first_generation();

  // decode-1: for decode recveive new request from prefill
  bool decode_schedule(std::shared_ptr<Request>& request,
                       const std::string& prefill_instance_name);
  // decode-2: for decode receive first token from prefill
  bool decode_recv_first_generation(
      const std::string& req_id,
      const std::string& reservation_id,
      int64_t token_id,
      bool has_logprob,
      float logprob,
      double time_to_first_token_latency_seconds,
      double upstream_elapsed_seconds,
      std::vector<int64_t> top_tokens,
      std::vector<float> top_logprobs,
      const std::string& kv_cache_transfer_mode,
      std::vector<uint64_t> src_cluster_ids,
      std::vector<std::string> src_addrs,
      std::vector<KVTransferMapping> source_mappings,
      int32_t src_dp_size,
      int32_t src_dp_rank,
      torch::Tensor mtp_bootstrap_embedding = torch::Tensor(),
      int32_t num_cached_tokens = 0);

  // Only consumes a matching waiting reservation, never a running request.
  bool release_reservation(const std::string& req_id,
                           const std::string& reservation_id);

  // RPC and handoff threads retain the request until the scheduler has
  // retired its blocks. Acknowledgement follows the actual release.
  void release_request_blocks(std::shared_ptr<Request> request,
                              bool cache_prefix);

  // Synchronous per-sequence release for RPC-thread paths that must not
  // enqueue through the release queue.
  void deallocate_sequences(Request& request, bool cache_prefix);

  // decode allocate blocks with prefix cache.
  bool try_allocate(Sequence* sequence);

  // Classifies a failed allocation as permanently oversized.
  // DSV4 multi-manager and VirtualMemory layouts conservatively return false
  // because their effective token capacity cannot be derived from the flat KV
  // count.
  bool exceeds_decode_capacity(Sequence* sequence) const;

  bool enable_schedule_overlap() const {
    return options_.enable_schedule_overlap();
  }

  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt) override;

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

  PDScheduler(
      BlockManagerPool* block_manager_pool,
      const ModelArgs* model_args,
      const Tokenizer* tokenizer,
      SchedulerCallbacks callbacks,
      const Options& options,
      std::shared_ptr<DistributedWorkerManager> distributed_worker_manager,
      std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator =
          nullptr,
      std::shared_ptr<KVTransferCoordinator> kv_transfer_coordinator = nullptr)
      : ContinuousScheduler(block_manager_pool,
                            model_args,
                            tokenizer,
                            std::move(callbacks),
                            options,
                            std::move(model_residency_coordinator),
                            std::move(distributed_worker_manager)),
        kv_transfer_coordinator_(std::move(kv_transfer_coordinator)),
        server_name_("PDServer"),
        requires_bootstrap_hidden_(
            options_.num_speculative_tokens() > 0 &&
            SpeculativeConfig::requires_bootstrap_hidden(
                SpeculativeConfig::get_instance().speculative_algorithm())) {
    if (!options_.instance_role().has_value()) {
      LOG(FATAL) << "Instance type is not set in disagg pd mode.";
    }
  }

 protected:
  void release_failed_request(const std::shared_ptr<Request>& request) override;

  std::future<void> enqueue_request_block_release(
      std::shared_ptr<Request> request,
      bool cache_prefix);
  void drain_request_releases() override;

  // Caller holds received_request_map_mutex_. Deallocation stays outside it.
  std::shared_ptr<Request> take_waiting_request(const std::string& req_id);

  void do_permanent_rejection(const std::shared_ptr<Request>& request,
                              int32_t status_code);

  size_t num_queued_requests() const override;
  bool enqueue_ready_request(std::shared_ptr<Request> request) override;
  bool enqueue_dispatched_request(std::shared_ptr<Request> request);
  void release_dispatch_admission();

  // Pre-execute prefill requests of different lengths at startup and obtain the
  // corresponding TTFT for calculating the estimated TTFT of requests.
  void profile_ttft();

  void profile_tpot();

  void restore_disaggregated_latency(Request* request,
                                     double time_to_first_token_latency_seconds,
                                     double upstream_elapsed_seconds);

  void start_rpc_server(PDServiceOptions response_options);

  // Initialize RPC server and xservice client
  // This method waits for the RPC server to be initialized and sets up the
  // xservice client connection.
  void initialize_rpc_server(const std::string& server_name);

  // Register instance information including name, RPC address, type, and cache
  // info
  void register_instance_info(const std::string& server_name);

  // Both P->D handshake halves must agree on carrying the bootstrap hidden
  // state.
  bool requires_bootstrap_hidden_;
  std::shared_ptr<KVTransferCoordinator> kv_transfer_coordinator_;

  struct RequestBlockRelease {
    std::shared_ptr<Request> request;
    bool cache_prefix;
    std::promise<void> completion;
  };

  void complete_request_block_release(RequestBlockRelease& release);

  // Producers enqueue owning commands; only the scheduling thread consumes
  // them. Destruction takes over consumption after the master loop joins.
  moodycamel::BlockingConcurrentQueue<std::unique_ptr<RequestBlockRelease>>
      request_block_releases_;

  // rpc server for prefill/decode instance
  std::unique_ptr<std::thread> rpc_server_thread_;

  // request_id -> brpc channel
  // brpc channel is connected to remote instance rpc server
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      req_to_channel_map_;
  std::unordered_map<std::string, std::unique_ptr<proto::DisaggPDService_Stub>>
      instance_channel_map_;
  std::mutex req_to_channel_map_mutex_;
  std::mutex instance_channel_map_mutex_;

  // for prefill, dispatch request to Decode instance
  std::unique_ptr<std::thread> dispatch_thread_;

  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_offline_;

  // Includes both dispatch queues, retries and the in-flight allocation RPC.
  // Guarded by request_admission_mutex_ through the transfer to
  // request_queue_.
  size_t pending_dispatch_requests_ = 0;

  // Release retries must not delay FirstGeneration or local KV reclamation.
  ThreadPool reservation_release_threadpool_{
      /*num_threads=*/1,
      /*cpu_binding=*/false,
      /*pool_name=*/"PDScheduler.reservation_release"};

  // use threadpool to handle prefill-completed request
  ThreadPool prefill_threadpool_{/*num_threads=*/1,
                                 /*cpu_binding=*/false,
                                 /*pool_name=*/"PDScheduler.prefill"};

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
  using ReservationIdentity = std::pair<std::string, std::string>;
  std::unordered_map<ReservationIdentity,
                     std::chrono::steady_clock::time_point,
                     absl::Hash<ReservationIdentity>>
      released_reservations_;

  // Lock for multi-threaded read-write linked instances
  std::mutex linked_instances_mutex_;
  std::unordered_set<std::string> linked_instance_;

  std::string server_name_;
};

}  // namespace xllm
