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
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "perf_model.h"
#include "runtime/xservice_client.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

// Show the type of requests handled by current step.
// Status DECODE means both online and offline decoding requests.
// Status IDLE means current step is not handling any request.
enum class StepStatus { ONLINE_PREFILL, OFFLINE_PREFILL, DECODE, IDLE };

// Online-offline co-location scheduler in Disaggregated PD mode
class PDOOCScheduler : public DisaggPDScheduler {
 public:
  PDOOCScheduler(Engine* engine, const Options& options);

  virtual ~PDOOCScheduler();

  void step(const absl::Duration& timeout) override;

  // prefill-1: for prefill send new request to decode
  void dispatch_requests() override;
  // prefill-2: for prefill send first token to decode
  void prefill_send_first_generation() override;
  // prefill-2b: for prefill send multiple tokens to decode
  void prefill_send_multi_generations();

  // decode-1: for decode recveive new request from prefill
  bool decode_schedule(std::shared_ptr<Request>& request,
                       const std::string& prefill_instance_name) override;
  // decode-2: for decode receive first token from prefill
  // bool decode_recv_first_generation(...);

  // decode-2b: for decode receive multiple tokens from prefill
  bool decode_recv_multi_generations(
      const std::string& req_id,
      const std::vector<proto::RemoteToken>& migration_tokens,
      const std::string& kv_cache_transfer_mode,
      std::vector<uint64_t> src_cluster_ids,
      std::vector<std::string> src_addrs,
      std::vector<uint64_t> src_block_ids,
      int32_t src_linear_state_id,
      int32_t src_dp_size,
      int32_t src_dp_rank);

  void prefill_step(const absl::Duration& timeout);

  void decode_step(const absl::Duration& timeout);

  void decode_send_pull_signal();

  bool check_able_to_pull();

  bool write_pull_signal(const proto::PullSignal& pull_signal);

  void prepare_offline_dispatch_queue();

  void dispatch_offline_requests();

  std::vector<Batch> prepare_batch() override;

  void handle_decode_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_offline_decode_preempt_offline_requests,
      size_t& num_online_decode_preempt_online_requests,
      size_t& num_online_decode_preempt_offline_requests,
      RequestPriorityQueue* running_queue);

 private:
  void handle_prefill_requests_impl(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      RequestPriorityQueue* waiting_priority_queue,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
  void handle_decode_requests_impl(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_offline_decode_preempt_offline_requests,
      size_t& num_online_decode_preempt_online_requests,
      size_t& num_online_decode_preempt_offline_requests,
      RequestPriorityQueue* running_queue);

  void handle_prefill_interruption();

  // Methods moved from ContinuousScheduler (only used by PDOOCScheduler).
  void cache_in_batch_prefix(
      const std::vector<Sequence*>& sequences,
      const std::vector<size_t>& current_step_token_budgets);

  void handle_abnormal_request(
      RequestPriorityQueue* running_queue,
      const std::vector<Sequence*>& candidate_sequences,
      const std::vector<size_t>& candidate_token_budgets,
      const size_t& allocated_tokens,
      const size_t& allocated_seqs,
      double& allocated_estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      double& estimate_latency,
      bool budget_exhausted,
      bool blocks_exhausted);

  void handle_running_requests(std::shared_ptr<Request> request);

  void start_rpc_server() override;

  // Build DisaggRequests proto from Request objects
  void build_disagg_requests(
      const std::vector<std::shared_ptr<Request>>& requests,
      proto::DisaggRequests& reqs);

  // Select a decode instance for dispatching requests
  std::string select_decode_instance();

  // Select a prefill instance for pulling requests
  std::string select_prefill_instance();

  StepStatus step_status_ = StepStatus::IDLE;

  std::mutex decode_send_pull_signal_mtx_;
  std::condition_variable decode_send_pull_signal_cv_;
  std::atomic<bool> decode_send_pull_signal_pending_ = true;
  std::atomic<bool> waiting_pull_finished_ = false;

  moodycamel::BlockingConcurrentQueue<proto::PullSignal> pull_signals_;

  std::vector<std::string> prefill_inst_names_;
  int current_prefill_idx_ = 0;

  std::unique_ptr<std::thread> send_pull_signal_thread_;

  std::unique_ptr<std::thread> dispatch_offline_thread_;

  // moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
  // offline_requests_to_dispatch_;
  moodycamel::BlockingConcurrentQueue<
      std::pair<std::shared_ptr<Request>, std::string>>
      offline_requests_to_dispatch_;  // Requests to dispatch and their
                                      // specified decoding instance names.
  moodycamel::BlockingConcurrentQueue<
      std::pair<std::shared_ptr<Request>, std::string>>
      offline_requests_to_transfer_;

  perf_model::LLMFlops llm_flops_;
  int linear_saturation_bs_;
  vector<int> decode_step_global_batch_req_lens_;
  double decode_last_step_latency_ = 0;
  vector<int> last_decode_step_global_batch_req_lens_;

  // Offline queues for PD-OOC online/offline batch exclusivity.
  std::unique_ptr<RequestPriorityQueue> prefill_queue_offline_;
  std::unique_ptr<RequestPriorityQueue> decode_queue_offline_;

  // for prefill save all remote requests
  std::unordered_map<std::string, std::shared_ptr<Request>>
      remote_requests_map_;
  std::mutex remote_requests_map_mutex_;
};

}  // namespace xllm
