/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <list>
#include <memory>
#include <queue>

#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"

namespace xllm {

// MixScheduler does not explicitly specify whether decoding or prefilling takes
// priority; instead, it mixes prefilling and decoding requests in a single
// queue. Currently, it's only for multi-priority scheduling algorithm ProSched.
//
// MIX inherits from DisaggPDScheduler so it picks up the disagg_pd RPC server
// and `register_instance_info` plumbing required for service-side routing
// (etcd entry must carry rpc_address / cluster_ids / addrs / ports for the
// service `LinkInstance` callback to land on us). The base ctor wires those.
// MIX overrides add_request / step / prepare_batch to keep using its own
// running_queue_ and skip the prefill->decode dispatch path that PREFILL/
// DECODE rely on.
class MixScheduler : public DisaggPDScheduler {
 public:
  MixScheduler(Engine* engine, const Options& options);
  virtual ~MixScheduler();

  // MIX serves requests locally; do not enqueue into the prefill->decode
  // dispatch queue that the base DisaggPDScheduler uses.
  bool add_request(std::shared_ptr<Request>& request) override;

  // Skip the prefill_send_first_generation step the base runs; MIX does not
  // forward generation tokens to a remote decode instance.
  void step(const absl::Duration& timeout) override;

 protected:
  std::list<std::shared_ptr<Request>> running_queue_;

  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch() override;

  virtual bool if_queue_not_empty() override;

  bool allocate_blocks_for(Sequence* sequence,
                           size_t token_budget,
                           size_t kv_cache_tokens_num,
                           size_t needed_copy_blocks_num,
                           size_t* current_step_handle_tokens);

 private:
  void handle_running_queue_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_preempted_requests,
      std::vector<Sequence*>& prefill_stage_sequences,
      std::list<std::shared_ptr<Request>>& running_queue,
      bool& budget_exhausted,
      bool& blocks_exhausted);

  void get_latency_budget_and_request_order(
      std::list<std::shared_ptr<Request>>& running_queue,
      double& latency_budget);
  int32_t get_max_chunk(Sequence* sequence,
                        size_t num_tokens,
                        size_t kv_cache_tokens_num,
                        int32_t latency_budget,
                        bool use_quadratic_formula = false);
  size_t get_max_copy_block_num(
      std::list<std::shared_ptr<Request>>& running_queue,
      double& latency_budget);

  size_t get_needed_copy_block_num(
      std::vector<std::shared_ptr<Request>>& req_vec,
      std::vector<size_t>& per_req_copy_block_num_vec,
      double max_h2d_transfer_time,
      double min_total_exec_time,
      size_t max_h2d_block_num);
};

}  // namespace xllm
