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

#include "scheduler/pd_ooc_scheduler.h"

#include <absl/strings/str_join.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/server.h>

#include <chrono>
#include <random>

#include "common/global_flags.h"
#include "common/interruption_bus.h"
#include "common/macros.h"
#include "core/distributed_runtime/pd_ooc_service.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/scheduler_config.h"
#include "disagg_pd.pb.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "pd_ooc_scheduler.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "util/env_var.h"
#include "util/utils.h"

namespace xllm {

namespace {

size_t estimate_decode_extra_blocks(Sequence* sequence,
                                    size_t updated_num_tokens,
                                    size_t block_size) {
  const size_t num_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t num_blocks_needed =
      (updated_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > num_blocks) {
    return num_blocks_needed - num_blocks;
  }

  // Beam swap may still require one extra block when reusing source blocks.
  if (sequence->check_beam_search() &&
      !sequence->kv_state().src_blocks().empty() &&
      sequence->kv_state().need_swap()) {
    return 1;
  }
  return 0;
}

size_t get_sequence_free_blocks_for_rank(KVCacheManager* kv_cache_manager,
                                         int32_t dp_rank) {
  const auto free_blocks = kv_cache_manager->num_free_blocks();
  if (free_blocks.empty()) {
    return 0;
  }
  if (dp_rank >= 0 && static_cast<size_t>(dp_rank) < free_blocks.size()) {
    return free_blocks[dp_rank];
  }
  return util::max(free_blocks);
}

inline size_t maybe_align_cp_prefill_tokens(const Sequence* sequence,
                                            size_t num_tokens,
                                            int32_t cp_size) {
  if (sequence == nullptr || cp_size <= 1 || num_tokens == 0) {
    return num_tokens;
  }
  if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
    return num_tokens;
  }
  if (!sequence->is_prefill_stage()) {
    return num_tokens;
  }
  const size_t alignment = static_cast<size_t>(cp_size) * 2;
  return xllm::util::align_up(num_tokens, alignment);
}

}  // namespace

void PDOOCScheduler::cache_in_batch_prefix(
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& current_step_token_budgets) {
  if (!enable_prefix_cache_ || !enable_in_batch_prefix_cache_ ||
      sequences.empty()) {
    return;
  }
  CHECK_EQ(sequences.size(), current_step_token_budgets.size());
  for (size_t i = 0; i < sequences.size(); ++i) {
    Sequence* sequence = sequences[i];
    if (sequence == nullptr || !sequence->is_prefill_stage()) {
      continue;
    }
    const size_t max_handle_num_tokens =
        sequence->kv_state().kv_cache_tokens_num() +
        current_step_token_budgets[i];
    kv_cache_manager_->cache(sequence, max_handle_num_tokens);
  }
}

void PDOOCScheduler::handle_abnormal_request(
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
    bool blocks_exhausted) {
  std::shared_ptr<Request> request = running_queue->top();
  if (candidate_sequences.empty()) {
    if (!running_sequences_.empty()) {
      return;
    }

    // unknown case, maybe a schedule bug.
    if (!budget_exhausted && !blocks_exhausted) {
      LOG(FATAL) << "Unknown case, budget and blocks are not exhausted, but "
                    "there are no running sequences."
                 << " budget_exhausted = " << budget_exhausted
                 << " blocks_exhausted = " << blocks_exhausted
                 << " candidate_sequences.size = " << candidate_sequences.size()
                 << ", running_sequences.size = " << running_sequences_.size();
    }

    // budget exhausted
    if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, please set a larger "
                    "max_tokens value via --max_tokens_per_batch.";
    } else {
      CHECK(running_queue->size() == 1)
          << "Running queue size is not 1, there maybe a bug of request "
             "preemption logic. decode_queue_.size ="
          << running_queue->size();
      if (util::sum(kv_cache_manager_->num_used_blocks()) !=
          request->total_num_blocks()) {
        // blocks_exhausted is true.
        // NOTE: consider dp > 1, here we need get all num blocks in use.
        // Total num blocks in use not equal request->total_num_blocks() means
        // some sequences are not scheduled but hold blocks in disagg PD mode.
        return;
      }
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
    }

    // request is too long, budget or memory no enough.
    running_queue->pop_top();
    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough resource to schedule a single sequence"});
  } else {
    // partially schedule the sequences in request
    if (!request->check_beam_search()) {
      running_queue->pop_top();
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      estimate_latency += allocated_estimate_latency;
    }
  }
}

void PDOOCScheduler::handle_running_requests(std::shared_ptr<Request> request) {
  if (request->finished() || request->cancelled()) {
    LOG(FATAL) << "Unknow error, finished/cancelled request have be handled "
                  "before. request_id is "
               << request->request_id();
  }

  // check if the request can be expanded
  if (request->expand_sequences()) {
    // cache the blocks to share among the sequences
    kv_cache_manager_->cache(request->sequences()[0].get());
  }

  // release blocks for finished sequences here
  for (auto& sequence : request->sequences()) {
    if (sequence->finished()) {
      kv_cache_manager_->deallocate(sequence.get());
    }
  }
}

PDOOCScheduler::PDOOCScheduler(Engine* engine, const Options& options)
    : DisaggPDScheduler(engine, options),
      llm_flops_(engine->model_args().n_layers(),
                 engine->model_args().vocab_size(),
                 engine->model_args().hidden_size(),
                 engine->model_args().intermediate_size(),
                 engine->model_args().n_kv_heads().has_value()
                     ? engine->model_args().n_heads() /
                           engine->model_args().n_kv_heads().value()
                     : 1,
                 engine->model_args().dtype() == "int8" ? 1 : 2,  // FIXME
                 options_.nnodes() / options_.dp_size()) {
  CHECK(options_.enable_pd_ooc());
  VLOG(1) << "Creating a PD OOC Scheduler";

  server_name_ = "PDOOCServer";

  // PerfModel::PerfModel(double flop_s_gemm,
  // double flop_s_attn,
  // double memory_bw_byte_s_gemm,
  // double memory_bw_byte_s_attn,
  // double overhead_prefill_ms,
  // double overhead_decode_ms,
  // std::optional<double> network_bw_byte_s)

  perf_model::set_perf_model(std::make_shared<perf_model::PerfModel>(
      390 * 1e12 * 0.68,  // FLOPs/s GEMM
      // 390 * 1e12 * 0.59,  // FLOPs/s ATTN_P
      390 * 1e12 * 0.60,  // FLOPs/s ATTN_D
      1600 * 1e9 * 0.58,  // MEM BW GEMM
      1600 * 1e9 * 0.38,  // MEM BW ATTN
      18,                 // prefill overhead
      0,                  // decode overhead
      10 * 1e9            // net
      ));

  linear_saturation_bs_ = llm_flops_.linear_saturation_bs();

  LOG(INFO) << "LLM linear saturation batch size: " << linear_saturation_bs_;

  // Create offline queues for online/offline batch exclusivity.
  prefill_queue_offline_ = std::make_unique<DequeQueue>();
  decode_queue_offline_ = std::make_unique<DequeQueue>();

  // OOC-specific threads based on instance role
  if (options_.instance_role().value() == InstanceRole::PREFILL) {
    VLOG(1) << "Running dispatch_thread_";
    // start dispatch thread for prefill instance
    dispatch_thread_ =
        std::make_unique<std::thread>(&PDOOCScheduler::dispatch_requests, this);
    dispatch_offline_thread_ = std::make_unique<std::thread>(
        &PDOOCScheduler::dispatch_offline_requests, this);
  }

  if (options_.instance_role().value() == InstanceRole::DECODE) {
    VLOG(1) << "Running send_pull_signal_thread_";
    send_pull_signal_thread_ = std::make_unique<std::thread>(
        &PDOOCScheduler::decode_send_pull_signal, this);
  }

  server_name_.append(std::to_string(options.server_idx()));

  // Start RPC server thread (must be done in subclass constructor to ensure
  // PDOOCScheduler::start_rpc_server is called, not the base class version)
  rpc_server_thread_ =
      std::make_unique<std::thread>(&PDOOCScheduler::start_rpc_server, this);
  initialize_rpc_server(server_name_);
  register_instance_info(server_name_, engine);
}

PDOOCScheduler::~PDOOCScheduler() {
  // Clean up OOC-specific threads only
  // Common threads (rpc_server_thread_, dispatch_thread_) are cleaned up by
  // base class destructor
  if (dispatch_offline_thread_ && dispatch_offline_thread_->joinable()) {
    dispatch_offline_thread_->join();
  }

  if (send_pull_signal_thread_ && send_pull_signal_thread_->joinable()) {
    send_pull_signal_thread_->join();
  }

  LOG(INFO) << "Stop scheduler rpc server " << server_name_ << ".";
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name_);
  if (rpc_server != nullptr) {
    rpc_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

void PDOOCScheduler::start_rpc_server() {
  std::unique_ptr<PDOOCService> service =
      std::make_unique<PDOOCService>(this, engine_);
  auto rpc_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!rpc_server->start(std::move(service))) {
    LOG(ERROR) << "Failed to start brpc disagg pd server on port "
               << ::xllm::DisaggPDConfig::get_instance().disagg_pd_port();
    return;
  }
}

void PDOOCScheduler::step(const absl::Duration& timeout) {
  if (options_.instance_role() == InstanceRole::PREFILL) {
    prefill_step(timeout);
  } else {
    decode_step(timeout);
  }
}

void PDOOCScheduler::prefill_step(const absl::Duration& timeout) {
  try {
    prepare_offline_dispatch_queue();
    /*
    WIP Determine the status of current step
    If request_queue_ has online requests or prefill_queue_ is not
    empty, set current status to ONLINE_PREFILL If decode_queue_ is not
    empty, set current status to OFFLINE_PREFILL If request_queue_ has offline
    requests or prefill_queue_ is not empty, set current status
    to OFFLINE_PREFILL
    */
    InterruptionBus::get_instance().publish(false);
    ContinuousScheduler::step(timeout);
    step_status_ = StepStatus::IDLE;  // Reset status to idle to maintain
                                      // consistency with actual state
    prefill_send_first_generation();
    prefill_send_multi_generations();
  } catch (const ForwardInterruptedException& e) {
    VLOG(1) << "PDOOCScheduler catched a ForwardInterruptedException";
    handle_prefill_interruption();
  }
}

std::vector<Batch> PDOOCScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to prefill_queue_
  // Include those requests that are preempted by others.
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue

  std::vector<std::shared_ptr<xllm::Request>> deferred_reqs;
  while (request_queue_.read(request)) {
    CHECK(request);

    // In disagg PD, expansion of best_of_n sequences is always handled
    // on the DECODE instance (where prefix cache lets seq[1..best_of-1]
    // reuse seq[0]'s prompt KV). On the PREFILL/MIX instance we keep a
    // single sequence -- expanding here would waste N x prefill compute
    // on candidates that are never shipped to the DECODE instance.
    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      if (request->offline()) {
        int current_offline_decode_bs =
            running_requests_.size() + prefill_queue_offline_->size();
        VLOG(1) << "Current offline decode batch size: "
                << current_offline_decode_bs
                << ", linear_saturation_bs_: " << linear_saturation_bs_;
        if (current_offline_decode_bs < linear_saturation_bs_) {
          prefill_queue_offline_->push(request);
        } else {
          deferred_reqs.emplace_back(request);
        }
      } else {
        prefill_queue_->push(request);
      }
    } else {
      // request from prefill instance in disagge pd mode.
      running_requests_.emplace_back(request);
    }
  }

  for (auto& req : deferred_reqs) {
    request_queue_.write(req);
  }
  deferred_reqs.clear();

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  if (options_.priority_strategy() == "fcfs") {
    if (last_step_prefill_) {
      // insert all requests to the back of decode_queue_
      // 1. last step is prefill step:
      // new prefill has high priority, but these requests has lower priority
      // then existed requests in decode_queue_ in decoding stage.
      // so we need to push them to the back of decode_queue_.
      for (auto it = running_requests_.cbegin(); it != running_requests_.cend();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->offline()) {
          decode_queue_offline_->push(*it, last_step_prefill_);
        } else {
          decode_queue_->push(*it, last_step_prefill_);
        }
      }
    } else {
      // insert all requests to the front of decode_queue_
      // 2. last step is decode step:
      // We need to traverse running_requests_ array in reverse order.
      // Because there may be some unexecuted requests with
      // lower priorities remaining in the decode_queue_.
      // For the requests in running_requests_,
      // their priorities are all higher than those of the
      // remaining requests. Therefore, the `push_front`
      // method needs to be used.
      //
      for (auto it = running_requests_.crbegin();
           it != running_requests_.crend();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->offline()) {
          decode_queue_offline_->push(*it, last_step_prefill_);
        } else {
          decode_queue_->push(*it, last_step_prefill_);
        }
      }
    }
  } else {
    for (auto it = running_requests_.cbegin(); it != running_requests_.cend();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      if ((*it)->offline()) {
        decode_queue_offline_->push(*it);
      } else {
        decode_queue_->push(*it);
      }
    }
  }

  // clear previous batch
  last_step_prefill_ = false;
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // maintain estimate_latency for current batch for support requests with
  // different ttft. TO IMPROVE: use min remaining time (i.e. slo -
  // elapsed_time) of the reuquest in current decode queue to replace current
  // latency_budget.
  double latency_budget = options_.max_global_ttft_ms();
  double estimate_latency = 0;
  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;
  size_t num_offline_decode_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_online_requests = 0;
  size_t num_online_prefill_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_offline_requests = 0;
  // TO IMPROVE?: handle online decode request before prefill offline request
  bool previous_idle = (step_status_ == StepStatus::IDLE);
  // PD-OOC has its own budgeting and never goes through the chunked-prefill
  // helper; use own handle_prefill_requests_impl directly.
  handle_prefill_requests_impl(latency_budget,
                               estimate_latency,
                               remaining_token_budget,
                               remaining_seq_budget,
                               prefill_queue_.get(),
                               num_online_prefill_preempt_offline_requests,
                               finished_requests);
  if (!running_sequences_.empty()) {
    step_status_ = StepStatus::ONLINE_PREFILL;
    VLOG(1) << "Set step status to ONLINE PREFILL";
  } else {
    // In PD OOC mode, a batch can only consist entirely of online requests or
    // entirely of offline requests
    handle_prefill_requests_impl(latency_budget,
                                 estimate_latency,
                                 remaining_token_budget,
                                 remaining_seq_budget,
                                 prefill_queue_offline_.get(),
                                 num_online_prefill_preempt_offline_requests,
                                 finished_requests);
    if (!running_sequences_.empty()) {
      step_status_ = StepStatus::OFFLINE_PREFILL;
      VLOG(1) << "Set step status to OFFLINE PREFILL";
    } else {
      latency_budget = options_.max_global_tpot_ms();
      // Handle decoding requests.
      // no prefill request, schedule the decode requests in the running
      // priority queue
      handle_decode_requests(latency_budget,
                             estimate_latency,
                             remaining_token_budget,
                             remaining_seq_budget,
                             num_offline_decode_preempt_offline_requests,
                             num_online_decode_preempt_online_requests,
                             num_online_decode_preempt_offline_requests,
                             decode_queue_.get());
      handle_decode_requests(latency_budget,
                             estimate_latency,
                             remaining_token_budget,
                             remaining_seq_budget,
                             num_offline_decode_preempt_offline_requests,
                             num_online_decode_preempt_online_requests,
                             num_online_decode_preempt_offline_requests,
                             decode_queue_offline_.get());
      if (!running_sequences_.empty()) {
        step_status_ = StepStatus::DECODE;
        VLOG(1) << "Set step status to DECODE";
      } else {
        step_status_ = StepStatus::IDLE;
        if (!previous_idle) {
          VLOG(1) << "Reset step status to IDLE";
        }
      }
    }
  }

  num_preempted_requests = num_offline_decode_preempt_offline_requests +
                           num_online_decode_preempt_online_requests +
                           num_online_decode_preempt_offline_requests +
                           num_online_prefill_preempt_offline_requests;
  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(running_requests_,
                                      running_sequences_,
                                      running_sequences_budgets_);

  bool is_batches_empty =
      (std::all_of(batches.begin(), batches.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      }));
  if (!is_batches_empty) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
    kv_cache_manager_->transfer_blocks(batches);
  } else {
    kv_cache_manager_->transfer_blocks();
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            prefill_queue_->size() + decode_queue_->size());

  GAUGE_ADD(num_preempted_requests, num_preempted_requests);
  GAUGE_ADD(num_offline_decode_preempt_offline_requests,
            num_offline_decode_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_online_requests,
            num_online_decode_preempt_online_requests);
  GAUGE_ADD(num_online_prefill_preempt_offline_requests,
            num_online_prefill_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_offline_requests,
            num_online_decode_preempt_offline_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(kv_cache_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(kv_cache_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(kv_cache_manager_->num_used_blocks()));
  return batches;
}

void PDOOCScheduler::handle_prefill_interruption() {
  std::vector<std::shared_ptr<Request>> offline_requests_to_preempt;

  // Find all offline requests in running_requests_ and mark them for preemption
  for (auto it = running_requests_.begin(); it != running_requests_.end();
       ++it) {
    if (*it && (*it)->offline()) {
      offline_requests_to_preempt.emplace_back(*it);
      *it = nullptr;  // Remove from running_requests_
    }
  }

  // Preempt offline requests and move them back to waiting queue
  for (auto& request : offline_requests_to_preempt) {
    // Deallocate KV cache blocks
    kv_cache_manager_->deallocate(request.get());

    // Mark request as preempted
    request->set_preempted();

    // Add back to offline waiting queue for rescheduling
    VLOG(1) << "Preempting offline request due to interruption: "
            << request->request_id();
    VLOG(1) << "prefill_queue_offline_->size() before push: "
            << prefill_queue_offline_->size();
    prefill_queue_offline_->push(request);

    VLOG(1) << "Preempted offline request due to interruption: "
            << request->request_id();
  }

  LOG(INFO) << "Handled prefill interruption: preempted "
            << offline_requests_to_preempt.size() << " offline requests";
}

void PDOOCScheduler::decode_step(const absl::Duration& timeout) {
  decode_step_global_batch_req_lens_.clear();
  ContinuousScheduler::step(timeout);
  // DEBUG ONLY
  if (last_batch_lengths_.size()) {
    VLOG(1) << " - PERF_MODEL_DEBUG: "
            << llm_flops_.decode(last_batch_lengths_).latency * 1000 << " ms";
  }

  // Check memory utilization rate to see if the scheduler is able to pull an
  // offline request from a P node
  if (check_able_to_pull()) {
    // Trigger decode_send_pull_signal()
    decode_send_pull_signal_pending_.store(false);
    decode_send_pull_signal_cv_.notify_all();
  }
  last_decode_step_global_batch_req_lens_ = decode_step_global_batch_req_lens_;
}

void PDOOCScheduler::handle_prefill_requests_impl(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    RequestPriorityQueue* waiting_priority_queue,
    size_t& num_online_prefill_preempt_offline_requests,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // Handle new request prompt first.
  // Include those requests that are preempted by others.
  //
  // schedule the prefill requests in the waiting priority queue until budgets
  // are exhausted.
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  //
  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contian many sequences, so we should check here.

  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  while (!waiting_priority_queue->empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 && latency_budget > estimate_latency) {
    if (!options_.enable_disagg_pd() &&
        kv_cache_manager_->kv_cache_utilization() >=
            ::xllm::SchedulerConfig::get_instance()
                .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(waiting_priority_queue->top());
    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get());
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the request priority queue
      waiting_priority_queue->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    if (!kv_cache_manager_->update_prefetch_result(
            request, options_.prefetch_timeout())) {
      waiting_priority_queue->pop_top();
      waiting_priority_queue->push(request);
      continue;
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    // TODO: can refactor like handle_decode otherwise request with multiple
    // long sequences may stuck when n>1
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      // FIXME: use actual num_tokens to handle
      // Currently overestimating the number of tokens actually processed when
      // enable prefix cache
      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      num_tokens = maybe_align_cp_prefill_tokens(
          prefill_sequence.get(), num_tokens, options_.cp_size());
      const size_t target_num_tokens =
          prefill_sequence->kv_cache_tokens_num() + num_tokens;
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // preempt offline decode
      if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                       target_num_tokens)) {
        can_schedule = false;
        kv_cache_manager_->deallocate(prefill_sequence.get());
        blocks_exhausted = true;
        break;
      }

      // OPTIMIZE for multi-slo requests
      // for prefill requests, check latency after prefix cache match
      double seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        seq_estimate_latency =
            profile_manager_->predict_step_time(prefill_sequence.get(), false);
        if ((estimate_latency + allocated_estimate_latency +
                 seq_estimate_latency >
             latency_budget) &&
            (!running_sequences_.empty())) {
          // release shared prefix blocks
          kv_cache_manager_->deallocate(prefill_sequence.get());
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_->deallocate(seq);
      }
      break;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    estimate_latency += allocated_estimate_latency;
    waiting_priority_queue->pop_top();
    running_requests_.emplace_back(request);
    request->record_num_prefix_cache_tokens();
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget);
  }
  // maybe can pre-compute if prompt beyond length
  if (running_sequences_.empty() && !waiting_priority_queue->empty() &&
      decode_queue_->empty()) {
    std::shared_ptr<Request> request(waiting_priority_queue->top());
    waiting_priority_queue->pop_top();
    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budegt.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(INFO) << "latency budegt: " << latency_budget
                << ", estimate latency: " << estimate_latency;
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

void PDOOCScheduler::handle_decode_requests_impl(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_offline_decode_preempt_offline_requests,
    size_t& num_online_decode_preempt_online_requests,
    size_t& num_online_decode_preempt_offline_requests,
    RequestPriorityQueue* running_queue) {
  std::vector<Sequence*> candidate_sequences;
  std::vector<size_t> candidate_token_budgets;

  while (!running_queue->empty() &&
         remaining_token_budget > min_speculative_tokens_required_ &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = running_queue->top();
    // TODO: check if request is timeout

    const size_t num_sequences = request->sequences().size();
    candidate_sequences.clear();
    candidate_token_budgets.clear();
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;

    if (request->check_beam_search()) {
      std::vector<Sequence*> active_sequences;
      active_sequences.reserve(num_sequences);
      for (auto& seq : request->sequences()) {
        if (!seq->finished()) {
          active_sequences.emplace_back(seq.get());
        }
      }
      if (active_sequences.empty()) {
        running_queue->pop_top();
        continue;
      }

      const size_t decode_step_tokens = min_speculative_tokens_required_ + 1;
      if (decode_step_tokens * active_sequences.size() >
              remaining_token_budget ||
          active_sequences.size() > remaining_seq_budget) {
        has_enough_budget = false;
      }

      if (has_enough_budget && options_.enable_latency_aware_schedule() &&
          !(options_.instance_role().has_value() &&
            options_.instance_role().value() == InstanceRole::PREFILL)) {
        for (auto* sequence : active_sequences) {
          const double seq_estimate_latency =
              profile_manager_->predict_step_time(sequence, false);
          if (estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              latency_budget) {
            has_enough_budget = false;
            break;
          }
          allocated_estimate_latency += seq_estimate_latency;
        }
      }

      // Reset estimation value. It will be recomputed on successful allocation.
      allocated_estimate_latency = 0.0;

      if (has_enough_budget) {
        const size_t block_size = kv_cache_manager_->block_size();
        size_t needed_blocks = 0;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + min_speculative_tokens_required_;
          needed_blocks += estimate_decode_extra_blocks(
              sequence, updated_num_tokens, block_size);
        }

        const int32_t dp_rank = active_sequences.front()->dp_rank();
        const size_t free_blocks =
            get_sequence_free_blocks_for_rank(kv_cache_manager_, dp_rank);
        if (needed_blocks > free_blocks) {
          has_enough_blocks = false;
        }
      }

      if (has_enough_budget && has_enough_blocks) {
        bool allocate_failed = false;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + min_speculative_tokens_required_;
          if (!kv_cache_manager_->allocate(sequence, updated_num_tokens)) {
            allocate_failed = true;
            break;
          }
          if (sequence->if_cache_block_for_prefill()) {
            kv_cache_manager_->cache(sequence);
          }
          candidate_sequences.emplace_back(sequence);
          candidate_token_budgets.emplace_back(decode_step_tokens);
          allocated_tokens += decode_step_tokens;
          allocated_seqs += 1;
        }

        if (allocate_failed) {
          LOG(ERROR) << "Beam strict scheduling allocation failed. "
                     << "request_id=" << request->request_id()
                     << ", beam=" << request->check_beam_search();
          // Fallback to full request deallocation to avoid inconsistent
          // per-sequence states.
          clear_mtp_bootstrap(request.get());
          kv_cache_manager_->deallocate(request.get());
          running_queue->pop_top();
          request->set_preempted();
          if (request->offline()) {
            prefill_queue_offline_->push(request);
          } else {
            prefill_queue_->push(request);
          }
          continue;
        }

        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          for (auto* sequence : candidate_sequences) {
            allocated_estimate_latency +=
                profile_manager_->predict_step_time(sequence, false);
          }
        }
      }
    } else {
      for (auto& sequence : request->sequences()) {
        if (sequence->finished()) {
          continue;
        }
        // no budget left
        double seq_estimate_latency = 0;
        if (options_.enable_latency_aware_schedule()
            // force not enabled on prefill node (only offline req decode here)
            && !(options_.instance_role().has_value() &&
                 options_.instance_role().value() == InstanceRole::PREFILL)) {
          seq_estimate_latency =
              profile_manager_->predict_step_time(sequence.get(), false);
          if (estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              latency_budget) {
            has_enough_budget = false;
            break;
          }
        }
        if (allocated_tokens + min_speculative_tokens_required_ >=
                remaining_token_budget ||
            allocated_seqs >= remaining_seq_budget) {
          has_enough_budget = false;
          break;
        }
        // sequence token already appended
        size_t updated_num_tokens =
            sequence->num_tokens() + min_speculative_tokens_required_;
        // no blocks left
        if (!kv_cache_manager_->allocate(sequence.get(), updated_num_tokens)) {
          has_enough_blocks = false;
          break;
        }

        if (sequence->if_cache_block_for_prefill()) {
          kv_cache_manager_->cache(sequence.get());
        }

        // update the allocated tokens for the sequence
        allocated_tokens += min_speculative_tokens_required_ + 1;
        allocated_seqs += 1;
        allocated_estimate_latency += seq_estimate_latency;
        candidate_sequences.emplace_back(sequence.get());
        candidate_token_budgets.emplace_back(min_speculative_tokens_required_ +
                                             1);
      }
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    // schedule candidates in the request if there are enough blocks
    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue->pop_top();
      // add the request to the batch
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      estimate_latency += allocated_estimate_latency;

      continue;
    }

    // budget exhausted, do partially schedule the request
    if (!has_enough_budget) {
      handle_abnormal_request(running_queue,
                              candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              allocated_estimate_latency,
                              remaining_token_budget,
                              remaining_seq_budget,
                              estimate_latency,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      break;
    }

    // memory exhausted, try to preempt lowest priority request
    // continue to evict blocks until enough or no other requests that can be
    // preempted
    if (options_.enable_online_preempt_offline() && !request->offline() &&
        !decode_queue_->empty()) {
      std::shared_ptr<Request> request_to_preempt = decode_queue_->back();
      ++num_online_decode_preempt_offline_requests;
      clear_mtp_bootstrap(request_to_preempt.get());
      kv_cache_manager_->deallocate(request_to_preempt.get());
      decode_queue_->pop_back();
      // add preemptable request to waiting priority queue
      request_to_preempt->set_preempted();
      prefill_queue_offline_->push(request_to_preempt);
      continue;
    } else if (running_queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue->back();
      if (request_to_preempt.get() != request.get()) {
        // TO IMPROVE: kv cache offload to cpu
        clear_mtp_bootstrap(request_to_preempt.get());
        kv_cache_manager_->deallocate(request_to_preempt.get());
        running_queue->pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        if (request_to_preempt->offline()) {
          ++num_offline_decode_preempt_offline_requests;
          prefill_queue_offline_->push(request_to_preempt);
        } else {
          ++num_online_decode_preempt_online_requests;
          prefill_queue_->push(request_to_preempt);
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(running_queue,
                            candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            allocated_estimate_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            estimate_latency,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    break;
  }
}

// copy+modify from ContinuousScheduler::handle_decode_requests
// Due to limitations in superclass' implementation, manual maintenance of
// decode_step_global_batch_req_lens_ is required
void PDOOCScheduler::handle_decode_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_offline_decode_preempt_offline_requests,
    size_t& num_online_decode_preempt_online_requests,
    size_t& num_online_decode_preempt_offline_requests,
    RequestPriorityQueue* running_queue) {
  // only used in decode step
  if (options_.instance_role().value() != InstanceRole::DECODE) {
    return handle_decode_requests_impl(
        latency_budget,
        estimate_latency,
        remaining_token_budget,
        remaining_seq_budget,
        num_offline_decode_preempt_offline_requests,
        num_online_decode_preempt_online_requests,
        num_online_decode_preempt_offline_requests,
        running_queue);
  }

  // LOG(INFO) << "PDOOCScheduler::handle_decode_requests, start."
  //           << options_.enable_latency_aware_schedule()
  //           << ", max_global_tpot_ms=" << options_.max_global_tpot_ms();

  double DECODE_SLO = options_.max_global_tpot_ms() / 1000.0;
  int CHECK_INTERVAL = 3;

  int num_offline = 0;
  double new_batch_latency = 0.0;
  std::vector<Sequence*> candidate_sequences;
  std::vector<size_t> candidate_token_budgets;

  while (!running_queue->empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = running_queue->top();
    // TODO: check if request is timeout

    const size_t num_sequences = request->sequences().size();
    candidate_sequences.clear();
    candidate_token_budgets.clear();
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_batch_latency = 0;
    if (request->offline()) {
      ++num_offline;
    }

    for (auto& sequence : request->sequences()) {
      // skip finished sequence.
      if (sequence->finished()) {
        continue;
      }
      // no budget left

      decode_step_global_batch_req_lens_.push_back(
          sequence.get()->num_tokens());
      if (decode_step_global_batch_req_lens_.size() % CHECK_INTERVAL == 0 ||
          !new_batch_latency) {
        new_batch_latency =
            llm_flops_.decode(decode_step_global_batch_req_lens_).latency;
        decode_last_step_latency_ = new_batch_latency;

        if (new_batch_latency > DECODE_SLO * 0.98) {
          LOG(INFO) << "DEBUG - Estimated decode latency for request "
                    << request->request_id() << " with "
                    << decode_step_global_batch_req_lens_.size() << " reqs ("
                    << num_offline << " offline): " << new_batch_latency << "s";
          LOG(INFO)
              << "DEBUG - Estimated decode latency is close to or exceeds "
                 "SLO, stop scheduling more requests in this batch.";
          has_enough_budget = false;
          break;
        }
      }

      // size_t seq_estimate_latency = 0;
      // if (options_.enable_latency_aware_schedule()) {
      //   seq_estimate_latency =
      //       profile_manager_->predict_step_time(sequence.get(), false);
      //   if (estimate_latency + allocated_estimate_latency +
      //           seq_estimate_latency >
      //       latency_budget) {
      //     has_enough_budget = false;
      //     break;
      //   }
      // }

      if (allocated_tokens + options_.num_speculative_tokens() >=
              remaining_token_budget ||
          allocated_seqs >= remaining_seq_budget) {
        has_enough_budget = false;
        break;
      }
      // sequence token already appended
      size_t updated_num_tokens =
          sequence->num_tokens() + options_.num_speculative_tokens();
      // no blocks left
      if (!kv_cache_manager_->allocate(sequence.get(), updated_num_tokens)) {
        has_enough_blocks = false;
        break;
      }

      if (sequence->if_cache_block_for_prefill()) {
        kv_cache_manager_->cache(sequence.get());
      }

      // update the allocated tokens for the sequence
      allocated_tokens += options_.num_speculative_tokens() + 1;
      allocated_seqs += 1;
      allocated_estimate_batch_latency = new_batch_latency * 1000;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(options_.num_speculative_tokens() +
                                           1);
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    // schedule candidates in the request if there are enough blocks
    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue->pop_top();
      // add the request to the batch
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      estimate_latency = allocated_estimate_batch_latency;

      // LOG(INFO) << "Scheduled request " << request->request_id()
      //           << "remaining_token_budget: " << remaining_token_budget
      //           << ", remaining_seq_budget: " << remaining_seq_budget
      //           << ", estimate_latency: " << estimate_latency;

      continue;
    }

    // budget exhausted, do partially schedule the request
    if (!has_enough_budget) {
      handle_abnormal_request(running_queue,
                              candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              allocated_estimate_batch_latency,
                              remaining_token_budget,
                              remaining_seq_budget,
                              estimate_latency,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      break;
    }

    // memory exhausted, try to preempt lowest priority request
    // continue to evict blocks until enough or no other requests that can be
    // preempted
    if (options_.enable_online_preempt_offline() && !request->offline() &&
        !decode_queue_->empty()) {
      std::shared_ptr<Request> request_to_preempt = decode_queue_->back();
      ++num_online_decode_preempt_offline_requests;
      kv_cache_manager_->deallocate(request_to_preempt.get());
      decode_queue_->pop_back();
      // add preemptable request to waiting priority queue
      request_to_preempt->set_preempted();
      prefill_queue_offline_->push(request_to_preempt);
      continue;
    } else if (running_queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue->back();
      if (request_to_preempt.get() != request.get()) {
        // TO IMPROVE: kv cache offload to cpu
        kv_cache_manager_->deallocate(request_to_preempt.get());
        running_queue->pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        if (request_to_preempt->offline()) {
          ++num_offline_decode_preempt_offline_requests;
          prefill_queue_offline_->push(request_to_preempt);
        } else {
          ++num_online_decode_preempt_online_requests;
          prefill_queue_->push(request_to_preempt);
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(running_queue,
                            candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            allocated_estimate_batch_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            estimate_latency,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    break;
  }
}

void PDOOCScheduler::decode_send_pull_signal() {
  while (true) {
    // Wait until step thread triggers
    std::unique_lock<std::mutex> lock(decode_send_pull_signal_mtx_);
    decode_send_pull_signal_cv_.wait(
        lock, [this] { return !decode_send_pull_signal_pending_.load(); });

    if (waiting_pull_finished_.load()) {
      // FIXME Add timeout for waiting_pull_finished_ in unreliable network
      // conditions.
      decode_send_pull_signal_pending_.store(true);
      absl::SleepFor(absl::Milliseconds(100));
      continue;
    }

    VLOG(1) << "Sending a pull signal to a P node";

    // WIP Send a pull signal to a P node

    // Select a P node
    std::string selected_prefill_instance = select_prefill_instance();
    VLOG(1) << "Selected prefill instance: " << selected_prefill_instance;

    // Build a stub
    proto::DisaggPDService_Stub* stub =
        create_rpc_channel(selected_prefill_instance);
    if (!stub) {
      LOG(ERROR) << "Failed to create RPC channel to prefill instance: "
                 << selected_prefill_instance;
      decode_send_pull_signal_pending_.store(true);
      absl::SleepFor(absl::Milliseconds(100));
      continue;
    }

    // Send a pull signal to the selected prefill instance
    proto::PullSignal pull_signal;
    pull_signal.set_source_instance_name(xservice_client_->get_instance_name());

    google::protobuf::uint64 preferred_len = 0;
    auto available_tokens = kv_cache_manager_->num_free_blocks()[0] *
                            kv_cache_manager_->block_size();
    pull_signal.set_max_total_len(available_tokens);

    preferred_len = llm_flops_.decode_preferred_req_len(
        last_decode_step_global_batch_req_lens_,
        linear_saturation_bs_,
        options_.max_global_tpot_ms(),
        available_tokens);
    pull_signal.set_preferred_req_len(preferred_len);

    proto::Status resp;
    brpc::Controller cntl;
    stub->SendPullSignal(&cntl, &pull_signal, &resp, nullptr);

    // Pend until next trigger
    if (cntl.Failed() || !resp.ok()) {
      VLOG(1) << "SendPullSignal failed";
      if (cntl.Failed()) {
        VLOG(1) << "cntl.Failed";
      } else {
        VLOG(1) << "!resp.ok()";
      }
      waiting_pull_finished_.store(false);
    } else {
      waiting_pull_finished_.store(true);
    }
    decode_send_pull_signal_pending_.store(true);
    absl::SleepFor(absl::Milliseconds(100));
  }
}

// prefill send new request to remote instance
void PDOOCScheduler::dispatch_requests() {
  while (true) {
    const auto timeout = std::chrono::milliseconds(100);
    // Wait for online request until timeout.
    // If timeout, try to get offline request once. If no offline request,
    // continue to wait for online request. This can avoid offline request
    // blocking online request for too long time.
    std::shared_ptr<Request> request;
    if (!prefill_request_queue_.wait_dequeue_timed(request, timeout)) {
      if (!prefill_request_queue_offline_.try_dequeue(request)) {
        continue;
      }
    }

    if (request == nullptr) {
      // nullptr is a signal to exit
      break;
    }

    if (request->offline()) {
      // Handle offline requests locally. No need to dispatch them to decoding
      // instances.
      request_queue_.write(request);
      continue;
    }

    // Create a RPC stub with given decoding instance.
    std::vector<std::shared_ptr<Request>> requests;
    requests.emplace_back(request);
    std::string selected_instance = "";
    proto::DisaggPDService_Stub* stub = nullptr;
    if (!request->state().decode_address.empty() && requests.size() == 1) {
      selected_instance = request->state().decode_address;
      stub = create_rpc_channel(request->state().decode_address);
    }

    // If no decoding instance is specified, randomly select one to create a
    // stub.
    if (selected_instance.empty() && !stub) {
      int try_decode_count = 0;
      while (!stub) {
        if (try_decode_count == decode_inst_names_.size()) {
          LOG(FATAL) << "Can not connect to all decode instances.";
        }
        ++try_decode_count;
        selected_instance = select_decode_instance();
        stub = create_rpc_channel(selected_instance);
      }
    }

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      for (auto& req : requests) {
        req_to_channel_map_[req->request_id()] = stub;
      }
    }

    // TODO: send the request to the selected D instance
    // Send 'DisaggRequests' and recv 'DisaggResponses'
    xllm::proto::DisaggRequests reqs;
    xllm::proto::DisaggResponses resps;

    // Build DisaggRequests proto from Request objects
    build_disagg_requests(requests, reqs);

    // TODO: sync rpc here currently
    brpc::Controller cntl;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);

    // check reqs which can not dispatch to D instance,
    // and push back to prefill_request_queue_
    CHECK_EQ(requests.size(), resps.resps().size());
    for (size_t i = 0; i < requests.size(); ++i) {
      CHECK(!requests[i]->offline());
      if (resps.resps()[i].status_code() != 200) {
        // push back to prefill_request_queue_
        if (requests[i]->offline()) {
          prefill_request_queue_offline_.enqueue(requests[i]);
        } else {
          prefill_request_queue_.enqueue(requests[i]);
        }

      } else {
        for (auto& sequence : requests[i]->sequences()) {
          TransferKVInfo info;
          info.request_id = requests[i]->request_id();
          for (auto& bid : resps.resps()[i].blocks_ids()) {
            info.remote_blocks_ids.emplace_back(bid);
          }
          if (resps.resps()[i].linear_state_id() >= 0) {
            info.remote_linear_state_ids.emplace_back(
                resps.resps()[i].linear_state_id());
          }
          info.dp_rank = resps.resps()[i].dp_rank();
          // TODO: remote_instances_info_ is not multi-thread safe.
          info.remote_instance_info = remote_instances_info_[selected_instance];
          sequence->kv_state().set_transfer_kv_info(std::move(info));
        }

        // push to request_queue_, and will be executed by engine.
        request_queue_.write(requests[i]);
        VLOG(1) << "Put a request into request_queue_";
      }
    }
    // WIP Interrupt ongoing offline prefill requests when online requests come
    if (!requests.empty()) {
      if (options_.enable_forward_interruption() &&
          step_status_ == StepStatus::OFFLINE_PREFILL) {
        InterruptionBus::get_instance().publish(true);
        // VLOG(1) << "Sent an interruption signal";
        // VLOG(1) << "Interruption disabled";
      }
    }
  }
}

void PDOOCScheduler::prefill_send_first_generation() {
  if (running_sequences_.size() == 0) {
    return;
  }

  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(running_requests_.size());
  {
    std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
    for (size_t i = 0; i < running_requests_.size(); ++i) {
      auto request = running_requests_[i];
      if (request == nullptr) {
        continue;
      }
      if (request->offline()) {
        // Do not send offline running requests to D initiatively
        continue;
      }
      // Check if the request is a recently completed prefill request
      if (request->sequences()[0]->num_generated_tokens() == 1) {
        if (remote_requests_map_.find(request->request_id()) !=
            remote_requests_map_.end()) {
          LOG(FATAL)
              << "Two request has the same request_id, check the requests map.";
        }
        remote_requests_map_[request->request_id()] = request;
        requests.emplace_back(request);

        running_requests_[i] = nullptr;
      }
    }
  }

  // No prefill request needs to be transferred to decode.
  if (requests.size() == 0) {
    return;
  }

  prefill_threadpool_.schedule([this,
                                requests = std::move(requests)]() mutable {
    // send request first token to remote instance
    // TODO: here we only support one sequence for now.
    for (auto& request : requests) {
      // TODO: support batch request later
      proto::DisaggGenerationsRequests gens;
      auto gen = gens.mutable_multi_gens()->Add();
      gen->set_req_id(request->request_id());
      if (request->sequences()[0]->first_token().has_value()) {
        auto token = gen->mutable_tokens()->Add();
        token->set_token_id(
            request->sequences()[0]->first_token().value().token_id);
        token->set_time_to_first_token_latency_seconds(
            request->sequences()[0]->time_to_first_token_latency_seconds());
        if (request->sequences()[0]
                ->first_token()
                .value()
                .token_logprob.has_value()) {
          token->set_logprob(request->sequences()[0]
                                 ->first_token()
                                 .value()
                                 .token_logprob.value());
          token->set_has_logprob(true);
        } else {
          token->set_has_logprob(false);
        }
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_tokens(),
            request->sequences()[0]->first_token().value().token_top_tokens);
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_logprobs(),
            request->sequences()[0]->first_token().value().token_top_logprobs);
      }
      gen->set_kv_cache_transfer_mode(options_.kv_cache_transfer_mode());
      if (options_.kv_cache_transfer_mode() == "PULL") {
        ADD_VECTOR_TO_PROTO(gen->mutable_cluster_ids(),
                            instance_info_.cluster_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_addrs(), instance_info_.addrs);

        const auto blocks =
            request->sequences()[0]->kv_state().blocks(BlockType::KV);
        std::vector<uint64_t> block_ids;
        block_ids.reserve(blocks.size());
        for (const auto& block : blocks) {
          block_ids.push_back(block.id());
        }
        ADD_VECTOR_TO_PROTO(gen->mutable_block_ids(), block_ids);
        gen->set_linear_state_id(
            request->sequences()[0]->get_single_block_id());
        gen->set_dp_size(instance_info_.dp_size);
        gen->set_dp_rank(request->sequences()[0]->dp_rank());
      }

      // send first gens to remote instance
      proto::DisaggPDService_Stub* stub = nullptr;
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        // now we only support one request once.
        stub = req_to_channel_map_[request->request_id()];
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      stub->FirstGeneration(&cntl, &gens, &resp, nullptr);

      if (cntl.Failed() || !resp.ok()) {
        LOG(ERROR) << "Failed to send first generation, " << cntl.ErrorText()
                   << ", staus: " << resp.ok();
      }
      {
        std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
        remote_requests_map_.erase(request->request_id());
      }
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      kv_cache_manager_->deallocate(request.get());
    }
  });
}

// request is received from prefill
bool PDOOCScheduler::decode_schedule(std::shared_ptr<Request>& request,
                                     const std::string& prefill_instance_name) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  proto::DisaggPDService_Stub* stub = create_rpc_channel(prefill_instance_name);
  if (!stub) {
    LOG(ERROR) << "Failed to create rpc channel for prefill instance: "
               << prefill_instance_name;
    kv_cache_manager_->deallocate(request.get());
    return false;
  }

  // TODO: check request_id, duplicate ids are not allowed
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    if (received_request_map_.find(request->request_id()) !=
        received_request_map_.end()) {
      LOG(FATAL) << "Decode receive same request_id from prefill.";
    }
    received_request_map_[request->request_id()] = request;
  }

  {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    req_to_channel_map_[request->request_id()] = stub;
  }

  if (request->offline()) {
    waiting_pull_finished_.store(false);
  }

  return true;
}

bool PDOOCScheduler::decode_recv_multi_generations(
    const std::string& req_id,
    const std::vector<proto::RemoteToken>& migration_tokens,
    const std::string& kv_cache_transfer_mode,
    std::vector<uint64_t> src_cluster_ids,
    std::vector<std::string> src_addrs,
    std::vector<uint64_t> src_block_ids,
    int32_t src_linear_state_id,
    int32_t src_dp_size,
    int32_t src_dp_rank) {
  // push to request_queue_, and will be executed by engine.
  std::shared_ptr<Request> request = nullptr;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << req_id;
      return false;
    }
    request = it->second;
    received_request_map_.erase(it);
  }

  // Enable checking whether to skip the prefill token
  if (request->state().stream) {
    request->sequences()[0]->enable_checking_prefill_token();
  }

  // Add all migration tokens to the sequence
  for (const auto& remote_token : migration_tokens) {
    if (remote_token.time_to_first_token_latency_seconds() > 0 &&
        request->sequences()[0]->time_to_first_token_latency_seconds() <= 0) {
      request->sequences()[0]->set_time_to_first_token_latency_seconds(
          remote_token.time_to_first_token_latency_seconds());
    }

    Token token(remote_token.token_id());
    if (remote_token.has_logprob()) {
      token.logprob = remote_token.logprob();
      if (!remote_token.top_tokens().empty() &&
          !remote_token.top_logprobs().empty()) {
        // Convert from repeated fields to vectors
        std::vector<int64_t> top_tokens(remote_token.top_tokens().begin(),
                                        remote_token.top_tokens().end());
        std::vector<float> top_logprobs(remote_token.top_logprobs().begin(),
                                        remote_token.top_logprobs().end());
        token.top_tokens = top_tokens;
        token.top_logprobs = top_logprobs;
      }
    }

    // Add token to sequence
    if (enable_schedule_overlap()) {
      Token fake_token(-1);
      request->sequences()[0]->append_token(fake_token);
      request->sequences()[0]->update_last_step_token(token);
    } else {
      request->sequences()[0]->append_token(token);
    }
  }

  // pull kv cache (only needed once for the entire request)
  if (kv_cache_transfer_mode == "PULL") {
    const auto blocks =
        request->sequences()[0]->kv_state().blocks(BlockType::KV);
    std::vector<uint64_t> dst_block_ids;
    dst_block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      dst_block_ids.push_back(block.id());
    }
    std::vector<uint64_t> src_linear_state_ids;
    std::vector<uint64_t> dst_linear_state_ids;
    if (src_linear_state_id >= 0 &&
        request->sequences()[0]->get_single_block_id() >= 0) {
      src_linear_state_ids.emplace_back(src_linear_state_id);
      dst_linear_state_ids.emplace_back(
          request->sequences()[0]->get_single_block_id());
    }

    int32_t dst_dp_rank = request->sequences()[0]->dp_rank();
    if (!engine_->pull_kv_blocks(src_dp_size,
                                 src_dp_rank,
                                 src_cluster_ids,
                                 src_addrs,
                                 src_block_ids,
                                 dst_dp_rank,
                                 dst_block_ids,
                                 src_linear_state_ids,
                                 dst_linear_state_ids)) {
      LOG(ERROR) << "Failed to pull KV blocks for offline migration, req_id: "
                 << req_id;
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
  }

  request_queue_.write(request);
  return true;
}

// TODO Need parameters tuning
bool PDOOCScheduler::check_able_to_pull() {
  // Estimated usage of current requests: half of current used blocks.
  return kv_cache_manager_->kv_cache_utilization() < 0.9 &&
         decode_last_step_latency_ <
             options_.max_global_tpot_ms() / 1000.0 * 0.9;
}

bool PDOOCScheduler::write_pull_signal(const proto::PullSignal& pull_signal) {
  if (pull_signals_.enqueue(pull_signal)) {
    VLOG(1) << "Wrote a pull signal into a queue: "
            << pull_signal.source_instance_name();
    return true;
  } else {
    VLOG(1) << "Failed to write a pull signal into a queue";
    return false;
  }
}

void PDOOCScheduler::prepare_offline_dispatch_queue() {
  // Read pull signals from pull_signals_ queue
  proto::PullSignal pull_signal;
  std::deque<proto::PullSignal> unused_signals;
  while (pull_signals_.try_dequeue(pull_signal)) {
    auto preferred_len = pull_signal.preferred_req_len();
    auto max_len = pull_signal.max_total_len();

    // Find an offline decoding request in running_requests_ to move to dispatch
    // queue
    size_t selected_red_idx = running_requests_.size();
    int minimal_diff = std::numeric_limits<int>::max();
    std::shared_ptr<Request> offline_request = nullptr;
    for (size_t i = 0; i < running_requests_.size(); ++i) {
      auto& request = running_requests_[i];
      if (request && request->offline() && !request->sequences().empty() &&
          !request->sequences()[0]->is_chunked_prefill_stage()) {
        size_t req_len = request->sequences()[0]->num_tokens();
        if (req_len <= max_len) {
          size_t diff = preferred_len > req_len ? preferred_len - req_len
                                                : req_len - preferred_len;
          if (diff < minimal_diff) {
            minimal_diff = diff;
            selected_red_idx = i;
            offline_request = request;
          }
        }
      }
    }

    if (offline_request) {
      running_requests_[selected_red_idx] =
          nullptr;  // Remove the request from running_requests_
      // Add to offline dispatch queue with the source instance name
      std::pair<std::shared_ptr<Request>, std::string> dispatch_pair =
          std::make_pair(offline_request, pull_signal.source_instance_name());
      offline_requests_to_dispatch_.enqueue(dispatch_pair);

      VLOG(1) << "Moved offline request " << offline_request->request_id()
              << " to dispatch queue for instance "
              << pull_signal.source_instance_name()
              << "\n        preferred_len: " << preferred_len
              << ", max_len: " << max_len << ", selected len: "
              << offline_request->sequences()[0]->num_tokens();
    } else {
      // If no offline request, put the signal back for future use.
      unused_signals.push_back(pull_signal);
    }
  }

  while (!unused_signals.empty()) {
    pull_signal = unused_signals.front();
    pull_signals_.enqueue(pull_signal);
    unused_signals.pop_front();
  }
}

void PDOOCScheduler::dispatch_offline_requests() {
  while (true) {
    const auto timeout = std::chrono::milliseconds(100);
    // Get offline request with target instance from dispatch queue
    std::pair<std::shared_ptr<Request>, std::string> dispatch_pair;
    if (!offline_requests_to_dispatch_.wait_dequeue_timed(dispatch_pair,
                                                          timeout)) {
      continue;
    }

    VLOG(1) << "Dispatching offline requests";

    auto request = dispatch_pair.first;
    auto target_instance = dispatch_pair.second;

    if (request == nullptr) {
      // nullptr is a signal to exit
      break;
    }

    // Create a RPC stub with the target decoding instance
    proto::DisaggPDService_Stub* stub = create_rpc_channel(target_instance);
    if (!stub) {
      LOG(ERROR) << "Failed to create RPC channel to target instance: "
                 << target_instance;
      // Put the request back to dispatch queue for retry
      offline_requests_to_dispatch_.enqueue(dispatch_pair);
      absl::SleepFor(absl::Milliseconds(100));
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      req_to_channel_map_[request->request_id()] = stub;
    }

    // Build DisaggRequests proto from Request object
    std::vector<std::shared_ptr<Request>> requests;
    requests.emplace_back(request);
    xllm::proto::DisaggRequests reqs;
    xllm::proto::DisaggResponses resps;
    build_disagg_requests(requests, reqs);

    // Send to target decode instance
    brpc::Controller cntl;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);

    // Check response and handle accordingly
    if (cntl.Failed() || resps.resps().empty() ||
        resps.resps()[0].status_code() != 200) {
      LOG(ERROR) << "Failed to dispatch offline request "
                 << request->request_id() << " to " << target_instance
                 << ". Status: "
                 << (resps.resps().empty() ? -1
                                           : resps.resps()[0].status_code());
      // Put the request back to dispatch queue for retry
      offline_requests_to_dispatch_.enqueue(dispatch_pair);
    } else {
      // Successfully dispatched, set up KV transfer info
      for (auto& sequence : request->sequences()) {
        TransferKVInfo info;
        info.request_id = request->request_id();
        for (auto& bid : resps.resps()[0].blocks_ids()) {
          info.remote_blocks_ids.emplace_back(bid);
        }
        if (resps.resps()[0].linear_state_id() >= 0) {
          info.remote_linear_state_ids.emplace_back(
              resps.resps()[0].linear_state_id());
        }
        info.dp_rank = resps.resps()[0].dp_rank();
        info.remote_instance_info = remote_instances_info_[target_instance];
        sequence->kv_state().set_transfer_kv_info(std::move(info));
      }

      // Move to transfer queue for KV cache transfer
      std::pair<std::shared_ptr<Request>, std::string> transfer_pair =
          std::make_pair(request, target_instance);
      offline_requests_to_transfer_.enqueue(transfer_pair);
    }
  }
}

std::string PDOOCScheduler::select_decode_instance() {
  // get allocated decode instance list from Master
  while (decode_inst_names_.empty()) {
    decode_inst_names_ = xservice_client_->get_static_decode_list();
    if (!decode_inst_names_.empty()) {
      LOG(INFO) << "Get PD decode instance list: "
                << absl::StrJoin(decode_inst_names_, "; ");
      break;
    }
    sleep(1);
  }

  // select a D instance use RR currently.
  // TODO: use better decode selection strategy later. maybe different
  // strategy for offline and online request. or implement in xllm service.
  std::string selected_instance = decode_inst_names_[current_decode_idx_];
  current_decode_idx_ = (++current_decode_idx_) % decode_inst_names_.size();

  return selected_instance;
}

std::string PDOOCScheduler::select_prefill_instance() {
  // get allocated prefill instance list from Master
  while (prefill_inst_names_.empty()) {
    prefill_inst_names_ = xservice_client_->get_static_prefill_list();
    if (!prefill_inst_names_.empty()) {
      LOG(INFO) << "Get PD prefill instance list: "
                << absl::StrJoin(prefill_inst_names_, "; ");
      break;
    }
    sleep(1);
  }

  // select a P instance use RR currently.
  // TODO: use better prefill selection strategy later.
  std::string selected_instance = prefill_inst_names_[current_prefill_idx_];
  current_prefill_idx_ = (++current_prefill_idx_) % prefill_inst_names_.size();

  return selected_instance;
}

void PDOOCScheduler::prefill_send_multi_generations() {
  // Process offline requests from transfer queue
  std::vector<std::pair<std::shared_ptr<Request>, std::string>> transfer_pairs;
  std::pair<std::shared_ptr<Request>, std::string> transfer_pair;

  // Dequeue all available offline requests to transfer
  while (offline_requests_to_transfer_.try_dequeue(transfer_pair)) {
    transfer_pairs.push_back(transfer_pair);
  }

  // No offline request needs to be transferred to decode.
  if (transfer_pairs.size() == 0) {
    return;
  }

  prefill_threadpool_.schedule([this,
                                transfer_pairs =
                                    std::move(transfer_pairs)]() mutable {
    // Add requests to remote_requests_map_ for response handling
    {
      std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
      for (auto& pair : transfer_pairs) {
        auto& request = pair.first;
        if (remote_requests_map_.find(request->request_id()) !=
            remote_requests_map_.end()) {
          LOG(FATAL)
              << "Two request has the same request_id, check the requests map.";
        }
        remote_requests_map_[request->request_id()] = request;
      }
    }

    // send multiple tokens to remote instance
    for (auto& pair : transfer_pairs) {
      auto request = pair.first;
      auto target_instance = pair.second;
      proto::DisaggGenerationsRequests multi_reqs;
      auto multi_req = multi_reqs.mutable_multi_gens()->Add();
      multi_req->set_req_id(request->request_id());

      // Get all generated token IDs from the sequence
      auto* sequence = request->sequences()[0].get();
      auto generated_token_ids = sequence->get_generated_tokens();

      // Add all generated token IDs to migration_tokens
      for (size_t token_index = 0; token_index < generated_token_ids.size();
           ++token_index) {
        auto remote_token = multi_req->mutable_tokens()->Add();
        remote_token->set_token_id(generated_token_ids[token_index]);
        remote_token->set_has_logprob(false);
        if (token_index == 0) {
          remote_token->set_time_to_first_token_latency_seconds(
              sequence->time_to_first_token_latency_seconds());
        }
      }

      auto kv_cache_transfer_mode = options_.kv_cache_transfer_mode();
#if defined(USE_DCU)
      if (kv_cache_transfer_mode == "PUSH") {
        // Offline migration happens after prefill forward has produced tokens.
        // Use decode-side pull so KV is copied before the request continues.
        kv_cache_transfer_mode = "PULL";
      }
#endif
      multi_req->set_kv_cache_transfer_mode(kv_cache_transfer_mode);
      if (kv_cache_transfer_mode == "PULL") {
        for (auto cluster_id : instance_info_.cluster_ids) {
          multi_req->mutable_cluster_ids()->Add(cluster_id);
        }
        for (auto& addr : instance_info_.addrs) {
          // multi_req->mutable_addrs()->Add(addr);
          multi_req->add_addrs(addr);
        }

        const auto blocks = sequence->kv_state().blocks(BlockType::KV);
        for (const auto& block : blocks) {
          multi_req->mutable_block_ids()->Add(block.id());
        }
        multi_req->set_linear_state_id(sequence->get_single_block_id());
        multi_req->set_dp_size(instance_info_.dp_size);
        multi_req->set_dp_rank(sequence->dp_rank());
      }

      // send multi generations to remote instance
      proto::DisaggPDService_Stub* stub = create_rpc_channel(target_instance);
      if (!stub) {
        LOG(ERROR) << "Failed to create RPC channel to target instance: "
                   << target_instance;
        continue;
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      stub->MultiGenerations(&cntl, &multi_reqs, &resp, nullptr);
      if (cntl.Failed() || !resp.ok()) {
        LOG(ERROR) << "Failed to send multi generations, " << cntl.ErrorText()
                   << ", status: " << resp.ok();
      }
      {
        std::lock_guard<std::mutex> lock(remote_requests_map_mutex_);
        remote_requests_map_.erase(request->request_id());
      }
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      kv_cache_manager_->deallocate(request.get());
    }
  });
}

void PDOOCScheduler::build_disagg_requests(
    const std::vector<std::shared_ptr<Request>>& requests,
    proto::DisaggRequests& reqs) {
  // prefill name (ID)
  reqs.set_prefill_name(xservice_client_->get_instance_name());
  reqs.mutable_reqs()->Reserve(requests.size());

  // Build proto::DisaggRequest for each request
  for (size_t i = 0; i < requests.size(); ++i) {
    auto req = reqs.mutable_reqs()->Add();
    req->set_req_id(requests[i]->request_id());
    req->set_service_req_id(requests[i]->service_request_id());
    req->set_source_xservice_addr(requests[i]->source_xservice_addr());
    req->set_tokens_num(requests[i]->state().prompt_tokens.size());
    req->set_prompt(requests[i]->state().prompt);
    ADD_VECTOR_TO_PROTO(req->mutable_prompt_tokens(),
                        requests[i]->state().prompt_tokens);
    req->set_stream(requests[i]->state().stream);
    req->set_x_request_id(requests[i]->x_request_id());
    req->set_x_request_time(requests[i]->x_request_time());
    req->set_seq_capacity(requests[i]->state().seq_capacity);
    req->set_max_tokens(
        requests[i]->state().stopping_checker.get_max_generated_tokens());
    req->set_max_context_len(
        requests[i]->state().stopping_checker.get_max_context_len());
    req->set_ignore_eos(requests[i]->state().stopping_checker.get_ignore_eos());
    req->set_eos_token_id(
        requests[i]->state().stopping_checker.get_eos_token());
    if (requests[i]->state().stopping_checker.get_stop_tokens().size() > 0) {
      ADD_VECTOR_TO_PROTO(
          req->mutable_stop_token_ids(),
          requests[i]->state().stopping_checker.get_stop_tokens());
    }
    if (requests[i]->state().stopping_checker.get_stop_sequences().size() > 0) {
      for (auto& stop_sequence :
           requests[i]->state().stopping_checker.get_stop_sequences()) {
        auto proto_seq = req->mutable_stop_sequences()->Add();
        ADD_VECTOR_TO_PROTO(proto_seq->mutable_seq_tokens(), stop_sequence);
      }
    }
    req->set_n(requests[i]->state().n);
    req->set_best_of(requests[i]->state().best_of);
    req->set_frequency_penalty(
        requests[i]->state().sampling_param.frequency_penalty);
    req->set_presence_penalty(
        requests[i]->state().sampling_param.presence_penalty);
    req->set_repetition_penalty(
        requests[i]->state().sampling_param.repetition_penalty);
    req->set_temperature(requests[i]->state().sampling_param.temperature);
    req->set_top_p(requests[i]->state().sampling_param.top_p);
    req->set_top_k(requests[i]->state().sampling_param.top_k);
    req->set_logprobs(requests[i]->state().sampling_param.logprobs);
    req->set_top_logprobs(requests[i]->state().sampling_param.top_logprobs);
    req->set_is_embeddings(requests[i]->state().sampling_param.is_embeddings);
    req->set_echo(requests[i]->state().echo);
    req->set_skip_special_tokens(requests[i]->state().skip_special_tokens);
    req->set_offline(requests[i]->offline());
  }

  // Add cluster info
  reqs.mutable_cluster_infos()->mutable_cluster_ids()->Add(
      instance_info_.cluster_ids.begin(), instance_info_.cluster_ids.end());
  reqs.mutable_cluster_infos()->mutable_addrs()->Add(
      instance_info_.addrs.begin(), instance_info_.addrs.end());
  reqs.mutable_cluster_infos()->mutable_ports()->Add(
      instance_info_.ports.begin(), instance_info_.ports.end());
  reqs.mutable_cluster_infos()->set_dp_size(options_.dp_size());
}

}  // namespace xllm
