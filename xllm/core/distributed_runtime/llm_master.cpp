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

#include "llm_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "core/platform/device_name_utils.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "models/model_registry.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#include "speculative_engine.h"
#include "util/net.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
namespace {

bool should_use_ssm_engine(const Options& options) {
  return !options.draft_model_path().value_or("").empty() ||
         (options.speculative_algorithm() == "Suffix" &&
          options.num_speculative_tokens() > 0);
}

}  // namespace

LLMMaster::LLMMaster(const Options& options)
    : Master(
          options,
          should_use_ssm_engine(options) ? EngineType::SSM : EngineType::LLM) {
  if (!is_leader()) {
    return;
  }

  CHECK(engine_->init(master_status_));
  task_type_ = options_.task_type();

  model_args_ = engine_->model_args();

  if (options_.enable_service_routing()) {
    xservice_client_ = XServiceClient::get_instance();
    if (!xservice_client_->init(options_.etcd_addr().value_or(""),
                                options_.instance_name().value_or(""),
                                engine_->block_manager_pool(),
                                options_.etcd_namespace().value_or(""))) {
      LOG(FATAL) << "XServiceClient init fail!";
      return;
    }
  }

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .nnodes(options_.nnodes())
      .dp_size(options_.dp_size())
      .cp_size(options_.cp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_pd_ooc(options_.enable_pd_ooc())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .disable_log_stats(options_.disable_log_stats())
      .priority_strategy(options_.priority_strategy())
      .enable_online_preempt_offline(options_.enable_online_preempt_offline())
      .enable_profile_step_time(options_.enable_profile_step_time())
      .enable_profile_token_budget(options_.enable_profile_token_budget())
      .enable_latency_aware_schedule(options_.enable_latency_aware_schedule())
      .profile_max_prompt_length(options_.profile_max_prompt_length())
      .enable_profile_kv_blocks(options_.enable_profile_kv_blocks())
      .disable_ttft_profiling(options_.disable_ttft_profiling())
      .enable_forward_interruption(options_.enable_forward_interruption())
      .max_global_ttft_ms(options_.max_global_ttft_ms())
      .max_global_tpot_ms(options_.max_global_tpot_ms())
      .server_idx(options_.server_idx())
      .prefetch_timeout(options_.prefetch_timeout())
      .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());
  scheduler_ = create_continuous_scheduler(engine_.get(), scheduler_options);

  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }

  // construct chat template
  chat_template_ =
      ChatTemplate::create(engine_->tokenizer_args(), model_args_.model_type());

  tokenizer_ = engine_->tokenizer()->clone();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"LLMMaster.request");

  request_factory_ = std::make_unique<LLMRequestFactory>(
      tokenizer_.get(),
      chat_template_.get(),
      &model_args_,
      &options_,
      get_rate_limiter(),
      task_type_,
      [this](const std::vector<RequestOutput>& outputs) {
        return handle_rpc_responses(outputs);
      });
}

LLMMaster::~LLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  LOG(INFO) << "LLMMaster stopping...";

  // Drain and join the request thread pool before any of the members its
  // worker lambdas touch are destroyed. Those lambdas dereference
  // request_factory_ (as well as scheduler_ and the rate limiter), but
  // request_factory_ is declared after threadpool_ in the header, so member
  // destruction would otherwise free the factory while pool workers are still
  // in flight. ~ThreadPool only signals/joins its workers when the pool is
  // destroyed, so reset it here explicitly while all dependencies are alive.
  // Done before joining loop_thread_ so the scheduler keeps advancing while the
  // pool drains.
  threadpool_.reset();

  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void LLMMaster::handle_batch_request(std::vector<std::string> prompts,
                                     std::vector<RequestParams> sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_batch_request(
    std::vector<std::vector<Message>> conversations,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  const size_t num_requests = conversations.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_request(std::string prompt,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               std::optional<Call*> call,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  // add into the queue
  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(callback),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry. If we bail
    // before the factory has a chance to create the Request, this
    // releases the slot; otherwise Request itself takes ownership.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(
        std::move(prompt), std::move(prompt_token), sp, call, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request",
                          sp.service_request_id,
                          sp.source_xservice_addr);
    }
  });
}

void LLMMaster::handle_request(std::vector<Message> messages,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               std::optional<Call*> call,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  // add into the queue
  threadpool_->schedule([this,
                         messages = std::move(messages),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(callback),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_chat);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(
        messages, std::move(prompt_token), sp, call, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request",
                          sp.service_request_id,
                          sp.source_xservice_addr);
    }
  });
}

void LLMMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "LLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void LLMMaster::generate() {
  DCHECK(options_.enable_schedule_overlap())
      << "Mode generate does not support schedule overlap yet.";
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

bool LLMMaster::handle_rpc_response(const RequestOutput& output) {
  // response to xllm service to avoid the redirect cost.
  if (xservice_client_ == nullptr) return false;
  auto return_status = xservice_client_->generations({output});
  CHECK_EQ(return_status.size(), 1)
      << "return size of generations is not equal to 1";
  return return_status[0];
}

std::vector<bool> LLMMaster::handle_rpc_responses(
    const std::vector<RequestOutput>& outputs) {
  // response to xllm service to avoid the redirect cost.
  if (xservice_client_ == nullptr)
    return std::vector<bool>(outputs.size(), false);
  return xservice_client_->generations(outputs);
}

bool LLMMaster::sleep() { return engine_->sleep(master_status_); }

bool LLMMaster::wakeup() {
  WakeupOptions options;
  options.master_status = master_status_;
  const bool ok = engine_->wakeup(options);
  // RL deep sleep discards the KV cache; on wake the physical memory is
  // re-mapped but its contents are garbage. Drop all prefix-cache entries so a
  // subsequent request never reuses a stale (now-garbage) cached prefix.
  if (ok && options_.enable_sleep_mode() && scheduler_ != nullptr) {
    scheduler_->reset_prefix_cache();
  }
  return ok;
}

bool LLMMaster::wakeup(const WakeupOptions& options) {
  WakeupOptions opts = options;
  opts.master_status = master_status_;
  return engine_->wakeup(opts);
}

bool LLMMaster::update_weights(const std::string& weights_path) {
  return engine_->update_weights(weights_path);
}

bool LLMMaster::link_p2p(const std::vector<std::string>& remote_addrs) {
  return engine_->link_p2p(remote_addrs);
}

bool LLMMaster::unlink_p2p(const std::vector<std::string>& remote_addrs) {
  return engine_->unlink_p2p(remote_addrs);
}

// ============== Async RL training support: Pause/Resume ==============
void LLMMaster::pause_scheduler(const std::string& mode) {
  LOG(INFO) << "LLMMaster: pausing scheduler (mode=" << mode << ")";

  auto* continuous_scheduler =
      dynamic_cast<ContinuousScheduler*>(scheduler_.get());
  if (!continuous_scheduler) {
    LOG(ERROR) << "Scheduler is not a ContinuousScheduler";
    return;
  }

  ContinuousScheduler::PauseMode pause_mode =
      ContinuousScheduler::PauseMode::KEEP;
  if (mode == "abort") {
    pause_mode = ContinuousScheduler::PauseMode::ABORT;
  } else if (mode == "wait") {
    pause_mode = ContinuousScheduler::PauseMode::WAIT;
  } else if (mode != "keep" && !mode.empty()) {
    LOG(WARNING) << "Unknown pause mode '" << mode << "', defaulting to keep";
  }

  continuous_scheduler->pause(pause_mode);

  // Block until the scheduler loop thread has actually reached PAUSED, so that
  // when this call returns it is safe to update weights (KEEP/ABORT: running
  // requests handled and KV cache freed; WAIT: all in-flight requests done).
  continuous_scheduler->wait_until_paused();
  LOG(INFO) << "LLMMaster: scheduler fully paused (mode=" << mode << ")";
}

void LLMMaster::resume_scheduler() {
  LOG(INFO) << "LLMMaster: resuming scheduler";

  auto* continuous_scheduler =
      dynamic_cast<ContinuousScheduler*>(scheduler_.get());
  if (!continuous_scheduler) {
    LOG(ERROR) << "Scheduler is not a ContinuousScheduler";
    return;
  }

  continuous_scheduler->resume();
}

bool LLMMaster::is_scheduler_paused() const {
  auto* continuous_scheduler =
      dynamic_cast<ContinuousScheduler*>(scheduler_.get());
  if (!continuous_scheduler) {
    return false;
  }

  return continuous_scheduler->is_paused();
}

}  // namespace xllm
