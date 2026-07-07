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

#include "llm_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
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
#include "util/utils.h"

namespace xllm {
namespace {

bool should_use_ssm_engine(const Options& options) {
  return !options.draft_model_path().value_or("").empty() ||
         (options.speculative_algorithm() == "Suffix" &&
          options.num_speculative_tokens() > 0);
}

}  // namespace

volatile bool LLMAssistantMaster::running_ = false;

LLMMaster::LLMMaster(const Options& options)
    : Master(
          options,
          should_use_ssm_engine(options) ? EngineType::SSM : EngineType::LLM) {
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
}

LLMMaster::~LLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  LOG(INFO) << "LLMMaster stopping...";
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

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(
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

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request =
        generate_request(messages, std::move(prompt_token), sp, call, callback);
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

std::shared_ptr<Request> LLMMaster::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback) {
  // A request is valid as long as it carries either text or pre-tokenized
  // prompt tokens; pure-token input (no text) is a first-class input.
  const bool has_prompt_tokens = prompt_tokens.has_value();
  if (prompt.empty() && !has_prompt_tokens) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt is empty",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  // encode the prompt
  Timer timer;
  std::vector<int> local_prompt_tokens;

  if (has_prompt_tokens) {
    local_prompt_tokens = std::move(prompt_tokens.value());
  } else {
    if (!tokenizer_->encode(
            prompt, &local_prompt_tokens, sp.add_special_tokens)) {
      LOG(ERROR) << "Failed to encode prompt: " << prompt;
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to encode prompt",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  // Validate directly-supplied prompt tokens against the vocabulary range to
  // avoid out-of-bounds embedding lookups. Encoded tokens are trusted, so only
  // scan when tokens were provided and the vocab range is known.
  const int64_t vocab_size = model_args_.vocab_size();
  if (has_prompt_tokens && vocab_size > 0) {
    const auto invalid_token =
        util::find_out_of_vocab_token(local_prompt_tokens, vocab_size);
    if (invalid_token.has_value()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Prompt token id out of vocabulary range: " +
                              std::to_string(invalid_token.value()),
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
  }

  const int32_t max_context_len = model_args_.max_position_embeddings();
  int32_t prompt_token_limit = max_context_len;
  if (!options_.enable_chunked_prefill()) {
    prompt_token_limit =
        std::min(prompt_token_limit, options_.max_tokens_per_batch());
  }
  if (local_prompt_tokens.size() >= prompt_token_limit) {
    LOG(ERROR) << "Prompt is too long: " << local_prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt is too long",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }
  uint32_t effective_max_tokens = max_tokens;
  if (sp.is_sample_request) {
    const uint32_t sample_slot_tokens =
        static_cast<uint32_t>(sp.sample_slots.size());
    if (sample_slot_tokens > effective_max_tokens) {
      effective_max_tokens = sample_slot_tokens;
    }
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  size_t capacity = local_prompt_tokens.size() + effective_max_tokens +
                    options_.num_speculative_tokens() + /*bouns_token*/ 1;
  if (options_.enable_schedule_overlap()) {
    capacity += options_.num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  sampling_param.logprobs = sp.logprobs;
  sampling_param.top_logprobs = sp.top_logprobs;
  sampling_param.is_embeddings = sp.is_embeddings;
  sampling_param.beam_width = sp.beam_width;
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  if (sampling_param.beam_width > 1) {
    // beam search requires logprobs, and needs at least one top_logprob
    // candidate for beam expansion.
    sampling_param.logprobs = true;
    if (sampling_param.top_logprobs == 0) {
      sampling_param.top_logprobs =
          static_cast<int64_t>(sampling_param.beam_width);
    }
  }
  // sampling_param.do_sample = sp.do_sample;

  SchedulerParam scheduler_param;
  scheduler_param.offline = sp.offline;
  scheduler_param.priority = sp.priority;
  if (!sp.offline) {
    scheduler_param.ttft_slo_ms = sp.ttft_slo_ms;
    scheduler_param.tpot_slo_ms = sp.tpot_slo_ms;
    scheduler_param.ttlt_slo_ms = sp.ttlt_slo_ms;
    scheduler_param.tpot_priority_weight = sp.tpot_priority_weight;
    scheduler_param.ttft_priority_weight = sp.ttft_priority_weight;
    scheduler_param.ttlt_priority_weight = sp.ttlt_priority_weight;
    scheduler_param.priority_weight = sp.priority_weight;
  }

  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else {
    stop_tokens = model_args_.stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence",
                            sp.service_request_id,
                            sp.source_xservice_addr);
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  StoppingChecker stopping_checker(
      effective_max_tokens,
      max_context_len - options_.num_speculative_tokens(),
      model_args_.eos_token_id(),
      sp.ignore_eos,
      std::move(stop_tokens),
      std::move(stop_sequences));

  if (task_type_ != "embed" && task_type_ != "mm_embed") {
    auto finish_reason =
        stopping_checker.check(local_prompt_tokens, local_prompt_tokens.size());
    if (finish_reason != FinishReason::NONE) {
      LOG(INFO) << " finish_reason " << finish_reason.to_string().value();
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Invalid Prompt",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      LOG(ERROR) << "Invalid Prompt EndWith Token_ID:"
                 << local_prompt_tokens[local_prompt_tokens.size() - 1];
      return nullptr;
    }
  }

  bool stream = sp.streaming;
  // results cannot be streamed when best_of != n
  if (best_of != sp.n) {
    stream = false;
  }

  OutputsFunc batch_callback = nullptr;
  if (options_.enable_service_routing()) {
    batch_callback = [this](const std::vector<RequestOutput>& req_outputs) {
      size_t decrease_requests_num = 0;
      for (const auto& req_output : req_outputs) {
        req_output.log_request_status();
        if (req_output.status.has_value() && !req_output.status.value().ok()) {
          decrease_requests_num++;
          continue;
        }
        // Reduce the number of concurrent requests when a request is
        // finished or canceled.
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          decrease_requests_num++;
        }
      }
      get_rate_limiter()->decrease_requests(decrease_requests_num);
      return handle_rpc_responses(req_outputs);
    };
  }

  RequestState req_state(std::move(prompt),
                         std::move(local_prompt_tokens),
                         std::move(sampling_param),
                         std::move(scheduler_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_.enable_schedule_overlap(),
                         callback,
                         batch_callback,
                         sp.decode_address,
                         call);
  req_state.sample_slots = sp.sample_slots;

  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id,
                                           sp.source_xservice_addr);

  // add one sequence, rest will be added by scheduler
  return request;
}

std::shared_ptr<Request> LLMMaster::generate_request(
    const std::vector<Message>& messages,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback) {
  Timer timer;

  std::optional<std::string> prompt;
  prompt = chat_template_->apply(messages, sp.tools, sp.chat_template_kwargs);
  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }

  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  return generate_request(
      std::move(prompt.value()), std::move(prompt_tokens), sp, call, callback);
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

LLMAssistantMaster::LLMAssistantMaster(const Options& options)
    : Master(
          options,
          should_use_ssm_engine(options) ? EngineType::SSM : EngineType::LLM) {
  // setup process workers
  auto master_node_addr = options_.master_node_addr().value_or("");
  // TODO: support local unix domain socket later.
  if (master_node_addr.empty()) {
    LOG(FATAL)
        << "MultiNodeEngine required master_node_addr, current value is empty.";
    return;
  }

  running_ = true;
}

LLMAssistantMaster::~LLMAssistantMaster() {
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void LLMAssistantMaster::run() {
  signal(SIGINT, LLMAssistantMaster::handle_signal);
  signal(SIGTERM, LLMAssistantMaster::handle_signal);

  loop_thread_ = std::thread([this]() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
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
