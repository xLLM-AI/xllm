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

#include "vlm_master.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/platform/device_name_utils.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#include "speculative_engine.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "vlm_engine.h"

namespace xllm {

namespace {

std::vector<Message> build_user_messages_from_image_urls(
    std::string prompt,
    const std::vector<std::string>& image_urls) {
  MMContentVec contents;
  contents.reserve(image_urls.size() + 1);
  for (const auto& url : image_urls) {
    contents.emplace_back("image_url", ImageURL{url});
  }
  contents.emplace_back("text", std::move(prompt));

  std::vector<Message> messages;
  messages.emplace_back("user", std::move(contents));
  return messages;
}

}  // namespace

VLMMaster::VLMMaster(const Options& options)
    : Master(options, EngineType::VLM) {
  CHECK(engine_->init());

  model_args_ = engine_->model_args();

  if (options_.enable_service_routing()) {
    XServiceClient* xservice_client = XServiceClient::get_instance();
    if (!xservice_client->init(options_.etcd_addr().value_or(""),
                               options_.instance_name().value_or(""),
                               engine_->block_manager_pool(),
                               options_.etcd_namespace().value_or(""))) {
      LOG(FATAL) << "XServiceClient init fail!";
      return;
    }
  }

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .dp_size(options_.dp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .disable_log_stats(options_.disable_log_stats())
      .disable_ttft_profiling(options_.disable_ttft_profiling())
      .enable_forward_interruption(options_.enable_forward_interruption())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .server_idx(options_.server_idx());
  scheduler_ = create_continuous_scheduler(engine_.get(), scheduler_options);

  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }

  chat_template_ =
      std::make_unique<JinjaChatTemplate>(engine_->tokenizer_args());
  tokenizer_ = engine_->tokenizer()->clone();
  processor_ = create_multimodal_processor(model_args_, tokenizer_);

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"VLMMaster.request");
}

VLMMaster::~VLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void VLMMaster::handle_request(std::string prompt,
                               MMData mm_data,
                               RequestParams sp,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         mm_data = std::move(mm_data),
                         sp = std::move(sp),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(std::move(prompt),
                                    std::move(mm_data),
                                    std::move(sp),
                                    std::move(callback));
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_request(std::vector<Message> messages,
                               RequestParams sp,
                               std::string payload,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         messages = std::move(messages),
                         sp = std::move(sp),
                         payload = std::move(payload),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_chat);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = generate_request(std::move(messages),
                                    std::move(sp),
                                    std::move(payload),
                                    std::move(callback));
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_batch_request(std::vector<std::string> prompts,
                                     std::vector<MMData> mm_datas,
                                     std::vector<RequestParams> sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   std::move(mm_datas[i]),
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void VLMMaster::handle_batch_request_with_image_urls(
    std::vector<std::string> prompts,
    std::vector<std::vector<std::string>> image_urls,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(prompts.size() == image_urls.size())
      << "Number of prompts and image urls should be the same";
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  std::vector<std::vector<Message>> conversations;
  conversations.reserve(prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    conversations.push_back(build_user_messages_from_image_urls(
        std::move(prompts[i]), image_urls[i]));
  }

  handle_batch_request(
      std::move(conversations), std::move(sps), std::move(callback));
}

void VLMMaster::handle_batch_request(
    std::vector<std::vector<Message>> conversations,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  std::string payload;
  const size_t num_requests = conversations.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::move(payload),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void VLMMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "VLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    running_.store(true, std::memory_order_relaxed);
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void VLMMaster::generate() {
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

bool VLMMaster::shutdown_remote_workers() {
  return engine_->shutdown_remote_workers();
}

std::shared_ptr<Request> VLMMaster::generate_request(std::string prompt,
                                                     MMData mm_data,
                                                     RequestParams sp,
                                                     OutputCallback callback) {
  if (prompt.empty() && mm_data.empty()) {
    LOG(ERROR) << "Prompt and multimodal data cannot be both empty.";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt and multimodal data are both empty.");
    return nullptr;
  }

  std::vector<int32_t> prompt_tokens;
  if (!processor_->process_prompt(prompt, mm_data, prompt_tokens)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to process prompt.");
    return nullptr;
  }

  return build_request(std::move(prompt),
                       std::move(prompt_tokens),
                       std::move(mm_data),
                       std::move(sp),
                       std::move(callback));
}

std::shared_ptr<Request> VLMMaster::build_request(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    RequestParams sp,
    OutputCallback callback) {
  const int32_t max_context_len = model_args_.max_position_embeddings();
  int32_t prompt_token_limit = max_context_len;
  if (!options_.enable_chunked_prefill()) {
    prompt_token_limit =
        std::min(prompt_token_limit, options_.max_tokens_per_batch());
  }
  if (prompt_tokens.size() >= static_cast<size_t>(prompt_token_limit)) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens, TODO: add image token size as well.
  const size_t capacity = prompt_tokens.size() + max_tokens + 1;
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
  if (best_of > sp.n) {
    // enable logprobs for best_of to generate sequence logprob
    sampling_param.logprobs = true;
  }
  // sampling_param.do_sample = sp.do_sample;

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
      std::vector<int32_t> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  StoppingChecker stopping_checker(max_tokens,
                                   max_context_len,
                                   model_args_.eos_token_id(),
                                   sp.ignore_eos,
                                   std::move(stop_tokens),
                                   std::move(stop_sequences));

  // results cannot be streamed when best_of != n
  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
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
                         nullptr);
  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id,
                                           sp.source_xservice_addr);

  // add one sequence, rest will be added by scheduler
  return request;
}

std::shared_ptr<Request> VLMMaster::generate_request(
    std::vector<Message> messages,
    RequestParams sp,
    std::string payload,
    OutputCallback callback) {
  static MMInputTransfer mm_input_transfer;

  MMInput mm_inputs(std::move(payload));
  MMErrCode code = mm_input_transfer.trans(messages, mm_inputs);
  if (code != MMErrCode::SUCCESS) {
    std::string error_message = MMErrToString(code);
    LOG(ERROR) << error_message;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, error_message);
    return nullptr;
  }

  MMData mm_data;
  if (!mm_inputs.empty() &&
      !processor_->process_multimodal(mm_inputs, mm_data)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to process multimodal input.");
    return nullptr;
  }

  Timer timer;
  std::optional<std::string> prompt =
      chat_template_->apply(messages, sp.tools, sp.chat_template_kwargs);
  if (!prompt.has_value()) {
    std::string error_message = "Failed to construct prompt from messages";
    LOG(ERROR) << error_message;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, error_message);
    return nullptr;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  return generate_request(std::move(prompt.value()),
                          std::move(mm_data),
                          std::move(sp),
                          std::move(callback));
}

volatile bool VLMAssistantMaster::running_ = false;

VLMAssistantMaster::VLMAssistantMaster(const Options& options)
    : Master(options, EngineType::VLM) {
  auto master_node_addr = options_.master_node_addr().value_or("");
  if (master_node_addr.empty()) {
    LOG(FATAL)
        << "MultiNodeEngine required master_node_addr, current value is empty.";
    return;
  }
  running_ = true;
}

VLMAssistantMaster::~VLMAssistantMaster() {
  running_ = false;
  wait();
}

void VLMAssistantMaster::run() {
  signal(SIGINT, VLMAssistantMaster::handle_signal);
  signal(SIGTERM, VLMAssistantMaster::handle_signal);

  loop_thread_ = std::thread([this]() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

void VLMAssistantMaster::wait() {
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

}  // namespace xllm
