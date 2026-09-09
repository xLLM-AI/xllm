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

#include "vlm_master.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/multimodal/mm_data.h"
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

bool should_use_vlm_speculative_engine(const Options& options) {
  return options.speculative_algorithm() != "Suffix" &&
         !options.draft_model_path().value_or("").empty();
}

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

// Extracts the final prompt from a single message, concatenating its text
// contents in order and skipping non-text (image/video/audio) contents.
//
// The offline "prompt as-is" contract expects the caller to hand in exactly
// one message whose text is the final prompt already assembled with the
// chat template (images may ride along as non-text contents). Multi-turn
// conversations must go through the chat template path instead; joining
// their texts would silently mash the roles together, so any other message
// shape returns std::nullopt and fails the request.
std::optional<std::string> join_message_texts(
    const std::vector<Message>& messages) {
  if (messages.size() != 1) {
    return std::nullopt;
  }
  const auto& message = messages.front();
  if (auto text = std::get_if<std::string>(&message.content)) {
    return *text;
  }
  if (auto contents = std::get_if<MMContentVec>(&message.content)) {
    std::string prompt;
    for (const auto& content : *contents) {
      if (content.type == "text") {
        prompt += content.text;
      }
    }
    return prompt;
  }
  return std::nullopt;
}

// Throws std::invalid_argument describing the violation when messages do not
// satisfy the offline "prompt as-is" contract (see join_message_texts).
// Offline-only: raised on the caller thread so pybind surfaces it to the
// Python caller at the call site, instead of returning an empty prompt that
// is hard to trace.
[[noreturn]] void throw_prompt_as_is_error(
    const std::vector<Message>& messages) {
  std::string roles;
  for (const auto& message : messages) {
    if (!roles.empty()) {
      roles += ", ";
    }
    roles += message.role;
  }
  throw std::invalid_argument(
      "use_prompt_as_is expects exactly 1 message holding the final prompt "
      "(images allowed as non-text contents), got " +
      std::to_string(messages.size()) + " messages [" + roles +
      "]; apply the chat template for multi-turn conversations");
}

}  // namespace

VLMMaster::VLMMaster(const Options& options)
    : Master(options,
             should_use_vlm_speculative_engine(options) ? EngineType::VLMSSM
                                                        : EngineType::VLM) {
  if (!is_leader()) {
    return;
  }

  CHECK(engine_->init(master_status_));

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
      .num_speculative_tokens(options_.num_speculative_tokens())
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
  processor_ = create_multimodal_processor(model_args_,
                                           tokenizer_,
                                           options_.max_processor_cache_items(),
                                           engine_->tokenizer_args());

  request_factory_ = std::make_unique<VLMRequestFactory>(processor_.get(),
                                                         chat_template_.get(),
                                                         tokenizer_.get(),
                                                         &model_args_,
                                                         &options_,
                                                         get_rate_limiter());

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"VLMMaster.request");
}

VLMMaster::~VLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);

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

    // Guard the rate-limit slot acquired at the service entry.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(
        std::move(prompt), std::move(mm_data), sp, std::move(callback));
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
                               OutputCallback callback,
                               bool use_prompt_as_is) {
  // Offline-only guard: fail loudly on the caller thread (pybind surfaces
  // the exception to the Python caller) before any scheduling side effect.
  if (use_prompt_as_is && !join_message_texts(messages).has_value()) {
    throw_prompt_as_is_error(messages);
  }

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
                         use_prompt_as_is,
                         callback = std::move(cb)]() mutable {
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
    auto request = request_factory_->create(std::move(messages),
                                            sp,
                                            std::move(payload),
                                            std::move(callback),
                                            use_prompt_as_is);
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

// Offline batch entry aligned with the vllm-ascend offline behavior: each
// prompt is handed in as the final prompt and is used as-is, without applying
// the chat template again (double assembly duplicates the vision placeholders
// and crashes prompt processing). This only affects offline inference; the
// online serving path still applies the chat template.
void VLMMaster::handle_batch_request_with_image_urls(
    std::vector<std::string> prompts,
    std::vector<std::vector<std::string>> image_urls,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(prompts.size() == image_urls.size())
      << "Number of prompts and image urls should be the same";
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  std::vector<std::vector<Message>> conversations;
  conversations.reserve(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    conversations.push_back(build_user_messages_from_image_urls(
        std::move(prompts[i]), image_urls[i]));
  }

  std::string payload;
  for (size_t i = 0; i < num_requests; ++i) {
    // Offline vLLM-style requests hand in a final prompt. The text-only
    // offline path never applies the chat template either, so use the
    // prompt as-is instead of re-assembling it (double assembly duplicates
    // the vision placeholders and crashes prompt processing).
    handle_request(
        std::move(conversations[i]),
        // the sampling parameter may be shared
        sps.size() == 1 ? sps[0] : std::move(sps[i]),
        std::move(payload),
        [i, callback](const RequestOutput& output) {
          output.log_request_status();
          return callback(i, output);
        },
        /*use_prompt_as_is=*/true);
  }
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
  if (!is_leader()) {
    Master::run();
    return;
  }

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

}  // namespace xllm
