/* Copyright 2025-2026 The xLLM Authors.
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

#include "completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_output.h"
#include "core/util/utils.h"

#ifdef likely
#undef likely
#endif
#define likely(x) __builtin_expect(!!(x), 1)

#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {
void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

bool send_delta_to_client_brpc(std::shared_ptr<CompletionCall> call,
                               bool include_usage,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model,
                               const RequestOutput& output) {
  auto& response = call->response();

  for (const auto& seq_output : output.outputs) {
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      set_logprobs(choice, seq_output.logprobs);
      if (!call->write(response)) {
        return false;
      }
    }

    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
    if (!call->write(response)) {
      return false;
    }
  }

  if (output.finished || output.cancelled) {
    response.Clear();
    return call->finish();
  }
  return true;
}

bool send_result_to_client_brpc(std::shared_ptr<CompletionCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
  }

  return call->write_and_finish(response);
}

}  // namespace

CompletionServiceImpl::CompletionServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
  add_model_master(models[0], master);
}

void CompletionServiceImpl::add_model_master(const std::string& model,
                                             LLMMaster* master) {
  CHECK(master != nullptr);
  std::unique_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  llm_model_to_master_.insert_or_assign(model, master);
  models_.insert(model);
}

LLMMaster* CompletionServiceImpl::get_model_master(
    const std::string& model) const {
  std::shared_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  auto it = llm_model_to_master_.find(model);
  if (it == llm_model_to_master_.end()) {
    return nullptr;
  }
  return it->second;
}

// complete_async for brpc from xllm_service
void CompletionServiceImpl::process_async_rpc_impl(
    const proto::CompletionRequest* request) {
  const auto& service_request_id = request->service_request_id();
  const auto& target_xservice_addr = request->source_xservice_addr();
  auto callback = [master = master_](const RequestOutput& req_output) -> bool {
    req_output.log_request_status();
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        // Reduce the number of concurrent requests when a request is
        // finished with error.
        master->get_rate_limiter()->decrease_one_request();
        return master->handle_rpc_response(req_output);
      }
    }
    // Reduce the number of concurrent requests when a request is finished
    // or canceled.
    if (req_output.finished || req_output.cancelled ||
        req_output.finished_on_prefill_instance) {
      master->get_rate_limiter()->decrease_one_request();
    }
    return master->handle_rpc_response(req_output);
  };

  // Check if the request is being rate-limited.
  if (unlikely(master_->get_rate_limiter()->is_limited())) {
    CALLBACK_WITH_ERROR(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.",
        service_request_id,
        target_xservice_addr);
    return;
  }

  // check if model is supported
  const auto& rpc_request = *request;
  const auto& model = rpc_request.model();
  if (unlikely(!models_.contains(model))) {
    CALLBACK_WITH_ERROR(StatusCode::UNKNOWN,
                        "Model not supported",
                        service_request_id,
                        target_xservice_addr);
    return;
  }

  RequestParams request_params(rpc_request, "", "");

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.token_ids_size() > 0) {
    prompt_tokens = std::vector<int>(rpc_request.token_ids().begin(),
                                     rpc_request.token_ids().end());
  }
  if (rpc_request.has_routing()) {
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  // schedule the request
  master_->handle_request(std::move(rpc_request.prompt()),
                          std::move(prompt_tokens),
                          std::move(request_params),
                          std::nullopt,
                          callback);
}

// complete_async for brpc
void CompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  LLMMaster* master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited or model is sleeping.
  // is_limited() returns true if sleeping or rate-limited.
  if (unlikely(master->get_rate_limiter()->is_limited())) {
    if (master->get_rate_limiter()->is_sleeping()) {
      call->finish_with_error(StatusCode::UNAVAILABLE,
                              "Model is currently in sleep state.");
    } else {
      call->finish_with_error(
          StatusCode::RESOURCE_EXHAUSTED,
          "The number of concurrent requests has reached the limit.");
    }
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.token_ids_size() > 0) {
    prompt_tokens = std::vector<int>(rpc_request.token_ids().begin(),
                                     rpc_request.token_ids().end());
  }
  if (rpc_request.has_routing()) {
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  // schedule the request
  master->handle_request(
      std::move(rpc_request.prompt()),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a request is
            // finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request is finished
        // or canceled.
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          return send_delta_to_client_brpc(
              call, include_usage, request_id, created_time, model, req_output);
        }
        // NOTE: maybe need to refactor along with service, currently for
        // non-stream request in prefill instance, we send a virtual response
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
