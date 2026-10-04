/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "api_service/openai_batch.h"
#include "api_service/utils.h"
#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/master_manager.h"
#include "core/framework/request/request_output.h"
#include "core/runtime/xservice_client.h"
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
                  const std::optional<std::vector<LogProb>>& logprobs,
                  int32_t offset = 0) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  // One entry per generated token, so the three parallel fields would otherwise
  // grow from empty and re-copy themselves O(log n) times per response.
  const int num_logprobs = static_cast<int>(logprobs.value().size());
  proto_logprobs->mutable_tokens()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_ids()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_logprobs()->Reserve(num_logprobs);
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(std::max(-9999.0f, logprob.logprob));
    proto_logprobs->add_text_offset(offset);
    offset += static_cast<int32_t>(std::count_if(
        logprob.token.begin(), logprob.token.end(), [](unsigned char byte) {
          return (byte & 0xc0) != 0x80;
        }));
    auto* top = proto_logprobs->add_top_logprobs()->mutable_values();
    if (logprob.top_logprobs.has_value()) {
      for (const auto& candidate : logprob.top_logprobs.value()) {
        (*top)[candidate.token] = std::max(-9999.0f, candidate.logprob);
      }
    }
  }
}

bool send_delta_to_client_brpc(std::shared_ptr<CompletionCall> call,
                               bool include_usage,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model,
                               const RequestOutput& output,
                               std::unordered_map<size_t, int32_t>* offsets) {
  auto& response = call->response();
  if (output.usage.has_value()) {
    proto::Usage usage;
    api_service::set_proto_usage(&usage, output.usage.value());
    call->set_stream_usage(usage);
  }

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
      int32_t& offset = (*offsets)[seq_output.index];
      set_logprobs(choice, seq_output.logprobs, offset);
      offset += static_cast<int32_t>(std::count_if(
          seq_output.text.begin(),
          seq_output.text.end(),
          [](unsigned char byte) { return (byte & 0xc0) != 0x80; }));
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
      api_service::set_proto_stop_reason(seq_output.stop_reason,
                                         choice->mutable_stop_reason());
      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value() &&
      (output.finished || output.cancelled)) {
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
    api_service::set_proto_stop_reason(output.stop_reason,
                                       choice->mutable_stop_reason());
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
    const std::vector<std::string>& models,
    std::shared_ptr<MasterManager> master_manager)
    : APIServiceImpl(models), master_manager_(std::move(master_manager)) {
  CHECK(master != nullptr);
  CHECK(master_manager_ != nullptr);
}

std::shared_ptr<LLMMaster> CompletionServiceImpl::get_model_master(
    const std::string& model) const {
  return std::dynamic_pointer_cast<LLMMaster>(
      master_manager_->find_master(model));
}

// complete_async for brpc from xllm_service
void CompletionServiceImpl::process_async_rpc_impl(
    const proto::CompletionRequest* request) {
  const auto& service_request_id = request->service_request_id();
  const auto& target_xservice_addr = request->source_xservice_addr();
  OutputCallback callback = [](RequestOutput output) {
    const auto results = XServiceClient::get_instance()->generations({output});
    return results.size() == 1 && results.front();
  };
  const auto& rpc_request = *request;
  const auto& model = rpc_request.model();
  auto master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    CALLBACK_WITH_ERROR(StatusCode::UNKNOWN,
                        "Model not supported",
                        service_request_id,
                        target_xservice_addr);
    return;
  }
  const std::weak_ptr<LLMMaster> weak_master = master;
  callback = [weak_master](const RequestOutput& req_output) -> bool {
    const auto master = weak_master.lock();
    if (master == nullptr) {
      return false;
    }
    req_output.log_request_status();
    return master->handle_rpc_response(req_output);
  };

  // Check if the request is being rate-limited.
  const Status admission = master->get_rate_limiter()->acquire();
  if (unlikely(!admission.ok())) {
    CALLBACK_WITH_ERROR(admission.code(),
                        admission.message(),
                        service_request_id,
                        target_xservice_addr);
    return;
  }

  // check if model is supported
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
  master->handle_request(rpc_request.prompt(),
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
  std::shared_ptr<LLMMaster> master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + model + "` does not exist.",
                            "model");
    return;
  }

  const Status admission = master->get_rate_limiter()->acquire();
  if (unlikely(!admission.ok())) {
    call->finish_with_error(admission.code(), admission.message());
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

  const bool stream = request_params.streaming;
  const std::string request_id = request_params.request_id;
  const int64_t created_time = absl::ToUnixSeconds(absl::Now());
  auto offsets = std::make_shared<std::unordered_map<size_t, int32_t>>();
  OutputCallback send =
      [call, model, stream, include_usage, request_id, created_time, offsets](
          RequestOutput output) {
        output.log_request_status();
        if (output.status.has_value() && !output.status->ok()) {
          return call->finish_with_error(output.status->code(),
                                         output.status->message());
        }
        if (stream) {
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           request_id,
                                           created_time,
                                           model,
                                           output,
                                           offsets.get());
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, output);
      };
  if (rpc_request.prompts().empty()) {
    master->handle_request(rpc_request.prompt(),
                           std::move(prompt_tokens),
                           std::move(request_params),
                           call.get(),
                           std::move(send));
    return;
  }
  const size_t choices_per_prompt =
      request_params.beam_width > 1
          ? static_cast<size_t>(request_params.num_return_sequences > 0
                                    ? request_params.num_return_sequences
                                    : request_params.beam_width)
          : request_params.n;
  auto batch = std::make_shared<api_service::OpenAIBatch>(
      rpc_request.prompts_size(), choices_per_prompt, stream);
  for (int32_t index = 0; index < rpc_request.prompts_size(); ++index) {
    const auto& prompt = rpc_request.prompts(index);
    RequestParams params(
        rpc_request, call->get_x_request_id(), call->get_x_request_time());
    params.request_id += "-" + std::to_string(index);
    if (rpc_request.has_routing()) {
      params.decode_address = rpc_request.routing().decode_name();
    }
    std::optional<std::vector<int>> tokens;
    if (!prompt.token_ids().empty()) {
      tokens.emplace(prompt.token_ids().begin(), prompt.token_ids().end());
    }
    master->handle_request(prompt.text(),
                           std::move(tokens),
                           std::move(params),
                           call.get(),
                           [batch, index, send](RequestOutput output) {
                             return batch->accept(
                                 index, std::move(output), send);
                           });
  }
}

}  // namespace xllm
