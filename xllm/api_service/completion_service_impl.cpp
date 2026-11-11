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
#include <unordered_map>
#include <utility>

#include "api_service/completion_response_builder.h"
#include "api_service/openai_batch.h"
#include "api_service/response_usage.h"
#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/master_manager.h"
#include "core/distributed_runtime/xservice_client.h"
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
bool send_completion_delta_to_client(
    const std::shared_ptr<CompletionCall>& call,
    bool include_usage,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output,
    std::unordered_map<size_t, int32_t>* offsets) {
  if (output.usage.has_value()) {
    proto::Usage usage;
    api_service::set_proto_usage(&usage, output.usage.value());
    call->set_stream_usage(usage);
  }
  const bool written = api_service::build_completion_delta(
      call->response(),
      include_usage,
      request_id,
      created_time,
      model,
      output,
      offsets,
      [&call](proto::CompletionResponse& frame) { return call->write(frame); });
  if (!written) {
    return false;
  }
  if (output.finished || output.cancelled) {
    call->response().Clear();
    return call->finish();
  }
  return true;
}

bool send_completion_result_to_client(
    const std::shared_ptr<CompletionCall>& call,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output) {
  api_service::build_completion_result(
      call->response(), request_id, created_time, model, output);
  return call->write_and_finish(call->response());
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
  process_request_from_service(*request);
}

void CompletionServiceImpl::process_request_from_service(
    const proto::CompletionRequest& rpc_request,
    const std::string& x_request_id,
    const std::string& x_request_time) {
  const auto& service_request_id = rpc_request.service_request_id();
  const auto& target_xservice_addr = rpc_request.source_xservice_addr();
  OutputCallback callback = [service_request_id,
                             target_xservice_addr,
                             stream =
                                 rpc_request.stream()](RequestOutput output) {
    // Validation failures may contain only a Status. Bind their owner here too.
    output.service_request_id = service_request_id;
    output.target_xservice_addr = target_xservice_addr;
    output.log_request_status();
    output.stream = stream;
    const auto results = XServiceClient::get_instance()->generations({output});
    return results.size() == 1 && results.front();
  };
  const auto& model = rpc_request.model();
  auto master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    CALLBACK_WITH_ERROR(StatusCode::UNKNOWN,
                        "Model not supported",
                        service_request_id,
                        target_xservice_addr);
    return;
  }
  if (rpc_request.has_routing() &&
      (!master->options().enable_disagg_pd() ||
       rpc_request.routing().decode_name().empty() ||
       rpc_request.token_ids().empty())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "PD service requests require PD mode, a Decode route "
                        "and prompt token IDs",
                        service_request_id,
                        target_xservice_addr);
    return;
  }
  if (!rpc_request.prompts().empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Service completion requests require a single prompt",
                        service_request_id,
                        target_xservice_addr);
    return;
  }

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
  RequestParams request_params(rpc_request, x_request_id, x_request_time);

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
                         std::move(callback));
}

// complete_async for brpc
void CompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  const auto& rpc_request = call->request();
  if (!rpc_request.source_xservice_addr().empty() &&
      rpc_request.has_routing()) {
    if (rpc_request.service_request_id().empty()) {
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Missing service callback request ID",
                              "service_request_id");
      return;
    }
    process_request_from_service(
        rpc_request, call->get_x_request_id(), call->get_x_request_time());
    call->accept_async();
    return;
  }
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
          return send_completion_delta_to_client(call,
                                                 include_usage,
                                                 request_id,
                                                 created_time,
                                                 model,
                                                 output,
                                                 offsets.get());
        }
        return send_completion_result_to_client(
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
  const size_t choices_per_prompt = request_params.n;
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
