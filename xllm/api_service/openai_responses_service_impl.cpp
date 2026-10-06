/* Copyright 2026 The xLLM Authors.

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

#include "api_service/openai_responses_service_impl.h"

#include "api_service/openai_responses_output.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/config/service_config.h"
#include "core/util/scope_guard.h"
#include "parser/detector_registry.h"
#include "parser/reasoning_parser.h"

namespace xllm {
namespace {

Status validate_capabilities(const LLMMaster& master,
                             api_service::ResponsesRequest* request,
                             std::string* tool_parser,
                             std::string* reasoning_parser,
                             std::string* error_param) {
  const Options& options = master.options();
  const std::string model_type = master.model_args()->model_type();
  const std::string configured_reasoning =
      options.reasoning_parser().value_or("");
  auto& registry = DetectorRegistry::get_instance();
  *reasoning_parser =
      configured_reasoning == "auto"
          ? registry.resolve_parser_name(model_type).value_or("")
          : configured_reasoning;
  if ((!configured_reasoning.empty() && reasoning_parser->empty()) ||
      (!reasoning_parser->empty() &&
       !registry.has_detector(*reasoning_parser))) {
    *error_param = "reasoning";
    return Status(StatusCode::INVALID_ARGUMENT,
                  "The selected model has no supported reasoning parser");
  }
  request->params.responses_reasoning_parser = *reasoning_parser;
  if (!request->params.tools.empty()) {
    const std::string configured = options.tool_call_parser().value_or("");
    if (configured.empty()) {
      *error_param = "tools";
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Function tools require a configured model tool parser");
    }
    if (configured != "qwen25" && configured != "qwen3_coder" &&
        configured != "glm45" && configured != "glm47" &&
        configured != "glm5") {
      *error_param = "tools";
      return Status(StatusCode::INVALID_ARGUMENT,
                    "The selected model does not support Responses function "
                    "tools with parser: " +
                        configured);
    }
    *tool_parser = configured;
  }
  if (request->params.response_format == ResponseFormatType::JSON_OBJECT &&
      !ServiceConfig::get_instance().enable_json_object_output()) {
    *error_param = "text.format";
    return Status(StatusCode::INVALID_ARGUMENT,
                  "JSON object constrained decoding is disabled");
  }
  if (!tool_parser->empty() || !reasoning_parser->empty()) {
    request->params.skip_special_tokens = false;
  }
  return Status();
}

}  // namespace

ResponsesServiceImpl::ResponsesServiceImpl(
    std::shared_ptr<MasterManager> master_manager)
    : master_manager_(std::move(master_manager)) {
  CHECK(master_manager_ != nullptr);
}

void ResponsesServiceImpl::process_async(
    std::shared_ptr<ResponsesCall> call,
    api_service::ResponsesRequest request) {
  auto master = std::dynamic_pointer_cast<LLMMaster>(
      master_manager_->find_master(request.model));
  if (master == nullptr) {
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + request.model + "` does not exist.",
                            "model");
    return;
  }
  std::string tool_parser;
  std::string reasoning_parser;
  std::string error_param;
  Status capability = validate_capabilities(
      *master, &request, &tool_parser, &reasoning_parser, &error_param);
  if (!capability.ok()) {
    call->finish_with_error(
        capability.code(), capability.message(), error_param);
    return;
  }
  if (call->is_disconnected()) {
    return;
  }
  const Status admission = master->get_rate_limiter()->acquire();
  if (!admission.ok()) {
    call->finish_with_error(admission.code(), admission.message());
    return;
  }
  ScopeGuard preparation_guard(
      [master] { master->get_rate_limiter()->decrease_one_request(); });
  request.params.x_request_id = call->get_x_request_id();
  request.params.x_request_time = call->get_x_request_time();
  const bool stream = request.params.streaming;
  auto output = std::make_shared<api_service::ResponsesOutput>(
      api_service::responses_initial_response(request),
      stream,
      request.params.tools,
      std::move(tool_parser),
      std::move(reasoning_parser),
      /*force_reasoning=*/false,
      [call](const nlohmann::json& event) { return call->write_event(event); });
  preparation_guard.dismiss();
  master->handle_request(
      std::move(request.messages),
      std::nullopt,
      std::move(request.params),
      call.get(),
      [call, output, stream](RequestOutput result) -> bool {
        if (call->is_disconnected() || result.cancelled) {
          call->finish();
          return false;
        }
        if (result.status.has_value() && !result.status->ok()) {
          if (!output->started()) {
            return call->finish_with_error(result.status->code(),
                                           result.status->message());
          }
          const bool written =
              output->fail(result.status->code(), result.status->message());
          if (!stream) {
            return call->write_response(output->snapshot());
          }
          call->finish();
          return written;
        }
        const bool accepted = output->append(result);
        if (!accepted) {
          if (!stream && output->status().code() != StatusCode::CANCELLED &&
              output->snapshot().value("status", "") == "failed") {
            call->write_response(output->snapshot());
          }
          call->finish();
          return false;
        }
        if (output->finished()) {
          if (!stream) {
            return call->write_response(output->snapshot());
          }
          return call->finish();
        }
        return true;
      });
}

}  // namespace xllm
