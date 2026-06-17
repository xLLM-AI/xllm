/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <folly/executors/CPUThreadPoolExecutor.h>

#include "core/common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/util/uuid.h"
#include "types.h"

namespace xllm {

struct LLMCore {
  // List of loaded model identifiers
  std::vector<std::string> model_ids;

  // Master controller for LLM runtime management
  std::unique_ptr<LLMMaster> master;

  // Thread pool for asynchronous task execution
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

namespace detail {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_request_id() {
  return "xllm-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}
}  // namespace

enum class InterfaceType { COMPLETIONS, CHAT_COMPLETIONS };

RequestParams transfer_request_params(
    const XLLM_RequestParams& request_params) {
  RequestParams xllm_request_params;

  xllm_request_params.echo = request_params.echo;
  xllm_request_params.offline = request_params.offline;
  xllm_request_params.logprobs = request_params.logprobs;
  xllm_request_params.best_of = request_params.best_of;
  xllm_request_params.slo_ms = request_params.slo_ms;
  xllm_request_params.top_k = request_params.top_k;
  xllm_request_params.top_p = request_params.top_p;
  xllm_request_params.ignore_eos = request_params.ignore_eos;
  xllm_request_params.skip_special_tokens = request_params.skip_special_tokens;
  xllm_request_params.n = request_params.n;
  xllm_request_params.max_tokens = request_params.max_tokens;
  xllm_request_params.frequency_penalty = request_params.frequency_penalty;
  xllm_request_params.presence_penalty = request_params.presence_penalty;
  xllm_request_params.repetition_penalty = request_params.repetition_penalty;
  xllm_request_params.stop = request_params.stop;
  xllm_request_params.stop_token_ids = request_params.stop_token_ids;
  xllm_request_params.beam_width = request_params.beam_width;
  xllm_request_params.num_return_sequences =
      request_params.num_return_sequences;
  xllm_request_params.top_logprobs = request_params.top_logprobs;

  return xllm_request_params;
}

XLLM_Response build_success_response(const RequestOutput& output,
                                     const InterfaceType& if_type,
                                     const std::string& request_id,
                                     int64_t created_time,
                                     const std::string& model) {
  XLLM_Response response;

  response.status_code = XLLM_StatusCode::kSuccess;

  response.id = request_id;
  response.created = created_time;
  response.model = model;
  if (if_type == InterfaceType::COMPLETIONS) {
    response.object = "text_completion";
  } else if (if_type == InterfaceType::CHAT_COMPLETIONS) {
    response.object = "chat.completion";
  }

  response.choices.reserve(output.outputs.size());
  for (const auto& output : output.outputs) {
    XLLM_Choice choice;
    choice.index = output.index;

    if (output.logprobs.has_value()) {
      std::vector<XLLM_LogProb> xllm_logprobs;
      xllm_logprobs.reserve(output.logprobs.value().size());
      for (const auto& logprob : output.logprobs.value()) {
        XLLM_LogProb xllm_logprob;
        xllm_logprob.token = logprob.token;
        xllm_logprob.token_id = logprob.token_id;
        xllm_logprob.logprob = logprob.logprob;

        if (logprob.top_logprobs.has_value()) {
          xllm_logprob.top_logprobs.reserve(
              logprob.top_logprobs.value().size());
          for (const auto& top_logprob : logprob.top_logprobs.value()) {
            XLLM_LogProbData xllm_logprob_data;
            xllm_logprob_data.token = top_logprob.token;
            xllm_logprob_data.token_id = top_logprob.token_id;
            xllm_logprob_data.logprob = top_logprob.logprob;
            xllm_logprob.top_logprobs.emplace_back(xllm_logprob_data);
          }
        }
        xllm_logprobs.emplace_back(xllm_logprob);
      }

      choice.logprobs = xllm_logprobs;
    }

    if (if_type == InterfaceType::COMPLETIONS) {
      choice.text = output.text;
    } else if (if_type == InterfaceType::CHAT_COMPLETIONS) {
      XLLM_ChatMessage chat_message;
      chat_message.role = "assistant";
      chat_message.content = output.text;
      choice.message = chat_message;
    }

    if (output.finish_reason.has_value()) {
      choice.finish_reason = output.finish_reason.value();
    }

    response.choices.emplace_back(choice);
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.usage.prompt_tokens = usage.num_prompt_tokens;
    response.usage.completion_tokens = usage.num_generated_tokens;
    response.usage.total_tokens = usage.num_total_tokens;
    response.usage.cached_tokens = usage.num_cached_tokens;
  }

  return response;
}

XLLM_Response build_error_response(const std::string& request_id,
                                   XLLM_StatusCode status_code,
                                   const std::string& error_info) {
  XLLM_Response response;
  response.status_code = status_code;
  response.error_info = error_info;
  response.id = request_id.empty() ? "unknown_request" : request_id;

  LOG(ERROR) << "Request [" << response.id << "] error: " << error_info
             << " (code: " << static_cast<int>(response.status_code) << ")";

  return response;
}

template <typename InputType>
XLLM_Response handle_inference_request(LLMCore* llm_core,
                                       const std::string& model_id,
                                       const InputType& input,
                                       uint32_t timeout_ms,
                                       const XLLM_RequestParams& request_params,
                                       InterfaceType interface_type) {
  if (!llm_core) {
    return build_error_response(
        "", XLLM_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  auto it = std::find(
      llm_core->model_ids.begin(), llm_core->model_ids.end(), model_id);
  if (it == llm_core->model_ids.end()) {
    return build_error_response("",
                                XLLM_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  RequestParams xllm_request_params = transfer_request_params(request_params);
  std::string request_id = xllm_request_params.request_id.empty()
                               ? generate_request_id()
                               : xllm_request_params.request_id;
  xllm_request_params.request_id = request_id;
  int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<XLLM_Response>>();
    auto weak_promise =
        std::weak_ptr<folly::Promise<XLLM_Response>>(promise_ptr);
    auto future = promise_ptr->getSemiFuture();

    llm_core->master->handle_request(
        input,
        std::nullopt,
        xllm_request_params,
        std::nullopt,
        [model_id,
         request_id,
         created_time,
         interface_type,
         weak_promise,
         timeout_ms](const RequestOutput& req_output) -> bool {
          auto promise_ptr = weak_promise.lock();
          if (!promise_ptr) {
            return false;
          }

          try {
            XLLM_Response response = build_success_response(
                req_output, interface_type, request_id, created_time, model_id);
            promise_ptr->setValue(std::move(response));
          } catch (const folly::PromiseAlreadySatisfied& e) {
            return false;
          }

          return true;
        });

    return std::move(future)
        .via(llm_core->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([](folly::Try<XLLM_Response>&& result) {
          if (result.hasValue()) {
            return std::move(result).value();
          } else {
            result.throwUnlessValue();
            return XLLM_Response{};
          }
        })
        .get();

  } catch (const folly::FutureTimeout& e) {
    return build_error_response(
        request_id, XLLM_StatusCode::kTimeout, "Request timed out");
  } catch (const std::exception& e) {
    return build_error_response(
        request_id,
        XLLM_StatusCode::kInternalError,
        "Failed to handle request: " + std::string(e.what()));
  }
}

}  // namespace detail
}  // namespace xllm
