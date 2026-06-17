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

#include "rec_completion_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "common/global_flags.h"
#include "common/instance_name.h"
#include "completion.pb.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/request/request_output.h"

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
struct RecEmitRecord {
  int32_t output_index = 0;
  int64_t item_id = 0;
  std::optional<RecItemInfo> item_info;
};

void append_rec_logprobs(proto::InferTensorContents* logprobs_context,
                         const SequenceOutput& output,
                         int32_t expected_count) {
  const auto& token_logprobs = output.token_ids_logprobs;
  const int32_t actual_count = static_cast<int32_t>(token_logprobs.size());

  for (int32_t i = 0; i < expected_count; ++i) {
    if (i < actual_count && token_logprobs[i].has_value()) {
      logprobs_context->mutable_fp32_contents()->Add(token_logprobs[i].value());
    } else {
      logprobs_context->mutable_fp32_contents()->Add(0.0f);
    }
  }
}

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

bool send_result_to_client_brpc_rec(std::shared_ptr<CompletionCall> call,
                                    const std::string& request_id,
                                    int64_t created_time,
                                    const std::string& model,
                                    const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  // add choices into response
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

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
  }

  // Add rec specific output tensors
  auto output_tensor = response.mutable_output_tensors()->Add();
  output_tensor->set_name("rec_result");
  proto::InferOutputTensor* logprobs_tensor = nullptr;
  int32_t logprob_width = 0;
  if (::xllm::RecConfig::get_instance().enable_output_sku_logprobs() &&
      !req_output.outputs.empty()) {
    logprobs_tensor = response.mutable_output_tensors()->Add();
    logprobs_tensor->set_name("sku_logprobs");
    logprobs_tensor->set_datatype(proto::DataType::FLOAT);
    logprob_width =
        static_cast<int32_t>(req_output.outputs[0].token_ids_logprobs.size());
  }

  if (::xllm::RecConfig::get_instance().enable_convert_tokens_to_item()) {
    output_tensor->set_datatype(proto::DataType::INT64);
    proto::InferOutputTensor* did_tensor = nullptr;
    proto::InferOutputTensor* type_tensor = nullptr;
    if (::xllm::RecConfig::get_instance().enable_extended_item_info()) {
      did_tensor = response.mutable_output_tensors()->Add();
      did_tensor->set_name("item_did");
      did_tensor->set_datatype(proto::DataType::STRING);

      type_tensor = response.mutable_output_tensors()->Add();
      type_tensor->set_name("item_type");
      type_tensor->set_datatype(proto::DataType::STRING);
    }

    std::vector<RecEmitRecord> emitted_items;
    emitted_items.reserve(req_output.outputs.size());
    const int32_t total_threshold =
        ::xllm::RecConfig::get_instance().total_conversion_threshold();
    for (int32_t i = 0; i < static_cast<int32_t>(req_output.outputs.size());
         ++i) {
      const auto& output = req_output.outputs[i];
      if (!output.item_ids_list.empty()) {
        const bool has_item_infos =
            output.item_infos_list.size() == output.item_ids_list.size();
        for (size_t item_idx = 0; item_idx < output.item_ids_list.size();
             ++item_idx) {
          if (static_cast<int32_t>(emitted_items.size()) >= total_threshold) {
            break;
          }
          std::optional<RecItemInfo> item_info;
          if (has_item_infos) {
            item_info = output.item_infos_list[item_idx];
          }
          RecEmitRecord emitted_item;
          emitted_item.output_index = i;
          emitted_item.item_id = output.item_ids_list[item_idx];
          emitted_item.item_info = std::move(item_info);
          emitted_items.emplace_back(std::move(emitted_item));
        }
      } else if (output.item_ids.has_value() &&
                 static_cast<int32_t>(emitted_items.size()) < total_threshold) {
        RecEmitRecord emitted_item;
        emitted_item.output_index = i;
        emitted_item.item_id = output.item_ids.value();
        emitted_item.item_info = output.item_info;
        emitted_items.emplace_back(std::move(emitted_item));
      }
      if (static_cast<int32_t>(emitted_items.size()) >= total_threshold) {
        break;
      }
    }

    const int32_t emitted_count = static_cast<int32_t>(emitted_items.size());
    output_tensor->mutable_shape()->Add(emitted_count);
    if (logprobs_tensor != nullptr) {
      logprobs_tensor->mutable_shape()->Add(emitted_count);
      logprobs_tensor->mutable_shape()->Add(logprob_width);
    }
    if (did_tensor != nullptr && type_tensor != nullptr) {
      did_tensor->mutable_shape()->Add(emitted_count);
      type_tensor->mutable_shape()->Add(emitted_count);
    }

    auto* output_context = output_tensor->mutable_contents();
    auto* logprobs_context = logprobs_tensor == nullptr
                                 ? nullptr
                                 : logprobs_tensor->mutable_contents();
    auto append_output_logprobs = [&](int32_t output_index) {
      if (logprobs_context != nullptr) {
        append_rec_logprobs(
            logprobs_context, req_output.outputs[output_index], logprob_width);
      }
    };
    for (const RecEmitRecord& emitted_item : emitted_items) {
      output_context->mutable_int64_contents()->Add(emitted_item.item_id);
      append_output_logprobs(emitted_item.output_index);
      if (did_tensor != nullptr && type_tensor != nullptr) {
        did_tensor->mutable_contents()->add_bytes_contents(
            emitted_item.item_info.has_value() ? emitted_item.item_info->did
                                               : "");
        type_tensor->mutable_contents()->add_bytes_contents(
            emitted_item.item_info.has_value() ? emitted_item.item_info->type
                                               : "");
      }
    }
  } else {
    output_tensor->set_datatype(proto::DataType::INT32);

    if (req_output.outputs.empty()) {
      output_tensor->mutable_shape()->Add(0);
      output_tensor->mutable_shape()->Add(0);
      if (logprobs_tensor != nullptr) {
        logprobs_tensor->mutable_shape()->Add(0);
        logprobs_tensor->mutable_shape()->Add(0);
      }
      return call->write_and_finish(response);
    }

    const int32_t output_count =
        static_cast<int32_t>(req_output.outputs.size());
    output_tensor->mutable_shape()->Add(output_count);
    output_tensor->mutable_shape()->Add(req_output.outputs[0].token_ids.size());
    if (logprobs_tensor != nullptr) {
      logprobs_tensor->mutable_shape()->Add(output_count);
      logprobs_tensor->mutable_shape()->Add(logprob_width);
    }

    auto* context = output_tensor->mutable_contents();
    auto* logprobs_context = logprobs_tensor == nullptr
                                 ? nullptr
                                 : logprobs_tensor->mutable_contents();
    auto append_output_logprobs = [&](int32_t output_index) {
      if (logprobs_context != nullptr) {
        append_rec_logprobs(
            logprobs_context, req_output.outputs[output_index], logprob_width);
      }
    };
    for (int32_t i = 0; i < output_count; ++i) {
      // LOG(INFO) << req_output.outputs[i].token_ids;
      context->mutable_int_contents()->Add(
          req_output.outputs[i].token_ids.begin(),
          req_output.outputs[i].token_ids.end());
      append_output_logprobs(i);
    }
  }

  return call->write_and_finish(response);
}

}  // namespace

RecCompletionServiceImpl::RecCompletionServiceImpl(
    RecMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

void RecCompletionServiceImpl::process_async_impl(
    std::shared_ptr<CompletionCall> call) {
  const auto& rpc_request = call->request();

  // check if model is supported
  const auto& model = rpc_request.model();
  if (unlikely(!models_.contains(model))) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited.
  if (unlikely(master_->get_rate_limiter()->is_limited())) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  if (::xllm::RecConfig::get_instance().enable_output_sku_logprobs()) {
    request_params.logprobs = true;
  }
  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  const auto& rpc_request_ref = call->request();
  std::optional<std::vector<proto::InferInputTensor>> input_tensors =
      std::nullopt;
  if (rpc_request_ref.input_tensors_size()) {
    std::vector<proto::InferInputTensor> tensors;
    tensors.reserve(rpc_request_ref.input_tensors_size());
    for (int i = 0; i < rpc_request_ref.input_tensors_size(); ++i) {
      tensors.push_back(rpc_request_ref.input_tensors(i));
    }
    input_tensors = std::move(tensors);
  }

  // schedule the request
  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  master_->handle_request(
      std::move(rpc_request_ref.prompt()),
      std::move(prompt_tokens),
      std::move(input_tensors),
      std::move(request_params),
      [call,
       model,
       master = master_,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       request_id = saved_request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
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
        if (req_output.finished || req_output.cancelled) {
          master->get_rate_limiter()->decrease_one_request();
        }

        return send_result_to_client_brpc_rec(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
