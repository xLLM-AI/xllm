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

#include "embedding_service_impl.h"

#include <glog/logging.h>

#include <string>
#include <vector>

#include "common/instance_name.h"
#include "distributed_runtime/llm_master.h"
#include "embedding_output_builder.h"
#include "framework/request/request_params.h"
#include "mm_service_utils.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {
namespace {

template <typename EmbeddingCall>
bool send_result_to_client_brpc(std::shared_ptr<EmbeddingCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("list");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_data()->Reserve(req_output.outputs.size());
  std::string encoding_format = call->request().encoding_format();
  bool use_binary_format = encoding_format == "binary";
  EmbeddingOutputBuilder mm_embeddings_output_builder(use_binary_format, false);
  std::string binary_payload;
  for (const auto& output : req_output.outputs) {
    // add data into response
    auto* data = response.add_data();
    data->set_index(output.index);
    data->set_object("embedding");
    if (output.embeddings.has_value()) {
      data->mutable_embedding()->Add(
          output.embeddings->data(),
          output.embeddings->data() + output.embeddings->size());
    }
    if (output.mm_embeddings.has_value()) {
      call->set_bytes_to_base64(true);
      mm_embeddings_output_builder.build_repeated_embedding_output(
          *output.mm_embeddings,
          *(data->mutable_mm_embeddings()),
          binary_payload);
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

  return call->write_and_finish(response, binary_payload);
}

}  // namespace

void BatchEmbeddingContext::on_complete(size_t index, RequestOutput output) {
  req_outputs_[index] = std::move(output);
  if (pending_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    finalize();
  }
}

void BatchEmbeddingContext::finalize() {
  auto call = std::move(call_);
  auto& response = call->response();
  response.set_object("list");
  response.set_id(request_id_);
  response.set_created(created_time_);
  response.set_model(model_);

  std::string encoding_format = call->request().encoding_format();
  bool use_binary_format = encoding_format == "binary";
  EmbeddingOutputBuilder mm_embeddings_output_builder(use_binary_format, false);
  std::string binary_payload;

  int32_t total_prompt_tokens = 0;
  int32_t total_generated_tokens = 0;
  int32_t total_tokens = 0;

  response.mutable_data()->Reserve(static_cast<int32_t>(req_outputs_.size()));
  for (size_t i = 0; i < req_outputs_.size(); ++i) {
    const auto& req_output = req_outputs_[i];

    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        call->finish_with_error(status.code(), status.message());
        return;
      }
    }

    for (const auto& output : req_output.outputs) {
      auto* data = response.add_data();
      data->set_index(static_cast<int32_t>(i));
      data->set_object("embedding");
      if (output.embeddings.has_value()) {
        data->mutable_embedding()->Add(
            output.embeddings->data(),
            output.embeddings->data() + output.embeddings->size());
      }
      if (output.mm_embeddings.has_value()) {
        call->set_bytes_to_base64(true);
        mm_embeddings_output_builder.build_repeated_embedding_output(
            *output.mm_embeddings,
            *(data->mutable_mm_embeddings()),
            binary_payload);
      }
    }

    if (req_output.usage.has_value()) {
      const auto& usage = req_output.usage.value();
      total_prompt_tokens += usage.num_prompt_tokens;
      total_generated_tokens += usage.num_generated_tokens;
      total_tokens += usage.num_total_tokens;
    }
  }

  if (total_tokens > 0) {
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(total_prompt_tokens);
    proto_usage->set_completion_tokens(total_generated_tokens);
    proto_usage->set_total_tokens(total_tokens);
  }

  call->write_and_finish(response, binary_payload);
}

EmbeddingServiceImpl::EmbeddingServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

// embedding_async for brpc
void EmbeddingServiceImpl::process_async_impl(
    std::shared_ptr<EmbeddingCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // create RequestParams for embeddings request
  // set is_embeddings and max_tokens = 1 to control engine step once.
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  // TODO only support input_str for now
  auto& input = rpc_request.input();

  auto saved_request_id = request_params.request_id;
  // schedule the request
  master_->handle_request(
      std::move(input),
      std::nullopt,
      std::move(request_params),
      call.get(),
      [call,
       model,
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }

        return send_result_to_client_brpc<EmbeddingCall>(
            call, request_id, created_time, model, req_output);
      });
}

void EmbeddingServiceImpl::process_batch_async(
    std::shared_ptr<EmbeddingCall> call,
    std::vector<std::string> inputs) {
  const auto& rpc_request = call->request();
  std::string model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  std::string request_id = request_params.request_id;
  int64_t created_time = absl::ToUnixSeconds(absl::Now());

  size_t num_requests = inputs.size();
  std::vector<RequestParams> sps(num_requests, request_params);

  auto ctx = std::make_shared<BatchEmbeddingContext>(
      std::move(call),
      std::move(model),
      std::move(request_id),
      created_time,
      num_requests);

  auto batch_callback = [ctx](size_t index, RequestOutput output) -> bool {
    ctx->on_complete(index, std::move(output));
    return true;
  };

  master_->handle_batch_request(
      std::move(inputs), std::move(sps), batch_callback);
}

MMEmbeddingServiceImpl::MMEmbeddingServiceImpl(
    VLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

void MMEmbeddingServiceImpl::process_async_impl(
    std::shared_ptr<MMEmbeddingCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // create RequestParams for embeddings request
  // set is_embeddings and max_tokens = 1 to control engine step once.
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  auto& req_messages = rpc_request.messages();

  std::vector<Message> messages;
  if (!mm_service_utils::build_messages<MMEmbeddingCall>(
          req_messages, messages, call, master_->get_image_limit())) {
    return;
  }
  auto request_id = request_params.request_id;

  auto payload = call->take_request_payload();

  // schedule the request
  master_->handle_request(
      std::move(messages),
      std::move(request_params),
      std::move(payload),
      [call,
       model,
       request_id = request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }

        return send_result_to_client_brpc<MMEmbeddingCall>(
            call, request_id, created_time, model, req_output);
      });
}
}  // namespace xllm
