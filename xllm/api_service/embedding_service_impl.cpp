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

#include "embedding_service_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>

#include "api_service/openai_batch.h"
#include "common/instance_name.h"
#include "distributed_runtime/llm_master.h"
#include "distributed_runtime/master_manager.h"
#include "embedding_output_builder.h"
#include "framework/config/model_config.h"
#include "framework/request/request_params.h"
#include "mm_service_utils.h"
#include "util/scope_guard.h"
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
      if (!mm_embeddings_output_builder.build_repeated_embedding_output(
              *output.mm_embeddings,
              *(data->mutable_mm_embeddings()),
              binary_payload)) {
        return call->finish_with_error(
            StatusCode::UNKNOWN, "Failed to serialize multimodal embeddings");
      }
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

EmbeddingServiceImpl::EmbeddingServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models,
    std::shared_ptr<MasterManager> master_manager)
    : APIServiceImpl(models), master_manager_(std::move(master_manager)) {
  CHECK(master != nullptr);
  CHECK(master_manager_ != nullptr);
}

// embedding_async for brpc
void EmbeddingServiceImpl::process_async_impl(
    std::shared_ptr<EmbeddingCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  auto master =
      std::dynamic_pointer_cast<LLMMaster>(master_manager_->find_master(model));
  if (master == nullptr) {
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + model + "` does not exist.",
                            "model");
    return;
  }

  if (rpc_request.has_dimensions()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "dimensions is not supported by this embedding backend.");
    return;
  }

  auto* rate_limiter = master->get_rate_limiter();
  int32_t reserved_slots = 0;
  ScopeGuard rate_limit_guard([rate_limiter, &reserved_slots] {
    while (reserved_slots > 0) {
      rate_limiter->decrease_one_request();
      --reserved_slots;
    }
  });
  const int32_t request_count =
      rpc_request.inputs().empty() ? 1 : rpc_request.inputs_size();
  for (int32_t index = 0; index < request_count; ++index) {
    const Status admission = rate_limiter->acquire();
    if (!admission.ok()) {
      call->finish_with_error(admission.code(), admission.message());
      return;
    }
    ++reserved_slots;
  }

  // create RequestParams for embeddings request
  // set is_embeddings and max_tokens = 1 to control engine step once.
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  const std::string request_id = request_params.request_id;
  const int64_t created_time = absl::ToUnixSeconds(absl::Now());
  OutputCallback send =
      [call, model, request_id, created_time](RequestOutput output) {
        if (output.status.has_value() && !output.status->ok()) {
          return call->finish_with_error(output.status->code(),
                                         output.status->message());
        }
        return send_result_to_client_brpc<EmbeddingCall>(
            call, request_id, created_time, model, output);
      };
  if (rpc_request.inputs().empty()) {
    master->handle_request(rpc_request.input(),
                           std::nullopt,
                           std::move(request_params),
                           call.get(),
                           std::move(send));
    --reserved_slots;
    return;
  }
  auto batch = std::make_shared<api_service::OpenAIBatch>(
      rpc_request.inputs_size(), /*choices_per_prompt=*/1, /*streaming=*/false);
  for (int32_t index = 0; index < rpc_request.inputs_size(); ++index) {
    const auto& input = rpc_request.inputs(index);
    RequestParams params(
        rpc_request, call->get_x_request_id(), call->get_x_request_time());
    params.request_id += "-" + std::to_string(index);
    std::optional<std::vector<int>> tokens;
    if (!input.token_ids().empty()) {
      tokens.emplace(input.token_ids().begin(), input.token_ids().end());
    }
    master->handle_request(input.text(),
                           std::move(tokens),
                           std::move(params),
                           call.get(),
                           [batch, index, send](RequestOutput output) {
                             return batch->accept(
                                 index, std::move(output), send);
                           });
    --reserved_slots;
  }
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
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + model + "` does not exist.",
                            "model");
    return;
  }

  if (rpc_request.has_dimensions()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "dimensions is not supported by this embedding backend.");
    return;
  }

  const Status admission = master_->get_rate_limiter()->acquire();
  if (!admission.ok()) {
    call->finish_with_error(admission.code(), admission.message());
    return;
  }
  ScopeGuard rate_limit_guard(
      [this] { master_->get_rate_limiter()->decrease_one_request(); });

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

  // mm_embed encodes multimodal inputs, so a text-only request is invalid.
  if (::xllm::ModelConfig::get_instance().task() == "mm_embed") {
    const bool has_multimodal_content =
        std::any_of(messages.begin(), messages.end(), [](Message& msg) {
          return msg.has_mm_content();
        });
    if (!has_multimodal_content) {
      call->finish_with_error(
          StatusCode::INVALID_ARGUMENT,
          "mm_embed request must contain multimodal content.");
      return;
    }
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
  rate_limit_guard.dismiss();
}
}  // namespace xllm
