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

#include "text_generation_service_impl.h"

#include <glog/logging.h>

#include "distributed_runtime/dit_master.h"
#include "framework/request/dit_request_output.h"
#include "framework/request/dit_request_params.h"

namespace xllm {
namespace {

bool send_result_to_client_brpc(std::shared_ptr<TextGenerationCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const DiTRequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("list");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);
  auto* proto_output = response.mutable_output();
  const std::vector<DiTGenerationOutput>& outputs = req_output.outputs;
  proto_output->mutable_results()->Reserve(
      static_cast<int32_t>(outputs.size()));

  for (const auto& output : outputs) {
    auto* proto_result = proto_output->add_results();
    proto_result->set_text(output.text);
    if (output.seed_is_set) {
      proto_result->set_seed(output.seed);
    }
  }
  return call->write_and_finish(response);
}

}  // namespace

TextGenerationServiceImpl::TextGenerationServiceImpl(
    DiTMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_{master} {
  CHECK(master_ != nullptr);
}

void TextGenerationServiceImpl::process_async_impl(
    std::shared_ptr<TextGenerationCall> call) {
  const auto& rpc_request = call->request();
  const std::string& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  DiTRequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  const std::string saved_request_id = request_params.request_id;
  master_->handle_request(
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master_,
       request_id = saved_request_id,
       created_time = absl::ToUnixSeconds(absl::Now())](
          const DiTRequestOutput& req_output) -> bool {
        master->get_rate_limiter()->decrease_one_request();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
