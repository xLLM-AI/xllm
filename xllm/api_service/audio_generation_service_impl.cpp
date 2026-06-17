/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "audio_generation_service_impl.h"

#include <glog/logging.h>

#include "butil/base64.h"
#include "common/instance_name.h"
#include "distributed_runtime/dit_master.h"
#include "framework/request/dit_request_output.h"
#include "framework/request/dit_request_params.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {
namespace {

bool send_result_to_client_brpc(std::shared_ptr<AudioGenerationCall> call,
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
    std::string audio_b64;
    butil::Base64Encode(output.audio, &audio_b64);
    proto_result->set_audio(audio_b64);
    proto_result->set_seed(output.seed);
  }
  return call->write_and_finish(response);
}

}  // namespace

AudioGenerationServiceImpl::AudioGenerationServiceImpl(
    DiTMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_{master} {
  CHECK(master_ != nullptr);
}

void AudioGenerationServiceImpl::process_async_impl(
    std::shared_ptr<AudioGenerationCall> call) {
  const auto& rpc_request = call->request();
  const std::string& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  DiTRequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  std::string saved_request_id = request_params.request_id;
  master_->handle_request(
      std::move(request_params),
      call.get(),
      [call,
       model,
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const DiTRequestOutput& req_output) -> bool {
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
