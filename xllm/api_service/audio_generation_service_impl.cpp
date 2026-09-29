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

#include "audio_generation_service_impl.h"

#include <glog/logging.h>

#include "common/instance_name.h"
#include "core/util/binary_payload.h"
#include "distributed_runtime/dit_master.h"
#include "framework/request/dit_request_output.h"
#include "framework/request/dit_request_params.h"
#include "mm_service_utils.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {
namespace {

bool send_result_to_client_brpc(std::shared_ptr<AudioGenerationCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const std::string& output_type,
                                const DiTRequestOutput& req_output) {
  auto& response = call->response();
  auto* proto_output = mm_service_utils::initialize_generation_response(
      response, request_id, created_time, model);
  const std::vector<DiTGenerationOutput>& outputs = req_output.outputs;
  proto_output->mutable_results()->Reserve(
      static_cast<int32_t>(outputs.size()));

  const bool use_binary_output = output_type == "binary";
  std::string binary_payload;
  if (use_binary_output &&
      !mm_service_utils::reserve_binary_payload(
          outputs,
          [](const DiTGenerationOutput& output) -> std::string_view {
            return output.audio;
          },
          binary_payload)) {
    return call->finish_with_error(StatusCode::UNKNOWN,
                                   "Binary audio payload size overflow");
  }
  for (const auto& output : outputs) {
    auto* proto_result = proto_output->add_results();
    mm_service_utils::fill_media_source(output.audio,
                                        /*name=*/"audio",
                                        use_binary_output,
                                        *proto_result->mutable_audio(),
                                        binary_payload);
    proto_result->set_seed(output.seed);
  }
  return use_binary_output ? call->write_and_finish(response, binary_payload)
                           : call->write_and_finish(response);
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

  DiTRequestParams request_params(rpc_request,
                                  call->get_x_request_id(),
                                  call->get_x_request_time(),
                                  BinaryPayload(call->take_request_iobuf()));

  std::string saved_request_id = request_params.request_id;
  std::string output_type = request_params.output_type;
  master_->handle_request(
      std::move(request_params),
      call.get(),
      [call,
       model,
       request_id = std::move(saved_request_id),
       output_type = std::move(output_type),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const DiTRequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, output_type, req_output);
      });
}

}  // namespace xllm
