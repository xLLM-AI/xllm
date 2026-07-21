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

#include "video_generation_service_impl.h"

#include <butil/base64.h>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "api_service/utils.h"
#include "core/framework/request/dit_request_params.h"
#include "distributed_runtime/dit_master.h"

namespace xllm {

namespace {

bool send_result_to_client_brpc(std::shared_ptr<VideoGenerationCall> call,
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
  proto_output->mutable_results()->Reserve(outputs.size());

  std::string video;
  for (const auto& output : outputs) {
    auto* proto_result = proto_output->add_results();

    video.clear();
    butil::Base64Encode(output.image, &video);
    proto_result->set_video(video);

    proto_result->set_width(output.width);
    proto_result->set_height(output.height);
    proto_result->set_seed(output.seed);
    proto_result->set_num_frames(output.num_frames);
    proto_result->set_fps(output.video_fps);
  }
  return call->write_and_finish(response);
}

}  // namespace

VideoGenerationServiceImpl::VideoGenerationServiceImpl(
    DiTMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models), master_{master} {
  CHECK(master_ != nullptr);
}

void VideoGenerationServiceImpl::process_async_impl(
    std::shared_ptr<VideoGenerationCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  DiTRequestParams request_params(
      rpc_request, call->x_request_id(), call->x_request_time());

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
