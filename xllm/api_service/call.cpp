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

#include "call.h"

#include "api_service/api_error.h"
#include "core/common/constants.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {

Call::Call(brpc::Controller* controller) : controller_(controller) { init(); }

void Call::init() {
  if (controller_->http_request().GetHeader("x-request-id")) {
    raw_header_x_request_id_ =
        *controller_->http_request().GetHeader("x-request-id");
  } else if (controller_->http_request().GetHeader("x-ms-client-request-id")) {
    raw_header_x_request_id_ =
        *controller_->http_request().GetHeader("x-ms-client-request-id");
  }

  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }

  // Resolve the request-scoped x-request-id once, here: the client-supplied
  // value when present, otherwise a server-generated id. Shared across logs,
  // the verbose trace and the engine so a request can be correlated end to end.
  x_request_id_ = raw_header_x_request_id_.empty()
                      ? api_service::generate_x_request_id()
                      : raw_header_x_request_id_;
  XLLM_VERBOSE_TRACE() << "event=request_received x-request-id="
                       << x_request_id_
                       << " path=" << controller_->http_request().uri().path();

  init_request_payload();
}

void Call::init_request_payload() {
  const auto infer_content_len =
      controller_->http_request().GetHeader(kInferContentLength);
  const auto content_len =
      controller_->http_request().GetHeader(kContentLength);

  if (infer_content_len == nullptr || content_len == nullptr) return;

  auto infer_len = std::stoul(*infer_content_len);
  auto len = std::stoul(*content_len);

  if (infer_len > len) {
    LOG(ERROR) << " content length is invalid:"
               << " infer content len is " << infer_len
               << " , content length is " << len;
    return;
  }

  controller_->request_attachment().copy_to(
      &request_payload_, len - infer_len, infer_len);
}

}  // namespace xllm
