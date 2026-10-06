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

#include "api_service/openai_responses_call.h"

#include <brpc/callback.h>
#include <glog/logging.h>

#include <utility>

#include "api_service/openai_http.h"

namespace xllm {

ResponsesCall::ResponsesCall(brpc::Controller* controller,
                             google::protobuf::Closure* done,
                             bool stream)
    : Call(controller, /*body_x_request_id=*/"", /*is_http_request=*/true),
      stream_(stream),
      done_(done),
      connection_(std::make_shared<ConnectionState>()) {
  CHECK(done_ != nullptr);
  controller_->http_response().set_content_type("application/json");
  controller_->NotifyOnCancel(brpc::NewCallback(on_cancel, connection_));
}

ResponsesCall::~ResponsesCall() {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_ = true;
  attachment_.reset();
  complete_locked();
}

void ResponsesCall::on_cancel(std::shared_ptr<ConnectionState> state) {
  // brpc also calls NotifyOnCancel after successful completion. The attachment
  // owns disconnect detection once the controller has been handed back.
  if (state->pre_commit.load(std::memory_order_acquire)) {
    state->cancelled.store(true, std::memory_order_release);
  }
}

void ResponsesCall::on_stopped(std::shared_ptr<ConnectionState> state) {
  state->stopped.store(true, std::memory_order_release);
}

void ResponsesCall::complete_locked() {
  if (done_ == nullptr) {
    return;
  }
  connection_->pre_commit.store(false, std::memory_order_release);
  controller_ = nullptr;
  auto* done = std::exchange(done_, nullptr);
  done->Run();
}

bool ResponsesCall::start_stream_locked() {
  if (stream_started_) {
    return true;
  }
  if (connection_->cancelled.load(std::memory_order_acquire)) {
    return false;
  }
  CHECK(stream_);
  CHECK(controller_ != nullptr);
  attachment_ = controller_->CreateProgressiveAttachment();
  attachment_->NotifyOnStopped(brpc::NewCallback(on_stopped, connection_));
  controller_->http_response().set_status_code(200);
  controller_->http_response().set_content_type(
      "text/event-stream; charset=utf-8");
  controller_->http_response().SetHeader("Cache-Control", "no-cache");
  controller_->http_response().SetHeader("Connection", "keep-alive");
  stream_started_ = true;
  complete_locked();
  return !connection_->stopped.load(std::memory_order_acquire);
}

bool ResponsesCall::write_response(const nlohmann::json& response) {
  const std::string body = response.dump();
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ || stream_started_) {
    return false;
  }
  finished_ = true;
  if (connection_->cancelled.load(std::memory_order_acquire)) {
    complete_locked();
    return false;
  }
  controller_->http_response().set_status_code(200);
  controller_->http_response().set_content_type("application/json");
  controller_->response_attachment().append(body);
  complete_locked();
  return true;
}

bool ResponsesCall::write_event(const nlohmann::json& event) {
  const std::string type = event.at("type").get<std::string>();
  const int64_t sequence_number = event.at("sequence_number").get<int64_t>();
  const std::string body = event.dump();
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ || !stream_ ||
      connection_->cancelled.load(std::memory_order_acquire) ||
      connection_->stopped.load(std::memory_order_acquire)) {
    return false;
  }
  CHECK_EQ(sequence_number, next_sequence_number_);
  if (!start_stream_locked()) {
    return false;
  }
  butil::IOBuf frame;
  frame.append("event: " + type + "\ndata: " + body + "\n\n");
  if (attachment_->Write(frame) != 0) {
    connection_->stopped.store(true, std::memory_order_release);
    return false;
  }
  ++next_sequence_number_;
  return true;
}

bool ResponsesCall::finish_with_error(StatusCode code,
                                      const std::string& message,
                                      const std::string& param) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_) {
    return false;
  }
  finished_ = true;
  if (!stream_started_) {
    if (!connection_->cancelled.load(std::memory_order_acquire)) {
      api_service::write_openai_error(controller_, code, message, param);
    }
    complete_locked();
    return false;
  }
  if (!connection_->stopped.load(std::memory_order_acquire)) {
    const nlohmann::json error = {
        {"type", "error"},
        {"sequence_number", next_sequence_number_++},
        {"code", "server_error"},
        {"message", message},
        {"param",
         param.empty() ? nlohmann::json(nullptr) : nlohmann::json(param)}};
    butil::IOBuf frame;
    frame.append("event: error\ndata: " + error.dump() + "\n\n");
    if (attachment_->Write(frame) != 0) {
      connection_->stopped.store(true, std::memory_order_release);
    }
  }
  attachment_.reset();
  return false;
}

bool ResponsesCall::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_) {
    return true;
  }
  finished_ = true;
  attachment_.reset();
  complete_locked();
  return true;
}

bool ResponsesCall::is_disconnected() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_) {
    return connection_->cancelled.load(std::memory_order_acquire);
  }
  return connection_->cancelled.load(std::memory_order_acquire) ||
         (stream_started_ &&
          connection_->stopped.load(std::memory_order_acquire));
}

bool ResponsesCall::stream_started() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stream_started_;
}

}  // namespace xllm
