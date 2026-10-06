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

#pragma once

#include <brpc/progressive_attachment.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

#include "api_service/call.h"
#include "core/common/types.h"

namespace xllm {

// The HTTP controller belongs to brpc and ceases to be usable after done runs.
class ResponsesCall final : public Call {
 public:
  ResponsesCall(brpc::Controller* controller,
                google::protobuf::Closure* done,
                bool stream);
  ~ResponsesCall() override;

  bool write_response(const nlohmann::json& response);
  bool write_event(const nlohmann::json& event);
  bool finish_with_error(StatusCode code,
                         const std::string& message,
                         const std::string& param = "");
  bool finish();
  bool is_disconnected() const override;
  bool stream_started() const;

 private:
  struct ConnectionState {
    std::atomic<bool> pre_commit{true};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> stopped{false};
  };

  static void on_cancel(std::shared_ptr<ConnectionState> state);
  static void on_stopped(std::shared_ptr<ConnectionState> state);
  bool start_stream_locked();
  void complete_locked();

  const bool stream_;
  google::protobuf::Closure* done_;
  std::shared_ptr<ConnectionState> connection_;
  mutable std::mutex mutex_;
  butil::intrusive_ptr<brpc::ProgressiveAttachment> attachment_;
  bool stream_started_ = false;
  bool finished_ = false;
  int64_t next_sequence_number_ = 0;
};

}  // namespace xllm
