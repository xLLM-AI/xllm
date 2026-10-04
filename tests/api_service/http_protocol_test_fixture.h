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

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "core/common/rate_limiter.h"
#include "core/framework/request/request.h"
#include "xllm_service.pb.h"

namespace xllm::api_service {

class AdmissionTestService : public proto::XllmAPIService {
 public:
  bool wait_held();
  bool wait_released();
  void release(bool cancel = false);
  void stop();
  bool observe_disconnect();
  RateLimiter& rate_limiter() { return rate_limiter_; }

 protected:
  template <typename CallType, typename Start, typename Complete>
  void serve_admitted(CallType& call,
                      const std::string& content,
                      Start start,
                      Complete complete) {
    const Status status = rate_limiter_.acquire();
    if (!status.ok()) {
      call.finish_with_error(status.code(), status.message());
      return;
    }
    RequestState state("hi",
                       {1},
                       RequestSamplingParam{},
                       SchedulerParam{},
                       StoppingChecker{},
                       /*seq_capacity=*/8,
                       /*n=*/1,
                       /*best_of=*/1,
                       /*logprobs=*/false,
                       call.request().stream(),
                       /*echo=*/false,
                       /*skip_special_tokens=*/true,
                       /*enable_schedule_overlap=*/false,
                       OutputFunc{},
                       OutputsFunc{},
                       /*decode_address=*/"",
                       &call);
    auto owner = std::make_shared<xllm::Request>(
        "fixture", "", "", std::move(state), "", "", &rate_limiter_);
    const bool held = content == "hold" || content == "hold_started";
    if (content == "hold_started") {
      start();
    }
    if (held) {
      std::unique_lock<std::mutex> lock(mutex_);
      released_ = false;
      held_request_ = owner;
      changed_.notify_all();
      changed_.wait(lock, [this] { return released_ || stopping_; });
      if (stopping_) {
        owner->set_cancel();
      }
    }
    if (owner->cancelled()) {
      call.finish_with_error(StatusCode::CANCELLED, "Request cancelled.");
    } else {
      if (call.request().stream() && content != "hold_started") {
        start();
      }
      complete();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (held) {
        held_request_.reset();
      }
      owner.reset();
      changed_.notify_all();
    }
  }

 private:
  RateLimiter rate_limiter_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::shared_ptr<xllm::Request> held_request_;
  bool released_ = false;
  bool stopping_ = false;
};

class HttpProtocolTestFixture : public testing::Test {
 protected:
  void start_service(AdmissionTestService& service, const char* routes);
  void TearDown() override;
  void post(const char* path,
            const std::string& body,
            brpc::Controller& controller,
            const char* content_type = "application/json");
  int32_t open_socket(const char* path, const std::string& body);
  std::string base_url() const;
  void run_sdk(const char* script,
               const std::string& url,
               const std::string& phase);

 private:
  AdmissionTestService* service_ = nullptr;
  brpc::Server server_;
  brpc::Channel channel_;
  int32_t previous_limit_ = 0;
};

}  // namespace xllm::api_service
