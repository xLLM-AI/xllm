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

#include "api_service/api_error.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>

#include "anthropic.pb.h"
#include "api_service/non_stream_call.h"
#include "api_service/xllm_metrics.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm {
namespace {

class NoopClosure final : public google::protobuf::Closure {
 public:
  void Run() override {}
};

TEST(ApiErrorTest, OpenAIErrorReplacesInvalidUtf8) {
  const std::string invalid_message("invalid-\xff", 9);

  std::string body;
  EXPECT_NO_THROW(body = api_service::make_openai_error_json(
                      StatusCode::INVALID_ARGUMENT, invalid_message));

  const auto parsed = nlohmann::json::parse(body);
  EXPECT_EQ(parsed["error"]["message"], "invalid-\xef\xbf\xbd");
  EXPECT_EQ(parsed["error"]["type"], "invalid_request_error");
}

TEST(ApiErrorTest, AnthropicErrorReplacesInvalidUtf8) {
  const std::string invalid_message("invalid-\xff", 9);

  std::string body;
  EXPECT_NO_THROW(body = api_service::make_anthropic_error_json(
                      StatusCode::INVALID_ARGUMENT, invalid_message));

  const auto parsed = nlohmann::json::parse(body);
  EXPECT_EQ(parsed["type"], "error");
  EXPECT_EQ(parsed["error"]["message"], "invalid-\xef\xbf\xbd");
}

TEST(XllmMetricsTest, UsesHttpStatusForStructuredErrors) {
  brpc::Controller success;
  success.http_response().set_status_code(200);
  EXPECT_FALSE(is_failed_request(&success));

  brpc::Controller client_error;
  client_error.http_response().set_status_code(400);
  EXPECT_TRUE(is_failed_request(&client_error));

  brpc::Controller server_error;
  server_error.http_response().set_status_code(500);
  EXPECT_TRUE(is_failed_request(&server_error));
}

TEST(CallTest, RequestIdHeaderTakesPrecedenceOverBody) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "header-id");
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
      &controller, &done, &request, &response, true);

  EXPECT_EQ(call.x_request_id(), "header-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "header-id");
}

TEST(CallTest, RequestIdFallsBackToBody) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
      &controller, &done, &request, &response, true);

  EXPECT_EQ(call.x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(CallTest, RequestIdIsGeneratedWhenHeaderAndBodyAreMissing) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;

  NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
      &controller, &done, &request, &response, true);

  EXPECT_FALSE(call.x_request_id().empty());
  EXPECT_EQ(call.x_request_id().find("req-"), 0);
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            call.x_request_id());
}

TEST(CallTest, ExtractsRequestIdFromAllSupportedBodyTypes) {
  proto::CompletionRequest completion;
  completion.set_x_request_id("completion-id");
  EXPECT_EQ(request_body_x_request_id(&completion), "completion-id");

  proto::ChatRequest chat;
  chat.set_x_request_id("chat-id");
  EXPECT_EQ(request_body_x_request_id(&chat), "chat-id");

  proto::AnthropicMessagesRequest anthropic;
  anthropic.set_x_request_id("anthropic-id");
  EXPECT_EQ(request_body_x_request_id(&anthropic), "anthropic-id");
}

}  // namespace
}  // namespace xllm
