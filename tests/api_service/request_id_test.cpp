/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "api_service/request_id.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <string>

#include "anthropic.pb.h"
#include "api_service/non_stream_call.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/instance_name.h"

namespace xllm {
namespace {

class NoopClosure final : public google::protobuf::Closure {
 public:
  void Run() override {}
};

using CompletionCallForTest =
    NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;

using ChatCallForTest = NonStreamCall<proto::ChatRequest, proto::ChatResponse>;

bool is_generated_numeric_id(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  for (char character : value) {
    if (character < '0' || character > '9') {
      return false;
    }
  }
  return true;
}

TEST(RequestIdTest, HeaderTakesPrecedenceOverBody) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "header-id");
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "header-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "header-id");
}

TEST(RequestIdTest, BodyIsUsedWhenHeaderIsMissing) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(RequestIdTest, ProtoRequestIdIsUsedWhenHeaderAndXRequestIdAreMissing) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_request_id("proto-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "proto-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "proto-id");
}

TEST(RequestIdTest, HeaderTakesPrecedenceOverProtoRequestId) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "header-id");
  proto::CompletionRequest request;
  request.set_request_id("proto-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "header-id");
}

TEST(RequestIdTest, BodyXRequestIdTakesPrecedenceOverProtoRequestId) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-x-id");
  request.set_request_id("proto-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "body-x-id");
}

TEST(RequestIdTest, ChatProtoRequestIdIsUsedWhenXRequestIdIsMissing) {
  brpc::Controller controller;
  proto::ChatRequest request;
  request.set_request_id("chat-proto-id");
  proto::ChatResponse response;
  NoopClosure done;

  ChatCallForTest call(&controller,
                       &done,
                       &request,
                       &response,
                       /*use_arena=*/true,
                       /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "chat-proto-id");
}

TEST(RequestIdTest, InvalidPrimaryHeaderFallsBackToSecondaryHeader) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "invalid\r\nid");
  controller.http_request().SetHeader("x-ms-client-request-id", "fallback-id");
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "fallback-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            "fallback-id");
}

TEST(RequestIdTest, InvalidHeadersAreReplacedBeforeResponseHeader) {
  brpc::Controller controller;
  const std::string invalid_primary_id = "invalid\r\nprimary";
  const std::string invalid_secondary_id = "invalid\nsecondary";
  controller.http_request().SetHeader("x-request-id", invalid_primary_id);
  controller.http_request().SetHeader("x-ms-client-request-id",
                                      invalid_secondary_id);
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_TRUE(is_generated_numeric_id(call.get_x_request_id()));
  EXPECT_NE(call.get_x_request_id(), invalid_primary_id);
  EXPECT_NE(call.get_x_request_id(), invalid_secondary_id);
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            call.get_x_request_id());
}

TEST(RequestIdTest, BodyOverridesIdGeneratedBeforeParsing) {
  brpc::Controller controller;
  const std::string early_x_request_id =
      api_service::ensure_http_x_request_id(&controller);
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_NE(early_x_request_id, "body-id");
  EXPECT_EQ(call.get_x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(RequestIdTest, MissingIdIsGeneratedAndReturned) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_TRUE(is_generated_numeric_id(call.get_x_request_id()));
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            call.get_x_request_id());
}

TEST(RequestIdTest, InvalidBodyIdIsReplacedBeforeResponseHeader) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id\r\nx-injected: true");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_TRUE(is_generated_numeric_id(call.get_x_request_id()));
  EXPECT_EQ(call.get_x_request_id().find('\r'), std::string::npos);
  EXPECT_EQ(call.get_x_request_id().find('\n'), std::string::npos);
}

TEST(RequestIdTest, EnforcesLengthLimit) {
  EXPECT_TRUE(api_service::is_valid_x_request_id(std::string(256, 'a')));
  EXPECT_FALSE(api_service::is_valid_x_request_id(std::string(257, 'a')));
}

TEST(RequestIdTest, TypedCallUsesBodyIdWithoutHttpResponseHeader) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("typed-body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true);

  EXPECT_EQ(call.get_x_request_id(), "typed-body-id");
  EXPECT_EQ(controller.http_response().GetHeader("x-request-id"), nullptr);
}

TEST(RequestIdTest, EarlyHttpIdSurvivesSetFailed) {
  brpc::Controller controller;
  const std::string x_request_id =
      api_service::ensure_http_x_request_id(&controller);

  controller.SetFailed("request failed before parsing");

  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            x_request_id);
}

TEST(RequestIdTest, ExtractsIdFromSupportedRequestBodies) {
  proto::CompletionRequest completion;
  completion.set_x_request_id("completion-id");
  EXPECT_EQ(api_service::request_body_x_request_id(&completion),
            "completion-id");

  proto::ChatRequest chat;
  chat.set_x_request_id("chat-id");
  EXPECT_EQ(api_service::request_body_x_request_id(&chat), "chat-id");

  proto::ChatRequest chat_request_id_only;
  chat_request_id_only.set_request_id("chat-request-id");
  EXPECT_EQ(api_service::request_body_x_request_id(&chat_request_id_only),
            "chat-request-id");

  proto::ChatRequest chat_prefers_x_request_id;
  chat_prefers_x_request_id.set_x_request_id("chat-x-id");
  chat_prefers_x_request_id.set_request_id("chat-request-id");
  EXPECT_EQ(api_service::request_body_x_request_id(&chat_prefers_x_request_id),
            "chat-x-id");

  proto::AnthropicMessagesRequest anthropic;
  anthropic.set_x_request_id("anthropic-id");
  EXPECT_EQ(api_service::request_body_x_request_id(&anthropic), "anthropic-id");
}

TEST(RequestIdTest, EnsureReusesExistingValidResponseHeader) {
  brpc::Controller controller;
  controller.http_response().SetHeader("x-request-id", "already-set");
  EXPECT_EQ(api_service::ensure_http_x_request_id(&controller), "already-set");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            "already-set");
}

TEST(RequestIdTest, EnsureReplacesInvalidResponseHeader) {
  brpc::Controller controller;
  controller.http_response().SetHeader("x-request-id", "bad\nid");
  const std::string x_request_id =
      api_service::ensure_http_x_request_id(&controller);
  EXPECT_TRUE(is_generated_numeric_id(x_request_id));
  EXPECT_EQ(x_request_id.find('\n'), std::string::npos);
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            x_request_id);
}

TEST(RequestIdTest, HttpGuardFillsResponseHeaderWhenCallIsMissing) {
  brpc::Controller controller;
  {
    api_service::HttpXRequestIdGuard x_request_id_guard(&controller);
  }
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_TRUE(is_generated_numeric_id(
      *controller.http_response().GetHeader("x-request-id")));
}

TEST(RequestIdTest, HttpGuardKeepsCallResolvedBodyId) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);
  {
    api_service::HttpXRequestIdGuard x_request_id_guard(&controller);
  }

  EXPECT_EQ(call.get_x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(RequestIdTest, ReadsRequestTimeFromPrimaryThenFallbackHeader) {
  brpc::Controller primary;
  primary.http_request().SetHeader("x-request-time", "1000");
  EXPECT_EQ(api_service::get_header_x_request_time(&primary), "1000");

  brpc::Controller fallback;
  fallback.http_request().SetHeader("x-request-timems", "2000");
  EXPECT_EQ(api_service::get_header_x_request_time(&fallback), "2000");
}

TEST(RequestIdTest, NextRequestIdIsUniqueInt64) {
  const int64_t first = next_request_id();
  const int64_t second = next_request_id();
  EXPECT_NE(first, second);
}

TEST(RequestIdTest, GenerateRequestIdEncodesNumericIdOnce) {
  const std::string first = generate_request_id("cmpl-");
  const std::string second = generate_request_id("cmpl-");
  EXPECT_EQ(first.find("cmpl-"), 0);
  EXPECT_EQ(second.find("cmpl-"), 0);
  EXPECT_NE(first, second);

  const std::string suffix = first.substr(5);
  EXPECT_TRUE(is_generated_numeric_id(suffix));
}

TEST(RequestIdTest, GenerateRequestIdWithoutPrefixIsNumeric) {
  const std::string generated = generate_request_id(/*prefix=*/"");
  EXPECT_TRUE(is_generated_numeric_id(generated));
}

}  // namespace
}  // namespace xllm
