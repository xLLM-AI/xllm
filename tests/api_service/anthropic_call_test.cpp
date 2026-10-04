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

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include "api_service/stream_call.h"
#include "xllm_service.pb.h"

namespace xllm {
namespace {

class StreamTestService final : public proto::XllmAPIService {
 public:
  void AnthropicMessagesHttp(google::protobuf::RpcController* controller,
                             const proto::HttpRequest* /*request*/,
                             proto::HttpResponse* /*response*/,
                             google::protobuf::Closure* done) override {
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    const std::string mode = ctrl->request_attachment().to_string();
    if (mode == "openai") {
      proto::ChatRequest request;
      request.set_stream(true);
      proto::ChatResponse response;
      StreamCall<proto::ChatRequest, proto::ChatResponse> call(
          ctrl, done, &request, &response, /*use_arena=*/true);
      call.finish();
      return;
    }
    proto::AnthropicMessagesRequest request;
    request.set_stream(mode != "count");
    proto::AnthropicMessagesResponse response;
    AnthropicCall call(ctrl,
                       done,
                       &request,
                       &response,
                       /*use_arena=*/true,
                       /*is_http_request=*/true);
    if (mode == "invalid") {
      call.finish_with_error(StatusCode::INVALID_ARGUMENT, "invalid request");
      return;
    }
    if (mode == "limited" || mode == "limited_then_write") {
      EXPECT_FALSE(call.finish_with_error(StatusCode::RATE_LIMITED, "busy"));
      if (mode == "limited_then_write") {
        EXPECT_FALSE(call.write_and_finish(response));
        EXPECT_FALSE(call.write("message_start", std::string("{}")));
        EXPECT_FALSE(call.finish_with_error(StatusCode::UNKNOWN, "late error"));
        call.finish();
      }
      return;
    }
    if (mode == "exhausted") {
      call.finish_with_error(StatusCode::RESOURCE_EXHAUSTED, "queue full");
      return;
    }
    if (mode == "count") {
      proto::AnthropicCountTokensResponse count;
      count.set_input_tokens(7);
      count.mutable_context_management()->set_original_input_tokens(7);
      call.write_and_finish(count);
      return;
    }
    proto::AnthropicStreamEvent event;
    event.set_type("message_start");
    event.mutable_message()->set_type("message");
    event.mutable_message()->set_role("assistant");
    event.mutable_message()->mutable_usage();
    call.write(event.type(), event);
    if (mode == "stream_error") {
      call.finish_with_error(StatusCode::UNKNOWN, "failed generation");
      return;
    }
    event.Clear();
    event.set_type("message_stop");
    call.write(event.type(), event);
    call.finish();
    call.finish();
  }
};

class AnthropicCallTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/test => AnthropicMessagesHttp"),
              0);
    ASSERT_EQ(server_.Start(/*port=*/0, /*options=*/nullptr), 0);
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.timeout_ms = 5000;
    options.max_retry = 0;
    ASSERT_EQ(channel_.Init(server_.listen_address(), &options), 0);
  }

  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  void request(const std::string& mode, brpc::Controller& controller) {
    controller.http_request().uri() = "/test";
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.request_attachment().append(mode);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  }

  static void check_json_error(const brpc::Controller& controller,
                               int32_t status,
                               const std::string& type,
                               const std::string& message) {
    EXPECT_EQ(controller.http_response().status_code(), status);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    EXPECT_EQ(controller.http_response().GetHeader("Retry-After"), nullptr);
    const auto* request_id = controller.http_response().GetHeader("request-id");
    ASSERT_NE(request_id, nullptr);
    EXPECT_FALSE(request_id->empty());
    const std::string body = controller.response_attachment().to_string();
    EXPECT_EQ(nlohmann::json::parse(body),
              nlohmann::json({{"type", "error"},
                              {"error", {{"type", type}, {"message", message}}},
                              {"request_id", *request_id}}));
    EXPECT_EQ(body.find("event:"), std::string::npos);
    EXPECT_EQ(body.find("data:"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
  }

  StreamTestService service_;
  brpc::Server server_;
  brpc::Channel channel_;
};

TEST_F(AnthropicCallTest, ValidationErrorRemainsJsonBeforeStreamStarts) {
  brpc::Controller controller;
  request("invalid", controller);
  check_json_error(controller, 400, "BadRequestError", "invalid request");
}

TEST_F(AnthropicCallTest, RateLimitUses429BeforeStreamStarts) {
  brpc::Controller controller;
  request("limited", controller);
  check_json_error(controller, 429, "rate_limit_error", "busy");
}

TEST_F(AnthropicCallTest, ResourceExhaustionIsNotRateLimiting) {
  brpc::Controller controller;
  request("exhausted", controller);
  check_json_error(controller, 500, "api_error", "queue full");
}

TEST_F(AnthropicCallTest, RejectionPreventsLaterWritesFromStartingStream) {
  brpc::Controller controller;
  request("limited_then_write", controller);
  check_json_error(controller, 429, "rate_limit_error", "busy");
}

TEST_F(AnthropicCallTest, StreamClosesAfterMessageStopWithoutDoneSentinel) {
  brpc::Controller controller;
  request("stream", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  EXPECT_EQ(controller.http_response().status_code(), 200);
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("event: message_start\n"), std::string::npos);
  EXPECT_NE(body.find("event: message_stop\n"), std::string::npos);
  EXPECT_EQ(body.find("[DONE]"), std::string::npos);
}

TEST_F(AnthropicCallTest, GenerationErrorIsAnSseEventAndClosesStream) {
  brpc::Controller controller;
  request("stream_error", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  EXPECT_EQ(controller.http_response().status_code(), 200);
  EXPECT_EQ(controller.http_response().content_type(),
            "text/event-stream; charset=utf-8");
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("event: message_start\n"), std::string::npos);
  const std::string prefix = "event: error\ndata: ";
  const size_t offset = body.find(prefix);
  ASSERT_NE(offset, std::string::npos);
  const size_t data_start = offset + prefix.size();
  const size_t data_end = body.find("\n\n", data_start);
  ASSERT_NE(data_end, std::string::npos);
  const auto* request_id = controller.http_response().GetHeader("request-id");
  ASSERT_NE(request_id, nullptr);
  EXPECT_EQ(
      nlohmann::json::parse(body.substr(data_start, data_end - data_start)),
      nlohmann::json(
          {{"type", "error"},
           {"error", {{"type", "api_error"}, {"message", "failed generation"}}},
           {"request_id", *request_id}}));
  EXPECT_EQ(body.find("event: error\n", data_end), std::string::npos);
  EXPECT_EQ(body.find("[DONE]"), std::string::npos);
}

TEST_F(AnthropicCallTest, CountResponseIsOrdinaryJson) {
  brpc::Controller controller;
  request("count", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const auto body =
      nlohmann::json::parse(controller.response_attachment().to_string());
  EXPECT_EQ(body["input_tokens"], 7);
  EXPECT_EQ(body["context_management"]["original_input_tokens"], 7);
}

TEST_F(AnthropicCallTest, OpenAiStreamStillEmitsDoneSentinel) {
  brpc::Controller controller;
  request("openai", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  EXPECT_EQ(controller.response_attachment().to_string(), "data: [DONE]\n\n");
}

}  // namespace
}  // namespace xllm
