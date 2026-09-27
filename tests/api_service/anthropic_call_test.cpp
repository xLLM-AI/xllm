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
    AnthropicCall call(ctrl, done, &request, &response, /*use_arena=*/true);
    if (mode == "invalid") {
      call.finish_with_error(StatusCode::INVALID_ARGUMENT, "invalid request");
      return;
    }
    if (mode == "limited") {
      call.finish_with_error(StatusCode::RESOURCE_EXHAUSTED, "busy");
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

  StreamTestService service_;
  brpc::Server server_;
  brpc::Channel channel_;
};

TEST_F(AnthropicCallTest, ValidationErrorRemainsJsonBeforeStreamStarts) {
  brpc::Controller controller;
  request("invalid", controller);
  EXPECT_EQ(controller.http_response().status_code(), 400);
  EXPECT_EQ(controller.http_response().content_type(), "application/json");
  const auto body =
      nlohmann::json::parse(controller.response_attachment().to_string());
  EXPECT_EQ(body["type"], "error");
  EXPECT_EQ(body["error"]["type"], "BadRequestError");
}

TEST_F(AnthropicCallTest, RateLimitUses429BeforeStreamStarts) {
  brpc::Controller controller;
  request("limited", controller);
  EXPECT_EQ(controller.http_response().status_code(), 429);
  const auto body =
      nlohmann::json::parse(controller.response_attachment().to_string());
  EXPECT_EQ(body["error"]["type"], "rate_limit_error");
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
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("event: error\n"), std::string::npos);
  EXPECT_NE(body.find("failed generation"), std::string::npos);
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
