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

#include <butil/fd_guard.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>

#include "api_service/anthropic_request_utils.h"
#include "api_service/chat_json_parser.h"
#include "api_service/stream_call.h"
#include "tests/api_service/http_protocol_test_fixture.h"
#include "util/scope_guard.h"

namespace xllm::api_service {
namespace {

class AnthropicMessagesTestService final : public AdmissionTestService {
 public:
  void AnthropicMessagesHttp(google::protobuf::RpcController* controller,
                             const proto::HttpRequest*,
                             proto::HttpResponse*,
                             google::protobuf::Closure* done) override {
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    if (ctrl->http_request().uri().path() == "/test") {
      direct_call(ctrl, done);
      return;
    }
    anthropic_request(ctrl, done, /*count_tokens=*/false);
  }

  void AnthropicCountTokensHttp(google::protobuf::RpcController* controller,
                                const proto::HttpRequest*,
                                proto::HttpResponse*,
                                google::protobuf::Closure* done) override {
    anthropic_request(static_cast<brpc::Controller*>(controller),
                      done,
                      /*count_tokens=*/true);
  }

 private:
  void anthropic_request(brpc::Controller* controller,
                         google::protobuf::Closure* done,
                         bool count_tokens) {
    auto [status, normalized] = ChatJsonParser::anthropic().preprocess(
        controller->request_attachment().to_string());
    proto::AnthropicMessagesRequest request;
    proto::AnthropicMessagesResponse response;
    if (status.ok()) {
      status = parse_anthropic_request(normalized, count_tokens, request);
    }
    if (status.ok()) {
      status = validate_anthropic_request(request, count_tokens);
    }
    AnthropicCall call(controller,
                       done,
                       &request,
                       &response,
                       /*use_arena=*/true,
                       /*is_http_request=*/true);
    if (!status.ok()) {
      call.finish_with_error(status.code(), status.message());
      return;
    }
    if (count_tokens) {
      status = rate_limiter().acquire();
      if (!status.ok()) {
        call.finish_with_error(status.code(), status.message());
        return;
      }
      // Count-token work owns the slot directly, like count_chat_tokens.
      ScopeGuard guard([this] { rate_limiter().decrease_one_request(); });
      proto::AnthropicCountTokensResponse count;
      count.set_input_tokens(1);
      call.write_and_finish(count);
      return;
    }

    response.set_id("msg-fixture");
    response.set_type("message");
    response.set_role("assistant");
    response.set_model("fixture");
    response.mutable_usage()->set_input_tokens(1);
    response.mutable_usage()->set_output_tokens(0);
    serve_admitted(
        call,
        request.messages(0).content_string(),
        [&] {
          proto::AnthropicStreamEvent event;
          event.set_type("message_start");
          *event.mutable_message() = response;
          call.write(event.type(), event);
          event.Clear();
          event.set_type("content_block_start");
          event.set_index(0);
          event.mutable_content_block()->set_type("text");
          event.mutable_content_block()->set_text("");
          call.write(event.type(), event);
          event.Clear();
          event.set_type("content_block_delta");
          event.set_index(0);
          event.mutable_delta()->set_type("text_delta");
          event.mutable_delta()->set_text("hi");
          call.write(event.type(), event);
        },
        [&] {
          if (!request.stream()) {
            auto* content = response.add_content();
            content->set_type("text");
            content->set_text("hi");
            response.set_stop_reason("end_turn");
            response.mutable_usage()->set_output_tokens(1);
            call.write_and_finish(response);
            return;
          }
          proto::AnthropicStreamEvent event;
          event.set_type("content_block_stop");
          event.set_index(0);
          call.write(event.type(), event);
          event.Clear();
          event.set_type("message_delta");
          event.mutable_delta()->set_stop_reason("end_turn");
          event.mutable_usage()->set_output_tokens(1);
          call.write(event.type(), event);
          event.Clear();
          event.set_type("message_stop");
          call.write(event.type(), event);
          call.finish();
        });
  }

  void direct_call(brpc::Controller* ctrl, google::protobuf::Closure* done) {
    const std::string mode = ctrl->request_attachment().to_string();
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

class AnthropicMessagesProtocolTest : public HttpProtocolTestFixture {
 protected:
  void SetUp() override {
    start_service(service_,
                  "/test => AnthropicMessagesHttp,"
                  "/v1/messages => AnthropicMessagesHttp,"
                  "/v1/messages/count_tokens => AnthropicCountTokensHttp");
  }
  void request(const std::string& mode, brpc::Controller& controller) {
    post("/test", mode, controller, /*content_type=*/nullptr);
  }
  std::string messages_body(bool stream,
                            const std::string& content = "hi") const {
    return nlohmann::json(
               {{"model", "fixture"},
                {"stream", stream},
                {"max_tokens", 8},
                {"messages", {{{"role", "user"}, {"content", content}}}}})
        .dump();
  }
  void messages(bool stream, brpc::Controller& controller) {
    post("/v1/messages", messages_body(stream), controller);
  }
  int32_t open_socket(bool stream, const std::string& content) {
    return HttpProtocolTestFixture::open_socket("/v1/messages",
                                                messages_body(stream, content));
  }
  void check_success(bool stream) {
    brpc::Controller controller;
    messages(stream, controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    EXPECT_EQ(controller.http_response().status_code(), 200);
    const std::string body = controller.response_attachment().to_string();
    if (stream) {
      EXPECT_EQ(controller.http_response().content_type(),
                "text/event-stream; charset=utf-8");
      EXPECT_NE(body.find("\"text\":\"hi\""), std::string::npos);
      EXPECT_NE(body.find("event: message_start\n"), std::string::npos);
      const size_t stop = body.find("event: message_stop\n");
      ASSERT_NE(stop, std::string::npos);
      EXPECT_EQ(body.find("event: message_stop\n", stop + 1),
                std::string::npos);
      EXPECT_TRUE(body.ends_with("data: {\"type\":\"message_stop\"}\n\n"));
      EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    } else {
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      const auto json = nlohmann::json::parse(body);
      EXPECT_EQ(json["type"], "message");
      EXPECT_EQ(json["role"], "assistant");
      EXPECT_EQ(json["model"], "fixture");
      EXPECT_EQ(json["content"][0]["text"], "hi");
      EXPECT_EQ(json["stop_reason"], "end_turn");
    }
    ASSERT_TRUE(service_.wait_released());
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
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

  AnthropicMessagesTestService service_;
};

TEST_F(AnthropicMessagesProtocolTest,
       RealAdmissionRejectsBothModesBeforeSseAndRecovers) {
  check_success(/*stream=*/false);
  check_success(/*stream=*/true);
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  pollfd readable{held, POLLIN, 0};
  EXPECT_EQ(poll(&readable, 1, 0), 0) << "SSE headers committed before output";
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  for (const bool stream : {false, true, false, true}) {
    brpc::Controller controller;
    messages(stream, controller);
    EXPECT_EQ(controller.http_response().status_code(), 429);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    EXPECT_EQ(controller.http_response().GetHeader("Retry-After"), nullptr);
    const std::string body = controller.response_attachment().to_string();
    const auto json = nlohmann::json::parse(body);
    const std::string message =
        "The number of concurrent requests has reached the limit.";
    const std::string* request_id =
        controller.http_response().GetHeader("request-id");
    ASSERT_NE(request_id, nullptr);
    EXPECT_FALSE(request_id->empty());
    const std::string* x_request_id =
        controller.http_response().GetHeader("x-request-id");
    ASSERT_NE(x_request_id, nullptr);
    EXPECT_EQ(*x_request_id, *request_id);
    EXPECT_EQ(
        json,
        nlohmann::json(
            {{"type", "error"},
             {"error", {{"type", "rate_limit_error"}, {"message", message}}},
             {"request_id", *request_id}}));
    EXPECT_EQ(body.find("data:"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  }
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  check_success(/*stream=*/false);
  check_success(/*stream=*/true);
}

TEST_F(AnthropicMessagesProtocolTest,
       SleepingAdmissionIsUnavailableNotRateLimited) {
  ASSERT_TRUE(service_.rate_limiter().try_set_sleeping());
  for (const bool stream : {false, true}) {
    brpc::Controller controller;
    messages(stream, controller);
    EXPECT_EQ(controller.http_response().status_code(), 503);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto error = nlohmann::json::parse(
        controller.response_attachment().to_string())["error"];
    EXPECT_EQ(error["type"], "overloaded_error");
    EXPECT_EQ(error["message"], "Model is currently in sleep state.");
    EXPECT_TRUE(service_.rate_limiter().is_sleeping());
  }
  EXPECT_TRUE(service_.rate_limiter().try_wakeup());
  check_success(/*stream=*/true);
}

TEST_F(AnthropicMessagesProtocolTest,
       ExplicitCancellationReleasesHeldRequestOnce) {
  for (const bool stream : {false, true}) {
    butil::fd_guard held(open_socket(stream, "hold"));
    ASSERT_GE(static_cast<int32_t>(held), 0);
    ASSERT_TRUE(service_.wait_held());
    service_.release(/*cancel=*/true);
    ASSERT_TRUE(service_.wait_released());
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    service_.release(/*cancel=*/true);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    check_success(stream);
  }
}

TEST_F(AnthropicMessagesProtocolTest,
       SocketDisconnectBeforeAndAfterSseReleasesRequest) {
  for (const auto& [stream, content] :
       {std::pair{false, "hold"}, {true, "hold"}, {true, "hold_started"}}) {
    butil::fd_guard held(open_socket(stream, content));
    ASSERT_GE(static_cast<int32_t>(held), 0);
    ASSERT_TRUE(service_.wait_held());
    pollfd readable{held, POLLIN, 0};
    if (std::string(content) == "hold_started") {
      ASSERT_EQ(poll(&readable, 1, 5000), 1);
      char buffer[4096];
      const ssize_t size = recv(held, buffer, sizeof(buffer), 0);
      ASSERT_GT(size, 0);
      EXPECT_NE(
          std::string(buffer, static_cast<size_t>(size)).find("HTTP/1.1 200"),
          std::string::npos);
    } else {
      EXPECT_EQ(poll(&readable, 1, 0), 0);
    }
    linger reset{1, 0};
    ASSERT_EQ(setsockopt(held, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    held.reset(-1);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool cancelled = service_.observe_disconnect();
    while (!cancelled && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      cancelled = service_.observe_disconnect();
    }
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
    service_.release();
    ASSERT_TRUE(service_.wait_released());
    check_success(stream);
  }
}

TEST_F(AnthropicMessagesProtocolTest, DISABLED_OfficialSdkCompatibility) {
  const std::string url = base_url();
  run_sdk(XLLM_MESSAGES_SDK_CLIENT, url, "success");
  ASSERT_TRUE(service_.wait_released());
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  run_sdk(XLLM_MESSAGES_SDK_CLIENT, url, "limited");
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  run_sdk(XLLM_MESSAGES_SDK_CLIENT, url, "success");
  ASSERT_TRUE(service_.wait_released());
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
}

TEST_F(AnthropicMessagesProtocolTest,
       AnthropicCountTokensRejectsAndRecoversWithSharedLimit) {
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  auto count_tokens = [this](brpc::Controller& controller) {
    post("/v1/messages/count_tokens",
         messages_body(/*stream=*/false),
         controller);
  };
  brpc::Controller rejected;
  count_tokens(rejected);
  EXPECT_EQ(rejected.http_response().status_code(), 429);
  EXPECT_EQ(rejected.http_response().content_type(), "application/json");
  EXPECT_EQ(rejected.http_response().GetHeader("Retry-After"), nullptr);
  const auto error =
      nlohmann::json::parse(rejected.response_attachment().to_string());
  EXPECT_EQ(error["type"], "error");
  EXPECT_EQ(error["error"],
            nlohmann::json({{"type", "rate_limit_error"},
                            {"message",
                             "The number of concurrent requests "
                             "has reached the limit."}}));
  const std::string* request_id =
      rejected.http_response().GetHeader("request-id");
  ASSERT_NE(request_id, nullptr);
  EXPECT_EQ(error["request_id"], *request_id);
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  brpc::Controller success;
  count_tokens(success);
  ASSERT_FALSE(success.Failed()) << success.ErrorText();
  EXPECT_EQ(success.http_response().status_code(), 200);
  EXPECT_EQ(success.http_response().content_type(), "application/json");
  EXPECT_EQ(nlohmann::json::parse(
                success.response_attachment().to_string())["input_tokens"],
            1);
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
}

TEST_F(AnthropicMessagesProtocolTest,
       ValidationErrorRemainsJsonBeforeStreamStarts) {
  brpc::Controller controller;
  request("invalid", controller);
  check_json_error(controller, 400, "BadRequestError", "invalid request");
}

TEST_F(AnthropicMessagesProtocolTest, RateLimitUses429BeforeStreamStarts) {
  brpc::Controller controller;
  request("limited", controller);
  check_json_error(controller, 429, "rate_limit_error", "busy");
}

TEST_F(AnthropicMessagesProtocolTest, ResourceExhaustionIsNotRateLimiting) {
  brpc::Controller controller;
  request("exhausted", controller);
  check_json_error(controller, 500, "api_error", "queue full");
}

TEST_F(AnthropicMessagesProtocolTest,
       RejectionPreventsLaterWritesFromStartingStream) {
  brpc::Controller controller;
  request("limited_then_write", controller);
  check_json_error(controller, 429, "rate_limit_error", "busy");
}

TEST_F(AnthropicMessagesProtocolTest,
       StreamClosesAfterMessageStopWithoutDoneSentinel) {
  brpc::Controller controller;
  request("stream", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  EXPECT_EQ(controller.http_response().status_code(), 200);
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("event: message_start\n"), std::string::npos);
  EXPECT_NE(body.find("event: message_stop\n"), std::string::npos);
  EXPECT_EQ(body.find("[DONE]"), std::string::npos);
}

TEST_F(AnthropicMessagesProtocolTest,
       GenerationErrorIsAnSseEventAndClosesStream) {
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

TEST_F(AnthropicMessagesProtocolTest, CountResponseIsOrdinaryJson) {
  brpc::Controller controller;
  request("count", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const auto body =
      nlohmann::json::parse(controller.response_attachment().to_string());
  EXPECT_EQ(body["input_tokens"], 7);
  EXPECT_EQ(body["context_management"]["original_input_tokens"], 7);
}

}  // namespace
}  // namespace xllm::api_service
