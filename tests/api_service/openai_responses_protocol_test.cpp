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
#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <butil/fd_guard.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "api_service/openai_http.h"
#include "api_service/openai_responses_call.h"
#include "api_service/openai_responses_output.h"
#include "api_service/openai_responses_request.h"
#include "core/common/rate_limiter.h"
#include "core/framework/config/service_config.h"
#include "core/framework/request/request.h"
#include "core/util/scope_guard.h"
#include "xllm_service.pb.h"

extern char** environ;

namespace xllm::api_service {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

class CountingClosure final : public google::protobuf::Closure {
 public:
  CountingClosure(google::protobuf::Closure* done, std::atomic<int32_t>* count)
      : done_(done), count_(count) {}

  void Run() override {
    count_->fetch_add(1, std::memory_order_relaxed);
    done_->Run();
    delete this;
  }

 private:
  google::protobuf::Closure* done_;
  std::atomic<int32_t>* count_;
};

RequestOutput fixture_chunk(std::string text,
                            bool finished = false,
                            const std::string& reason = "stop") {
  RequestOutput result;
  result.force_reasoning = false;
  result.finished = finished;
  result.outputs.emplace_back();
  result.outputs.back().index = 0;
  result.outputs.back().text = std::move(text);
  if (finished) {
    result.outputs.back().finish_reason = reason;
    Usage usage;
    usage.num_prompt_tokens = 9;
    usage.num_generated_tokens = 5;
    usage.num_total_tokens = 14;
    usage.num_cached_tokens = 3;
    result.usage = usage;
  }
  return result;
}

// Only generation is deterministic. Request decoding, HTTP/SSE transport,
// incremental parsers and admission-slot ownership use production code.
class ResponsesFixtureService final : public proto::XllmAPIService {
 public:
  void ResponsesHttp(google::protobuf::RpcController* controller,
                     const proto::HttpRequest* /*request*/,
                     proto::HttpResponse* /*response*/,
                     google::protobuf::Closure* done) override {
    requests_.fetch_add(1, std::memory_order_relaxed);
    brpc::ClosureGuard done_guard(new CountingClosure(done, &closures_));
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    if (ctrl->http_request().method() != brpc::HTTP_METHOD_POST) {
      write_openai_error(
          ctrl, StatusCode::INVALID_ARGUMENT, "Responses requires POST");
      ctrl->http_response().set_status_code(405);
      ctrl->http_response().SetHeader("Allow", "POST");
      return;
    }
    std::string error_param;
    auto [status, parsed] =
        parse_responses_request(ctrl->request_attachment().to_string(),
                                "responses-fixture",
                                &error_param);
    if (!status.ok()) {
      write_openai_error(ctrl, status.code(), status.message(), error_param);
      return;
    }
    ResponsesCall call(ctrl, done_guard.release(), parsed.params.streaming);
    if (parsed.model != "responses-fixture") {
      call.finish_with_error(
          StatusCode::NOT_FOUND, "Unknown fixture model", "model");
      return;
    }
    const Status admission = rate_limiter_.acquire();
    if (!admission.ok()) {
      call.finish_with_error(admission.code(), admission.message());
      EXPECT_FALSE(call.write_response(Json::object()));
      EXPECT_FALSE(call.write_event(
          {{"type", "response.created"}, {"sequence_number", 0}}));
      call.finish();
      return;
    }
    ScopeGuard preparation_guard([this] {
      std::lock_guard<std::mutex> lock(mutex_);
      rate_limiter_.decrease_one_request();
      changed_.notify_all();
    });
    const Message& last = parsed.messages.back();
    const std::string selector = last.role == "tool"
                                     ? "tool-output"
                                     : std::get<std::string>(last.content);
    if (selector == "__responses_fixture_preflight_failed__") {
      call.finish_with_error(StatusCode::RESOURCE_EXHAUSTED,
                             "Fixture engine resource exhaustion");
      call.finish();
      return;
    }
    RequestState state("fixture",
                       {1},
                       parsed.params.to_sampling_param(/*best_of=*/1),
                       parsed.params.to_scheduler_param(),
                       StoppingChecker{},
                       /*seq_capacity=*/32,
                       /*n=*/1,
                       /*best_of=*/1,
                       /*logprobs=*/false,
                       parsed.params.streaming,
                       /*echo=*/false,
                       /*skip_special_tokens=*/true,
                       /*enable_schedule_overlap=*/false,
                       OutputFunc{},
                       OutputsFunc{},
                       /*decode_address=*/"",
                       &call);
    state.responses_request = true;
    auto owner = std::make_shared<Request>(parsed.params.request_id,
                                           "",
                                           "",
                                           std::move(state),
                                           "",
                                           "",
                                           &rate_limiter_);
    preparation_guard.dismiss();
    const bool reasoning = selector == "__responses_fixture_reasoning__";
    ResponsesOutput output(
        responses_initial_response(parsed),
        parsed.params.streaming,
        parsed.params.tools,
        parsed.params.tools.empty() ? "" : "glm47",
        reasoning ? "qwen3" : "",
        /*force_reasoning=*/false,
        [&call](const Json& event) { return call.write_event(event); });
    bool accepted = true;
    if (selector == "__responses_fixture_hold__" ||
        selector == "__responses_fixture_hold_before__") {
      if (selector == "__responses_fixture_hold__") {
        accepted = output.append(fixture_chunk("hello "));
      }
      if (accepted) {
        std::unique_lock<std::mutex> lock(mutex_);
        held_request_ = owner;
        changed_.notify_all();
        while (!stopping_ && !owner->cancelled()) {
          owner->update_connection_status();
          if (!owner->cancelled()) {
            changed_.wait_for(lock, std::chrono::milliseconds(5));
          }
        }
        owner->set_cancel();
        cancelled_.fetch_add(1, std::memory_order_relaxed);
        held_request_.reset();
      }
      RequestOutput cancelled;
      cancelled.cancelled = true;
      output.append(cancelled);
      call.finish();
    } else {
      if (reasoning) {
        for (const std::string text : {"<th", "ink>why</th", "ink>answer"}) {
          accepted = output.append(fixture_chunk(text));
          if (!accepted) {
            break;
          }
        }
      } else if (selector == "__responses_fixture_tool__") {
        for (const std::string text :
             {"<tool_call>weat",
              "her<arg_key>city</arg_key><arg_value>Par",
              "is</arg_value></tool_call>"}) {
          accepted = output.append(fixture_chunk(text));
          if (!accepted) {
            break;
          }
        }
      } else if (selector == "__responses_fixture_utf8__") {
        for (const std::string text : {std::string("\xE4\xBD", 2),
                                       std::string("\xA0\xE5\xA5", 3),
                                       std::string("\xBD \xF0\x9F", 4),
                                       std::string("\x91\x8B", 2)}) {
          accepted = output.append(fixture_chunk(text));
          if (!accepted) {
            break;
          }
        }
      } else if (selector == "__responses_fixture_json__") {
        accepted = output.append(fixture_chunk("{\"ok\":")) &&
                   output.append(fixture_chunk("true}"));
      } else if (selector == "tool-output") {
        accepted = output.append(fixture_chunk("It is sunny in Paris."));
      } else if (selector == "__responses_fixture_incomplete__" ||
                 selector == "__responses_fixture_failed__") {
        accepted = output.append(fixture_chunk("partial"));
      } else {
        accepted = output.append(fixture_chunk("hello ")) &&
                   output.append(fixture_chunk("world"));
      }
      if (accepted) {
        if (selector == "__responses_fixture_failed__") {
          output.fail(StatusCode::UNKNOWN, "Fixture generation failed");
        } else {
          const std::string reason =
              selector == "__responses_fixture_incomplete__" ? "length"
              : selector == "__responses_fixture_tool__"     ? "function_call"
                                                             : "stop";
          output.append(fixture_chunk("", true, reason));
        }
      }
      if (!parsed.params.streaming &&
          output.status().code() != StatusCode::CANCELLED) {
        call.write_response(output.snapshot());
      }
      call.finish();
      call.finish();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      owner.reset();
      changed_.notify_all();
    }
  }

  bool wait_held() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
      return held_request_ != nullptr;
    });
  }

  bool wait_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
      return rate_limiter_.get_num_concurrent_requests() == 0;
    });
  }

  void cancel_held() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (held_request_ != nullptr) {
      held_request_->set_cancel();
    }
    changed_.notify_all();
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    changed_.notify_all();
  }

  RateLimiter& rate_limiter() { return rate_limiter_; }
  int32_t cancelled() const {
    return cancelled_.load(std::memory_order_relaxed);
  }
  int32_t requests() const { return requests_.load(std::memory_order_relaxed); }
  int32_t closures() const { return closures_.load(std::memory_order_relaxed); }

 private:
  RateLimiter rate_limiter_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::shared_ptr<Request> held_request_;
  bool stopping_ = false;
  std::atomic<int32_t> requests_{0};
  std::atomic<int32_t> closures_{0};
  std::atomic<int32_t> cancelled_{0};
};

std::vector<Json> sse_events(const std::string& body) {
  std::vector<Json> events;
  size_t start = 0;
  while (start < body.size()) {
    const size_t end = body.find("\n\n", start);
    EXPECT_NE(end, std::string::npos);
    if (end == std::string::npos) {
      break;
    }
    const std::string frame = body.substr(start, end - start);
    const size_t data = frame.find("\ndata: ");
    EXPECT_TRUE(frame.starts_with("event: "));
    EXPECT_NE(data, std::string::npos);
    if (data == std::string::npos) {
      break;
    }
    const auto event = Json::parse(frame.substr(data + 7));
    EXPECT_EQ(event["type"], frame.substr(7, data - 7));
    EXPECT_EQ(event["sequence_number"], events.size());
    events.emplace_back(event);
    start = end + 2;
  }
  return events;
}

class OpenAIResponsesProtocolTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_limit_ = ServiceConfig::get_instance().max_concurrent_requests();
    ServiceConfig::get_instance().max_concurrent_requests(1);
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/v1/responses => ResponsesHttp"),
              0);
    ASSERT_EQ(server_.Start(/*port=*/0, /*options=*/nullptr), 0);
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.timeout_ms = 5000;
    options.max_retry = 0;
    ASSERT_EQ(channel_.Init(server_.listen_address(), &options), 0);
  }

  void TearDown() override {
    service_.stop();
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    EXPECT_EQ(service_.closures(), service_.requests());
    ServiceConfig::get_instance().max_concurrent_requests(previous_limit_);
  }

  Json request_body(bool stream,
                    const std::string& input = "__responses_fixture_text__") {
    return {{"model", "responses-fixture"},
            {"input", input},
            {"stream", stream},
            {"max_output_tokens", 64},
            {"store", false}};
  }

  void request(const Json& body, brpc::Controller& controller) {
    request_raw(body.dump(), controller);
  }

  void request_raw(const std::string& body,
                   brpc::Controller& controller,
                   const std::string& path = "/v1/responses") {
    controller.http_request().uri() = path;
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().set_content_type("application/json");
    controller.request_attachment().append(body);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  }

  int32_t open_socket(const Json& request) {
    butil::fd_guard fd(socket(AF_INET, SOCK_STREAM, 0));
    if (static_cast<int32_t>(fd) < 0) {
      return -1;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(server_.listen_address().port);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
        0) {
      return -1;
    }
    const std::string body = request.dump();
    const std::string wire =
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t offset = 0;
    while (offset < wire.size()) {
      const ssize_t sent =
          send(fd, wire.data() + offset, wire.size() - offset, MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) {
        continue;
      }
      if (sent <= 0) {
        return -1;
      }
      offset += static_cast<size_t>(sent);
    }
    return fd.release();
  }

  std::string read_until(int32_t fd, const std::string& marker) {
    std::string wire;
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (wire.find(marker) == std::string::npos && Clock::now() < deadline) {
      pollfd readable{fd, POLLIN, 0};
      const int32_t ready = poll(&readable, 1, 100);
      if (ready < 0 && errno == EINTR) {
        continue;
      }
      if (ready <= 0) {
        continue;
      }
      char buffer[4096];
      const ssize_t size = recv(fd, buffer, sizeof(buffer), 0);
      if (size <= 0) {
        break;
      }
      wire.append(buffer, static_cast<size_t>(size));
    }
    return wire;
  }

  void check_success(bool stream) {
    brpc::Controller controller;
    request(request_body(stream), controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    ASSERT_EQ(controller.http_response().status_code(), 200);
    const std::string body = controller.response_attachment().to_string();
    Json final;
    if (stream) {
      EXPECT_EQ(controller.http_response().content_type(),
                "text/event-stream; charset=utf-8");
      const auto events = sse_events(body);
      ASSERT_FALSE(events.empty());
      EXPECT_EQ(events.front()["type"], "response.created");
      EXPECT_EQ(events.back()["type"], "response.completed");
      EXPECT_EQ(std::count_if(events.begin(),
                              events.end(),
                              [](const auto& event) {
                                return event["type"] == "response.completed";
                              }),
                1);
      std::string deltas;
      for (const auto& event : events) {
        if (event["type"] == "response.output_text.delta") {
          deltas += event["delta"].template get<std::string>();
        }
      }
      EXPECT_EQ(deltas, "hello world");
      final = events.back()["response"];
    } else {
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      final = Json::parse(body);
    }
    EXPECT_EQ(final["object"], "response");
    EXPECT_EQ(final["status"], "completed");
    EXPECT_TRUE(final["id"].get<std::string>().starts_with("resp_"));
    EXPECT_EQ(final["output"][0]["content"][0]["text"], "hello world");
    EXPECT_EQ(final["usage"]["input_tokens"], 9);
    EXPECT_EQ(final["usage"]["output_tokens"], 5);
    EXPECT_EQ(final["usage"]["total_tokens"], 14);
    EXPECT_EQ(final["usage"]["input_tokens_details"],
              (Json{{"cached_tokens", 3}, {"cache_write_tokens", 0}}));
    const auto& reasoning_tokens =
        final["usage"]["output_tokens_details"]["reasoning_tokens"];
    EXPECT_TRUE(reasoning_tokens.is_number_integer());
    EXPECT_EQ(reasoning_tokens, 0);
    EXPECT_EQ(final["store"], false);
    EXPECT_FALSE(final.contains("choices"));
    EXPECT_EQ(body.find("chat.completion"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    ASSERT_TRUE(service_.wait_idle());
  }

  ResponsesFixtureService service_;
  brpc::Server server_;
  brpc::Channel channel_;
  int32_t previous_limit_ = 0;
};

TEST_F(OpenAIResponsesProtocolTest,
       CompletedJsonAndNamedSseUseResponseObjects) {
  check_success(false);
  check_success(true);
}

TEST_F(OpenAIResponsesProtocolTest, OnlyPluralPostRouteIsRegistered) {
  brpc::Controller wrong_route;
  request_raw(request_body(false).dump(), wrong_route, "/v1/response");
  EXPECT_EQ(wrong_route.http_response().status_code(), 404);
  brpc::Controller get;
  get.http_request().uri() = "/v1/responses";
  get.http_request().set_method(brpc::HTTP_METHOD_GET);
  channel_.CallMethod(nullptr, &get, nullptr, nullptr, nullptr);
  EXPECT_EQ(get.http_response().status_code(), 405);
  ASSERT_NE(get.http_response().GetHeader("Allow"), nullptr);
  EXPECT_EQ(*get.http_response().GetHeader("Allow"), "POST");
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
}

TEST_F(OpenAIResponsesProtocolTest, ValidationAndPreflightFailuresStayJson) {
  for (const bool stream : {false, true}) {
    for (const auto& patch : {Json{{"store", true}},
                              Json{{"previous_response_id", "resp_old"}},
                              Json{{"max_output_tokens", 0}},
                              Json{{"background", true}},
                              Json{{"input", {{{"type", "input_image"}}}}}}) {
      auto body = request_body(stream);
      body.update(patch);
      brpc::Controller controller;
      request(body, controller);
      EXPECT_EQ(controller.http_response().status_code(), 400);
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      const std::string wire = controller.response_attachment().to_string();
      EXPECT_EQ(wire.find("event:"), std::string::npos);
      EXPECT_TRUE(Json::parse(wire).contains("error"));
      EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    }
    brpc::Controller exhausted;
    request(request_body(stream, "__responses_fixture_preflight_failed__"),
            exhausted);
    EXPECT_EQ(exhausted.http_response().status_code(), 500);
    EXPECT_EQ(exhausted.http_response().content_type(), "application/json");
    const auto error = Json::parse(exhausted.response_attachment().to_string());
    EXPECT_EQ(error["error"]["type"], "InternalServerError");
    EXPECT_EQ(error["error"]["code"], 500);
    ASSERT_TRUE(service_.wait_idle());
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    check_success(stream);
  }
  brpc::Controller malformed;
  request_raw("{", malformed);
  EXPECT_EQ(malformed.http_response().status_code(), 400);
}

TEST_F(OpenAIResponsesProtocolTest,
       LengthAndLateFailureHaveDistinctTerminalResponses) {
  for (const bool stream : {false, true}) {
    for (const bool failure : {false, true}) {
      brpc::Controller controller;
      request(request_body(stream,
                           failure ? "__responses_fixture_failed__"
                                   : "__responses_fixture_incomplete__"),
              controller);
      ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
      EXPECT_EQ(controller.http_response().status_code(), 200);
      const std::string body = controller.response_attachment().to_string();
      Json final;
      if (stream) {
        const auto events = sse_events(body);
        ASSERT_FALSE(events.empty());
        EXPECT_EQ(events.front()["type"], "response.created");
        EXPECT_EQ(events.back()["type"],
                  failure ? "response.failed" : "response.incomplete");
        EXPECT_EQ(std::count_if(events.begin(),
                                events.end(),
                                [](const auto& event) {
                                  return event["type"] == "response.failed" ||
                                         event["type"] ==
                                             "response.incomplete" ||
                                         event["type"] == "response.completed";
                                }),
                  1);
        final = events.back()["response"];
      } else {
        final = Json::parse(body);
      }
      EXPECT_EQ(final["status"], failure ? "failed" : "incomplete");
      EXPECT_EQ(final["output"][0]["content"][0]["text"], "partial");
      EXPECT_EQ(final["output"][0]["status"], "incomplete");
      if (failure) {
        EXPECT_EQ(final["error"]["message"], "Fixture generation failed");
      } else {
        EXPECT_EQ(final["incomplete_details"]["reason"], "max_output_tokens");
      }
      ASSERT_TRUE(service_.wait_idle());
      check_success(stream);
    }
  }
}

TEST_F(OpenAIResponsesProtocolTest, RawReasoningIsNotInventedSummary) {
  for (const bool stream : {false, true}) {
    brpc::Controller controller;
    request(request_body(stream, "__responses_fixture_reasoning__"),
            controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    const std::string body = controller.response_attachment().to_string();
    Json final;
    if (stream) {
      const auto events = sse_events(body);
      ASSERT_FALSE(events.empty());
      final = events.back()["response"];
      EXPECT_EQ(body.find("reasoning_summary"), std::string::npos);
      EXPECT_NE(body.find("response.reasoning_text.delta"), std::string::npos);
    } else {
      final = Json::parse(body);
    }
    ASSERT_EQ(final["output"].size(), 2);
    EXPECT_EQ(final["output"][0]["type"], "reasoning");
    EXPECT_TRUE(final["output"][0]["summary"].empty());
    EXPECT_EQ(final["output"][0]["content"][0]["text"], "why");
    EXPECT_EQ(final["output"][1]["content"][0]["text"], "answer");
    const auto& reasoning_tokens =
        final["usage"]["output_tokens_details"]["reasoning_tokens"];
    EXPECT_TRUE(reasoning_tokens.is_number_integer());
    EXPECT_EQ(reasoning_tokens, 0);
    ASSERT_TRUE(service_.wait_idle());
  }
}

TEST_F(OpenAIResponsesProtocolTest,
       RealAdmissionRejectsBothModesBeforeHeadersAndRecovers) {
  butil::fd_guard held(
      open_socket(request_body(true, "__responses_fixture_hold_before__")));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  pollfd readable{held, POLLIN, 0};
  EXPECT_EQ(poll(&readable, 1, 0), 0);
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  for (const bool stream : {false, true, false, true}) {
    brpc::Controller controller;
    request(request_body(stream), controller);
    EXPECT_EQ(controller.http_response().status_code(), 429);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    EXPECT_EQ(controller.http_response().GetHeader("Retry-After"), nullptr);
    const std::string body = controller.response_attachment().to_string();
    EXPECT_EQ(
        Json::parse(body),
        (Json{{"error",
               {{"message",
                 "The number of concurrent requests has reached the limit."},
                {"type", "RateLimitError"},
                {"param", nullptr},
                {"code", 429}}}}));
    EXPECT_EQ(body.find("data:"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  }
  service_.cancel_held();
  ASSERT_TRUE(service_.wait_idle());
  EXPECT_EQ(service_.cancelled(), 1);
  service_.cancel_held();
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
  check_success(false);
  check_success(true);
}

TEST_F(OpenAIResponsesProtocolTest,
       SocketDisconnectBeforeAndAfterFirstByteReleasesOnce) {
  for (const auto& [stream, selector] :
       {std::pair{false, "__responses_fixture_hold_before__"},
        std::pair{true, "__responses_fixture_hold_before__"},
        std::pair{true, "__responses_fixture_hold__"}}) {
    const int32_t cancelled_before = service_.cancelled();
    const int32_t closures_before = service_.closures();
    butil::fd_guard held(open_socket(request_body(stream, selector)));
    ASSERT_GE(static_cast<int32_t>(held), 0);
    ASSERT_TRUE(service_.wait_held());
    if (std::string(selector) == "__responses_fixture_hold__") {
      const std::string wire =
          read_until(held, "event: response.output_text.delta");
      EXPECT_NE(wire.find("HTTP/1.1 200"), std::string::npos);
      EXPECT_NE(wire.find("text/event-stream"), std::string::npos);
      EXPECT_NE(wire.find("event: response.created"), std::string::npos);
      EXPECT_NE(wire.find("event: response.output_text.delta"),
                std::string::npos);
    } else {
      pollfd readable{held, POLLIN, 0};
      EXPECT_EQ(poll(&readable, 1, 0), 0);
    }
    linger reset{1, 0};
    ASSERT_EQ(setsockopt(held, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    held.reset(-1);
    ASSERT_TRUE(service_.wait_idle())
        << "Real Request still owns its admission slot";
    EXPECT_EQ(service_.cancelled(), cancelled_before + 1);
    EXPECT_EQ(service_.closures(), closures_before + 1);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
    check_success(stream);
  }
}

TEST_F(OpenAIResponsesProtocolTest, SleepingIs503NotAdmission429) {
  ASSERT_TRUE(service_.rate_limiter().try_set_sleeping());
  for (const bool stream : {false, true}) {
    brpc::Controller controller;
    request(request_body(stream), controller);
    EXPECT_EQ(controller.http_response().status_code(), 503);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto error =
        Json::parse(controller.response_attachment().to_string());
    EXPECT_EQ(error["error"]["type"], "ServiceUnavailableError");
    EXPECT_TRUE(service_.rate_limiter().is_sleeping());
  }
  EXPECT_TRUE(service_.rate_limiter().try_wakeup());
  check_success(true);
}

TEST_F(OpenAIResponsesProtocolTest, DISABLED_OfficialSdkCompatibility) {
  std::string interpreter(PYTHON3_EXECUTABLE);
  std::string script(XLLM_RESPONSES_SDK_CLIENT_PATH);
  std::string base_url =
      "http://127.0.0.1:" + std::to_string(server_.listen_address().port) +
      "/v1";
  std::string base_url_flag = "--base-url";
  std::string model_flag = "--model";
  std::string model = "responses-fixture";
  std::string fixture_flag = "--fixture";
  char* args[] = {interpreter.data(),
                  script.data(),
                  base_url_flag.data(),
                  base_url.data(),
                  model_flag.data(),
                  model.data(),
                  fixture_flag.data(),
                  nullptr};
  pid_t pid = -1;
  ASSERT_EQ(
      posix_spawnp(&pid, interpreter.c_str(), nullptr, nullptr, args, environ),
      0);
  int32_t status = 0;
  pid_t waited;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  ASSERT_EQ(waited, pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  ASSERT_TRUE(service_.wait_idle());
}

}  // namespace
}  // namespace xllm::api_service
