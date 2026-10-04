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

#include <brpc/callback.h>
#include <brpc/controller.h>
#include <butil/fd_guard.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "api_service/chat_request_decoder.h"
#include "api_service/openai_json.h"
#include "api_service/openai_request.h"
#include "api_service/stream_call.h"
#include "tests/api_service/http_protocol_test_fixture.h"

namespace xllm::api_service {
namespace {

TEST(ChatCompletionRequestTest, ChatAliasesAndNullableDefaults) {
  auto [status, body] = normalize_openai_request(
      R"({"messages":[{"role":"assistant","content":null,"reasoning":"why"}],
          "max_tokens":99,"max_completion_tokens":7,"stop":"END",
          "temperature":null,"tool_choice":null})",
      OpenAIEndpoint::CHAT,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  const auto json = nlohmann::json::parse(body);
  EXPECT_EQ(json["max_tokens"], 7);
  EXPECT_EQ(json["stop"], nlohmann::json::array({"END"}));
  EXPECT_EQ(json["messages"][0]["reasoning_content"], "why");
  EXPECT_EQ(json["temperature"], 1.0);
  EXPECT_EQ(json["model"], "model");
  EXPECT_FALSE(json.contains("tool_choice"));
}

TEST(ChatCompletionRequestTest, InvalidParametersAreClientErrors) {
  const auto base =
      nlohmann::json::parse(R"({"messages":[{"role":"user","content":"hi"}]})");
  for (const auto& patch :
       {nlohmann::json{{"stop", 3}},
        {{"stop", ""}},
        {{"top_p", 0}},
        {{"n", -1}},
        {{"max_completion_tokens", 0}},
        {{"max_completion_tokens", 1.5}},
        {{"stream", "true"}},
        {{"stream_options", {{"include_usage", true}}}},
        {{"stream", true}, {"stream_options", {{"include_usage", 1}}}},
        {{"top_logprobs", 2}, {"logprobs", false}},
        {{"tools", nlohmann::json::array()}},
        {{"tool_choice", "auto"}},
        {{"messages", nlohmann::json::array()}}}) {
    auto request = base;
    request.update(patch);
    auto [status, body] =
        normalize_openai_request(request.dump(), OpenAIEndpoint::CHAT, "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << request;
  }
}

TEST(ChatCompletionRequestTest, GreedySamplingRequiresOneChoice) {
  for (const bool stream : {false, true}) {
    for (const uint32_t n : {1U, 2U}) {
      for (const auto& temperature : {nlohmann::json(0),
                                      nlohmann::json(0.5),
                                      nlohmann::json(1e-8),
                                      nlohmann::json(nullptr)}) {
        nlohmann::json request = {
            {"prompt", {"hello", "world"}},
            {"messages", {{{"role", "user"}, {"content", "hello"}}}},
            {"temperature", temperature},
            {"n", n},
            {"stream", stream}};
        std::string param = "stale";
        const auto [status, body] = normalize_openai_request(
            request.dump(), OpenAIEndpoint::CHAT, "model", &param);
        const bool greedy_multiple = temperature == 0 && n > 1;
        EXPECT_EQ(status.ok(), !greedy_multiple) << request;
        EXPECT_TRUE(param.empty());
        if (status.ok() && temperature == 1e-8) {
          EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 0.01);
        }
      }
    }
  }
}

TEST(ChatCompletionRequestTest, Vllm023AcceptsTemperaturesAboveTwo) {
  const auto [status, body] = normalize_openai_request(
      R"({"prompt":"hi","messages":[{"role":"user","content":"hi"}],"temperature":3})",
      OpenAIEndpoint::CHAT,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 3);
}

TEST(ChatCompletionRequestTest, SchemaAndSamplingErrorsHaveDistinctTypes) {
  bool schema_error = true;
  const auto [status, body] = normalize_openai_request(R"({"messages":[]})",
                                                       OpenAIEndpoint::CHAT,
                                                       "model",
                                                       nullptr,
                                                       &schema_error);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(schema_error);
}

TEST(ChatCompletionRequestTest, ValidationErrorsIdentifyTheirParameters) {
  const nlohmann::json base = {
      {"messages", {{{"role", "user"}, {"content", "hello"}}}}};
  for (const auto& [patch, expected] :
       std::vector<std::pair<nlohmann::json, std::string>>{
           {{{"max_tokens", -1}}, "max_tokens"},
           {{{"max_tokens", 8}, {"max_completion_tokens", -1}}, "max_tokens"},
           {{{"temperature", -1}}, "temperature"},
           {{{"top_p", 0}}, "top_p"},
           {{{"stream_options", {{"include_usage", true}}}}, "stream_options"},
           {{{"top_logprobs", 2}}, "top_logprobs"},
           {{{"tool_choice", "auto"}}, "tool_choice"},
           {{{"n", 0}}, ""}}) {
    auto request = base;
    request.update(patch);
    std::string param = "stale";
    const auto [status, body] = normalize_openai_request(
        request.dump(), OpenAIEndpoint::CHAT, "model", &param);
    ASSERT_FALSE(status.ok()) << request;
    EXPECT_EQ(param, expected) << request;
    const auto error = nlohmann::json::parse(
        openai_error_json(status.code(), status.message(), param));
    EXPECT_EQ(
        error["error"]["param"],
        expected.empty() ? nlohmann::json(nullptr) : nlohmann::json(expected));
  }
  for (const auto& [choice, expected] :
       std::vector<std::pair<nlohmann::json, std::string>>{
           {42, "tool_choice"},
           {{{"type", "function"}}, "tool_choice.function"},
           {{{"type", "function"}, {"function", nlohmann::json::object()}},
            "tool_choice.function.name"},
           {{{"type", "function"}, {"function", {{"name", "missing"}}}},
            "tool_choice"}}) {
    auto request = base;
    request["tools"] = {
        {{"type", "function"}, {"function", {{"name", "weather"}}}}};
    request["tool_choice"] = choice;
    std::string param;
    EXPECT_FALSE(normalize_openai_request(
                     request.dump(), OpenAIEndpoint::CHAT, "model", &param)
                     .first.ok());
    EXPECT_EQ(param, expected);
  }
  std::string param = "stale";
  EXPECT_FALSE(
      normalize_openai_request("{", OpenAIEndpoint::CHAT, "model", &param)
          .first.ok());
  EXPECT_TRUE(param.empty());
  const auto [status, body] = normalize_openai_request(
      R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":-1,"max_completion_tokens":8})",
      OpenAIEndpoint::CHAT,
      "model",
      &param);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["max_tokens"], 8);
}

// The body-length utility is shared; keep its regression with Chat-typed
// ingress.
TEST(ChatCompletionRequestTest,
     InferenceBodyLengthIsCheckedWithoutRequiringContentLength) {
  brpc::Controller controller;
  controller.request_attachment().append("{}binary");
  controller.http_request().SetHeader(kInferContentLength, "2");
  auto [status, body] = openai_request_body(controller);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(body, "{}");
  for (const char* value : {"-1", "9999", "nan", "2x"}) {
    controller.http_request().SetHeader(kInferContentLength, value);
    std::tie(status, body) = openai_request_body(controller);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    std::tie(status, body) =
        openai_request_body(controller, /*binary_input=*/false);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(body, "{}binary");
  }
}

TEST(ChatCompletionRequestTest, CallPayloadParsingHandlesInvalidLengthHeaders) {
  for (const char* value :
       {"-1", "9999", "nan", "2x", "184467440737095516160"}) {
    for (const char* header : {kInferContentLength, kContentLength}) {
      brpc::Controller controller;
      controller.request_attachment().append("{}binary");
      controller.http_request().SetHeader(kInferContentLength, "2");
      controller.http_request().SetHeader(kContentLength, "8");
      controller.http_request().SetHeader(header, value);
      StreamCall<proto::ChatRequest, proto::ChatResponse> call(
          &controller,
          brpc::DoNothing(),
          new proto::ChatRequest(),
          new proto::ChatResponse(),
          /*use_arena=*/false,
          /*is_http_request=*/true);
      EXPECT_TRUE(call.take_request_payload().empty());
    }
  }
  brpc::Controller controller;
  controller.request_attachment().append("{}binary");
  controller.http_request().SetHeader(kInferContentLength, "2");
  controller.http_request().SetHeader(kContentLength, "8");
  StreamCall<proto::ChatRequest, proto::ChatResponse> call(
      &controller,
      brpc::DoNothing(),
      new proto::ChatRequest(),
      new proto::ChatResponse(),
      /*use_arena=*/false,
      /*is_http_request=*/true);
  EXPECT_EQ(call.take_request_payload(), "binary");
}

TEST(ChatCompletionResponseTest, NamedToolChoicePreservesStopAndLength) {
  for (const bool stream : {false, true}) {
    for (const std::string reason : {"tool_calls", "length", "stop"}) {
      proto::ChatResponse response;
      response.set_object(stream ? "chat.completion.chunk" : "chat.completion");
      response.add_choices()->set_finish_reason(reason);
      const auto named = openai_response_json(response, true);
      EXPECT_EQ(named["choices"][0]["finish_reason"],
                reason == "tool_calls" ? "stop" : reason);
      const auto automatic = openai_response_json(response);
      EXPECT_EQ(automatic["choices"][0]["finish_reason"], reason);
    }
  }
}

TEST(ChatCompletionResponseTest, StopReasonsKeepTheirJsonTypes) {
  for (const StopReason& reason : {StopReason{},
                                   StopReason{int32_t{123}},
                                   StopReason{std::string("END")}}) {
    proto::ChatResponse chat;
    set_proto_stop_reason(reason, chat.add_choices()->mutable_stop_reason());
    const nlohmann::json expected =
        std::holds_alternative<int32_t>(reason)       ? nlohmann::json(123)
        : std::holds_alternative<std::string>(reason) ? nlohmann::json("END")
                                                      : nlohmann::json(nullptr);
    for (const bool stream : {false, true}) {
      chat.set_object(stream ? "chat.completion.chunk" : "chat.completion");
      chat.mutable_choices(0)->set_finish_reason("stop");
      EXPECT_EQ(openai_response_json(chat)["choices"][0]["stop_reason"],
                expected);
    }
  }
}

// Fingerprints are shared by Chat and Text Completion; test the utility once.
TEST(ChatCompletionResponseTest,
     FingerprintIsStableAndAppearsOnlyOnFinalStreamMessages) {
  const std::string fingerprint = openai_system_fingerprint("model=a;tp=16");
  EXPECT_TRUE(fingerprint.starts_with("xllm-"));
  EXPECT_EQ(fingerprint, openai_system_fingerprint("model=a;tp=16"));
  EXPECT_NE(fingerprint, openai_system_fingerprint("model=a;tp=8"));
  EXPECT_NE(fingerprint, openai_system_fingerprint("model=b;tp=16"));
  for (const bool include_usage : {false, true}) {
    for (const bool terminal : {false, true}) {
      nlohmann::json chunk = {
          {"choices",
           {{{"finish_reason",
              terminal ? nlohmann::json("stop") : nlohmann::json(nullptr)}}}}};
      set_openai_system_fingerprint(
          chunk, fingerprint, /*stream=*/true, include_usage);
      EXPECT_EQ(chunk.contains("system_fingerprint"),
                terminal && !include_usage);
    }
    nlohmann::json usage = {{"choices", nlohmann::json::array()},
                            {"usage", {{"total_tokens", 3}}}};
    set_openai_system_fingerprint(
        usage, fingerprint, /*stream=*/true, include_usage);
    EXPECT_EQ(usage.contains("system_fingerprint"), include_usage);
  }
  nlohmann::json full = {{"choices", nlohmann::json::array()}};
  set_openai_system_fingerprint(
      full, fingerprint, /*stream=*/false, /*include_usage=*/false);
  EXPECT_EQ(full["system_fingerprint"], fingerprint);
}

TEST(ChatCompletionResponseTest, ChatFinishAndUsageChunksKeepRequiredFields) {
  proto::ChatResponse response;
  response.set_object("chat.completion.chunk");
  auto* choice = response.add_choices();
  choice->set_index(0);
  choice->set_finish_reason("stop");
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["delta"].is_object());
  EXPECT_TRUE(json["choices"][0]["delta"].empty());
  EXPECT_TRUE(json["choices"][0]["logprobs"].is_null());
  response.clear_choices();
  response.mutable_usage()->set_prompt_tokens(2);
  json = openai_response_json(response);
  EXPECT_EQ(json["choices"], nlohmann::json::array());
  EXPECT_EQ(json["usage"]["prompt_tokens"], 2);
}

TEST(ChatCompletionResponseTest, ReasoningAndLogprobBytesMatchVllm) {
  proto::ChatResponse response;
  auto* choice = response.add_choices();
  choice->mutable_message()->set_reasoning_content("why");
  auto* logprob = choice->mutable_logprobs()->add_content();
  logprob->set_token("你");
  logprob->set_logprob(-0.5);
  const auto json = openai_response_json(response);
  EXPECT_EQ(json["choices"][0]["message"]["reasoning"], "why");
  EXPECT_FALSE(json["choices"][0]["message"].contains("reasoning_content"));
  EXPECT_FALSE(
      json["choices"][0]["logprobs"]["content"][0].contains("token_id"));
  EXPECT_EQ(json["choices"][0]["logprobs"]["content"][0]["bytes"],
            nlohmann::json::array({228, 189, 160}));
  EXPECT_EQ(json["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
            nlohmann::json::array());
}

TEST(ChatCompletionResponseTest, FullChatPreservesNullsAndForcedToolContent) {
  proto::ChatResponse response;
  response.set_object("chat.completion");
  auto* message = response.add_choices()->mutable_message();
  message->set_role("assistant");
  message->set_content("");
  message->set_reasoning_content("thinking");
  response.mutable_usage()->mutable_prompt_tokens_details()->set_cached_tokens(
      0);
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["message"]["content"].is_null());
  EXPECT_FALSE(json["choices"][0]["message"].contains("tool_calls"));
  EXPECT_TRUE(json["choices"][0]["message"].contains("refusal"));
  EXPECT_TRUE(json.contains("system_fingerprint"));
  EXPECT_TRUE(json["usage"]["prompt_tokens_details"].is_null());
  EXPECT_TRUE(json["usage"].contains("completion_tokens_details"));
  message->add_tool_calls()->mutable_function()->set_name("weather");
  for (const bool named : {false, true}) {
    json = openai_response_json(response, named, !named);
    EXPECT_EQ(json["choices"][0]["message"]["content"], "");
  }
  response.set_object("chat.completion.chunk");
  response.mutable_choices(0)->mutable_delta()->set_reasoning_content("");
  json = openai_response_json(response);
  EXPECT_FALSE(json.contains("service_tier"));
  EXPECT_TRUE(json["choices"][0]["delta"].empty());
  EXPECT_FALSE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_FALSE(json["usage"].contains("completion_tokens_details"));
}

TEST(ChatCompletionResponseTest,
     ToolArgumentDeltasOnlyRepeatIndexAndArguments) {
  proto::ChatResponse response;
  response.set_object("chat.completion.chunk");
  auto* delta = response.add_choices()->mutable_delta();
  auto* tool = delta->add_tool_calls();
  tool->set_index(0);
  tool->set_id("call_test");
  tool->mutable_function()->set_name("weather");
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["delta"]["content"].is_null());
  EXPECT_EQ(json["choices"][0]["delta"]["tool_calls"][0]["type"], "function");
  tool->clear_id();
  tool->mutable_function()->clear_name();
  tool->mutable_function()->set_arguments("{}");
  json = openai_response_json(response);
  const auto& call = json["choices"][0]["delta"]["tool_calls"][0];
  EXPECT_EQ(
      call,
      (nlohmann::json{{"index", 0}, {"function", {{"arguments", "{}"}}}}));
}

class ChatCompletionTestService final : public AdmissionTestService {
 public:
  void ChatCompletionsHttp(google::protobuf::RpcController* controller,
                           const proto::HttpRequest*,
                           proto::HttpResponse*,
                           google::protobuf::Closure* done) override {
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    const std::string mode = ctrl->request_attachment().to_string();
    if (ctrl->http_request().uri().path() == "/v1/chat/completions") {
      chat_request(ctrl, done, mode);
      return;
    }
    if (mode == "openai") {
      proto::ChatRequest request;
      request.set_stream(true);
      proto::ChatResponse response;
      StreamCall<proto::ChatRequest, proto::ChatResponse> call(
          ctrl, done, &request, &response, /*use_arena=*/true);
      call.finish();
      return;
    }
    proto::ChatRequest request;
    request.set_stream(mode != "named_full");
    request.mutable_stream_options()->set_include_usage(true);
    request.mutable_stream_options()->set_continuous_usage_stats(mode ==
                                                                 "continuous");
    if (mode == "named_full" || mode == "named_stream") {
      request.set_tool_choice(
          R"({"type":"function","function":{"name":"weather"}})");
    } else if (mode == "required_stream") {
      request.set_tool_choice("required");
    }
    proto::ChatResponse response;
    StreamCall<proto::ChatRequest, proto::ChatResponse> call(
        ctrl,
        done,
        &request,
        &response,
        /*use_arena=*/true,
        /*is_http_request=*/true);
    if (mode == "invalid" || mode == "missing" || mode == "limited" ||
        mode == "exhausted") {
      call.finish_with_error(mode == "invalid"   ? StatusCode::INVALID_ARGUMENT
                             : mode == "missing" ? StatusCode::NOT_FOUND
                             : mode == "limited"
                                 ? StatusCode::RATE_LIMITED
                                 : StatusCode::RESOURCE_EXHAUSTED,
                             "failure",
                             mode == "missing" ? "model" : "");
      return;
    }
    if (mode == "named_full") {
      response.set_object("chat.completion");
      response.add_choices()->set_finish_reason("tool_calls");
      call.write_and_finish(response);
      return;
    }
    proto::Usage usage;
    usage.set_prompt_tokens(2);
    usage.set_completion_tokens(1);
    usage.set_total_tokens(3);
    call.set_stream_usage(usage);
    response.set_object("chat.completion.chunk");
    auto* choice = response.add_choices();
    choice->mutable_delta()->set_content("hi");
    if (mode == "named_stream" || mode == "required_stream") {
      choice->set_finish_reason("tool_calls");
    }
    call.write(response);
    if (mode == "stream_error") {
      call.finish_with_error(StatusCode::UNKNOWN, "generation failed");
      return;
    }
    response.clear_choices();
    response.mutable_usage()->set_total_tokens(3);
    call.write(response);
    call.finish();
    call.finish();
  }

 private:
  void chat_request(brpc::Controller* controller,
                    google::protobuf::Closure* done,
                    const std::string& body) {
    auto [status, normalized] =
        normalize_openai_request(body, OpenAIEndpoint::CHAT, "fixture");
    proto::ChatRequest request;
    proto::ChatResponse response;
    if (status.ok()) {
      status = decode_chat_request(std::move(normalized), &request);
    }
    StreamCall<proto::ChatRequest, proto::ChatResponse> call(
        controller,
        done,
        &request,
        &response,
        /*use_arena=*/true,
        /*is_http_request=*/true);
    if (!status.ok()) {
      call.finish_with_error(status.code(), status.message());
      return;
    }
    response.set_id("chatcmpl-fixture");
    response.set_created(1);
    response.set_model("fixture");
    response.set_object(request.stream() ? "chat.completion.chunk"
                                         : "chat.completion");
    auto* choice = response.add_choices();
    choice->set_index(0);
    if (request.stream()) {
      choice->mutable_delta()->set_role("assistant");
      choice->mutable_delta()->set_content("hi");
    } else {
      choice->mutable_message()->set_role("assistant");
      choice->mutable_message()->set_content("hi");
    }
    serve_admitted(
        call,
        request.messages(0).content(),
        [&] { call.write(response); },
        [&] {
          if (request.stream()) {
            choice->mutable_delta()->clear_content();
            choice->set_finish_reason("stop");
            call.write(response);
            call.finish();
          } else {
            choice->set_finish_reason("stop");
            response.mutable_usage()->set_prompt_tokens(1);
            response.mutable_usage()->set_completion_tokens(1);
            response.mutable_usage()->set_total_tokens(2);
            call.write_and_finish(response);
          }
        });
  }
};

class ChatCompletionProtocolTest : public HttpProtocolTestFixture {
 protected:
  void SetUp() override {
    start_service(service_,
                  "/test => ChatCompletionsHttp,"
                  "/v1/chat/completions => ChatCompletionsHttp");
  }
  void request(const std::string& mode, brpc::Controller& controller) {
    post("/test", mode, controller, /*content_type=*/nullptr);
  }
  std::string chat_body(bool stream, const std::string& content = "hi") const {
    return nlohmann::json(
               {{"model", "fixture"},
                {"stream", stream},
                {"messages", {{{"role", "user"}, {"content", content}}}}})
        .dump();
  }
  void chat(bool stream, brpc::Controller& controller) {
    post("/v1/chat/completions", chat_body(stream), controller);
  }
  int32_t open_socket(bool stream, const std::string& content) {
    return HttpProtocolTestFixture::open_socket("/v1/chat/completions",
                                                chat_body(stream, content));
  }
  void check_success(bool stream) {
    brpc::Controller controller;
    chat(stream, controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    EXPECT_EQ(controller.http_response().status_code(), 200);
    const std::string body = controller.response_attachment().to_string();
    if (stream) {
      EXPECT_EQ(controller.http_response().content_type(),
                "text/event-stream; charset=utf-8");
      EXPECT_NE(body.find("\"content\":\"hi\""), std::string::npos);
      EXPECT_TRUE(body.ends_with("data: [DONE]\n\n"));
      EXPECT_EQ(body.find("[DONE]"), body.rfind("[DONE]"));
    } else {
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      const auto json = nlohmann::json::parse(body);
      EXPECT_EQ(json["choices"][0]["message"]["content"], "hi");
    }
    ASSERT_TRUE(service_.wait_released());
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
  }
  ChatCompletionTestService service_;
};

TEST_F(ChatCompletionProtocolTest,
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
    chat(stream, controller);
    EXPECT_EQ(controller.http_response().status_code(), 429);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    EXPECT_EQ(controller.http_response().GetHeader("Retry-After"), nullptr);
    const std::string body = controller.response_attachment().to_string();
    const auto json = nlohmann::json::parse(body);
    const std::string message =
        "The number of concurrent requests has reached the limit.";
    EXPECT_EQ(json,
              nlohmann::json({{"error",
                               {{"message", message},
                                {"type", "RateLimitError"},
                                {"param", nullptr},
                                {"code", 429}}}}));
    EXPECT_EQ(body.find("data:"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  }
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  check_success(/*stream=*/false);
  check_success(/*stream=*/true);
}

TEST_F(ChatCompletionProtocolTest,
       SleepingAdmissionIsUnavailableNotRateLimited) {
  ASSERT_TRUE(service_.rate_limiter().try_set_sleeping());
  for (const bool stream : {false, true}) {
    brpc::Controller controller;
    chat(stream, controller);
    EXPECT_EQ(controller.http_response().status_code(), 503);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto error = nlohmann::json::parse(
        controller.response_attachment().to_string())["error"];
    EXPECT_EQ(error["type"], "ServiceUnavailableError");
    EXPECT_EQ(error["message"], "Model is currently in sleep state.");
    EXPECT_EQ(error["code"], 503);
    EXPECT_TRUE(service_.rate_limiter().is_sleeping());
  }
  EXPECT_TRUE(service_.rate_limiter().try_wakeup());
  check_success(/*stream=*/true);
}

TEST_F(ChatCompletionProtocolTest,
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

TEST_F(ChatCompletionProtocolTest,
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

TEST_F(ChatCompletionProtocolTest, DISABLED_OfficialSdkCompatibility) {
  const std::string url = base_url() + "/v1";
  run_sdk(XLLM_CHAT_COMPLETION_SDK_CLIENT, url, "success");
  ASSERT_TRUE(service_.wait_released());
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  run_sdk(XLLM_CHAT_COMPLETION_SDK_CLIENT, url, "limited");
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  run_sdk(XLLM_CHAT_COMPLETION_SDK_CLIENT, url, "success");
  ASSERT_TRUE(service_.wait_released());
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
}

TEST_F(ChatCompletionProtocolTest, PreStreamErrorsUseJsonAndCorrectHttpStatus) {
  for (const auto& [mode, code] : {std::pair{"invalid", 400},
                                   {"missing", 404},
                                   {"limited", 429},
                                   {"exhausted", 500}}) {
    brpc::Controller controller;
    request(mode, controller);
    EXPECT_EQ(controller.http_response().status_code(), code);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto json =
        nlohmann::json::parse(controller.response_attachment().to_string());
    EXPECT_EQ(json["error"]["code"], code);
    if (std::string(mode) == "exhausted") {
      EXPECT_EQ(json["error"]["type"], "InternalServerError");
    }
    EXPECT_EQ(json["error"]["param"],
              std::string(mode) == "missing" ? nlohmann::json("model")
                                             : nlohmann::json(nullptr));
  }
}

TEST_F(ChatCompletionProtocolTest, NamedToolChoiceIsAppliedToHttpAndSse) {
  for (const std::string mode :
       {"named_full", "named_stream", "required_stream"}) {
    brpc::Controller controller;
    request(mode, controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    const std::string body = controller.response_attachment().to_string();
    const auto json = nlohmann::json::parse(
        mode == "named_full" ? body : body.substr(6, body.find("\n\n") - 6));
    EXPECT_EQ(json["choices"][0]["finish_reason"],
              mode == "required_stream" ? "tool_calls" : "stop");
  }
}

TEST_F(ChatCompletionProtocolTest,
       StreamsEndWithExactlyOneDoneAndUsageChoicesArray) {
  brpc::Controller controller;
  request("stream", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  const auto first =
      nlohmann::json::parse(body.substr(6, body.find("\n\n") - 6));
  EXPECT_FALSE(first.contains("usage"));
  EXPECT_NE(body.find("\"choices\":[]"), std::string::npos);
  const size_t done = body.find("data: [DONE]\n\n");
  ASSERT_NE(done, std::string::npos);
  EXPECT_EQ(body.find("[DONE]", done + 12), std::string::npos);
}

TEST_F(ChatCompletionProtocolTest,
       ContinuousUsageIncludesCountsOnContentChunks) {
  brpc::Controller controller;
  request("continuous", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  const auto first =
      nlohmann::json::parse(body.substr(6, body.find("\n\n") - 6));
  EXPECT_EQ(first["usage"]["prompt_tokens"], 2);
  EXPECT_EQ(first["usage"]["completion_tokens"], 1);
  EXPECT_EQ(first["choices"].size(), 1);
}

TEST_F(ChatCompletionProtocolTest, GenerationErrorsAreSseJsonAndCloseStream) {
  brpc::Controller controller;
  request("stream_error", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("data: {\"error\":"), std::string::npos);
  EXPECT_NE(body.find("generation failed"), std::string::npos);
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n"));
}

TEST_F(ChatCompletionProtocolTest, OpenAiStreamStillEmitsDoneSentinel) {
  brpc::Controller controller;
  request("openai", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  EXPECT_EQ(controller.response_attachment().to_string(), "data: [DONE]\n\n");
}

}  // namespace
}  // namespace xllm::api_service
