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
#include <butil/fd_guard.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "api_service/anthropic_request_utils.h"
#include "api_service/chat_json_parser.h"
#include "api_service/chat_request_decoder.h"
#include "api_service/non_stream_call.h"
#include "api_service/openai_batch.h"
#include "api_service/openai_json.h"
#include "api_service/openai_request.h"
#include "api_service/stream_call.h"
#include "core/common/rate_limiter.h"
#include "core/framework/config/service_config.h"
#include "core/framework/request/request.h"
#include "util/scope_guard.h"
#include "xllm_service.pb.h"

extern char** environ;

namespace xllm::api_service {
namespace {

TEST(OpenAIRequestTest, ChatAliasesAndNullableDefaults) {
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

TEST(OpenAIRequestTest, BatchedInputsKeepTheirTypesAndOrder) {
  for (const auto& [endpoint, field, target] :
       {std::tuple{OpenAIEndpoint::COMPLETION, "prompt", "prompts"},
        std::tuple{OpenAIEndpoint::EMBEDDING, "input", "inputs"}}) {
    for (const auto& input : {nlohmann::json::array({"a", "b"}),
                              nlohmann::json::array({{1, 2}, {3}})}) {
      auto [status, body] = normalize_openai_request(
          nlohmann::json({{field, input}}).dump(), endpoint, "model");
      ASSERT_TRUE(status.ok()) << status.message();
      const auto json = nlohmann::json::parse(body);
      ASSERT_EQ(json[target].size(), 2);
      EXPECT_EQ(json[target][0][input[0].is_string() ? "text" : "token_ids"],
                input[0]);
      EXPECT_EQ(json[target][1][input[1].is_string() ? "text" : "token_ids"],
                input[1]);
    }
  }
}

TEST(OpenAIRequestTest, InvalidParametersAreClientErrors) {
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

TEST(OpenAIRequestTest, GreedySamplingRequiresOneChoice) {
  for (const OpenAIEndpoint endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
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
              request.dump(), endpoint, "model", &param);
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
}

TEST(OpenAIRequestTest, Vllm023AcceptsTemperaturesAboveTwo) {
  for (const OpenAIEndpoint endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
    const auto [status, body] = normalize_openai_request(
        R"({"prompt":"hi","messages":[{"role":"user","content":"hi"}],"temperature":3})",
        endpoint,
        "model");
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 3);
  }
}

TEST(OpenAIRequestTest, SchemaAndSamplingErrorsHaveDistinctTypes) {
  for (const auto& [body, expected_type] :
       {std::pair{"{", "Bad Request"},
        {R"({"prompt":[1,"hi"]})", "Bad Request"},
        {R"({"prompt":[]})", "BadRequestError"},
        {R"({"prompt":"hi","stream_options":{"include_usage":true}})",
         "Bad Request"},
        {R"({"prompt":"hi","temperature":-1})", "BadRequestError"},
        {R"({"prompt":"hi","temperature":0,"n":2})", "BadRequestError"}}) {
    bool schema_error = false;
    std::string param;
    const auto [status, normalized] = normalize_openai_request(
        body, OpenAIEndpoint::COMPLETION, "model", &param, &schema_error);
    ASSERT_FALSE(status.ok());
    const auto error = nlohmann::json::parse(openai_error_json(
        status.code(), status.message(), param, schema_error));
    EXPECT_EQ(error["error"]["type"], expected_type);
  }
  bool schema_error = true;
  const auto [status, body] = normalize_openai_request(R"({"messages":[]})",
                                                       OpenAIEndpoint::CHAT,
                                                       "model",
                                                       nullptr,
                                                       &schema_error);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(schema_error);
}

TEST(OpenAIRequestTest, ValidationErrorsIdentifyTheirParameters) {
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

TEST(OpenAIResponseTest, NamedToolChoicePreservesStopAndLength) {
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

TEST(OpenAIRequestTest, CompletionDefaultsAndExtendedStops) {
  auto [status, body] = normalize_openai_request(
      R"({"prompt":"hi","max_tokens":null,"stop":["1","2","3","4","5"],"frequency_penalty":-1})",
      OpenAIEndpoint::COMPLETION,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["max_tokens"], 16);
}

TEST(OpenAIRequestTest, MalformedPromptBatchesAreRejected) {
  for (const auto& input : {R"([])",
                            R"([1,"x"])",
                            R"([[-1]])",
                            R"([[2147483648]])",
                            R"(["a",[1]])"}) {
    auto [status, body] =
        normalize_openai_request(std::string("{\"input\":") + input + "}",
                                 OpenAIEndpoint::EMBEDDING,
                                 "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << input;
  }
}

TEST(OpenAIRequestTest, UnsupportedSamplingControlsFailExplicitly) {
  for (const char* field :
       {"seed", "min_p", "min_tokens", "logit_bias", "structured_outputs"}) {
    auto request = nlohmann::json({{"prompt", "hi"}, {field, 1}});
    const auto [status, body] = normalize_openai_request(
        request.dump(), OpenAIEndpoint::COMPLETION, "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << field;
    EXPECT_NE(status.message().find(field), std::string::npos);
  }
}

TEST(OpenAIRequestTest, InternalInputsAndOversizedBatchesAreRejected) {
  for (const auto& request :
       {nlohmann::json{{"prompt", "hi"}, {"prompts", {{{"text", "hidden"}}}}},
        nlohmann::json{{"prompt", std::vector<std::string>(1025, "hi")}}}) {
    EXPECT_EQ(normalize_openai_request(
                  request.dump(), OpenAIEndpoint::COMPLETION, "model")
                  .first.code(),
              StatusCode::INVALID_ARGUMENT);
  }
  EXPECT_EQ(
      normalize_openai_request(
          R"({"input":"","token_ids":[1]})", OpenAIEndpoint::EMBEDDING, "model")
          .first.code(),
      StatusCode::INVALID_ARGUMENT);
}

TEST(OpenAIRequestTest, CompletionTokenInputPreservesExistingExtension) {
  auto [status, body] = normalize_openai_request(
      R"({"prompt":[1,2,3]})", OpenAIEndpoint::COMPLETION, "model");
  ASSERT_TRUE(status.ok()) << status.message();
  const auto json = nlohmann::json::parse(body);
  EXPECT_EQ(json["token_ids"], nlohmann::json::array({1, 2, 3}));
  EXPECT_FALSE(json.contains("prompts"));
  std::tie(status, body) =
      normalize_openai_request(R"({"prompt":"","token_ids":[1,2,3]})",
                               OpenAIEndpoint::COMPLETION,
                               "model");
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(OpenAIRequestTest,
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

TEST(OpenAIRequestTest, CallPayloadParsingHandlesInvalidLengthHeaders) {
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

TEST(OpenAIJsonTest, StopReasonsKeepTheirJsonTypes) {
  for (const StopReason& reason : {StopReason{},
                                   StopReason{int32_t{123}},
                                   StopReason{std::string("END")}}) {
    proto::ChatResponse chat;
    proto::CompletionResponse completion;
    set_proto_stop_reason(reason, chat.add_choices()->mutable_stop_reason());
    set_proto_stop_reason(reason,
                          completion.add_choices()->mutable_stop_reason());
    const nlohmann::json expected =
        std::holds_alternative<int32_t>(reason)       ? nlohmann::json(123)
        : std::holds_alternative<std::string>(reason) ? nlohmann::json("END")
                                                      : nlohmann::json(nullptr);
    for (const bool stream : {false, true}) {
      chat.set_object(stream ? "chat.completion.chunk" : "chat.completion");
      chat.mutable_choices(0)->set_finish_reason("stop");
      completion.mutable_choices(0)->set_finish_reason("stop");
      EXPECT_EQ(openai_response_json(chat)["choices"][0]["stop_reason"],
                expected);
      EXPECT_EQ(
          openai_response_json(completion, stream)["choices"][0]["stop_reason"],
          expected);
    }
  }
}

TEST(OpenAIJsonTest, FingerprintIsStableAndAppearsOnlyOnFinalStreamMessages) {
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

TEST(OpenAIJsonTest, ChatFinishAndUsageChunksKeepRequiredFields) {
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

TEST(OpenAIJsonTest, ReasoningAndLogprobBytesMatchVllm) {
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

TEST(OpenAIJsonTest, FullChatPreservesNullsAndForcedToolContent) {
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

TEST(OpenAIJsonTest, CompletionSeparatesFullAndStreamMetadata) {
  proto::CompletionResponse response;
  response.set_object("text_completion");
  response.add_choices()->mutable_logprobs()->add_token_ids(42);
  response.mutable_usage()->set_prompt_tokens(2);
  for (const bool stream : {false, true}) {
    const auto json = openai_response_json(response, stream);
    EXPECT_EQ(json.contains("system_fingerprint"), !stream);
    EXPECT_EQ(json["choices"][0].contains("prompt_logprobs"), !stream);
    EXPECT_EQ(json["usage"].contains("prompt_tokens_details"), !stream);
    EXPECT_FALSE(json["choices"][0]["logprobs"].contains("token_ids"));
  }
}

TEST(OpenAIJsonTest, ToolArgumentDeltasOnlyRepeatIndexAndArguments) {
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

TEST(OpenAIJsonTest, EmbeddingsUseFloat32LittleEndianBase64) {
  proto::EmbeddingResponse response;
  response.add_data()->add_embedding(1.0f);
  response.mutable_data(0)->add_embedding(-2.0f);
  const auto json = openai_embedding_json(response, "base64");
  EXPECT_EQ(json["data"][0]["embedding"], "AACAPwAAAMA=");
  EXPECT_FALSE(json["usage"].contains("completion_tokens"));
  EXPECT_EQ(openai_embedding_json(response, "float")["data"][0]["embedding"],
            nlohmann::json::array({1.0, -2.0}));
}

TEST(OpenAIBatchTest, ReordersChoicesAndSumsFinalUsage) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/2, /*streaming=*/false);
  std::vector<RequestOutput> sent;
  OutputCallback send = [&sent](RequestOutput output) {
    sent.emplace_back(std::move(output));
    return true;
  };
  for (size_t index : {1, 0}) {
    RequestOutput output;
    output.finished = true;
    output.usage = Usage{3, 2, 5, 1};
    output.outputs.resize(2);
    output.outputs[0].index = 0;
    output.outputs[1].index = 1;
    batch.accept(index, std::move(output), send);
  }
  ASSERT_EQ(sent.size(), 1);
  ASSERT_EQ(sent[0].outputs.size(), 4);
  EXPECT_EQ(sent[0].outputs[0].index, 0);
  EXPECT_EQ(sent[0].outputs[3].index, 3);
  EXPECT_EQ(sent[0].usage->num_prompt_tokens, 6);
  EXPECT_EQ(sent[0].usage->num_total_tokens, 10);
  EXPECT_TRUE(sent[0].finished);
}

TEST(OpenAIBatchTest, StreamingWaitsForEveryPromptAndUsesLatestUsage) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/1, /*streaming=*/true);
  std::vector<RequestOutput> sent;
  OutputCallback send = [&sent](RequestOutput output) {
    sent.emplace_back(std::move(output));
    return true;
  };
  RequestOutput first;
  first.usage = Usage{3, 1, 4, 0};
  batch.accept(0, std::move(first), send);
  RequestOutput second;
  second.finished = true;
  second.usage = Usage{4, 2, 6, 0};
  batch.accept(1, std::move(second), send);
  RequestOutput last;
  last.finished = true;
  last.usage = Usage{3, 3, 6, 0};
  batch.accept(0, std::move(last), send);
  ASSERT_EQ(sent.size(), 3);
  EXPECT_FALSE(sent[0].finished);
  EXPECT_FALSE(sent[1].finished);
  EXPECT_TRUE(sent[2].finished);
  EXPECT_EQ(sent[2].usage->num_total_tokens, 12);
}

TEST(OpenAIBatchTest, ErrorClosesAllRemainingCallbacks) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/1, /*streaming=*/true);
  int32_t sent = 0;
  OutputCallback send = [&sent](RequestOutput) {
    ++sent;
    return true;
  };
  EXPECT_FALSE(batch.accept(
      0, RequestOutput(Status(StatusCode::INVALID_ARGUMENT, "bad")), send));
  EXPECT_FALSE(batch.accept(1, RequestOutput(), send));
  EXPECT_EQ(sent, 1);
}

class StreamTestService final : public proto::XllmAPIService {
 public:
  bool wait_held() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
      return held_request_ != nullptr;
    });
  }

  void release(bool cancel = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancel && held_request_ != nullptr) {
      held_request_->set_cancel();
    }
    released_ = true;
    changed_.notify_all();
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    if (held_request_ != nullptr) {
      held_request_->set_cancel();
    }
    changed_.notify_all();
  }

  bool wait_released() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
      return held_request_ == nullptr &&
             rate_limiter_.get_num_concurrent_requests() == 0;
    });
  }

  bool observe_disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (held_request_ == nullptr) {
      return false;
    }
    held_request_->update_connection_status();
    return held_request_->cancelled();
  }

  RateLimiter& rate_limiter() { return rate_limiter_; }

  void AnthropicMessagesHttp(google::protobuf::RpcController* controller,
                             const proto::HttpRequest*,
                             proto::HttpResponse*,
                             google::protobuf::Closure* done) override {
    anthropic_request(static_cast<brpc::Controller*>(controller),
                      done,
                      /*count_tokens=*/false);
  }

  void AnthropicCountTokensHttp(google::protobuf::RpcController* controller,
                                const proto::HttpRequest*,
                                proto::HttpResponse*,
                                google::protobuf::Closure* done) override {
    anthropic_request(static_cast<brpc::Controller*>(controller),
                      done,
                      /*count_tokens=*/true);
  }

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
      status = rate_limiter_.acquire();
      if (!status.ok()) {
        call.finish_with_error(status.code(), status.message());
        return;
      }
      // Count-token work owns the slot directly, like count_chat_tokens.
      ScopeGuard guard([this] { rate_limiter_.decrease_one_request(); });
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
    // Both protocols use the same real Request ownership and release boundary.
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

  RateLimiter rate_limiter_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::shared_ptr<xllm::Request> held_request_;
  bool released_ = false;
  bool stopping_ = false;
};

class OpenAICallTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_limit_ = ServiceConfig::get_instance().max_concurrent_requests();
    ServiceConfig::get_instance().max_concurrent_requests(1);
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/test => ChatCompletionsHttp,"
                                 "/v1/chat/completions => ChatCompletionsHttp,"
                                 "/v1/messages => AnthropicMessagesHttp,"
                                 "/v1/messages/count_tokens => "
                                 "AnthropicCountTokensHttp"),
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
    ServiceConfig::get_instance().max_concurrent_requests(previous_limit_);
  }
  void request(const std::string& mode, brpc::Controller& controller) {
    controller.http_request().uri() = "/test";
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.request_attachment().append(mode);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  }
  const char* endpoint() const {
    return anthropic_ ? "/v1/messages" : "/v1/chat/completions";
  }

  const char* sdk_python_env() const {
    return anthropic_ ? "XLLM_ANTHROPIC_SDK_PYTHON" : "XLLM_OPENAI_SDK_PYTHON";
  }

  void chat(bool stream, brpc::Controller& controller) {
    controller.http_request().uri() = endpoint();
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().set_content_type("application/json");
    controller.request_attachment().append(chat_body(stream));
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  }

  std::string chat_body(bool stream, const std::string& content = "hi") const {
    nlohmann::json body = {
        {"model", "fixture"},
        {"stream", stream},
        {"messages", {{{"role", "user"}, {"content", content}}}}};
    if (anthropic_) {
      body["max_tokens"] = 8;
    }
    return body.dump();
  }

  int32_t open_socket(bool stream, const std::string& content) {
    const int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return fd;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(server_.listen_address().port);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
        0) {
      close(fd);
      return -1;
    }
    const std::string body = chat_body(stream, content);
    const std::string wire = std::string("POST ") + endpoint() +
                             " HTTP/1.1\r\n"
                             "Host: localhost\r\nContent-Type: "
                             "application/json\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\n\r\n" + body;
    if (send(fd, wire.data(), wire.size(), MSG_NOSIGNAL) !=
        static_cast<ssize_t>(wire.size())) {
      close(fd);
      return -1;
    }
    return fd;
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
      if (anthropic_) {
        EXPECT_NE(body.find("\"text\":\"hi\""), std::string::npos);
        EXPECT_NE(body.find("event: message_start\n"), std::string::npos);
        const size_t stop = body.find("event: message_stop\n");
        ASSERT_NE(stop, std::string::npos);
        EXPECT_EQ(body.find("event: message_stop\n", stop + 1),
                  std::string::npos);
        EXPECT_TRUE(body.ends_with("data: {\"type\":\"message_stop\"}\n\n"));
        EXPECT_EQ(body.find("[DONE]"), std::string::npos);
      } else {
        EXPECT_NE(body.find("\"content\":\"hi\""), std::string::npos);
        EXPECT_TRUE(body.ends_with("data: [DONE]\n\n"));
        EXPECT_EQ(body.find("[DONE]"), body.rfind("[DONE]"));
      }
    } else {
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      const auto json = nlohmann::json::parse(body);
      if (anthropic_) {
        EXPECT_EQ(json["type"], "message");
        EXPECT_EQ(json["role"], "assistant");
        EXPECT_EQ(json["model"], "fixture");
        EXPECT_EQ(json["content"][0]["text"], "hi");
        EXPECT_EQ(json["stop_reason"], "end_turn");
      } else {
        EXPECT_EQ(json["choices"][0]["message"]["content"], "hi");
      }
    }
    ASSERT_TRUE(service_.wait_released());
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
  }

  void run_sdk(const std::string& phase) {
    const char* python = std::getenv(sdk_python_env());
    ASSERT_NE(python, nullptr);
    std::string interpreter(python);
    std::string script =
        anthropic_ ? XLLM_ANTHROPIC_SDK_CLIENT : XLLM_OPENAI_SDK_CLIENT;
    std::string url =
        "http://127.0.0.1:" + std::to_string(server_.listen_address().port);
    if (!anthropic_) {
      url += "/v1";
    }
    std::string selected_phase = phase;
    char* args[] = {interpreter.data(),
                    script.data(),
                    url.data(),
                    selected_phase.data(),
                    nullptr};
    pid_t pid = -1;
    ASSERT_EQ(posix_spawnp(
                  &pid, interpreter.c_str(), nullptr, nullptr, args, environ),
              0);
    int32_t status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
  }

  StreamTestService service_;
  brpc::Server server_;
  brpc::Channel channel_;
  int32_t previous_limit_ = 0;
  bool anthropic_ = false;
};

class AdmissionCallTest : public OpenAICallTest,
                          public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    anthropic_ = GetParam();
    OpenAICallTest::SetUp();
  }
};

INSTANTIATE_TEST_SUITE_P(Protocols,
                         AdmissionCallTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "Anthropic" : "OpenAI";
                         });

TEST_F(OpenAICallTest, PreStreamErrorsUseJsonAndCorrectHttpStatus) {
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

TEST_P(AdmissionCallTest, RealAdmissionRejectsBothModesBeforeSseAndRecovers) {
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
    if (anthropic_) {
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
    } else {
      EXPECT_EQ(json,
                nlohmann::json({{"error",
                                 {{"message", message},
                                  {"type", "RateLimitError"},
                                  {"param", nullptr},
                                  {"code", 429}}}}));
    }
    EXPECT_EQ(body.find("data:"), std::string::npos);
    EXPECT_EQ(body.find("[DONE]"), std::string::npos);
    EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  }
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  check_success(/*stream=*/false);
  check_success(/*stream=*/true);
}

TEST_P(AdmissionCallTest, SleepingAdmissionIsUnavailableNotRateLimited) {
  ASSERT_TRUE(service_.rate_limiter().try_set_sleeping());
  for (const bool stream : {false, true}) {
    brpc::Controller controller;
    chat(stream, controller);
    EXPECT_EQ(controller.http_response().status_code(), 503);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto error = nlohmann::json::parse(
        controller.response_attachment().to_string())["error"];
    EXPECT_EQ(error["type"],
              anthropic_ ? "overloaded_error" : "ServiceUnavailableError");
    EXPECT_EQ(error["message"], "Model is currently in sleep state.");
    if (!anthropic_) {
      EXPECT_EQ(error["code"], 503);
    }
    EXPECT_TRUE(service_.rate_limiter().is_sleeping());
  }
  EXPECT_TRUE(service_.rate_limiter().try_wakeup());
  check_success(/*stream=*/true);
}

TEST_P(AdmissionCallTest, ExplicitCancellationReleasesHeldRequestOnce) {
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

TEST_P(AdmissionCallTest, SocketDisconnectBeforeAndAfterSseReleasesRequest) {
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

TEST_P(AdmissionCallTest, SdkRecognizesRateLimitsAndSuccessfulResponses) {
  if (std::getenv(sdk_python_env()) == nullptr) {
    GTEST_SKIP() << "Set " << sdk_python_env()
                 << " to an interpreter with the official SDK installed";
  }
  run_sdk("success");
  ASSERT_TRUE(service_.wait_released());
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  run_sdk("limited");
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 1);
  service_.release();
  ASSERT_TRUE(service_.wait_released());
  run_sdk("success");
  ASSERT_TRUE(service_.wait_released());
  EXPECT_EQ(service_.rate_limiter().get_num_concurrent_requests(), 0);
}

TEST_F(OpenAICallTest, AnthropicCountTokensRejectsAndRecoversWithSharedLimit) {
  anthropic_ = true;
  butil::fd_guard held(open_socket(/*stream=*/true, "hold"));
  ASSERT_GE(static_cast<int32_t>(held), 0);
  ASSERT_TRUE(service_.wait_held());
  auto count_tokens = [this](brpc::Controller& controller) {
    controller.http_request().uri() = "/v1/messages/count_tokens";
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().set_content_type("application/json");
    controller.request_attachment().append(chat_body(/*stream=*/false));
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
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

TEST_F(OpenAICallTest, NamedToolChoiceIsAppliedToHttpAndSse) {
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

TEST_F(OpenAICallTest, StreamsEndWithExactlyOneDoneAndUsageChoicesArray) {
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

TEST_F(OpenAICallTest, ContinuousUsageIncludesCountsOnContentChunks) {
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

TEST_F(OpenAICallTest, GenerationErrorsAreSseJsonAndCloseStream) {
  brpc::Controller controller;
  request("stream_error", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("data: {\"error\":"), std::string::npos);
  EXPECT_NE(body.find("generation failed"), std::string::npos);
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n"));
}

}  // namespace
}  // namespace xllm::api_service
