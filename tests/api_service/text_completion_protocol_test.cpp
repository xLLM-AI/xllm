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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "api_service/openai_batch.h"
#include "api_service/openai_json.h"
#include "api_service/openai_request.h"

namespace xllm::api_service {
namespace {

TEST(TextCompletionRequestTest, BatchedPromptsKeepTheirTypesAndOrder) {
  for (const auto& input : {nlohmann::json::array({"a", "b"}),
                            nlohmann::json::array({{1, 2}, {3}})}) {
    auto [status, body] =
        normalize_openai_request(nlohmann::json({{"prompt", input}}).dump(),
                                 OpenAIEndpoint::COMPLETION,
                                 "model");
    ASSERT_TRUE(status.ok()) << status.message();
    const auto json = nlohmann::json::parse(body);
    ASSERT_EQ(json["prompts"].size(), 2);
    EXPECT_EQ(json["prompts"][0][input[0].is_string() ? "text" : "token_ids"],
              input[0]);
    EXPECT_EQ(json["prompts"][1][input[1].is_string() ? "text" : "token_ids"],
              input[1]);
  }
}

TEST(TextCompletionRequestTest, GreedySamplingRequiresOneChoice) {
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
            request.dump(), OpenAIEndpoint::COMPLETION, "model", &param);
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

TEST(TextCompletionRequestTest, AcceptsTemperaturesAboveTwo) {
  const auto [status, body] = normalize_openai_request(
      R"({"prompt":"hi","messages":[{"role":"user","content":"hi"}],"temperature":3})",
      OpenAIEndpoint::COMPLETION,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 3);
}

TEST(TextCompletionRequestTest, SchemaAndSamplingErrorsHaveDistinctTypes) {
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
}

TEST(TextCompletionRequestTest, CompletionDefaultsAndExtendedStops) {
  auto [status, body] = normalize_openai_request(
      R"({"prompt":"hi","max_tokens":null,"stop":["1","2","3","4","5"],"frequency_penalty":-1})",
      OpenAIEndpoint::COMPLETION,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["max_tokens"], 16);
}

TEST(TextCompletionRequestTest, UnsupportedSamplingControlsFailExplicitly) {
  for (const char* field :
       {"prompt_logprobs", "structured_outputs", "logprob_token_ids"}) {
    auto request = nlohmann::json({{"prompt", "hi"}, {field, 1}});
    const auto [status, body] = normalize_openai_request(
        request.dump(), OpenAIEndpoint::COMPLETION, "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << field;
    EXPECT_NE(status.message().find(field), std::string::npos);
  }
}

TEST(TextCompletionRequestTest, InternalPromptsAndOversizedBatchesAreRejected) {
  for (const auto& request :
       {nlohmann::json{{"prompt", "hi"}, {"prompts", {{{"text", "hidden"}}}}},
        nlohmann::json{{"prompt", std::vector<std::string>(1025, "hi")}}}) {
    EXPECT_EQ(normalize_openai_request(
                  request.dump(), OpenAIEndpoint::COMPLETION, "model")
                  .first.code(),
              StatusCode::INVALID_ARGUMENT);
  }
}

TEST(TextCompletionRequestTest,
     CompletionTokenInputPreservesExistingExtension) {
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

TEST(TextCompletionResponseTest, StopReasonsKeepTheirJsonTypes) {
  for (const StopReason& reason : {StopReason{},
                                   StopReason{int32_t{123}},
                                   StopReason{std::string("END")}}) {
    proto::CompletionResponse completion;
    set_proto_stop_reason(reason,
                          completion.add_choices()->mutable_stop_reason());
    const nlohmann::json expected =
        std::holds_alternative<int32_t>(reason)       ? nlohmann::json(123)
        : std::holds_alternative<std::string>(reason) ? nlohmann::json("END")
                                                      : nlohmann::json(nullptr);
    for (const bool stream : {false, true}) {
      completion.mutable_choices(0)->set_finish_reason("stop");
      EXPECT_EQ(
          openai_response_json(completion, stream)["choices"][0]["stop_reason"],
          expected);
    }
  }
}

TEST(TextCompletionResponseTest, CompletionSeparatesFullAndStreamMetadata) {
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

// OpenAIBatch is shared by Text Completion and Embeddings, not Chat batching.
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

}  // namespace
}  // namespace xllm::api_service
