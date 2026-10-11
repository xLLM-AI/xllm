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

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "api_service/openai_request.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm::api_service {
namespace {

nlohmann::json generation_request(OpenAIEndpoint endpoint) {
  if (endpoint == OpenAIEndpoint::CHAT) {
    return {{"messages", {{{"role", "user"}, {"content", "hi"}}}}};
  }
  return {{"prompt", "hi"}};
}

TEST(SamplingProtocolTest, NewControlsSurviveJsonAndProtobufDecoding) {
  for (const auto endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
    auto request = generation_request(endpoint);
    request.update({{"seed", 123},
                    {"min_p", 0.2},
                    {"min_tokens", 2},
                    {"max_tokens", 4},
                    {"logit_bias", {{"12", 250}, {"13", -200}}},
                    {"allowed_token_ids", {12, 13}},
                    {"bad_words", {"word"}}});
    const auto [status, body] =
        normalize_openai_request(request.dump(), endpoint, "model");
    ASSERT_TRUE(status.ok()) << status.message();
    const auto json = nlohmann::json::parse(body);
    EXPECT_EQ(json["logit_bias"]["12"], 100);
    EXPECT_EQ(json["logit_bias"]["13"], -100);
    if (endpoint == OpenAIEndpoint::CHAT) {
      proto::ChatRequest decoded;
      const auto parsed =
          google::protobuf::util::JsonStringToMessage(body, &decoded);
      ASSERT_TRUE(parsed.ok()) << parsed.ToString();
      EXPECT_EQ(decoded.seed(), 123);
      EXPECT_FLOAT_EQ(decoded.min_p(), 0.2F);
      EXPECT_EQ(decoded.min_tokens(), 2);
      EXPECT_EQ(decoded.logit_bias().at(12), 100);
      EXPECT_EQ(decoded.allowed_token_ids_size(), 2);
      EXPECT_EQ(decoded.bad_words(0), "word");
    } else {
      proto::CompletionRequest decoded;
      const auto parsed =
          google::protobuf::util::JsonStringToMessage(body, &decoded);
      ASSERT_TRUE(parsed.ok()) << parsed.ToString();
      EXPECT_EQ(decoded.seed(), 123);
      EXPECT_FLOAT_EQ(decoded.min_p(), 0.2F);
      EXPECT_EQ(decoded.min_tokens(), 2);
      EXPECT_EQ(decoded.logit_bias().at(12), 100);
      EXPECT_EQ(decoded.allowed_token_ids_size(), 2);
      EXPECT_EQ(decoded.bad_words(0), "word");
    }
  }
}

TEST(SamplingProtocolTest, RejectsInvalidControlsBeforeGeneration) {
  for (const auto endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
    for (const auto& patch : std::vector<nlohmann::json>{
             {{"temperature", -0.01}},
             {{"temperature", 1e100}},
             {{"top_p", 0}},
             {{"min_p", -0.1}},
             {{"min_p", 1.1}},
             {{"top_k", -2}},
             {{"seed", 1.5}},
             {{"seed", 18446744073709551615ULL}},
             {{"min_tokens", -1}},
             {{"min_tokens", 3}, {"max_tokens", 2}},
             {{"allowed_token_ids", nlohmann::json::array()}},
             {{"allowed_token_ids", {-1}}},
             {{"allowed_token_ids", {1.5}}},
             {{"logit_bias", {{"bad", 1}}}},
             {{"logit_bias", {{"1", "bad"}}}},
             {{"bad_words", {""}}},
             {{"bad_words", "word"}}}) {
      auto request = generation_request(endpoint);
      request.update(patch);
      const auto [status, body] =
          normalize_openai_request(request.dump(), endpoint, "model");
      EXPECT_FALSE(status.ok()) << request;
    }
  }
}

TEST(SamplingProtocolTest, NullControlsUseDefaultsAndTinyTemperatureIsClamped) {
  auto request = generation_request(OpenAIEndpoint::CHAT);
  request.update({{"seed", nullptr},
                  {"min_p", nullptr},
                  {"min_tokens", nullptr},
                  {"allowed_token_ids", nullptr},
                  {"logit_bias", nullptr},
                  {"bad_words", nullptr},
                  {"temperature", 0.00001}});
  const auto [status, body] =
      normalize_openai_request(request.dump(), OpenAIEndpoint::CHAT, "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 0.01);
}

}  // namespace
}  // namespace xllm::api_service
