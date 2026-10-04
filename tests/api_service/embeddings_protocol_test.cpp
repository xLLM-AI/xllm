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

#include <nlohmann/json.hpp>
#include <string>

#include "api_service/openai_json.h"
#include "api_service/openai_request.h"

namespace xllm::api_service {
namespace {

TEST(EmbeddingsRequestTest, BatchedInputsKeepTheirTypesAndOrder) {
  for (const auto& input : {nlohmann::json::array({"a", "b"}),
                            nlohmann::json::array({{1, 2}, {3}})}) {
    auto [status, body] =
        normalize_openai_request(nlohmann::json({{"input", input}}).dump(),
                                 OpenAIEndpoint::EMBEDDING,
                                 "model");
    ASSERT_TRUE(status.ok()) << status.message();
    const auto json = nlohmann::json::parse(body);
    ASSERT_EQ(json["inputs"].size(), 2);
    EXPECT_EQ(json["inputs"][0][input[0].is_string() ? "text" : "token_ids"],
              input[0]);
    EXPECT_EQ(json["inputs"][1][input[1].is_string() ? "text" : "token_ids"],
              input[1]);
  }
}

TEST(EmbeddingsRequestTest, MalformedInputBatchesAreRejected) {
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

TEST(EmbeddingsRequestTest, InternalTokenInputIsRejected) {
  EXPECT_EQ(
      normalize_openai_request(
          R"({"input":"","token_ids":[1]})", OpenAIEndpoint::EMBEDDING, "model")
          .first.code(),
      StatusCode::INVALID_ARGUMENT);
}

TEST(EmbeddingsResponseTest, EmbeddingsUseFloat32LittleEndianBase64) {
  proto::EmbeddingResponse response;
  response.add_data()->add_embedding(1.0f);
  response.mutable_data(0)->add_embedding(-2.0f);
  const auto json = openai_embedding_json(response, "base64");
  EXPECT_EQ(json["data"][0]["embedding"], "AACAPwAAAMA=");
  EXPECT_FALSE(json["usage"].contains("completion_tokens"));
  EXPECT_EQ(openai_embedding_json(response, "float")["data"][0]["embedding"],
            nlohmann::json::array({1.0, -2.0}));
}

}  // namespace
}  // namespace xllm::api_service
