/* Copyright 2025-2026 The xLLM Authors.

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

// Testing steps:
// 1. start the xllm server first
// 2. run the test with disabled cases enabled
// XLLM_TEST_BASE_URL=http://127.0.0.1:9977 XLLM_TEST_MODEL=Qwen3-8B
//     ./build/lib.linux-aarch64-cpython-311/xllm/openai_service_test
//     --gtest_also_run_disabled_tests

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace xllm {
namespace {

struct TestConfig {
  std::string base_url;
  std::string model;
  std::string api_key;

  static TestConfig get() {
    TestConfig config;
    const char* url_env = std::getenv("XLLM_TEST_BASE_URL");
    config.base_url = url_env ? url_env : "http://127.0.0.1:9977";

    const char* model_env = std::getenv("XLLM_TEST_MODEL");
    config.model = model_env ? model_env : "my_model";

    const char* key_env = std::getenv("XLLM_TEST_API_KEY");
    config.api_key = key_env ? key_env : "xllm-test-123456";
    return config;
  }
};

struct HttpResult {
  bool controller_failed = false;
  int status_code = 0;
  std::string content_type;
  std::string error_text;
  std::string body;
  nlohmann::json json = nullptr;
};

class HttpClient {
 public:
  explicit HttpClient(const std::string& base_url) {
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.connection_type = brpc::CONNECTION_TYPE_POOLED;
    options.timeout_ms = 60000;
    options.max_retry = 3;

    if (channel_.Init(base_url.c_str(), &options) != 0) {
      LOG(ERROR) << "Failed to init channel for " << base_url;
    }
  }

  HttpResult post(const std::string& path,
                  const nlohmann::json& body,
                  const std::map<std::string, std::string>& headers) {
    brpc::Controller cntl;
    cntl.http_request().uri() = path;
    cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
    cntl.http_request().set_content_type("application/json");
    for (const auto& [key, value] : headers) {
      cntl.http_request().SetHeader(key, value);
    }
    cntl.request_attachment().append(body.dump());

    channel_.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

    HttpResult result;
    result.controller_failed = cntl.Failed();
    result.status_code = cntl.http_response().status_code();
    result.content_type = cntl.http_response().content_type();
    result.error_text = cntl.ErrorText();
    result.body = cntl.response_attachment().to_string();
    if (!result.body.empty()) {
      try {
        result.json = nlohmann::json::parse(result.body);
      } catch (const std::exception&) {
        result.json = nullptr;
      }
    }
    return result;
  }

 private:
  brpc::Channel channel_;
};

std::string describe_result(const HttpResult& result) {
  std::string description = "status=" + std::to_string(result.status_code);
  if (!result.content_type.empty()) {
    description += ", content_type=" + result.content_type;
  }
  if (!result.error_text.empty()) {
    description += ", error=" + result.error_text;
  }
  if (!result.body.empty()) {
    description += ", body=" + result.body;
  }
  return description;
}

void expect_error_contains(const HttpResult& result,
                           const std::string& expected_fragment) {
  const std::string haystack =
      result.error_text + "\n" + result.body + "\n" + result.json.dump();
  EXPECT_NE(haystack.find(expected_fragment), std::string::npos)
      << "Expected fragment '" << expected_fragment
      << "' in response, got: " << describe_result(result);
}

void expect_logprobs_shape(const nlohmann::json& choice) {
  ASSERT_TRUE(choice.contains("logprobs"));
  ASSERT_TRUE(choice["logprobs"].is_object());
  ASSERT_TRUE(choice["logprobs"].contains("tokens"));
  ASSERT_TRUE(choice["logprobs"].contains("token_ids"));
  ASSERT_TRUE(choice["logprobs"].contains("token_logprobs"));
  ASSERT_TRUE(choice["logprobs"]["tokens"].is_array());
  ASSERT_TRUE(choice["logprobs"]["token_ids"].is_array());
  ASSERT_TRUE(choice["logprobs"]["token_logprobs"].is_array());
  EXPECT_EQ(choice["logprobs"]["tokens"].size(),
            choice["logprobs"]["token_ids"].size());
  EXPECT_EQ(choice["logprobs"]["tokens"].size(),
            choice["logprobs"]["token_logprobs"].size());
}

class DISABLED_OpenAIServerFeaturesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = TestConfig::get();
    client_ = std::make_unique<HttpClient>(config_.base_url);
  }

  std::map<std::string, std::string> get_headers() const {
    return {{"Authorization", "Bearer " + config_.api_key},
            {"Content-Type", "application/json"}};
  }

  nlohmann::json make_sample_request(const std::string& prompt,
                                     int logprobs = 3) const {
    return {{"model", config_.model},
            {"prompt", prompt},
            {"selector", {{"type", "literal"}, {"value", "<emb_0>"}}},
            {"logprobs", logprobs},
            {"request_id", "sample-it"}};
  }

  void expect_sample_choice(const nlohmann::json& choice,
                            std::size_t expected_index,
                            std::size_t max_logprobs) const {
    EXPECT_EQ(choice["index"], expected_index);
    ASSERT_TRUE(choice.contains("finish_reason"));
    ASSERT_TRUE(choice.contains("text"));
    expect_logprobs_shape(choice);
    EXPECT_LE(choice["logprobs"]["tokens"].size(), max_logprobs);
    const std::string finish_reason = choice["finish_reason"];
    EXPECT_TRUE(finish_reason == "selector_match" ||
                finish_reason == "empty_logprobs")
        << "Unexpected finish_reason: " << finish_reason;
    if (!choice["logprobs"]["tokens"].empty()) {
      EXPECT_EQ(choice["text"], choice["logprobs"]["tokens"][0]);
    } else {
      EXPECT_TRUE(choice["text"].empty());
      EXPECT_EQ(finish_reason, "empty_logprobs");
    }
  }

  TestConfig config_;
  std::unique_ptr<HttpClient> client_;
};

TEST_F(DISABLED_OpenAIServerFeaturesTest, SampleSingleMatch) {
  const HttpResult result =
      client_->post("/v1/sample",
                    make_sample_request("Classify <emb_0> in one token."),
                    get_headers());

  ASSERT_FALSE(result.controller_failed) << describe_result(result);
  ASSERT_EQ(result.status_code, 200) << describe_result(result);
  EXPECT_EQ(result.content_type, "application/json") << describe_result(result);
  ASSERT_TRUE(result.json.is_object()) << describe_result(result);
  EXPECT_EQ(result.json["id"], "sample-it");
  EXPECT_EQ(result.json["object"], "sample_completion");
  EXPECT_EQ(result.json["model"], config_.model);
  ASSERT_TRUE(result.json.contains("choices"));
  ASSERT_EQ(result.json["choices"].size(), 1);
  expect_sample_choice(result.json["choices"][0], 0, 3);
}

TEST_F(DISABLED_OpenAIServerFeaturesTest, SampleMultipleMatchesStayOrdered) {
  const HttpResult result = client_->post(
      "/v1/sample", make_sample_request("A<emb_0>B<emb_0>C"), get_headers());

  ASSERT_FALSE(result.controller_failed) << describe_result(result);
  ASSERT_EQ(result.status_code, 200) << describe_result(result);
  ASSERT_TRUE(result.json.is_object()) << describe_result(result);
  ASSERT_TRUE(result.json.contains("choices"));
  ASSERT_EQ(result.json["choices"].size(), 2);
  expect_sample_choice(result.json["choices"][0], 0, 3);
  expect_sample_choice(result.json["choices"][1], 1, 3);
}

TEST_F(DISABLED_OpenAIServerFeaturesTest,
       SampleSelectorMissReturnsEmptyChoices) {
  const HttpResult result = client_->post(
      "/v1/sample", make_sample_request("plain text"), get_headers());

  ASSERT_FALSE(result.controller_failed) << describe_result(result);
  ASSERT_EQ(result.status_code, 200) << describe_result(result);
  ASSERT_TRUE(result.json.is_object()) << describe_result(result);
  EXPECT_EQ(result.json["id"], "sample-it");
  EXPECT_EQ(result.json["object"], "sample_completion");
  ASSERT_TRUE(result.json.contains("choices"));
  EXPECT_TRUE(result.json["choices"].empty());
}

TEST_F(DISABLED_OpenAIServerFeaturesTest,
       SampleRejectsUnsupportedSelectorType) {
  nlohmann::json request = make_sample_request("A<emb_0>");
  request["selector"]["type"] = "regex";
  const HttpResult result = client_->post("/v1/sample", request, get_headers());

  EXPECT_NE(result.status_code, 200) << describe_result(result);
  expect_error_contains(result, "literal");
}

TEST_F(DISABLED_OpenAIServerFeaturesTest, SampleRejectsOutOfRangeLogprobs) {
  nlohmann::json request = make_sample_request("A<emb_0>");
  request["logprobs"] = 0;
  const HttpResult result = client_->post("/v1/sample", request, get_headers());

  EXPECT_NE(result.status_code, 200) << describe_result(result);
  expect_error_contains(result, "between 1 and 5");
}

TEST_F(DISABLED_OpenAIServerFeaturesTest, CompletionsRegressionSmoke) {
  nlohmann::json request = {{"model", config_.model},
                            {"prompt", "Say hi."},
                            {"max_tokens", 1},
                            {"temperature", 0.0}};
  const HttpResult result =
      client_->post("/v1/completions", request, get_headers());

  ASSERT_FALSE(result.controller_failed) << describe_result(result);
  ASSERT_EQ(result.status_code, 200) << describe_result(result);
  EXPECT_EQ(result.content_type, "application/json") << describe_result(result);
  ASSERT_TRUE(result.json.is_object()) << describe_result(result);
  ASSERT_TRUE(result.json.contains("choices"));
  ASSERT_EQ(result.json["choices"].size(), 1);
  EXPECT_EQ(result.json["choices"][0]["index"], 0);
  ASSERT_TRUE(result.json.contains("usage"));
}

TEST_F(DISABLED_OpenAIServerFeaturesTest, ChatCompletionsRegressionSmoke) {
  nlohmann::json request = {
      {"model", config_.model},
      {"messages", {{{"role", "user"}, {"content", "Say hi."}}}},
      {"max_tokens", 1},
      {"temperature", 0.0}};
  const HttpResult result =
      client_->post("/v1/chat/completions", request, get_headers());

  ASSERT_FALSE(result.controller_failed) << describe_result(result);
  ASSERT_EQ(result.status_code, 200) << describe_result(result);
  EXPECT_EQ(result.content_type, "application/json") << describe_result(result);
  ASSERT_TRUE(result.json.is_object()) << describe_result(result);
  ASSERT_TRUE(result.json.contains("choices"));
  ASSERT_EQ(result.json["choices"].size(), 1);
  ASSERT_TRUE(result.json["choices"][0].contains("message"));
  ASSERT_TRUE(result.json["choices"][0]["message"].contains("role"));
  ASSERT_TRUE(result.json["choices"][0]["message"].contains("content"));
  ASSERT_TRUE(result.json.contains("usage"));
}

}  // namespace
}  // namespace xllm
