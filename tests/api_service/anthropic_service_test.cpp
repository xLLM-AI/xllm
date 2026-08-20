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
// 2. run the test
// XLLM_TEST_BASE_URL=127.0.0.1:19977 XLLM_TEST_MODEL=Qwen3-8B
//      ./build/lib.linux-x86_64-cpython-311/xllm/anthropic_service_test
//      --gtest_also_run_disabled_tests
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace xllm {
namespace {

// Configuration loaded from environment or using sensible defaults
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

// Lightweight HTTP client wrapper using brpc for API requests
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

  // Execute a synchronous POST request with JSON payload
  bool post(const std::string& path,
            const nlohmann::json& body,
            const std::map<std::string, std::string>& headers,
            nlohmann::json& response_json,
            int& status_code) {
    brpc::Controller cntl;
    cntl.http_request().uri() = path;
    cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
    cntl.http_request().set_content_type("application/json");

    for (const auto& [key, value] : headers) {
      cntl.http_request().SetHeader(key, value);
    }
    cntl.request_attachment().append(body.dump());

    channel_.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

    if (cntl.Failed()) {
      LOG(ERROR) << "Request failed: " << cntl.ErrorText();
      return false;
    }

    status_code = cntl.http_response().status_code();
    std::string response_body = cntl.response_attachment().to_string();

    try {
      response_json = nlohmann::json::parse(response_body);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse JSON response: " << e.what();
      return false;
    }

    return true;
  }

  // Execute POST request and parse Server-Sent Events from the response
  bool post_stream(const std::string& path,
                   const nlohmann::json& body,
                   const std::map<std::string, std::string>& headers,
                   std::vector<std::pair<std::string, nlohmann::json>>& events,
                   int& status_code) {
    brpc::Controller cntl;
    cntl.http_request().uri() = path;
    cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
    cntl.http_request().set_content_type("application/json");

    // Set headers
    for (const auto& [key, value] : headers) {
      cntl.http_request().SetHeader(key, value);
    }

    // Set body
    cntl.request_attachment().append(body.dump());

    channel_.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

    if (cntl.Failed()) {
      LOG(ERROR) << "Request failed: " << cntl.ErrorText();
      return false;
    }

    status_code = cntl.http_response().status_code();
    std::string response_body = cntl.response_attachment().to_string();

    // Parse response as SSE stream
    std::istringstream stream(response_body);
    std::string line;
    std::string current_event_type;

    while (std::getline(stream, line)) {
      // Strip trailing carriage return for cross-platform compatibility
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }

      if (line.empty()) {
        continue;
      }

      if (line.rfind("event: ", 0) == 0) {
        current_event_type = line.substr(7);
      } else if (line.rfind("data: ", 0) == 0) {
        std::string data = line.substr(6);
        if (data != "[DONE]" && !data.empty()) {
          try {
            nlohmann::json event_data = nlohmann::json::parse(data);
            events.emplace_back(current_event_type, event_data);
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to parse event data: " << e.what();
          }
        }
      }
    }

    return true;
  }

 private:
  brpc::Channel channel_;
};

// Server feature tests - skipped in automated runs (DISABLED_ prefix).
// To run: ./anthropic_service_test --gtest_also_run_disabled_tests
class DISABLED_AnthropicServerFeaturesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = TestConfig::get();
    client_ = std::make_unique<HttpClient>(config_.base_url);
  }

  std::map<std::string, std::string> get_headers() {
    return {{"Authorization", "Bearer " + config_.api_key},
            {"Content-Type", "application/json"},
            {"anthropic-version", "2023-06-01"}};
  }

  // Construct weather tool schema with city and unit parameters
  nlohmann::json get_weather_tool() {
    return {{"name", "get_weather"},
            {"description", "Get the current weather in a given location"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"city",
                 {{"type", "string"},
                  {"description", "The city to find the weather for"}}},
                {"unit",
                 {{"type", "string"},
                  {"description", "Weather unit (celsius or fahrenheit)"},
                  {"enum", {"celsius", "fahrenheit"}}}}}},
              {"required", {"city", "unit"}}}}};
  }

  // Construct time lookup tool schema
  nlohmann::json get_time_tool() {
    return {{"name", "get_time"},
            {"description", "Get the current time in a given location"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"city",
                 {{"type", "string"},
                  {"description", "The city to find the time for"}}}}},
              {"required", {"city"}}}}};
  }

  // Check whether a specific content block type is present
  bool has_content_type(const nlohmann::json& content,
                        const std::string& type) {
    for (const auto& block : content) {
      if (block.contains("type") && block["type"] == type) {
        return true;
      }
    }
    return false;
  }

  TestConfig config_;
  std::unique_ptr<HttpClient> client_;
};

// Verify default tool_choice behavior (auto mode)
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceAuto) {
  nlohmann::json tools = {get_weather_tool(), get_time_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "What is the weather like in Paris?"}}}},
      {"temperature", 0.8},
      {"tools", tools}
      // For auto tool choice, we don't need to specify tool_choice explicitly
      // as it defaults to auto behavior when tools are provided
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Should have either text or tool_use blocks
  bool has_text = has_content_type(response["content"], "text");
  bool has_tool_use = has_content_type(response["content"], "tool_use");
  EXPECT_TRUE(has_text || has_tool_use)
      << "Should have either text or tool_use content blocks";
}

// Ensure tools work correctly without explicit tool_choice
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceSpecific) {
  // Create simplified tools for this test
  nlohmann::json weather_tool = {
      {"name", "get_weather"},
      {"description", "Get the current weather in a given location"},
      {"input_schema",
       {{"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city to find the weather for"}}}}},
        {"required", {"city"}}}}};

  nlohmann::json time_tool = get_time_tool();
  nlohmann::json tools = {weather_tool, time_tool};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "What time is it in London?"}}}},
      {"temperature", 0.8},
      {"tools", tools}
      // When tool_choice is not specified, it should default to auto behavior
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Should have either text or tool_use blocks
  bool has_text = has_content_type(response["content"], "text");
  bool has_tool_use = has_content_type(response["content"], "tool_use");
  EXPECT_TRUE(has_text || has_tool_use)
      << "Should have either text or tool_use content blocks";
}

// Validate that stop_reason returns valid enum values
TEST_F(DISABLED_AnthropicServerFeaturesTest, StopReasonVariations) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 10},  // Small max_tokens to trigger max_tokens stop reason
      {"messages", {{{"role", "user"}, {"content", "Count from 1 to 100"}}}},
      {"temperature", 0.1}  // Low temperature for deterministic results
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";

  // Check that stop_reason is one of the valid values
  std::vector<std::string> valid_stop_reasons = {
      "end_turn", "max_tokens", "stop_sequence", "tool_use"};
  std::string stop_reason = response["stop_reason"].get<std::string>();

  bool is_valid = std::find(valid_stop_reasons.begin(),
                            valid_stop_reasons.end(),
                            stop_reason) != valid_stop_reasons.end();
  EXPECT_TRUE(is_valid) << "Stop reason '" << stop_reason
                        << "' should be one of: end_turn, max_tokens, "
                           "stop_sequence, tool_use";
}

// Confirm explicit tool_choice=auto behaves identically to implicit auto
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceAutoExplicit) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "What is the weather like in Tokyo?"}}}},
      {"temperature", 0.5},
      {"tools", tools},
      {"tool_choice", {{"type", "auto"}}}  // Explicitly set to auto
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
}

// Force tool invocation with tool_choice=any
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceAny) {
  nlohmann::json tools = {get_weather_tool(), get_time_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "Tell me something about Berlin."}}}},
      {"temperature", 0.5},
      {"tools", tools},
      {"tool_choice", {{"type", "any"}}}  // Force use of any tool
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
}

// Force a specific tool via tool_choice with name
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceSpecificTool) {
  nlohmann::json tools = {get_weather_tool(), get_time_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "Tell me about New York."}}}},
      {"temperature", 0.5},
      {"tools", tools},
      {"tool_choice",
       {{"type", "tool"}, {"name", "get_weather"}}}  // Force specific tool
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
}

// Validate tool_use block contains required fields (id, name, input)
TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolUseBlockStructure) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"},
         {"content", "Get the weather in San Francisco in celsius"}}}},
      {"temperature", 0.1},
      {"tools", tools},
      {"tool_choice",
       {{"type", "tool"}, {"name", "get_weather"}}}  // Force tool use
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";

  // Find tool_use block and verify structure
  for (const auto& block : response["content"]) {
    if (block.contains("type") && block["type"] == "tool_use") {
      EXPECT_TRUE(block.contains("id")) << "tool_use block missing 'id'";
      EXPECT_TRUE(block.contains("name")) << "tool_use block missing 'name'";
      EXPECT_TRUE(block.contains("input")) << "tool_use block missing 'input'";

      // Verify tool name
      EXPECT_EQ(block["name"], "get_weather");

      // Verify input structure
      if (block.contains("input")) {
        EXPECT_TRUE(block["input"].is_object())
            << "'input' should be an object";
      }
    }
  }
}

// Ensure multiple tools with varying schemas are handled correctly
TEST_F(DISABLED_AnthropicServerFeaturesTest,
       MultipleToolsWithDifferentSchemas) {
  nlohmann::json calculator_tool = {
      {"name", "calculator"},
      {"description", "Perform mathematical calculations"},
      {"input_schema",
       {{"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "The mathematical expression to evaluate"}}},
          {"precision",
           {{"type", "integer"},
            {"description", "Number of decimal places"}}}}},
        {"required", {"expression"}}}}};

  nlohmann::json tools = {get_weather_tool(), get_time_tool(), calculator_tool};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"messages",
       {{{"role", "user"}, {"content", "What's the weather in Paris?"}}}},
      {"temperature", 0.5},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
}

// Confirm natural completion yields end_turn stop reason
TEST_F(DISABLED_AnthropicServerFeaturesTest, StopReasonEndTurn) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 100},
      {"messages", {{{"role", "user"}, {"content", "Say hello in one word."}}}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
  // With enough max_tokens for a simple response, should end naturally
  std::string stop_reason = response["stop_reason"].get<std::string>();
  EXPECT_TRUE(stop_reason == "end_turn" || stop_reason == "max_tokens")
      << "Stop reason should be 'end_turn' or 'max_tokens', got: "
      << stop_reason;
}

// Verify token limit triggers max_tokens stop reason
TEST_F(DISABLED_AnthropicServerFeaturesTest, StopReasonMaxTokens) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 1},  // Very small limit
      {"messages",
       {{{"role", "user"},
         {"content", "Write a long essay about the history of computing."}}}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
  // With max_tokens=1, should hit the limit
  EXPECT_EQ(response["stop_reason"], "max_tokens")
      << "Stop reason should be 'max_tokens'";
}

// Check stop_sequence reason when hitting a custom stop string
TEST_F(DISABLED_AnthropicServerFeaturesTest, StopReasonStopSequence) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 100},
      {"messages", {{{"role", "user"}, {"content", "Count: 1, 2, 3, 4, 5"}}}},
      {"stop_sequences", {"3"}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
  // May or may not hit the stop sequence depending on model output
  std::string stop_reason = response["stop_reason"].get<std::string>();
  std::vector<std::string> valid_reasons = {
      "end_turn", "max_tokens", "stop_sequence"};
  bool is_valid =
      std::find(valid_reasons.begin(), valid_reasons.end(), stop_reason) !=
      valid_reasons.end();
  EXPECT_TRUE(is_valid) << "Stop reason should be valid, got: " << stop_reason;
}

// Basic non-streaming message completion
TEST_F(DISABLED_AnthropicServerFeaturesTest, Completion) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 32},
      {"messages",
       {{{"role", "user"}, {"content", "The capital of France is"}}}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("id")) << "Response missing 'id'";
  EXPECT_TRUE(response.contains("type")) << "Response missing 'type'";
  EXPECT_EQ(response["type"], "message");
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";
  EXPECT_TRUE(response["content"][0].contains("text"))
      << "Content block missing 'text'";
}

// Streaming message completion with SSE event validation
TEST_F(DISABLED_AnthropicServerFeaturesTest, CompletionStream) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 32},
      {"messages",
       {{{"role", "user"}, {"content", "The capital of France is"}}}},
      {"temperature", 0},
      {"stream", true}};

  std::vector<std::pair<std::string, nlohmann::json>> events;
  int status_code;
  bool success = client_->post_stream(
      "/v1/messages", request_body, get_headers(), events, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  EXPECT_GT(events.size(), 0) << "Should receive some events";

  std::vector<std::string> event_types;
  for (const auto& [event_type, _] : events) {
    event_types.push_back(event_type);
  }

  auto contains = [&event_types](const std::string& type) {
    return std::find(event_types.begin(), event_types.end(), type) !=
           event_types.end();
  };

  // Validate required SSE event sequence
  EXPECT_TRUE(contains("message_start")) << "Missing 'message_start' event";
  EXPECT_TRUE(contains("content_block_start"))
      << "Missing 'content_block_start' event";
  EXPECT_TRUE(contains("content_block_delta"))
      << "Missing 'content_block_delta' event";
  EXPECT_TRUE(contains("content_block_stop"))
      << "Missing 'content_block_stop' event";
  EXPECT_TRUE(contains("message_stop")) << "Missing 'message_stop' event";
}

// Validate complete chat response structure and metadata
TEST_F(DISABLED_AnthropicServerFeaturesTest, ChatCompletion) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 64},
      {"messages",
       {{{"role", "user"},
         {"content",
          "What is the capital of France? Answer in a few words."}}}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("id")) << "Response missing 'id'";
  EXPECT_TRUE(response.contains("type")) << "Response missing 'type'";
  EXPECT_EQ(response["type"], "message");
  EXPECT_TRUE(response.contains("role")) << "Response missing 'role'";
  EXPECT_EQ(response["role"], "assistant");
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";
  EXPECT_TRUE(response["content"][0].contains("text"))
      << "Content block missing 'text'";
  EXPECT_TRUE(response.contains("model")) << "Response missing 'model'";
  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
  EXPECT_TRUE(response.contains("usage")) << "Response missing 'usage'";
  EXPECT_TRUE(response["usage"].contains("input_tokens"))
      << "Usage missing 'input_tokens'";
  EXPECT_TRUE(response["usage"].contains("output_tokens"))
      << "Usage missing 'output_tokens'";
}

// Streaming chat completion with full event lifecycle
TEST_F(DISABLED_AnthropicServerFeaturesTest, ChatCompletionStream) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 64},
      {"messages",
       {{{"role", "user"}, {"content", "What is the capital of France?"}}}},
      {"temperature", 0},
      {"stream", true}};

  std::vector<std::pair<std::string, nlohmann::json>> events;
  int status_code;
  bool success = client_->post_stream(
      "/v1/messages", request_body, get_headers(), events, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify we got some events
  EXPECT_GT(events.size(), 0) << "Should receive some events";

  // Collect event types
  std::vector<std::string> event_types;
  for (const auto& [event_type, _] : events) {
    event_types.push_back(event_type);
  }

  // Verify event structure
  auto contains = [&event_types](const std::string& type) {
    return std::find(event_types.begin(), event_types.end(), type) !=
           event_types.end();
  };

  EXPECT_TRUE(contains("message_start")) << "Missing 'message_start' event";
  EXPECT_TRUE(contains("content_block_start"))
      << "Missing 'content_block_start' event";
  EXPECT_TRUE(contains("content_block_delta"))
      << "Missing 'content_block_delta' event";
  EXPECT_TRUE(contains("content_block_stop"))
      << "Missing 'content_block_stop' event";
  EXPECT_TRUE(contains("message_stop")) << "Missing 'message_stop' event";
}

// Verify system message is properly applied to conversation
TEST_F(DISABLED_AnthropicServerFeaturesTest, SystemPrompt) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 32},
      {"system", "You are a helpful assistant specialized in geography."},
      {"messages",
       {{{"role", "user"}, {"content", "What is the capital of France?"}}}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  EXPECT_EQ(response["type"], "message");
  EXPECT_EQ(response["role"], "assistant");
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";
}

// Confirm custom stop_sequences terminates generation properly
TEST_F(DISABLED_AnthropicServerFeaturesTest, StopSequences) {
  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 64},
      {"messages", {{{"role", "user"}, {"content", "Count from 1 to 10."}}}},
      {"stop_sequences", {"5"}},
      {"temperature", 0}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // The response should have a stop_reason
  EXPECT_TRUE(response.contains("stop_reason"))
      << "Response missing 'stop_reason'";
}

class DISABLED_AnthropicFunctionCallingTest : public ::testing::Test {
 protected:
  static constexpr const char* SYSTEM_MESSAGE = "You are a helpful assistant.";

  void SetUp() override {
    config_ = TestConfig::get();
    client_ = std::make_unique<HttpClient>(config_.base_url);
  }

  std::map<std::string, std::string> get_headers() {
    return {{"Authorization", "Bearer " + config_.api_key},
            {"Content-Type", "application/json"},
            {"anthropic-version", "2023-06-01"}};
  }

  // Arithmetic addition tool with two integer parameters
  nlohmann::json get_add_tool() {
    return {{"name", "add"},
            {"description", "Compute the sum of two numbers"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"a", {{"type", "integer"}, {"description", "A number"}}},
                {"b", {{"type", "integer"}, {"description", "A number"}}}}},
              {"required", {"a", "b"}}}}};
  }

  // Weather query tool requiring city and temperature unit
  nlohmann::json get_weather_tool() {
    return {{"name", "get_current_weather"},
            {"description", "Get the current weather in a given location"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"city",
                 {{"type", "string"},
                  {"description", "The city to find the weather for"}}},
                {"unit",
                 {{"type", "string"},
                  {"description", "Weather unit (celsius or fahrenheit)"},
                  {"enum", {"celsius", "fahrenheit"}}}}}},
              {"required", {"city", "unit"}}}}};
  }

  // Simplified weather tool with only city parameter
  nlohmann::json get_simple_weather_tool() {
    return {{"name", "get_weather"},
            {"description", "Get the current weather in a given location"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"city",
                 {{"type", "string"},
                  {"description", "The city to find the weather for"}}}}},
              {"required", {"city"}}}}};
  }

  // Timezone-aware time lookup tool
  nlohmann::json get_time_tool() {
    return {{"name", "get_time"},
            {"description", "Get the current time in a given location"},
            {"input_schema",
             {{"type", "object"},
              {"properties",
               {{"city",
                 {{"type", "string"},
                  {"description", "The city to find the time for"}}}}},
              {"required", {"city"}}}}};
  }

  // Age calculation tool with name and birth_year inputs
  nlohmann::json get_calculate_age_tool() {
    return {
        {"name", "calculate_age"},
        {"description", "Calculate a person's age based on birth year"},
        {"input_schema",
         {{"type", "object"},
          {"properties",
           {{"name", {{"type", "string"}, {"description", "Person's name"}}},
            {"birth_year",
             {{"type", "integer"}, {"description", "Year of birth"}}}}},
          {"required", {"name", "birth_year"}}}}};
  }

  // Determine if a given block type exists in the content array
  bool has_content_type(const nlohmann::json& content,
                        const std::string& type) {
    for (const auto& block : content) {
      if (block.contains("type") && block["type"] == type) {
        return true;
      }
    }
    return false;
  }

  // Extract all content blocks matching the specified type
  std::vector<nlohmann::json> get_content_blocks_by_type(
      const nlohmann::json& content,
      const std::string& type) {
    std::vector<nlohmann::json> blocks;
    for (const auto& block : content) {
      if (block.contains("type") && block["type"] == type) {
        blocks.push_back(block);
      }
    }
    return blocks;
  }

  TestConfig config_;
  std::unique_ptr<HttpClient> client_;
};

// Validate tool_use block structure when model invokes a function
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingFormat) {
  nlohmann::json tools = {get_add_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages", {{{"role", "user"}, {"content", "Compute (3+5)"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Check if we have a tool_use content block
  auto tool_use_blocks =
      get_content_blocks_by_type(response["content"], "tool_use");

  // Since this is a simple test, we'll check if we have tool_use blocks or
  // just text. Either way is acceptable for a basic test
  if (!tool_use_blocks.empty()) {
    // If we have tool_use blocks, verify their structure
    EXPECT_TRUE(tool_use_blocks[0].contains("name"))
        << "Tool use block should have a name";
    std::string function_name = tool_use_blocks[0]["name"].get<std::string>();
    EXPECT_FALSE(function_name.empty()) << "Function name should not be empty";

    // Check input parameters
    EXPECT_TRUE(tool_use_blocks[0].contains("input"))
        << "Tool use block should have input";
    const auto& function_input = tool_use_blocks[0]["input"];
    EXPECT_TRUE(function_input.contains("a")) << "Should have parameter 'a'";
    EXPECT_TRUE(function_input.contains("b")) << "Should have parameter 'b'";
  } else {
    // If we don't have tool_use blocks, that's also acceptable for this test
    // Just verify we have a text response
    auto text_blocks = get_content_blocks_by_type(response["content"], "text");
    EXPECT_GT(text_blocks.size(), 0)
        << "Should have at least one text block if no tool_use blocks";
  }
}

// Weather tool invocation with location and unit parameters
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingWithWeatherTool) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content", "What is the temperature in Paris in celsius?"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Check content blocks - either tool_use or text is acceptable
  bool has_text = has_content_type(response["content"], "text");
  bool has_tool_use = has_content_type(response["content"], "tool_use");
  EXPECT_TRUE(has_text || has_tool_use)
      << "Should have either text or tool_use content blocks";
}

// Streaming mode tool call detection and SSE event validation
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingStreamingSimple) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content", "What is the temperature in Paris in celsius?"}}}},
      {"temperature", 0.8},
      {"tools", tools},
      {"stream", true}};

  std::vector<std::pair<std::string, nlohmann::json>> events;
  int status_code;
  bool success = client_->post_stream(
      "/v1/messages", request_body, get_headers(), events, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify we got some events
  EXPECT_GT(events.size(), 0) << "Streaming should return at least one chunk";

  // Look for tool_use events in content blocks
  bool found_tool_use = false;
  for (const auto& [event_type, event_data] : events) {
    if (event_type == "content_block_start") {
      if (event_data.contains("content_block")) {
        const auto& content_block = event_data["content_block"];
        if (content_block.contains("type") &&
            content_block["type"] == "tool_use") {
          if (content_block.contains("name")) {
            EXPECT_EQ(content_block["name"], "get_current_weather")
                << "Function name should be 'get_current_weather'";
          }
          found_tool_use = true;
          break;
        }
      }
    }
  }

  // It's acceptable if we don't find tool_use in streaming - model may respond
  // with text. Just verify we have proper event structure
  std::vector<std::string> event_types;
  for (const auto& [event_type, _] : events) {
    event_types.push_back(event_type);
  }

  auto contains = [&event_types](const std::string& type) {
    return std::find(event_types.begin(), event_types.end(), type) !=
           event_types.end();
  };

  EXPECT_TRUE(contains("message_start")) << "Missing 'message_start' event";
  EXPECT_TRUE(contains("content_block_start"))
      << "Missing 'content_block_start' event";
  EXPECT_TRUE(contains("content_block_stop"))
      << "Missing 'content_block_stop' event";
  EXPECT_TRUE(contains("message_stop")) << "Missing 'message_stop' event";
}

// Force tool invocation using tool_choice=any, regardless of query content
TEST_F(DISABLED_AnthropicFunctionCallingTest, ToolChoiceRequired) {
  nlohmann::json tools = {get_simple_weather_tool(), get_add_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"}, {"content", "What is the capital of France?"}}}},
      {"temperature", 0.8},
      {"tools", tools},
      {"tool_choice", {{"type", "any"}}}  // Force tool use
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";
}

// Direct a specific tool by name via tool_choice configuration
TEST_F(DISABLED_AnthropicFunctionCallingTest, ToolChoiceSpecific) {
  nlohmann::json tools = {get_simple_weather_tool(), get_add_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content", "What is the weather like in London?"}}}},
      {"temperature", 0.8},
      {"tools", tools},
      {"tool_choice",
       {{"type", "tool"}, {"name", "get_weather"}}}  // Force specific tool
  };

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";
}

// Verify tool arguments conform to the defined input schema
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingStrictMode) {
  nlohmann::json tools = {get_calculate_age_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content", "Calculate John's age who was born in 1990"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Check if we have a tool_use content block
  auto tool_use_blocks =
      get_content_blocks_by_type(response["content"], "tool_use");

  // If we have tool_use blocks, verify their structure
  if (!tool_use_blocks.empty()) {
    const auto& tool_call = tool_use_blocks[0];
    std::string function_name = tool_call["name"].get<std::string>();
    EXPECT_EQ(function_name, "calculate_age")
        << "Function name should be 'calculate_age'";

    // Check input parameters
    EXPECT_TRUE(tool_call.contains("input")) << "Tool call missing 'input'";
    const auto& function_input = tool_call["input"];
    EXPECT_TRUE(function_input.contains("name"))
        << "Should have parameter 'name'";
    EXPECT_TRUE(function_input.contains("birth_year"))
        << "Should have parameter 'birth_year'";

    // Verify parameter types
    EXPECT_TRUE(function_input["name"].is_string())
        << "Name should be a string";
    // birth_year could be int or string representation of int
    bool is_int = function_input["birth_year"].is_number_integer();
    bool is_string = function_input["birth_year"].is_string();
    EXPECT_TRUE(is_int || is_string)
        << "Birth year should be an integer or string representation of "
           "integer";
  }
}

// Confirm model responds with text when query doesn't warrant tool use
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingNoToolCall) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages", {{{"role", "user"}, {"content", "Who are you?"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Should have text content when no tool call is needed
  auto text_blocks = get_content_blocks_by_type(response["content"], "text");
  EXPECT_GT(text_blocks.size(), 0)
      << "Should have text response when no tool call is needed";

  if (!text_blocks.empty()) {
    std::string response_text = text_blocks[0]["text"].get<std::string>();
    EXPECT_GT(response_text.length(), 0) << "Response text should not be empty";
  }
}

// Model should not hallucinate tools that aren't in the provided list
TEST_F(DISABLED_AnthropicFunctionCallingTest,
       FunctionCallingInvalidFunctionName) {
  nlohmann::json tools = {get_weather_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content",
          "Please use the 'get_stock_price' function to check "
          "Apple's stock price"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // If there are tool_use blocks, verify they are only for valid functions
  auto tool_use_blocks =
      get_content_blocks_by_type(response["content"], "tool_use");
  for (const auto& tool_block : tool_use_blocks) {
    std::string function_name = tool_block["name"].get<std::string>();
    // Should only use functions from our tools list
    EXPECT_EQ(function_name, "get_current_weather")
        << "Should only call valid functions from the tools list";
  }
}

// Handle queries requiring multiple concurrent tool invocations
TEST_F(DISABLED_AnthropicFunctionCallingTest, FunctionCallingParallelTools) {
  nlohmann::json tools = {get_weather_tool(), get_time_tool()};

  nlohmann::json request_body = {
      {"model", config_.model},
      {"max_tokens", 2048},
      {"system", SYSTEM_MESSAGE},
      {"messages",
       {{{"role", "user"},
         {"content", "What is the weather and current time in New York?"}}}},
      {"temperature", 0.8},
      {"tools", tools}};

  nlohmann::json response;
  int status_code;
  bool success = client_->post(
      "/v1/messages", request_body, get_headers(), response, status_code);

  ASSERT_TRUE(success) << "Request failed";
  EXPECT_EQ(status_code, 200) << "Expected status 200, got " << status_code;

  // Verify response structure
  EXPECT_TRUE(response.contains("content")) << "Response missing 'content'";
  EXPECT_TRUE(response["content"].is_array()) << "'content' should be array";
  EXPECT_GT(response["content"].size(), 0) << "'content' should not be empty";

  // Check if we have tool_use content blocks
  auto tool_use_blocks =
      get_content_blocks_by_type(response["content"], "tool_use");

  // If we have tool_use blocks, verify their structure
  if (!tool_use_blocks.empty()) {
    // Collect all tool names called
    std::vector<std::string> called_tools;
    for (const auto& block : tool_use_blocks) {
      called_tools.push_back(block["name"].get<std::string>());
    }

    // Should only call tools from our tools list
    std::vector<std::string> valid_tools = {"get_current_weather", "get_time"};
    for (const auto& tool_name : called_tools) {
      bool is_valid =
          std::find(valid_tools.begin(), valid_tools.end(), tool_name) !=
          valid_tools.end();
      EXPECT_TRUE(is_valid)
          << "Should only call valid functions from the tools list, got "
          << tool_name;
    }

    // Should have proper input parameters for each tool
    for (const auto& tool_block : tool_use_blocks) {
      EXPECT_TRUE(tool_block.contains("input"))
          << "Tool block should have 'input'";
      const auto& function_input = tool_block["input"];
      std::string tool_name = tool_block["name"].get<std::string>();

      if (tool_name == "get_current_weather") {
        // Should have city and unit parameters
        EXPECT_TRUE(function_input.contains("city"))
            << "Weather tool should have 'city' parameter";
        EXPECT_TRUE(function_input.contains("unit"))
            << "Weather tool should have 'unit' parameter";
      } else if (tool_name == "get_time") {
        // Should have city parameter
        EXPECT_TRUE(function_input.contains("city"))
            << "Time tool should have 'city' parameter";
      }
    }
  }
}

}  // namespace
}  // namespace xllm
