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

#include "glm45_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace function_call {

class Glm45DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Glm45Detector>();

    // Setup test tools
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"location",
           {{"type", "string"},
            {"description", "The city and state, e.g. San Francisco, CA"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
        {"required", {"location"}}};

    JsonFunction weather_func("get_current_weather",
                              "Get the current weather in a given location",
                              weather_params);
    weather_tool_ = JsonTool("function", weather_func);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);
    calculator_tool_ = JsonTool("function", calculator_func);

    tools_ = {weather_tool_, calculator_tool_};
  }

  std::unique_ptr<Glm45Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

// Test constructor and basic properties
TEST_F(Glm45DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection
  std::string text_with_tool_call =
      "Some text "
      "<tool_call>test\n<arg_key>param</arg_key>\n<arg_value>value</"
      "arg_value>\n</tool_call>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(Glm45DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>Following content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</"
      "arg_value>\n</tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without closing"));
}

// Test single tool call parsing
TEST_F(Glm45DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>\n"
      "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["unit"], "celsius");
}

// Test multiple tool calls parsing
TEST_F(Glm45DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate an expression "
      "<tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Shanghai</arg_value>\n</"
      "tool_call>"
      "<tool_call>calculate\n"
      "<arg_key>expression</arg_key>\n<arg_value>2 + 3 * "
      "4</arg_value>\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate an expression");
  EXPECT_EQ(result.calls.size(), 2);

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_current_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["location"], "Shanghai");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "calculate");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["expression"], "2 + 3 * 4");
}

// Test GLM-4.5 specific format with Chinese characters
TEST_F(Glm45DetectorTest, Glm45SpecificFormatWithChinese) {
  std::string text =
      "I need weather info "
      "<tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>北京</arg_value>\n"
      "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "I need weather info");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "北京");
  EXPECT_EQ(params["unit"], "celsius");
}

// Test empty tool call content
TEST_F(Glm45DetectorTest, EmptyToolCallContent) {
  std::string text =
      "Test empty content <tool_call>test\n   \t\n  \n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call (only start tag)
TEST_F(Glm45DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Glm45DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool <tool_call>unknown_tool\n"
      "<arg_key>param</arg_key>\n<arg_value>value</arg_value>\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools, so should be 0 calls
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(Glm45DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(Glm45DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test whitespace-only input
TEST_F(Glm45DetectorTest, WhitespaceOnlyInput) {
  std::string text = "   \t\n\r   ";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
}

// Test complex nested parameters (JSON values)
TEST_F(Glm45DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>\n"
      "<arg_key>options</arg_key>\n<arg_value>{\"include_forecast\": true, "
      "\"days\": 7}</arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
}

// Test special characters handling
TEST_F(Glm45DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>New York City</arg_value>\n"
      "<arg_key>note</arg_key>\n<arg_value>Contains "
      "symbols！@#$%^&*()_+=</arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test tool call in the middle of text
TEST_F(Glm45DetectorTest, ToolCallInMiddleOfText) {
  std::string text =
      "Previous text <tool_call>calculate\n"
      "<arg_key>expression</arg_key>\n<arg_value>1+1</arg_value>\n"
      "</tool_call> Following text";

  auto result = detector_->detect_and_parse(text, tools_);

  // Note: According to GLM-4.5 implementation, only text before tool call is
  // preserved as normal_text
  EXPECT_EQ(result.normal_text, "Previous text");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_EQ(call.name.value(), "calculate");
}

// Test malformed arg tags handling
TEST_F(Glm45DetectorTest, MalformedArgTagsHandling) {
  std::string text =
      "Malformed args test <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key><arg_value>Beijing</arg_value>\n"  // Missing
                                                                     // newline
      "<arg_key>unit<arg_value>celsius</arg_value>\n"  // Missing closing tag
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Malformed args test");
  // Should still parse what it can
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "get_current_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  // The malformed unit parameter should not be parsed
  EXPECT_FALSE(params.contains("unit"));
}

// Test whitespace handling in arg values
TEST_F(Glm45DetectorTest, WhitespaceHandlingInArgValues) {
  std::string text =
      "Whitespace test <tool_call>get_current_weather\n"
      "<arg_key>  location  </arg_key>\n<arg_value>  Beijing  </arg_value>\n"
      "<arg_key>\t\nunit\r\n</arg_key>\n<arg_value>\n\tcelsius\r\n</"
      "arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Whitespace test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");  // Whitespace should be trimmed
  EXPECT_EQ(params["unit"], "celsius");
}

// Test multiple sections (edge case)
TEST_F(Glm45DetectorTest, MultipleSections) {
  std::string text =
      "First section <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>\n"
      "</tool_call> Middle text <tool_call>calculate\n"
      "<arg_key>expression</arg_key>\n<arg_value>1+1</arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  // Should extract text before first tool call
  EXPECT_EQ(result.normal_text, "First section");
  // Should parse all tool calls
  EXPECT_EQ(result.calls.size(), 2);

  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
  EXPECT_EQ(result.calls[1].name.value(), "calculate");
}

// Test performance with many tool calls
TEST_F(Glm45DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test ";

  // Build text containing multiple tool calls
  for (int i = 0; i < 100; ++i) {  // Reduced from 10000 for faster testing
    text +=
        "<tool_call>calculate\n"
        "<arg_key>expression</arg_key>\n<arg_value>" +
        std::to_string(i) + " + " + std::to_string(i + 1) +
        "</arg_value>\n</tool_call>";
  }

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test");
  EXPECT_EQ(result.calls.size(), 100);

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 100; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

// Test edge case: nested braces in JSON values
TEST_F(Glm45DetectorTest, NestedBracesInJsonValues) {
  std::string text =
      "Nested braces test <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>\n"
      "<arg_key>config</arg_key>\n<arg_value>{\"nested\": {\"deep\": "
      "\"value\"}}</arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Nested braces test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_EQ(call.name.value(), "get_current_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

// Test streaming parsing functionality
TEST_F(Glm45DetectorTest, StreamingParseBasicFunctionality) {
  std::string chunk1 = "<tool_call>get_current_weather\n";
  std::string chunk2 = "<arg_key>location</arg_key>\n<arg_value>";
  std::string chunk3 = "Beijing</arg_value>\n</tool_call>";

  // First chunk - should buffer and wait for complete tool call
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.normal_text, "");
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - still incomplete
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_EQ(result2.normal_text, "");
  EXPECT_EQ(result2.calls.size(), 0);

  // Third chunk - completes the tool call
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_EQ(result3.normal_text, "");
  EXPECT_EQ(result3.calls.size(), 1);
  EXPECT_EQ(result3.calls[0].name.value(), "get_current_weather");
}

// Test streaming parse with normal text
TEST_F(Glm45DetectorTest, StreamingParseWithNormalText) {
  std::string chunk1 = "Please check the weather ";
  std::string chunk2 = "<tool_call>get_current_weather\n";
  std::string chunk3 =
      "<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>";

  // First chunk - normal text should be returned
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.normal_text, "Please check the weather ");
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - tool call start, should be buffered
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_EQ(result2.normal_text, "");
  EXPECT_EQ(result2.calls.size(), 0);

  // Third chunk - complete tool call
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_EQ(result3.normal_text, "");
  EXPECT_EQ(result3.calls.size(), 1);
  EXPECT_EQ(result3.calls[0].name.value(), "get_current_weather");
}

// Test invalid JSON in arg values
TEST_F(Glm45DetectorTest, InvalidJsonInArgValues) {
  std::string text =
      "Invalid JSON test <tool_call>get_current_weather\n"
      "<arg_key>location</arg_key>\n<arg_value>Beijing</arg_value>\n"
      "<arg_key>config</arg_key>\n<arg_value>{invalid json}</arg_value>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Invalid JSON test");
  EXPECT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["config"], "{invalid json}");  // Should be treated as string
}

}  // namespace function_call
}  // namespace xllm
