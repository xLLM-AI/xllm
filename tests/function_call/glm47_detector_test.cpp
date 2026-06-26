/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "glm47_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "function_call_parser.h"

namespace xllm {
namespace function_call {

class Glm47DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Glm47Detector>();

    // Setup test tools
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city name, e.g. Beijing, Shanghai"}}},
          {"date",
           {{"type", "string"},
            {"description", "Date in YYYY-MM-DD format"}}}}},
        {"required", {"city"}}};

    JsonFunction weather_func("get_weather",
                              "Get the weather information for a given city",
                              weather_params);
    weather_tool_ = JsonTool("function", weather_func);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}},
          {"precision",
           {{"type", "number"}, {"description", "Number of decimal places"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);
    calculator_tool_ = JsonTool("function", calculator_func);

    tools_ = {weather_tool_, calculator_tool_};
  }

  std::unique_ptr<Glm47Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

// Test constructor and basic properties
TEST_F(Glm47DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection (GLM-4.7 compact format)
  std::string text_with_tool_call =
      "Some text "
      "<tool_call>test<arg_key>param</arg_key><arg_value>value</arg_value></"
      "tool_call>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(Glm47DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>Following content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</"
      "arg_value></tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without closing"));
}

// Test single tool call parsing (GLM-4.7 compact format)
TEST_F(Glm47DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test multiple tool calls parsing (GLM-4.7 format)
TEST_F(Glm47DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>"
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate");
  ASSERT_EQ(result.calls.size(), 2);  // Use ASSERT to stop test if this fails

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, -1);
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["city"], "上海");
  EXPECT_EQ(params1["date"], "2024-06-27");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, -1);
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "get_weather");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["city"], "北京");
  EXPECT_EQ(params2["date"], "2024-06-27");
}

// Test GLM-4.7 specific compact format
TEST_F(Glm47DetectorTest, Glm47CompactFormat) {
  std::string text =
      "Weather query "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Weather query");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
}

TEST_F(Glm47DetectorTest, Glm52AutoParserUsesGlm5Format) {
  EXPECT_EQ(FunctionCallParser::get_parser_auto("auto", "glm_moe_dsa"), "glm5");
  EXPECT_EQ(FunctionCallParser::get_parser_auto("auto", "glm_moe_dsa_mtp"),
            "glm5");
}

TEST_F(Glm47DetectorTest, Glm52CompactXmlToolCallParsing) {
  std::string text =
      "Need current weather "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>date</arg_key><arg_value>2026-06-26</arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm5");
  auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Need current weather");
  ASSERT_EQ(calls.size(), 1);

  const ToolCallItem& call = calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["date"], "2026-06-26");
}

TEST_F(Glm47DetectorTest, KeepsNormalTextBetweenAndAfterToolCalls) {
  std::string text =
      "First "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>"
      " middle "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>1 + 2</"
      "arg_value></tool_call>"
      " done";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "First  middle  done");
  ASSERT_EQ(result.calls.size(), 2);
  EXPECT_EQ(result.calls[0].name.value(), "get_weather");
  EXPECT_EQ(result.calls[1].name.value(), "calculate");
}

TEST_F(Glm47DetectorTest, DropsTrailingIncompleteToolCallFromNormalText) {
  std::string text =
      "First "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>"
      " after "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>1 +";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "First  after");
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value(), "get_weather");
}

// Test number type coercion
TEST_F(Glm47DetectorTest, NumberTypeCoercion) {
  std::string text =
      "Calculate with precision "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>3.14 * "
      "2</arg_value>"
      "<arg_key>precision</arg_key><arg_value>2</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Calculate with precision");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "calculate");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["expression"], "3.14 * 2");
  // precision should be parsed as number
  EXPECT_TRUE(params["precision"].is_number());
  EXPECT_EQ(params["precision"], 2);
}

// Test empty tool call content
TEST_F(Glm47DetectorTest, EmptyToolCallContent) {
  std::string text = "Test empty content <tool_call>test</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call
TEST_F(Glm47DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Glm47DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool "
      "<tool_call>unknown_tool<arg_key>param</arg_key><arg_value>value</"
      "arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(Glm47DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(Glm47DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test complex nested JSON parameters
TEST_F(Glm47DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>options</arg_key><arg_value>{\"include_forecast\": true, "
      "\"days\": 7}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
}

// Test special characters handling
TEST_F(Glm47DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York "
      "City</arg_value>"
      "<arg_key>note</arg_key><arg_value>Contains "
      "symbols！@#$%^&*()_+=</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test whitespace handling in arg values
TEST_F(Glm47DetectorTest, WhitespaceHandlingInArgValues) {
  std::string text =
      "Whitespace test "
      "<tool_call>get_weather<arg_key>  city  </arg_key><arg_value>  Beijing  "
      "</arg_value>"
      "<arg_key>\t\ndate\r\n</arg_key><arg_value>\n\t2024-06-27\r\n</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Whitespace test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");  // Whitespace should be trimmed
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test streaming parsing functionality
TEST_F(Glm47DetectorTest, StreamingParseBasicFunctionality) {
  std::string chunk1 = "<tool_call>get_weather";
  std::string chunk2 = "<arg_key>city</arg_key><arg_value>";
  std::string chunk3 = "Beijing</arg_value></tool_call>";

  // First chunk - function name not yet complete (no <arg_key> or </tool_call>)
  // Should wait for more data to avoid sending partial names
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - now we have <arg_key>, so function name is complete
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_EQ(result2.calls.size(), 1);
  EXPECT_TRUE(result2.calls[0].name.has_value());
  EXPECT_EQ(result2.calls[0].name.value(), "get_weather");

  // Third chunk - completes the tool call
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 0);  // Should complete the JSON
}

// Test streaming parse with normal text
TEST_F(Glm47DetectorTest, StreamingParseWithNormalText) {
  std::string chunk1 = "Please check the weather ";
  std::string chunk2 = "<tool_call>get_weather";
  std::string chunk3 =
      "<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>";

  // First chunk - normal text should be returned
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  EXPECT_EQ(result1.normal_text, "Please check the weather ");
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - tool call start, but function name not yet complete
  // (no <arg_key> or </tool_call>), should wait for more data
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  EXPECT_EQ(result2.calls.size(), 0);

  // Third chunk - now we have <arg_key>, function name is complete
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 1);
  EXPECT_EQ(result3.calls[0].name.value(), "get_weather");
}

// Test invalid JSON in arg values
TEST_F(Glm47DetectorTest, InvalidJsonInArgValues) {
  std::string text =
      "Invalid JSON test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{invalid json}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Invalid JSON test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"], "{invalid json}");  // Should be treated as string
}

// Test nested braces in JSON values
TEST_F(Glm47DetectorTest, NestedBracesInJsonValues) {
  std::string text =
      "Nested braces test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{\"nested\": {\"deep\": "
      "\"value\"}}</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Nested braces test");
  ASSERT_EQ(result.calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

// Test performance with many tool calls
TEST_F(Glm47DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test ";

  // Build text containing multiple tool calls
  for (int i = 0; i < 100; ++i) {
    text += "<tool_call>calculate<arg_key>expression</arg_key><arg_value>" +
            std::to_string(i) + " + " + std::to_string(i + 1) +
            "</arg_value></tool_call>";
  }

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test");
  ASSERT_EQ(result.calls.size(), 100);  // Use ASSERT to stop test if this fails

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 100; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, -1);
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

// Regression test for issue #751: std::regex stack overflow on large payloads
// The old regex implementation with [\s\S]*? pattern caused O(n) recursion
// depth, leading to stack overflow on inputs larger than ~46KB (depending on
// stack size). The fix uses string::find() and substr() for O(1) stack usage.
TEST_F(Glm47DetectorTest, LargePayloadNoStackOverflow) {
  // Create a tool that accepts large content
  nlohmann::json write_params = {
      {"type", "object"},
      {"properties",
       {{"filename", {{"type", "string"}, {"description", "Filename"}}},
        {"content",
         {{"type", "string"}, {"description", "Content to write"}}}}},
      {"required", {"filename", "content"}}};

  JsonFunction write_func("write_file", "Write content to file", write_params);
  JsonTool write_tool("function", write_func);
  std::vector<JsonTool> tools = {write_tool};

  // Test with 50KB payload (larger than the ~46KB that caused stack overflow)
  std::string large_content(50000, 'A');
  std::string text =
      "Test "
      "<tool_call>write_file"
      "<arg_key>filename</arg_key><arg_value>test.txt</arg_value>"
      "<arg_key>content</arg_key><arg_value>" +
      large_content +
      "</arg_value>"
      "</tool_call>";

  // This would crash with stack overflow before the fix
  auto result = detector_->detect_and_parse(text, tools);

  EXPECT_EQ(result.normal_text, "Test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "write_file");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["filename"], "test.txt");
  EXPECT_EQ(params["content"].get<std::string>().size(), 50000);
}

// Test with Chinese content to validate UTF-8 handling with large payloads
TEST_F(Glm47DetectorTest, LargeChinesePayloadNoStackOverflow) {
  nlohmann::json write_params = {
      {"type", "object"},
      {"properties",
       {{"filename", {{"type", "string"}}}, {"content", {{"type", "string"}}}}},
      {"required", {"filename", "content"}}};

  JsonFunction write_func("write_file", "Write content to file", write_params);
  JsonTool write_tool("function", write_func);
  std::vector<JsonTool> tools = {write_tool};

  // Generate ~50KB of Chinese content (each char is 3 bytes in UTF-8)
  std::string chinese_char = "测";  // 3 bytes in UTF-8
  std::string large_content;
  large_content.reserve(50000);
  for (int i = 0; i < 16667; ++i) {  // 16667 * 3 ≈ 50KB
    large_content += chinese_char;
  }

  std::string text =
      "<tool_call>write_file"
      "<arg_key>filename</arg_key><arg_value>中文.txt</arg_value>"
      "<arg_key>content</arg_key><arg_value>" +
      large_content +
      "</arg_value>"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools);

  ASSERT_EQ(result.calls.size(), 1);
  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "write_file");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["filename"], "中文.txt");
  // Verify content length (each Chinese char is 3 bytes)
  EXPECT_GE(params["content"].get<std::string>().size(), 49000);
}

// Test streaming with large payload
TEST_F(Glm47DetectorTest, StreamingLargePayloadNoStackOverflow) {
  nlohmann::json write_params = {
      {"type", "object"},
      {"properties", {{"content", {{"type", "string"}}}}},
      {"required", {"content"}}};

  JsonFunction write_func("write_file", "Write content", write_params);
  JsonTool write_tool("function", write_func);
  std::vector<JsonTool> tools = {write_tool};

  // Build a large tool call incrementally (simulating streaming)
  std::string large_content(20000, 'X');  // 20KB for streaming test
  std::string full_text =
      "<tool_call>write_file"
      "<arg_key>content</arg_key><arg_value>" +
      large_content +
      "</arg_value>"
      "</tool_call>";

  // Simulate streaming by sending chunks
  // Create fresh detector for streaming test
  auto streaming_detector = std::make_unique<Glm47Detector>();
  std::vector<StreamingParseResult> results;

  size_t chunk_size = 1000;  // 1KB chunks
  for (size_t i = 0; i < full_text.size(); i += chunk_size) {
    std::string chunk = full_text.substr(i, chunk_size);
    auto result = streaming_detector->parse_streaming_increment(chunk, tools);
    results.push_back(result);
  }

  // Verify we got the function name and completed without crash
  bool found_name = false;
  bool found_args = false;
  for (const auto& r : results) {
    for (const auto& call : r.calls) {
      if (call.name.has_value() && call.name.value() == "write_file") {
        found_name = true;
      }
      if (!call.parameters.empty()) {
        found_args = true;
      }
    }
  }

  EXPECT_TRUE(found_name);
  EXPECT_TRUE(found_args);
}

// =============================================================================
// UTF-8 Streaming Tests
// =============================================================================
// These tests verify that multi-byte UTF-8 characters are handled correctly
// when split across streaming chunks. The fix buffers incomplete UTF-8
// sequences until the next chunk completes them.

// Test streaming with Chinese characters split across chunks
// "北京" = 0xE5 0x8C 0x97 (北) + 0xE4 0xBA 0xAC (京)
TEST_F(Glm47DetectorTest, StreamingParseWithChineseCharactersSplit) {
  // Chunk 1: Function name and key, start of value
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";

  // Chunk 2: First 2 bytes of "北" (0xE5 0x8C) - incomplete 3-byte sequence
  std::string chunk2 = "\xE5\x8C";

  // Chunk 3: Last byte of "北" (0x97) + first 2 bytes of "京" (0xE4 0xBA)
  std::string chunk3 = "\x97\xE4\xBA";

  // Chunk 4: Last byte of "京" (0xAC) + closing tags
  std::string chunk4 = "\xAC</arg_value></tool_call>";

  // Process each chunk - should not crash with UTF-8 errors
  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  // Function name is returned when <arg_key> is seen
  EXPECT_GE(result1.calls.size(), 1);
  if (result1.calls.size() > 0) {
    EXPECT_TRUE(result1.calls[0].name.has_value());
    EXPECT_EQ(result1.calls[0].name.value(), "get_weather");
  }

  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  // Should buffer incomplete UTF-8, not crash

  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  // Should complete first character and buffer second incomplete

  auto result4 = detector_->parse_streaming_increment(chunk4, tools_);
  // Should complete the tool call
  EXPECT_GE(result4.calls.size(), 1);
}

// Test with single Chinese character split (simpler case)
TEST_F(Glm47DetectorTest, StreamingParseWithSingleChineseCharacterSplit) {
  // "北" = 0xE5 0x8C 0x97 (3-byte UTF-8)
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";
  std::string chunk2 = "\xE5\x8C";  // First 2 bytes (incomplete)
  std::string chunk3 = "\x97</arg_value></tool_call>";  // Last byte + closing

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  // Should not crash with "invalid UTF-8 byte at index 1: 0xE5"

  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 1);
}

// Test with 4-byte UTF-8 emoji split
// Emoji "😊" = 0xF0 0x9F 0x98 0x8A (4-byte UTF-8)
TEST_F(Glm47DetectorTest, StreamingParseWithEmojiSplit) {
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo ";
  std::string chunk2 = "\xF0\x9F";  // First 2 bytes of emoji (incomplete)
  std::string chunk3 =
      "\x98\x8A</arg_value></tool_call>";  // Last 2 bytes + closing

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  // Should buffer incomplete 4-byte sequence

  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 1);
}

// Test with complete UTF-8 characters (no splitting needed)
TEST_F(Glm47DetectorTest, StreamingParseWithCompleteUtf8) {
  // Complete Chinese characters in single chunks
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";
  std::string chunk2 = "北京";  // Complete UTF-8 characters
  std::string chunk3 = "</arg_value></tool_call>";

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  // Function name is returned when <arg_key> is seen
  EXPECT_GE(result1.calls.size(), 1);
  if (result1.calls.size() > 0) {
    EXPECT_TRUE(result1.calls[0].name.has_value());
    EXPECT_EQ(result1.calls[0].name.value(), "get_weather");
  }

  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);

  EXPECT_GE(result3.calls.size(), 1);
}

// Test with mixed UTF-8 and ASCII
TEST_F(Glm47DetectorTest, StreamingParseWithMixedUtf8AndAscii) {
  // Mix of ASCII and Chinese with UTF-8 split at boundary
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>City: ";
  std::string chunk2 = "\xE5";  // First byte of "北" (incomplete)
  std::string chunk3 =
      "\x8C\x97京</arg_value></tool_call>";  // Rest of "北" + complete "京"

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  // Should buffer the single incomplete byte

  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  EXPECT_GE(result3.calls.size(), 1);
}

// Test with Japanese characters (also 3-byte UTF-8)
// "東京" = 0xE6 0x9D 0xB1 (東) + 0xE4 0xBA 0xAC (京)
TEST_F(Glm47DetectorTest, StreamingParseWithJapaneseCharactersSplit) {
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";
  std::string chunk2 = "\xE6\x9D";  // First 2 bytes of "東" (incomplete)
  std::string chunk3 = "\xB1\xE4\xBA\xAC</arg_value></tool_call>";  // Rest

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);

  EXPECT_GE(result3.calls.size(), 1);
}

// Test with Korean characters (also 3-byte UTF-8)
// "서울" = 0xEC 0x84 0x9C (서) + 0xEC 0x9A 0xB8 (울)
TEST_F(Glm47DetectorTest, StreamingParseWithKoreanCharactersSplit) {
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";
  std::string chunk2 = "\xEC\x84";  // First 2 bytes of "서" (incomplete)
  std::string chunk3 = "\x9C\xEC\x9A\xB8</arg_value></tool_call>";  // Rest

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);

  EXPECT_GE(result3.calls.size(), 1);
}

// Test UTF-8 boundary: one byte at a time for 3-byte char
TEST_F(Glm47DetectorTest, StreamingParseWithUtf8OneByteAtATime) {
  // Split "北" (0xE5 0x8C 0x97) one byte at a time
  std::string chunk1 =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>";
  std::string chunk2 = "\xE5";                          // First byte
  std::string chunk3 = "\x8C";                          // Second byte
  std::string chunk4 = "\x97</arg_value></tool_call>";  // Third byte + closing

  auto result1 = detector_->parse_streaming_increment(chunk1, tools_);
  auto result2 = detector_->parse_streaming_increment(chunk2, tools_);
  auto result3 = detector_->parse_streaming_increment(chunk3, tools_);
  auto result4 = detector_->parse_streaming_increment(chunk4, tools_);

  EXPECT_GE(result4.calls.size(), 1);
}

// Test UTF-8 in non-streaming (full parse) mode still works
TEST_F(Glm47DetectorTest, NonStreamingParseWithUtf8) {
  std::string text =
      "Query for city "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Query for city");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

}  // namespace function_call
}  // namespace xllm
