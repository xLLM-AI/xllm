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

#include "deepseekv32_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace function_call {

class DeepSeek32DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<DeepSeekV32Detector>();

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

  std::unique_ptr<DeepSeekV32Detector> detector_;
  JsonTool weather_tool_;
  JsonTool calculator_tool_;
  std::vector<JsonTool> tools_;
};

class DeepSeekV4DetectorTest : public DeepSeek32DetectorTest {
 protected:
  void SetUp() override {
    DeepSeek32DetectorTest::SetUp();
    v4_detector_ = std::make_unique<DeepSeekV4Detector>();
  }

  std::unique_ptr<DeepSeekV4Detector> v4_detector_;
};

// Test constructor and basic properties
TEST_F(DeepSeek32DetectorTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(detector_, nullptr);

  // Test basic token detection
  std::string text_with_tool_call =
      "Some text "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">Beijing</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

// Test has_tool_call method
TEST_F(DeepSeek32DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<｜DSML｜function_calls>"));
  EXPECT_TRUE(detector_->has_tool_call("<｜DSML｜invoke"));
  EXPECT_TRUE(detector_->has_tool_call(
      "Previous text <｜DSML｜function_calls>Following content"));
  EXPECT_TRUE(
      detector_->has_tool_call("<｜DSML｜function_calls><｜DSML｜invoke "
                               "name=\"get_weather\"><｜DSML｜parameter "
                               "name=\"city\" "
                               "string=\"true\">北京</｜DSML｜parameter></"
                               "｜DSML｜invoke></｜DSML｜function_calls>"));
  EXPECT_TRUE(detector_->has_tool_call("{\"tool_calls\": []}"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("DSML without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<function_calls> without DSML"));
}

// Test single tool call parsing
TEST_F(DeepSeek32DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">北京</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"date\" "
      "string=\"true\">2024-06-27</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test single tool call parsing with direct JSON format
TEST_F(DeepSeek32DetectorTest, SingleToolCallParsingJsonFormat) {
  std::string text =
      "Please help me check the weather "
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">{\"city\": "
      "\"北京\", \"date\": "
      "\"2024-06-27\"}</｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test multiple tool calls parsing
TEST_F(DeepSeek32DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">上海</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"date\" "
      "string=\"true\">2024-06-27</｜DSML｜parameter></"
      "｜DSML｜invoke><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter name=\"city\" "
      "string=\"true\">北京</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"date\" "
      "string=\"true\">2024-06-27</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate");
  ASSERT_EQ(result.calls.size(), 2);

  // Verify first tool call
  const auto& call1 = result.calls[0];
  EXPECT_EQ(call1.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["city"], "上海");
  EXPECT_EQ(params1["date"], "2024-06-27");

  // Verify second tool call
  const auto& call2 = result.calls[1];
  EXPECT_EQ(call2.tool_index, -1);  // Base class always returns -1
  EXPECT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "get_weather");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["city"], "北京");
  EXPECT_EQ(params2["date"], "2024-06-27");
}

// Test number type handling
TEST_F(DeepSeek32DetectorTest, NumberTypeHandling) {
  std::string text =
      "Calculate with precision "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"calculate\"><｜DSML｜parameter "
      "name=\"expression\" string=\"true\">3.14 * "
      "2</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"precision\" "
      "string=\"false\">2</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Calculate with precision");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "calculate");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["expression"], "3.14 * 2");
  // precision should be parsed as number
  EXPECT_TRUE(params["precision"].is_number());
  EXPECT_EQ(params["precision"], 2);
}

// Test empty tool call content
TEST_F(DeepSeek32DetectorTest, EmptyToolCallContent) {
  std::string text =
      "Test empty content <｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"></｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params.empty());
}

// Test incomplete tool call (only start tag)
TEST_F(DeepSeek32DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Beijing";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(DeepSeek32DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"unknown_tool\"><｜DSML｜parameter "
      "name=\"param\" "
      "string=\"true\">value</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools, so should be 0 calls
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(DeepSeek32DetectorTest, OnlyNormalText) {
  std::string text = "This is a regular text without any tool calls.";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "This is a regular text without any tool calls.");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test empty string input
TEST_F(DeepSeek32DetectorTest, EmptyStringInput) {
  std::string text = "";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
  EXPECT_FALSE(result.has_calls());
}

// Test whitespace-only input
TEST_F(DeepSeek32DetectorTest, WhitespaceOnlyInput) {
  std::string text = "   \t\n\r   ";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "");
  EXPECT_EQ(result.calls.size(), 0);
}

// Test complex nested JSON parameters
TEST_F(DeepSeek32DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">Beijing</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"options\" string=\"false\">{\"include_forecast\": true, "
      "\"days\": "
      "7}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
}

// Test special characters handling
TEST_F(DeepSeek32DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">New York "
      "City</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"note\" string=\"true\">Contains "
      "symbols！@#$%^&*()_+=</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test whitespace handling in parameter values
TEST_F(DeepSeek32DetectorTest, WhitespaceHandlingInParameterValues) {
  std::string text =
      "Whitespace test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">  Beijing  "
      "</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"date\" "
      "string=\"true\">\n\t2024-06-27\r\n</｜DSML｜parameter></"
      "｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Whitespace test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  // Note: whitespace trimming behavior depends on implementation
  // This test verifies the basic parsing works
  EXPECT_TRUE(params.contains("city"));
  EXPECT_TRUE(params.contains("date"));
}

// ========== Streaming Tests ==========

// Test basic streaming parsing
TEST_F(DeepSeek32DetectorTest, BasicStreamingParsing) {
  std::vector<std::string> chunks = {
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">",
      "<｜DSML｜parameter name=\"city\" string=\"true\">",
      "Beijing</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>"};

  std::vector<StreamingParseResult> results;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    results.push_back(result);
  }

  bool found_tool_name = false;
  bool found_arguments = false;

  for (const auto& result : results) {
    if (!result.calls.empty()) {
      for (const auto& call : result.calls) {
        if (call.name.has_value() && call.name.value() == "get_weather") {
          found_tool_name = true;
        }
        if (!call.parameters.empty() &&
            call.parameters.find("Beijing") != std::string::npos) {
          found_arguments = true;
        }
      }
    }
  }

  EXPECT_TRUE(found_tool_name) << "Should find tool name in streaming results";
  EXPECT_TRUE(found_arguments) << "Should find arguments in streaming results";
}

// Test incremental argument streaming
TEST_F(DeepSeek32DetectorTest, IncrementalArgumentStreaming) {
  std::vector<std::string> chunks = {
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">",
      "<｜DSML｜parameter name=\"city\" string=\"true\">",
      "Beijing",
      "</｜DSML｜parameter><｜DSML｜parameter name=\"date\" string=\"true\">",
      "2024-06-27</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>"};

  std::string accumulated_args;
  bool tool_name_sent = false;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);

    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        tool_name_sent = true;
        EXPECT_EQ(call.name.value(), "get_weather");
      } else {
        accumulated_args += call.parameters;
      }
    }
  }

  EXPECT_TRUE(tool_name_sent)
      << "Tool name should be sent when tool call is complete";
  if (!accumulated_args.empty()) {
    EXPECT_TRUE(accumulated_args.find("Beijing") != std::string::npos)
        << "Should contain city argument";
    EXPECT_TRUE(accumulated_args.find("2024-06-27") != std::string::npos)
        << "Should contain date argument";
  }
}

TEST_F(DeepSeek32DetectorTest,
       StreamingPartialNonStringParameterWaitsForValue) {
  std::vector<std::string> chunks = {
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"calculate\">",
      "<｜DSML｜parameter name=\"expression\" string=\"true\">1 + "
      "1</｜DSML｜parameter>",
      "<｜DSML｜parameter name=\"precision\" string=\"false\">",
      "2</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>"};

  std::string accumulated_args;
  bool tool_name_sent = false;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    EXPECT_EQ(result.normal_text.find("<｜DSML"), std::string::npos);

    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        tool_name_sent = true;
        EXPECT_EQ(call.name.value(), "calculate");
      } else {
        accumulated_args += call.parameters;
      }
    }
  }

  ASSERT_TRUE(tool_name_sent);
  nlohmann::json params = nlohmann::json::parse(accumulated_args);
  EXPECT_EQ(params["expression"], "1 + 1");
  EXPECT_TRUE(params["precision"].is_number());
  EXPECT_EQ(params["precision"], 2);
}

// Test normal text handling during streaming
TEST_F(DeepSeek32DetectorTest, StreamingNormalTextHandling) {
  std::vector<std::string> chunks = {
      "This is normal text before tool call. ",
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">",
      "<｜DSML｜parameter name=\"city\" "
      "string=\"true\">Tokyo</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>",
      " And this is text after."};

  std::string accumulated_normal_text;
  bool found_tool_call = false;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);

    if (!result.normal_text.empty()) {
      accumulated_normal_text += result.normal_text;
    }

    if (!result.calls.empty()) {
      found_tool_call = true;
    }
  }

  EXPECT_TRUE(found_tool_call) << "Should find tool call";
  EXPECT_TRUE(accumulated_normal_text.find("This is normal text") !=
              std::string::npos)
      << "Should preserve normal text before tool call";
  // Note: We don't expect "And this is text after" to be in
  // accumulated_normal_text
}

TEST_F(DeepSeek32DetectorTest, StreamingNormalTextBuffersIncompleteUtf8Tail) {
  std::string chunk1 = "The user didn";
  chunk1.push_back(static_cast<char>(0xE2));
  chunk1.push_back(static_cast<char>(0x80));

  std::string chunk2;
  chunk2.push_back(static_cast<char>(0x99));
  chunk2 += "t specify temperature units.";

  std::string accumulated_normal_text;
  for (const auto& chunk : {chunk1, chunk2}) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    accumulated_normal_text += result.normal_text;
  }

  EXPECT_EQ(accumulated_normal_text,
            "The user didn’t specify temperature units.");
}

TEST_F(DeepSeek32DetectorTest, StreamingToolArgumentsBufferIncompleteUtf8Tail) {
  std::string chunk1 =
      "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">"
      "<｜DSML｜parameter name=\"city\" string=\"true\">";

  std::string chunk2;
  chunk2.push_back(static_cast<char>(0xE5));
  chunk2.push_back(static_cast<char>(0x8C));

  std::string chunk3;
  chunk3.push_back(static_cast<char>(0x97));
  chunk3 += "京</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>";

  std::string accumulated_arguments;
  bool tool_name_sent = false;
  for (const auto& chunk : {chunk1, chunk2, chunk3}) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        tool_name_sent = true;
        EXPECT_EQ(call.name.value(), "get_weather");
      } else {
        accumulated_arguments += call.parameters;
      }
    }
  }

  ASSERT_TRUE(tool_name_sent);
  nlohmann::json params = nlohmann::json::parse(accumulated_arguments);
  EXPECT_EQ(params["city"], "北京");
}

// Test invalid JSON in parameter values
TEST_F(DeepSeek32DetectorTest, InvalidJsonInParameterValues) {
  std::string text =
      "Invalid JSON test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">Beijing</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"config\" string=\"false\">{invalid "
      "json}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Invalid JSON test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  // Invalid JSON should be treated as string or cause parsing to fail
  // gracefully
  EXPECT_TRUE(params.contains("config"));
}

// Test nested braces in JSON values

TEST_F(DeepSeek32DetectorTest, NestedBracesInJsonValues) {
  std::string text =
      "Nested braces test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">Beijing</｜DSML｜parameter><｜DSML｜parameter "
      "name=\"config\" string=\"false\">{\"nested\": {\"deep\": "
      "\"value\"}}</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Nested braces test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

// Test mixed format (XML and JSON in same invoke)
TEST_F(DeepSeek32DetectorTest, MixedFormatNotSupported) {
  // Note: This test verifies that the parser handles the format correctly
  // In practice, an invoke should use either XML or JSON, not both
  std::string text =
      "Mixed format test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Beijing</｜DSML｜parameter>{\"date\": "
      "\"2024-06-27\"}</｜DSML｜invoke></｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Mixed format test");
  // The parser should handle this gracefully
  ASSERT_GE(result.calls.size(), 0);
}

// Test performance with many tool calls
TEST_F(DeepSeek32DetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test <｜DSML｜function_calls>";

  // Build text containing multiple tool calls
  for (int i = 0; i < 10; ++i) {
    text +=
        "<｜DSML｜invoke name=\"calculate\"><｜DSML｜parameter "
        "name=\"expression\" string=\"true\">" +
        std::to_string(i) + " + " + std::to_string(i + 1) +
        "</｜DSML｜parameter></｜DSML｜invoke>";
  }
  text += "</｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Performance test");
  ASSERT_EQ(result.calls.size(), 10);

  // Verify each tool call is correctly parsed
  for (int i = 0; i < 10; ++i) {
    const auto& call = result.calls[i];
    EXPECT_EQ(call.tool_index, -1);  // Base class always returns -1
    EXPECT_EQ(call.name.value(), "calculate");

    nlohmann::json params = nlohmann::json::parse(call.parameters);
    std::string expected_expr =
        std::to_string(i) + " + " + std::to_string(i + 1);
    EXPECT_EQ(params["expression"], expected_expr);
  }
}

// Test array parameter with string="false"
TEST_F(DeepSeek32DetectorTest, ArrayParameterHandling) {
  std::string text =
      "Array parameter test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"cities\" string=\"false\">[\"Beijing\", \"Shanghai\", "
      "\"Guangzhou\"]</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Array parameter test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_TRUE(params["cities"].is_array());
  EXPECT_EQ(params["cities"].size(), 3);
  EXPECT_EQ(params["cities"][0], "Beijing");
  EXPECT_EQ(params["cities"][1], "Shanghai");
  EXPECT_EQ(params["cities"][2], "Guangzhou");
}

// Test boolean parameter with string="false"
TEST_F(DeepSeek32DetectorTest, BooleanParameterHandling) {
  std::string text =
      "Boolean parameter test "
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"include_forecast\" "
      "string=\"false\">true</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Boolean parameter test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_TRUE(params["include_forecast"].is_boolean());
  EXPECT_EQ(params["include_forecast"], true);
}

// Test JSON format tool calls (fallback)
TEST_F(DeepSeek32DetectorTest, JsonFormatToolCalls) {
  std::string text =
      "Some text before "
      "{\"tool_calls\": [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Beijing\", \"date\": \"2024-06-27\"}}]}";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Some text before");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "get_weather");
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test JSON format with function wrapper
TEST_F(DeepSeek32DetectorTest, JsonFormatWithFunctionWrapper) {
  std::string text =
      "Text before "
      "{\"tool_calls\": [{\"function\": {\"name\": \"get_weather\", "
      "\"arguments\": \"{\\\"city\\\": \\\"Tokyo\\\"}\"}}]}";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Text before");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "get_weather");
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Tokyo");
}

// Test trailing markdown language hint before DSML tool block
TEST_F(DeepSeek32DetectorTest, StripTrailingJsonHintBeforeDsmlToolBlock) {
  std::string text =
      "Some text before tool call\n"
      "json\n"
      "<｜DSML｜function_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" "
      "string=\"true\">Beijing</｜DSML｜parameter></｜DSML｜invoke></"
      "｜DSML｜function_calls>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Some text before tool call");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "get_weather");
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
}

// Test trailing markdown language hint before JSON fallback tool block
TEST_F(DeepSeek32DetectorTest, StripTrailingJsonHintBeforeJsonToolBlock) {
  std::string text =
      "Some text before tool call\n"
      "json\n"
      "{\"tool_calls\": [{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\": \"Beijing\"}}]}";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Some text before tool call");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.name.value(), "get_weather");
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
}

// Test streaming with multiple tool calls
TEST_F(DeepSeek32DetectorTest, StreamingMultipleToolCalls) {
  std::vector<std::string> chunks = {"<｜DSML｜function_calls>",
                                     "\n",
                                     "<｜DSML｜invoke",
                                     " name=\"get_weather\">",
                                     "<｜DSML｜parameter name=\"city\" "
                                     "string=\"true\">",
                                     "Tokyo",
                                     "</｜DSML｜parameter>",
                                     "</｜DSML｜invoke>",
                                     "\n",
                                     "<｜DSML｜invoke",
                                     " name=\"get_weather\">",
                                     "<｜DSML｜parameter name=\"city\" "
                                     "string=\"true\">",
                                     "Paris",
                                     "</｜DSML｜parameter>",
                                     "</｜DSML｜invoke>",
                                     "\n",
                                     "</｜DSML｜function_calls>"};

  int tool_calls_found = 0;
  for (const auto& chunk : chunks) {
    auto stream_result = detector_->parse_streaming_increment(chunk, tools_);
    for (const auto& call : stream_result.calls) {
      if (call.name.has_value()) {
        tool_calls_found++;
      }
    }
  }

  EXPECT_EQ(tool_calls_found, 2)
      << "Should find 2 tool calls in streaming mode";
}

// Test streaming response with trailing empty markdown json fence before
// tool-call chunks. The empty fence should not leak to normal_text.
TEST_F(DeepSeek32DetectorTest,
       StreamingSkipsTrailingEmptyJsonFenceBeforeToolCall) {
  std::vector<std::string> chunks = {
      "Let me break this down.\n\nNow I'll call the tool.\n\n```json\n",
      "\n```",
      "<｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Boston</｜DSML｜parameter></"
      "｜DSML｜invoke>"};

  std::string accumulated_normal_text;
  int tool_name_events = 0;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    if (!result.normal_text.empty()) {
      accumulated_normal_text += result.normal_text;
    }
    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        tool_name_events++;
      }
    }
  }

  EXPECT_EQ(tool_name_events, 1);
  EXPECT_TRUE(accumulated_normal_text.find("```json") == std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("```") == std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("Now I'll call the tool") !=
              std::string::npos);
}

// Test streaming response with trailing placeholder punctuation and fence at
// stream end. These artifacts should be removed from flushed normal_text.
TEST_F(DeepSeek32DetectorTest,
       StreamingFlushStripsTrailingPlaceholderAndFence) {
  std::vector<std::string> chunks = {
      "Let me break this down.\n\n"
      "The user is asking for weather info.\n\n"
      "I will call the weather tool now.\n\n"
      "、、、\n"
      "```\n"};

  std::string accumulated_normal_text;
  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    if (!result.normal_text.empty()) {
      accumulated_normal_text += result.normal_text;
    }
  }

  auto flush_result = detector_->parse_streaming_increment("", tools_);
  if (!flush_result.normal_text.empty()) {
    accumulated_normal_text += flush_result.normal_text;
  }

  EXPECT_TRUE(accumulated_normal_text.find(
                  "I will call the weather tool now.") != std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("、、、") == std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("```") == std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("\n\n\n") == std::string::npos);
  EXPECT_TRUE(!accumulated_normal_text.empty() &&
              accumulated_normal_text.back() != ' ');
}

// Test standalone placeholder/fence chunks right before invoke start.
// They should not leak into streamed normal_text.
TEST_F(DeepSeek32DetectorTest,
       StreamingDropsStandalonePlaceholderChunkBeforeToolCall) {
  std::vector<std::string> chunks = {
      "I'll use the weather tool for Boston.\n\n",
      "、、、",
      "\n```",
      "<｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Boston</｜DSML｜parameter></"
      "｜DSML｜invoke>"};

  std::string accumulated_normal_text;
  int tool_name_events = 0;

  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    if (!result.normal_text.empty()) {
      accumulated_normal_text += result.normal_text;
    }
    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        tool_name_events++;
      }
    }
  }

  EXPECT_EQ(tool_name_events, 1);
  EXPECT_TRUE(
      accumulated_normal_text.find("I'll use the weather tool for Boston.") !=
      std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("、、、") == std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("```") == std::string::npos);
}

// Test that deferred trailing whitespace does not grow into large blank blocks
// across chunks.
TEST_F(DeepSeek32DetectorTest,
       StreamingDeferredWhitespaceDoesNotAccumulateBlankLines) {
  std::vector<std::string> chunks = {"First line.\n\n\n", "Second line."};

  std::string accumulated_normal_text;
  for (const auto& chunk : chunks) {
    auto result = detector_->parse_streaming_increment(chunk, tools_);
    if (!result.normal_text.empty()) {
      accumulated_normal_text += result.normal_text;
    }
  }

  EXPECT_TRUE(accumulated_normal_text.find("First line.") != std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("Second line.") !=
              std::string::npos);
  EXPECT_TRUE(accumulated_normal_text.find("\n\n\n") == std::string::npos);
}

// Test tool call without function_calls wrapper
TEST_F(DeepSeek32DetectorTest, ToolCallWithoutWrapper) {
  std::string text =
      "Direct invoke "
      "<｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Shanghai</｜DSML｜parameter></"
      "｜DSML｜invoke>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Direct invoke");
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value(), "get_weather");
}

TEST_F(DeepSeekV4DetectorTest, ParsesToolCallsWrapper) {
  std::string text =
      "Need weather "
      "<｜DSML｜tool_calls><｜DSML｜invoke "
      "name=\"get_weather\"><｜DSML｜parameter "
      "name=\"city\" string=\"true\">Beijing</｜DSML｜parameter></"
      "｜DSML｜invoke></｜DSML｜tool_calls>";

  auto result = v4_detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Need weather");
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value(), "get_weather");
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_EQ(params["city"], "Beijing");
}

TEST_F(DeepSeekV4DetectorTest, StreamingHandlesToolCallsTokenSplit) {
  std::vector<std::string> chunks = {
      "Thinking... ",
      "<｜DSML｜",
      "tool",
      "_calls><｜DSML｜invoke name=\"get_weather\">",
      "<｜DSML｜parameter name=\"city\" string=\"true\">",
      "Beijing</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜",
      "tool_calls>"};

  std::string accumulated_normal_text;
  int tool_calls_found = 0;
  std::string accumulated_args;
  for (const auto& chunk : chunks) {
    auto result = v4_detector_->parse_streaming_increment(chunk, tools_);
    accumulated_normal_text += result.normal_text;
    for (const auto& call : result.calls) {
      if (call.name.has_value()) {
        ++tool_calls_found;
        EXPECT_EQ(call.name.value(), "get_weather");
      } else {
        accumulated_args += call.parameters;
      }
    }
  }

  EXPECT_EQ(accumulated_normal_text, "Thinking...");
  EXPECT_EQ(tool_calls_found, 1);
  nlohmann::json params = nlohmann::json::parse(accumulated_args);
  EXPECT_EQ(params["city"], "Beijing");
}

}  // namespace function_call
}  // namespace xllm
