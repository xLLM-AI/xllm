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

#include "qwen3_coder_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core_types.h"
#include "function_call_parser.h"

namespace xllm {
namespace function_call {

namespace {

struct StreamCallAccumulator {
  std::string name;
  std::string parameters;
};

void merge_stream_result(const StreamingParseResult& result,
                         std::string* normal_text,
                         std::vector<StreamCallAccumulator>* calls) {
  if (normal_text != nullptr && !result.normal_text.empty()) {
    *normal_text += result.normal_text;
  }

  if (calls == nullptr) {
    return;
  }

  for (const auto& call : result.calls) {
    if (call.tool_index < 0) {
      continue;
    }

    while (calls->size() <= static_cast<size_t>(call.tool_index)) {
      calls->push_back(StreamCallAccumulator());
    }

    auto& acc = (*calls)[call.tool_index];
    if (call.name.has_value()) {
      acc.name = call.name.value();
    }
    if (!call.parameters.empty()) {
      acc.parameters += call.parameters;
    }
  }
}

}  // namespace

class Qwen3CoderDetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Qwen3CoderDetector>();

    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"location", {{"type", "string"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}},
          {"days", {{"type", "integer"}}},
          {"temperature", {{"type", "number"}}},
          {"metadata", {{"type", "object"}}}}}};
    tools_.emplace_back(
        "function",
        JsonFunction(
            "get_current_weather", "Get weather info", weather_params));

    nlohmann::json sql_params = {{"type", "object"},
                                 {"properties",
                                  {{"query", {{"type", "string"}}},
                                   {"dry_run", {{"type", "boolean"}}}}}};
    tools_.emplace_back("function",
                        JsonFunction("sql_interpreter", "Run SQL", sql_params));

    nlohmann::json todo_params = {
        {"type", "object"}, {"properties", {{"todos", {{"type", "array"}}}}}};
    tools_.emplace_back(
        "function", JsonFunction("TodoWrite", "Write TODO items", todo_params));
  }

  std::unique_ptr<Qwen3CoderDetector> detector_;
  std::vector<JsonTool> tools_;
};

// -----------------------------------------------------------------------------
// Basic behavior
// -----------------------------------------------------------------------------

TEST_F(Qwen3CoderDetectorTest, ConstructorInitializesCorrectly) {
  ASSERT_NE(detector_, nullptr);

  std::string text_with_tool_call =
      "Some text <tool_call><function=test></function></tool_call>";
  std::string text_without_tool_call =
      "Just normal text without any tool calls";

  EXPECT_TRUE(detector_->has_tool_call(text_with_tool_call));
  EXPECT_FALSE(detector_->has_tool_call(text_without_tool_call));
}

TEST_F(Qwen3CoderDetectorTest, HasToolCallDetection) {
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>"));
  EXPECT_TRUE(
      detector_->has_tool_call("prefix <tool_call> middle </tool_call>"));

  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("regular text only"));
  EXPECT_FALSE(detector_->has_tool_call("<function=get_current_weather>"));
}

TEST_F(Qwen3CoderDetectorTest, PlainTextOnly) {
  std::string text = "This is plain text without any tool calls.";
  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, text);
  EXPECT_TRUE(result.calls.empty());
}

TEST_F(Qwen3CoderDetectorTest, SingleToolCallParsing) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>Boston</parameter>\n"
      "<parameter=unit>celsius</parameter>\n"
      "<parameter=days>3</parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];
  EXPECT_EQ(call.tool_index, 0);
  ASSERT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Boston");
  EXPECT_EQ(params["unit"], "celsius");
  EXPECT_EQ(params["days"], 3);
}

TEST_F(Qwen3CoderDetectorTest, SingleToolCallWithTextPrefix) {
  std::string text =
      "Let me check this for you.\n\n"
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>New York</parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  EXPECT_EQ(result.normal_text, "Let me check this for you.\n\n");
  ASSERT_EQ(result.calls.size(), 1);
  ASSERT_TRUE(result.calls[0].name.has_value());
  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
}

TEST_F(Qwen3CoderDetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>New York</parameter>\n"
      "</function>\n"
      "</tool_call>\n"
      "<tool_call>\n"
      "<function=sql_interpreter>\n"
      "<parameter=query>SELECT * FROM users</parameter>\n"
      "<parameter=dry_run>True</parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 2);

  ASSERT_TRUE(result.calls[0].name.has_value());
  ASSERT_TRUE(result.calls[1].name.has_value());
  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
  EXPECT_EQ(result.calls[1].name.value(), "sql_interpreter");

  nlohmann::json params1 = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_EQ(params1["location"], "New York");

  nlohmann::json params2 = nlohmann::json::parse(result.calls[1].parameters);
  EXPECT_EQ(params2["query"], "SELECT * FROM users");
  EXPECT_EQ(params2["dry_run"], true);
}

TEST_F(Qwen3CoderDetectorTest, MultipleFunctionsInOneToolCallBlock) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>Paris</parameter>\n"
      "</function>\n"
      "<function=sql_interpreter>\n"
      "<parameter=query>SELECT 1</parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 2);

  EXPECT_EQ(result.calls[0].name.value_or(""), "get_current_weather");
  EXPECT_EQ(result.calls[1].name.value_or(""), "sql_interpreter");
}

TEST_F(Qwen3CoderDetectorTest, ParseWithoutToolCallWrapperFallback) {
  std::string text =
      "Prefix text\n"
      "<function=get_current_weather>\n"
      "<parameter=location>Tokyo</parameter>\n"
      "</function>";

  auto result = detector_->detect_and_parse(text, tools_);
  EXPECT_EQ(result.normal_text, "Prefix text\n");
  ASSERT_EQ(result.calls.size(), 1);

  ASSERT_TRUE(result.calls[0].name.has_value());
  EXPECT_EQ(result.calls[0].name.value(), "get_current_weather");
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_EQ(params["location"], "Tokyo");
}

TEST_F(Qwen3CoderDetectorTest, UnknownToolIsStillParsedLikeSgLang) {
  std::string text =
      "<tool_call>\n"
      "<function=unknown_tool>\n"
      "<parameter=x>42</parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name.value_or(""), "unknown_tool");

  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  // Unknown schema defaults to string type.
  EXPECT_EQ(params["x"], "42");
}

TEST_F(Qwen3CoderDetectorTest, EmptyParameterValue) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location></parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);

  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_EQ(params["location"], "");
}

TEST_F(Qwen3CoderDetectorTest, SpecialCharactersInParameterValue) {
  std::string text =
      "<tool_call>\n"
      "<function=sql_interpreter>\n"
      "<parameter=query>SELECT * FROM users WHERE name = 'John \"Doe\"'</"
      "parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_NE(params["query"].get<std::string>().find("John"), std::string::npos);
  EXPECT_NE(params["query"].get<std::string>().find("Doe"), std::string::npos);
}

TEST_F(Qwen3CoderDetectorTest, IncompleteToolCallDoesNotCrash) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>London";

  auto result = detector_->detect_and_parse(text, tools_);
  EXPECT_GE(result.calls.size(), 0U);
}

// -----------------------------------------------------------------------------
// Type conversion behavior
// -----------------------------------------------------------------------------

TEST_F(Qwen3CoderDetectorTest, IntegerParameterConversion) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Tokyo</parameter>"
      "<parameter=days>5</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["days"].is_number_integer());
  EXPECT_EQ(params["days"], 5);
}

TEST_F(Qwen3CoderDetectorTest, InvalidIntegerFallsBackToString) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Tokyo</parameter>"
      "<parameter=days>five</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["days"].is_string());
  EXPECT_EQ(params["days"], "five");
}

TEST_F(Qwen3CoderDetectorTest, NumberParameterConversion) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Tokyo</parameter>"
      "<parameter=temperature>12.5</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["temperature"].is_number_float());
  EXPECT_DOUBLE_EQ(params["temperature"], 12.5);
}

TEST_F(Qwen3CoderDetectorTest, NumberIntegerLikeStringConvertedToInt) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Tokyo</parameter>"
      "<parameter=temperature>12</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["temperature"].is_number_integer());
  EXPECT_EQ(params["temperature"], 12);
}

TEST_F(Qwen3CoderDetectorTest, BooleanParameterConversion) {
  std::string text =
      "<tool_call><function=sql_interpreter>"
      "<parameter=query>SELECT 1</parameter>"
      "<parameter=dry_run>True</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["dry_run"].is_boolean());
  EXPECT_EQ(params["dry_run"], true);
}

TEST_F(Qwen3CoderDetectorTest, InvalidBooleanFallsBackToFalse) {
  std::string text =
      "<tool_call><function=sql_interpreter>"
      "<parameter=query>SELECT 1</parameter>"
      "<parameter=dry_run>not_bool</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["dry_run"].is_boolean());
  EXPECT_EQ(params["dry_run"], false);
}

TEST_F(Qwen3CoderDetectorTest, NullValueConversion) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>null</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["location"].is_null());
}

TEST_F(Qwen3CoderDetectorTest, ObjectAndArrayConversion) {
  std::string text =
      "<tool_call>\n"
      "<function=get_current_weather>\n"
      "<parameter=location>Beijing</parameter>\n"
      "<parameter=metadata>{\"source\":\"api\",\"retry\":1}</parameter>\n"
      "</function>\n"
      "</tool_call>\n"
      "<tool_call>\n"
      "<function=TodoWrite>\n"
      "<parameter=todos>[{\"content\":\"A\",\"status\":\"pending\"}]</"
      "parameter>\n"
      "</function>\n"
      "</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 2);

  nlohmann::json params1 = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params1["metadata"].is_object());
  EXPECT_EQ(params1["metadata"]["source"], "api");

  nlohmann::json params2 = nlohmann::json::parse(result.calls[1].parameters);
  EXPECT_TRUE(params2["todos"].is_array());
  EXPECT_EQ(params2["todos"][0]["content"], "A");
}

TEST_F(Qwen3CoderDetectorTest, InvalidObjectFallsBackToString) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Beijing</parameter>"
      "<parameter=metadata>{invalid_json}</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["metadata"].is_string());
  EXPECT_EQ(params["metadata"], "{invalid_json}");
}

TEST_F(Qwen3CoderDetectorTest, UnknownParameterFallsBackToRawString) {
  std::string text =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Berlin</parameter>"
      "<parameter=unexpected_field>123</parameter>"
      "</function></tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);
  ASSERT_EQ(result.calls.size(), 1);
  nlohmann::json params = nlohmann::json::parse(result.calls[0].parameters);
  EXPECT_TRUE(params["unexpected_field"].is_string());
  EXPECT_EQ(params["unexpected_field"], "123");
}

// -----------------------------------------------------------------------------
// Streaming behavior
// -----------------------------------------------------------------------------

TEST_F(Qwen3CoderDetectorTest, StreamingSingleToolCall) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {"<tool_call>",
                                     "<function=get_current_weather>",
                                     "<parameter=location>",
                                     "Boston",
                                     "</parameter>",
                                     "<parameter=unit>celsius</parameter>",
                                     "<parameter=days>3</parameter>",
                                     "</function>",
                                     "</tool_call>"};

  std::string normal_text;
  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(
        parser.parse_streaming_increment(chunk), &normal_text, &acc);
  }

  EXPECT_TRUE(normal_text.empty());
  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "get_current_weather");
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Boston");
  EXPECT_EQ(params["unit"], "celsius");
  EXPECT_EQ(params["days"], 3);
}

TEST_F(Qwen3CoderDetectorTest, StreamingMultipleToolCalls) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {
      "<tool_call><function=get_current_weather><parameter=location>Paris</"
      "parameter></function></tool_call>",
      "<tool_call><function=sql_interpreter><parameter=query>SELECT 1</"
      "parameter><parameter=dry_run>false</parameter></function></tool_call>"};

  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);
  }

  ASSERT_EQ(acc.size(), 2);
  EXPECT_EQ(acc[0].name, "get_current_weather");
  EXPECT_EQ(acc[1].name, "sql_interpreter");

  nlohmann::json params0 = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params0["location"], "Paris");

  nlohmann::json params1 = nlohmann::json::parse(acc[1].parameters);
  EXPECT_EQ(params1["query"], "SELECT 1");
  EXPECT_EQ(params1["dry_run"], false);
}

TEST_F(Qwen3CoderDetectorTest, StreamingTextAndToolCall) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {"Let me ",
                                     "help you.\n\n",
                                     "<tool_call>",
                                     "<function=get_current_weather>",
                                     "<parameter=location>Paris</parameter>",
                                     "</function>",
                                     "</tool_call>"};

  std::string normal_text;
  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(
        parser.parse_streaming_increment(chunk), &normal_text, &acc);
  }

  EXPECT_EQ(normal_text, "Let me help you.\n\n");
  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "get_current_weather");
}

TEST_F(Qwen3CoderDetectorTest, StreamingParameterEndedByNextParameter) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::string chunk =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Boston"
      "<parameter=unit>celsius</parameter>"
      "</function></tool_call>";

  std::vector<StreamCallAccumulator> acc;
  merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);

  ASSERT_EQ(acc.size(), 1);
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Boston");
  EXPECT_EQ(params["unit"], "celsius");
}

TEST_F(Qwen3CoderDetectorTest, StreamingParameterEndedByFunctionEnd) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::string chunk =
      "<tool_call><function=get_current_weather>"
      "<parameter=location>Boston"
      "</function></tool_call>";

  std::vector<StreamCallAccumulator> acc;
  merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);

  ASSERT_EQ(acc.size(), 1);
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Boston");
}

TEST_F(Qwen3CoderDetectorTest,
       StreamingFunctionWithoutParametersEmitsEmptyJson) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::string chunk =
      "<tool_call><function=sql_interpreter></function></tool_call>";

  std::vector<StreamCallAccumulator> acc;
  merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);

  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "sql_interpreter");
  EXPECT_EQ(acc[0].parameters, "{}");
}

TEST_F(Qwen3CoderDetectorTest, StreamingIgnoresTextInsideToolCallRegion) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {"before ",
                                     "<tool_call>\n",
                                     "THIS_SHOULD_BE_IGNORED",
                                     "<function=get_current_weather>",
                                     "<parameter=location>Rome</parameter>",
                                     "</function>",
                                     "</tool_call>",
                                     " after"};

  std::string normal_text;
  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(
        parser.parse_streaming_increment(chunk), &normal_text, &acc);
  }

  EXPECT_EQ(normal_text, "before  after");
  ASSERT_EQ(acc.size(), 1);
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Rome");
}

TEST_F(Qwen3CoderDetectorTest,
       StreamingKeepsLiteralAngleBracketOutsideToolCall) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {"2 < 3 and ", "5 > 4"};

  std::string normal_text;
  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(
        parser.parse_streaming_increment(chunk), &normal_text, &acc);
  }

  EXPECT_EQ(normal_text, "2 < 3 and 5 > 4");
  EXPECT_TRUE(acc.empty());
}

TEST_F(Qwen3CoderDetectorTest, StreamingPartialTagWaitsForMoreData) {
  FunctionCallParser parser(tools_, "qwen3_coder");

  auto result1 = parser.parse_streaming_increment("prefix <tool_ca");
  EXPECT_EQ(result1.normal_text, "prefix ");
  EXPECT_TRUE(result1.calls.empty());

  auto result2 = parser.parse_streaming_increment(
      "ll><function=get_current_weather><parameter=location>Paris</parameter>"
      "</function></tool_call>");
  std::vector<StreamCallAccumulator> acc;
  merge_stream_result(result2, nullptr, &acc);

  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "get_current_weather");
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Paris");
}

TEST_F(Qwen3CoderDetectorTest,
       StreamingUnknownTagInsideToolCallDoesNotBreakParsing) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {
      "<tool_call><unknown>abc</unknown><function=get_current_weather>",
      "<parameter=location>Madrid</parameter></function></tool_call>"};

  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);
  }

  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "get_current_weather");
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Madrid");
}

TEST_F(Qwen3CoderDetectorTest,
       StreamingTextAfterToolCallIsReturnedAsNormalText) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::vector<std::string> chunks = {
      "<tool_call><function=get_current_weather><parameter=location>Seoul</"
      "parameter></function></tool_call>",
      " done"};

  std::string normal_text;
  std::vector<StreamCallAccumulator> acc;
  for (const auto& chunk : chunks) {
    merge_stream_result(
        parser.parse_streaming_increment(chunk), &normal_text, &acc);
  }

  EXPECT_EQ(normal_text, " done");
  ASSERT_EQ(acc.size(), 1);
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Seoul");
}

TEST_F(Qwen3CoderDetectorTest, StreamingCharacterByCharacter) {
  FunctionCallParser parser(tools_, "qwen3_coder");
  std::string text =
      "<tool_call><function=get_current_weather><parameter=location>Tokyo</"
      "parameter><parameter=days>2</parameter></function></tool_call>";

  std::vector<StreamCallAccumulator> acc;
  for (char ch : text) {
    std::string chunk(1, ch);
    merge_stream_result(parser.parse_streaming_increment(chunk), nullptr, &acc);
  }

  ASSERT_EQ(acc.size(), 1);
  EXPECT_EQ(acc[0].name, "get_current_weather");
  nlohmann::json params = nlohmann::json::parse(acc[0].parameters);
  EXPECT_EQ(params["location"], "Tokyo");
  EXPECT_EQ(params["days"], 2);
}

// -----------------------------------------------------------------------------
// Robustness / scale
// -----------------------------------------------------------------------------

TEST_F(Qwen3CoderDetectorTest, PerformanceWithManyToolCalls) {
  std::string text = "Performance test ";
  for (int i = 0; i < 300; ++i) {
    text += "<tool_call><function=sql_interpreter><parameter=query>SELECT " +
            std::to_string(i) + "</parameter></function></tool_call>";
  }

  auto result = detector_->detect_and_parse(text, tools_);
  EXPECT_EQ(result.normal_text, "Performance test ");
  ASSERT_EQ(result.calls.size(), 300);
  for (int i = 0; i < 300; ++i) {
    EXPECT_EQ(result.calls[i].name.value_or(""), "sql_interpreter");
  }
}

}  // namespace function_call
}  // namespace xllm
