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

#include "api_service/anthropic_request_utils.h"

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include "api_service/chat_json_parser.h"
#include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/request/request_params.h"

namespace xllm {
namespace {

proto::AnthropicMessagesRequest parse_request(const std::string& json) {
  auto [status, normalized] = ChatJsonParser::anthropic().preprocess(json);
  EXPECT_TRUE(status.ok()) << status.message();
  proto::AnthropicMessagesRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto result = google::protobuf::util::JsonStringToMessage(
      normalized, &request, options);
  EXPECT_TRUE(result.ok()) << result.ToString();
  return request;
}

TEST(AnthropicRequestUtilsTest,
     PreservesThinkingAndLinksStructuredToolResults) {
  const auto request = parse_request(R"({
    "model":"test", "max_tokens":32,
    "messages":[
      {"role":"assistant","content":[
        {"type":"thinking","thinking":"reason","signature":"opaque"},
        {"type":"redacted_thinking","data":"opaque"},
        {"type":"tool_use","id":"call_1","name":"search","input":{"q":"x"}}
      ]},
      {"role":"user","content":[
        {"type":"tool_result","tool_use_id":"call_1","id":"wrong",
         "content":[{"type":"text","text":"first"},{"type":"text","text":"second"}]},
        {"type":"text","text":"continue"}
      ]}
    ]})");
  ASSERT_TRUE(api_service::validate_anthropic_request(request).ok());
  const auto messages = api_service::build_anthropic_messages(request);
  ASSERT_EQ(messages.size(), 3);
  EXPECT_EQ(messages[0].reasoning_content, "reason");
  ASSERT_TRUE(messages[0].tool_calls.has_value());
  EXPECT_EQ(messages[0].tool_calls->at(0).id, "call_1");
  EXPECT_EQ(messages[1].role, "tool");
  EXPECT_EQ(messages[1].tool_call_id, "call_1");
  EXPECT_EQ(std::get<std::string>(messages[1].content), "first\nsecond");
  EXPECT_EQ(messages[2].role, "user");
  EXPECT_EQ(std::get<std::string>(messages[2].content), "continue");
}

TEST(AnthropicRequestUtilsTest, ToolOnlyTurnDoesNotAppendEmptyUserMessage) {
  const auto request = parse_request(R"({"messages":[{"role":"user","content":[
    {"type":"tool_result","tool_use_id":"call_1","content":""}]}]})");
  const auto messages = api_service::build_anthropic_messages(request);
  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0].role, "tool");
  EXPECT_EQ(std::get<std::string>(messages[0].content), "");
}

TEST(AnthropicRequestUtilsTest, JoinsTextBlocksForTextModelTemplates) {
  const auto request = parse_request(R"({"model":"test","max_tokens":16,
    "messages":[{"role":"user","content":[
      {"type":"text","text":"first"},{"type":"text","text":"second"}]}]})");
  ASSERT_TRUE(api_service::validate_anthropic_request(request).ok());
  const auto messages = api_service::build_anthropic_messages(request);
  ASSERT_EQ(messages.size(), 1);
  ASSERT_TRUE(std::holds_alternative<std::string>(messages[0].content));
  EXPECT_EQ(std::get<std::string>(messages[0].content), "first\nsecond");
}

TEST(AnthropicRequestUtilsTest, StripsBillingBlocksAndMergesAllSystemMessages) {
  const auto request = parse_request(R"({
    "system":[{"type":"text","text":"x-anthropic-billing-header: changing"},
               {"type":"text","text":"base"}],
    "messages":[{"role":"user","content":"first"},
                {"role":"system","content":"reminder"},
                {"role":"user","content":"next"}]})");
  auto messages = api_service::build_anthropic_messages(request);
  ASSERT_EQ(messages.size(), 3);
  EXPECT_EQ(std::get<std::string>(messages[0].content), "basereminder");
  EXPECT_EQ(messages[1].role, "user");
}

TEST(AnthropicRequestUtilsTest, ValidatesRequiredFieldsAndNamedTool) {
  auto request = parse_request(R"({"model":"test","max_tokens":16,
    "messages":[{"role":"user","content":"hi"}]})");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  request.set_max_tokens(-1);
  EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
  EXPECT_TRUE(
      api_service::validate_anthropic_request(request, /*count_tokens=*/true)
          .ok());
  request.set_max_tokens(16);
  request.mutable_tool_choice()->set_type("tool");
  EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
  request.mutable_tool_choice()->set_name("missing");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  auto* tool = request.add_tools();
  tool->set_name("missing");
  tool->mutable_input_schema();
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  request.mutable_messages(0)->set_role("bad");
  EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
}

TEST(AnthropicRequestUtilsTest, IgnoresFieldsUnknownToVllm023) {
  const auto request = parse_request(R"({"model":"test","max_tokens":4096,
    "messages":[{"role":"user","content":"hi"}],
    "thinking":"unknown even with this type",
    "cache_salt":42,"vllm_xargs":"ignored","ec_transfer_params":false,
    "tools":[{"name":"search","input_schema":{},"strict":"ignored"}],
    "tool_choice":{"type":"auto","disable_parallel_tool_use":"ignored"}})");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  EXPECT_TRUE(api_service::validate_anthropic_backend(request).ok());
}

TEST(AnthropicRequestUtilsTest, SeparatesOutputSchemaFromBackendSupport) {
  auto request = parse_request(R"({"model":"test","max_tokens":32,
    "messages":[{"role":"user","content":"hi"}],
    "output_config":{"format":{"schema":{"type":"object"}}}})");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  EXPECT_FALSE(api_service::validate_anthropic_backend(request).ok());
  request.mutable_output_config()->mutable_format()->mutable_schema()->Clear();
  EXPECT_TRUE(api_service::validate_anthropic_backend(request).ok());
  request.mutable_output_config()->mutable_format()->set_type("bad");
  EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
}

TEST(AnthropicRequestUtilsTest, SamplingErrorsMatchVllm023ServingBoundary) {
  auto request = parse_request(R"({"model":"test","max_tokens":32,
    "messages":[{"role":"user","content":"hi"}],"temperature":3})");
  EXPECT_TRUE(api_service::validate_anthropic_backend(request).ok());
  request.set_temperature(-1);
  EXPECT_EQ(api_service::validate_anthropic_backend(request).code(),
            StatusCode::UNKNOWN);
  request.clear_temperature();
  request.set_top_p(0);
  EXPECT_EQ(api_service::validate_anthropic_backend(request).code(),
            StatusCode::UNKNOWN);
  request.clear_top_p();
  request.set_top_k(-2);
  EXPECT_EQ(api_service::validate_anthropic_backend(request).code(),
            StatusCode::UNKNOWN);
  EXPECT_TRUE(api_service::validate_anthropic_backend(request, true).ok());
}

TEST(AnthropicRequestUtilsTest, RejectsMalformedAndUnsupportedContent) {
  auto [status, normalized] = ChatJsonParser::anthropic().preprocess(
      R"({"messages":[{"role":"user","content":42}]})");
  EXPECT_FALSE(status.ok());
  auto request = parse_request(R"({"model":"test","max_tokens":16,
    "messages":[{"role":"user","content":[{"type":"unknown"}]}]})");
  EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
  request.mutable_messages(0)
      ->mutable_content_blocks()
      ->mutable_blocks(0)
      ->set_type("image");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  EXPECT_FALSE(api_service::validate_anthropic_backend(request).ok());
}

TEST(AnthropicRequestUtilsTest, RejectsMalformedSystemAndToolResults) {
  for (
      const std::string json :
      {R"({"system":42})",
       R"({"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":42}]}]})"}) {
    const auto [status, normalized] =
        ChatJsonParser::anthropic().preprocess(json);
    EXPECT_FALSE(status.ok()) << json;
  }
  for (const std::string content :
       {R"({"type":"tool_result","content":"missing id"})",
        R"({"type":"tool_use","name":"search"})",
        R"({"type":"tool_result","content":[{"type":"unknown"}]})"}) {
    const auto request = parse_request(
        R"({"model":"test","max_tokens":16,"messages":[{"role":"user","content":[)" +
        content + "]}]}");
    EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
    EXPECT_TRUE(api_service::validate_anthropic_backend(request).ok());
  }
  const auto request = parse_request(R"({"model":"test","max_tokens":16,
    "system":[{"type":"thinking","thinking":"ignored"}],
    "messages":[{"role":"user","content":"hi"}]})");
  EXPECT_TRUE(api_service::validate_anthropic_request(request).ok());
  const auto messages = api_service::build_anthropic_messages(request);
  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0].role, "user");
}

TEST(AnthropicRequestUtilsTest, CountTokensIgnoresGenerationOnlyOptions) {
  proto::AnthropicMessagesRequest request;
  const Status status = api_service::parse_anthropic_request(
      R"({"model":"test","messages":[{"role":"user","content_string":"hi"}],
          "max_tokens":"ignored","thinking":"ignored","stream":true,
          "output_config":{"effort":"max"},"chat_template_kwargs":{"custom":true}})",
      /*count_tokens=*/true,
      request);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(request.stream());
  EXPECT_FALSE(request.has_output_config());
  EXPECT_TRUE(request.has_chat_template_kwargs());
  EXPECT_TRUE(
      api_service::validate_anthropic_request(request, /*count_tokens=*/true)
          .ok());
}

TEST(AnthropicRequestUtilsTest,
     PreservesToolSchemaOrderThroughPromptRendering) {
  const std::vector<std::string> schemas = {
      R"({"type":"object","properties":{"z":{"description":"last","type":"string"},"a":{"type":"integer"}},"required":["z"]})",
      R"({"required":["z"],"properties":{"a":{"type":"integer"},"z":{"type":"string","description":"last"}},"type":"object"})",
      R"({"properties":{"z":{"anyOf":[{"enum":["x","y"],"type":"string"},{"type":"null"}]}},"type":"object"})",
      R"({"properties":{}})",
      R"({})"};
  TokenizerArgs args;
  args.chat_template("{% for tool in tools %}{{ tool | tojson }}{% endfor %}");
  JinjaChatTemplate chat_template(args);
  for (bool count_tokens : {false, true}) {
    for (const std::string& schema : schemas) {
      SCOPED_TRACE(schema);
      auto [status, normalized] = ChatJsonParser::anthropic().preprocess(
          R"({"model":"test","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"search","input_schema":)" +
          schema +
          R"(,"input_schema_json":"malicious","inputSchemaJson":42}]})");
      ASSERT_TRUE(status.ok()) << status.message();
      proto::AnthropicMessagesRequest request;
      status = api_service::parse_anthropic_request(
          normalized, count_tokens, request);
      ASSERT_TRUE(status.ok()) << status.message();
      ASSERT_TRUE(api_service::validate_anthropic_request(request).ok());
      const RequestParams params(request, "", "");
      const auto prompt =
          chat_template.apply(api_service::build_anthropic_messages(request),
                              params.tools,
                              params.chat_template_kwargs);
      ASSERT_TRUE(prompt.has_value());
      // Compare serialization, since JSON object equality ignores key order.
      auto expected = nlohmann::ordered_json::parse(schema);
      if (!expected.contains("type")) {
        expected["type"] = "object";
      }
      EXPECT_EQ(nlohmann::ordered_json::parse(*prompt)["function"]["parameters"]
                    .dump(),
                expected.dump());
    }
  }
}

TEST(AnthropicRequestUtilsTest, IgnoresClientSchemaMetadataWithoutSchema) {
  for (bool count_tokens : {false, true}) {
    proto::AnthropicMessagesRequest request;
    const Status status = api_service::parse_anthropic_request(
        R"({"model":"test","max_tokens":16,"messages":[],"tools":[{"name":"search","input_schema_json":"{}","inputSchemaJson":42}]})",
        count_tokens,
        request);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(request.tools_size(), 1);
    EXPECT_FALSE(request.tools(0).has_input_schema_json());
    EXPECT_FALSE(api_service::validate_anthropic_request(request).ok());
  }
}

TEST(AnthropicRequestUtilsTest, AcceptsPydanticNumericBooleanCoercions) {
  proto::AnthropicMessagesRequest request;
  const Status status = api_service::parse_anthropic_request(
      R"({"model":"test","max_tokens":true,"stream":1,"temperature":false,
          "messages":[{"role":"user","content_string":"hi"}],
          "tools":[{"name":"search","input_schema":{},"defer_loading":0}]})",
      /*count_tokens=*/false,
      request);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(request.max_tokens(), 1);
  EXPECT_TRUE(request.stream());
  EXPECT_EQ(request.temperature(), 0);
  ASSERT_EQ(request.tools_size(), 1);
  EXPECT_FALSE(request.tools(0).defer_loading());
}

}  // namespace
}  // namespace xllm
