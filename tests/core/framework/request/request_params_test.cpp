/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/request_params.h"

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm {
namespace {

TEST(RequestParamsTest,
     CompletionBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::CompletionRequest request;
  request.set_beam_width(3);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 3);
}

TEST(RequestParamsTest, CompletionBeamSearchKeepsExplicitLogprobsWhenSet) {
  proto::CompletionRequest request;
  request.set_beam_width(3);
  request.set_logprobs(5);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 5);
}

TEST(RequestParamsTest, CompletionNonBeamSearchKeepsLogprobsDisabled) {
  proto::CompletionRequest request;
  request.set_beam_width(1);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::ChatRequest request;
  request.set_beam_width(4);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 4);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitLogprobsDisabled) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(false);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(2);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 2);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitZeroTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(0);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, AnthropicPreservesIgnoreEos) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);
  request.set_ignore_eos(true);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.ignore_eos);
}

TEST(RequestParamsTest, AnthropicToolChoiceDefaults) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  RequestParams no_tool_params(request, "", "");
  EXPECT_EQ(no_tool_params.tool_choice, "");

  auto* tool = request.add_tools();
  tool->set_name("list_files");

  RequestParams auto_params(request, "", "");
  EXPECT_EQ(auto_params.tool_choice, "auto");

  request.mutable_tool_choice()->set_type("any");
  RequestParams required_params(request, "", "");
  EXPECT_EQ(required_params.tool_choice, "required");

  request.mutable_tool_choice()->set_type("tool");
  request.mutable_tool_choice()->clear_name();
  RequestParams fallback_params(request, "", "");
  EXPECT_EQ(fallback_params.tool_choice, "auto");

  request.mutable_tool_choice()->set_type("unknown");
  RequestParams unknown_params(request, "", "");
  EXPECT_EQ(unknown_params.tool_choice, "auto");
}

TEST(RequestParamsTest, AnthropicToolWithoutSchemaUsesEmptyJson) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  auto* tool = request.add_tools();
  tool->set_name("list_files");

  RequestParams params(request, "", "");

  ASSERT_EQ(params.tools.size(), 1);
  EXPECT_TRUE(params.tools[0].function.parameters.is_object());
  EXPECT_TRUE(params.tools[0].function.parameters.empty());
}

TEST(RequestParamsTest, AnthropicToolSchemaUsesPlainJson) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  auto* tool = request.add_tools();
  tool->set_name("list_files");
  tool->set_description("List files under a folder");
  const std::string schema = R"({
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Folder path"
      },
      "recursive": {
        "type": "boolean",
        "default": false
      }
    },
    "required": ["path"]
  })";
  auto status = google::protobuf::util::JsonStringToMessage(
      schema, tool->mutable_input_schema());
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto* tool_choice = request.mutable_tool_choice();
  tool_choice->set_type("tool");
  tool_choice->set_name("list_files");

  RequestParams params(request, "", "");

  ASSERT_EQ(params.tools.size(), 1);
  const auto& parsed_tool = params.tools[0];
  EXPECT_EQ(parsed_tool.type, "function");
  EXPECT_EQ(parsed_tool.function.name, "list_files");
  EXPECT_EQ(parsed_tool.function.description, "List files under a folder");

  const nlohmann::json& params_schema = parsed_tool.function.parameters;
  ASSERT_TRUE(params_schema.is_object());
  EXPECT_FALSE(params_schema.contains("fields"));
  EXPECT_EQ(params_schema.at("type"), "object");
  EXPECT_EQ(params_schema.at("properties").at("path").at("type"), "string");
  EXPECT_EQ(params_schema.at("properties").at("recursive").at("type"),
            "boolean");
  EXPECT_EQ(params_schema.at("properties").at("recursive").at("default"),
            false);
  ASSERT_TRUE(params_schema.at("required").is_array());
  EXPECT_EQ(params_schema.at("required").at(0), "path");

  nlohmann::json expected_tool_choice = {
      {"type", "function"}, {"function", {{"name", "list_files"}}}};
  EXPECT_EQ(nlohmann::json::parse(params.tool_choice), expected_tool_choice);
}

}  // namespace
}  // namespace xllm
