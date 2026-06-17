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

#include "api_service/anthropic_json.h"

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>
#include <json2pb/pb_to_json.h>

#include <nlohmann/json.hpp>
#include <string>

#include "anthropic.pb.h"

namespace xllm {
namespace {

google::protobuf::Struct json_to_struct(const nlohmann::json& value) {
  google::protobuf::Struct pb_struct;
  std::string json_str = value.dump();
  google::protobuf::util::JsonStringToMessage(json_str, &pb_struct);
  return pb_struct;
}

json2pb::Pb2JsonOptions json_options() {
  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  return options;
}

TEST(AnthropicJsonTest, ExpandsToolUseInputStruct) {
  proto::AnthropicMessagesResponse response;
  response.set_id("msg_123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("test_model");
  response.set_stop_reason("tool_use");

  auto* tool_block = response.add_content();
  tool_block->set_type("tool_use");
  tool_block->set_id("call_123");
  tool_block->set_name("get_weather");
  *tool_block->mutable_input() =
      json_to_struct({{"city", "San Francisco"}, {"unit", "celsius"}});

  std::string json;
  std::string err_msg;
  ASSERT_TRUE(api_service::proto_to_anthropic_json(
      response, json_options(), &json, &err_msg))
      << err_msg;

  nlohmann::json parsed = nlohmann::json::parse(json);
  ASSERT_TRUE(parsed["content"].is_array());
  ASSERT_EQ(parsed["content"].size(), 1);
  const nlohmann::json& input = parsed["content"][0]["input"];
  ASSERT_TRUE(input.is_object());
  EXPECT_FALSE(input.contains("fields"));
  EXPECT_EQ(input["city"], "San Francisco");
  EXPECT_EQ(input["unit"], "celsius");
  EXPECT_EQ(parsed["stop_reason"], "tool_use");
}

TEST(AnthropicJsonTest, ExpandsNestedToolInputStruct) {
  proto::AnthropicMessagesResponse response;
  response.set_id("msg_123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("test_model");
  response.set_stop_reason("tool_use");

  auto* tool_block = response.add_content();
  tool_block->set_type("tool_use");
  tool_block->set_id("call_123");
  tool_block->set_name("search");
  *tool_block->mutable_input() = json_to_struct({
      {"query", "error budget"},
      {"limit", 3},
      {"filters",
       {{"regions", {"us-east-1", "eu-west-1"}}, {"archived", nullptr}}},
      {"flags", {true, false}},
  });

  std::string json;
  std::string err_msg;
  ASSERT_TRUE(api_service::proto_to_anthropic_json(
      response, json_options(), &json, &err_msg))
      << err_msg;

  nlohmann::json parsed = nlohmann::json::parse(json);
  const nlohmann::json& input = parsed["content"][0]["input"];
  ASSERT_TRUE(input.is_object());
  EXPECT_FALSE(input.contains("fields"));
  EXPECT_EQ(input["query"], "error budget");
  EXPECT_EQ(input["limit"].get<double>(), 3.0);
  ASSERT_TRUE(input["filters"].is_object());
  EXPECT_FALSE(input["filters"].contains("fields"));
  ASSERT_TRUE(input["filters"]["regions"].is_array());
  EXPECT_EQ(input["filters"]["regions"][0], "us-east-1");
  EXPECT_EQ(input["filters"]["regions"][1], "eu-west-1");
  EXPECT_TRUE(input["filters"]["archived"].is_null());
  ASSERT_TRUE(input["flags"].is_array());
  EXPECT_EQ(input["flags"][0], true);
  EXPECT_EQ(input["flags"][1], false);
}

TEST(AnthropicJsonTest, KeepsEmptyContentArray) {
  proto::AnthropicMessagesResponse response;
  response.set_id("msg_123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("test_model");

  std::string json;
  std::string err_msg;
  ASSERT_TRUE(api_service::proto_to_anthropic_json(
      response, json_options(), &json, &err_msg))
      << err_msg;

  nlohmann::json parsed = nlohmann::json::parse(json);
  ASSERT_TRUE(parsed.contains("content"));
  EXPECT_TRUE(parsed["content"].is_array());
  EXPECT_TRUE(parsed["content"].empty());
}

TEST(AnthropicJsonTest, KeepsEmptyToolUseInputObject) {
  proto::AnthropicStreamEvent event;
  event.set_type("content_block_start");
  event.set_index(0);
  auto* content_block = event.mutable_content_block();
  content_block->set_type("tool_use");
  content_block->set_id("call_123");
  content_block->set_name("Bash");
  content_block->mutable_input();

  std::string json;
  std::string err_msg;
  ASSERT_TRUE(api_service::proto_to_anthropic_json(
      event, json_options(), &json, &err_msg))
      << err_msg;

  nlohmann::json parsed = nlohmann::json::parse(json);
  const nlohmann::json& input = parsed["content_block"]["input"];
  ASSERT_TRUE(input.is_object());
  EXPECT_TRUE(input.empty());
}

}  // namespace
}  // namespace xllm
