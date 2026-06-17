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

#include <gtest/gtest.h>
#include <json2pb/pb_to_json.h>

#include <nlohmann/json.hpp>
#include <string>

#include "api_service/utils.h"
#include "chat.pb.h"

namespace xllm {
namespace {

TEST(UsageJsonTest, ChatUsageSerializesOpenAICachedTokensField) {
  Usage usage;
  usage.num_prompt_tokens = 1024;
  usage.num_generated_tokens = 50;
  usage.num_total_tokens = 1074;
  usage.num_cached_tokens = 896;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json.contains("usage"));
  EXPECT_EQ(json["usage"]["prompt_tokens"], 1024);
  EXPECT_EQ(json["usage"]["completion_tokens"], 50);
  EXPECT_EQ(json["usage"]["total_tokens"], 1074);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 896);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

TEST(UsageJsonTest, ChatUsagePrintsZeroCachedTokens) {
  Usage usage;
  usage.num_prompt_tokens = 12;
  usage.num_generated_tokens = 3;
  usage.num_total_tokens = 15;
  usage.num_cached_tokens = 0;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

}  // namespace
}  // namespace xllm
