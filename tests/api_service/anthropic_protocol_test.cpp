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

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "anthropic.pb.h"

namespace xllm {
namespace {

using proto::AnthropicContentBlock;
using proto::AnthropicContentBlockList;
using proto::AnthropicMessage;
using proto::AnthropicMessagesRequest;
using proto::AnthropicMessagesResponse;
using proto::AnthropicTool;
using proto::AnthropicToolChoice;
using proto::AnthropicUsage;

class AnthropicProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Helper to convert nlohmann::json to google::protobuf::Struct
  static google::protobuf::Struct json_to_struct(const nlohmann::json& j) {
    google::protobuf::Struct pb_struct;
    std::string json_str = j.dump();
    google::protobuf::util::JsonStringToMessage(json_str, &pb_struct);
    return pb_struct;
  }

  // Helper to convert google::protobuf::Struct to nlohmann::json
  static nlohmann::json struct_to_json(
      const google::protobuf::Struct& pb_struct) {
    std::string json_str;
    google::protobuf::util::JsonPrintOptions options;
    options.preserve_proto_field_names = true;
    google::protobuf::util::MessageToJsonString(pb_struct, &json_str, options);
    return nlohmann::json::parse(json_str);
  }
};

// Test AnthropicContentBlock with text content
TEST_F(AnthropicProtocolTest, AnthropicContentBlockText) {
  AnthropicContentBlock content_block;
  content_block.set_type("text");
  content_block.set_text("Hello, world!");

  EXPECT_EQ(content_block.type(), "text");
  EXPECT_EQ(content_block.text(), "Hello, world!");
  EXPECT_FALSE(content_block.has_source());
  EXPECT_FALSE(content_block.has_id());
  EXPECT_FALSE(content_block.has_name());
  EXPECT_FALSE(content_block.has_input());
  EXPECT_EQ(content_block.tool_result_content_case(),
            AnthropicContentBlock::TOOL_RESULT_CONTENT_NOT_SET);
  EXPECT_FALSE(content_block.has_is_error());
}

// Test AnthropicContentBlock with tool use content
TEST_F(AnthropicProtocolTest, AnthropicContentBlockToolUse) {
  AnthropicContentBlock content_block;
  content_block.set_type("tool_use");
  content_block.set_id("toolu_123");
  content_block.set_name("get_weather");

  nlohmann::json input_json = {{"location", "Paris"}, {"unit", "celsius"}};
  *content_block.mutable_input() = json_to_struct(input_json);

  EXPECT_EQ(content_block.type(), "tool_use");
  EXPECT_EQ(content_block.id(), "toolu_123");
  EXPECT_EQ(content_block.name(), "get_weather");

  // Verify input
  auto parsed_input = struct_to_json(content_block.input());
  EXPECT_EQ(parsed_input["location"], "Paris");
  EXPECT_EQ(parsed_input["unit"], "celsius");
}

// Test AnthropicContentBlock with tool result content
TEST_F(AnthropicProtocolTest, AnthropicContentBlockToolResult) {
  AnthropicContentBlock content_block;
  content_block.set_type("tool_result");
  content_block.set_id("call_123");
  content_block.set_content_string("The weather in Paris is sunny.");
  content_block.set_is_error(false);

  EXPECT_EQ(content_block.type(), "tool_result");
  EXPECT_EQ(content_block.id(), "call_123");
  EXPECT_EQ(content_block.content_string(), "The weather in Paris is sunny.");
  EXPECT_FALSE(content_block.is_error());
}

// Test AnthropicMessage for user role
TEST_F(AnthropicProtocolTest, AnthropicMessageUser) {
  AnthropicMessage message;
  message.set_role("user");
  message.set_content_string("Hello, assistant!");

  EXPECT_EQ(message.role(), "user");
  EXPECT_EQ(message.content_string(), "Hello, assistant!");
}

// Test AnthropicMessage for assistant role with text content
TEST_F(AnthropicProtocolTest, AnthropicMessageAssistantText) {
  AnthropicMessage message;
  message.set_role("assistant");

  // Create content block list
  auto* content_blocks = message.mutable_content_blocks();
  auto* text_block = content_blocks->add_blocks();
  text_block->set_type("text");
  text_block->set_text("Hello, user!");

  EXPECT_EQ(message.role(), "assistant");
  EXPECT_TRUE(message.has_content_blocks());
  EXPECT_EQ(message.content_blocks().blocks_size(), 1);
  EXPECT_EQ(message.content_blocks().blocks(0).type(), "text");
  EXPECT_EQ(message.content_blocks().blocks(0).text(), "Hello, user!");
}

// Test AnthropicTool model
TEST_F(AnthropicProtocolTest, AnthropicTool) {
  nlohmann::json tool_schema = {
      {"type", "object"},
      {"properties",
       {{"location", {{"type", "string"}}},
        {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
      {"required", {"location"}}};

  AnthropicTool tool;
  tool.set_name("get_weather");
  tool.set_description("Get the current weather for a location");
  *tool.mutable_input_schema() = json_to_struct(tool_schema);

  EXPECT_EQ(tool.name(), "get_weather");
  EXPECT_EQ(tool.description(), "Get the current weather for a location");

  // Verify input_schema
  auto parsed_schema = struct_to_json(tool.input_schema());
  EXPECT_EQ(parsed_schema["type"], "object");
  EXPECT_TRUE(parsed_schema.contains("properties"));
  EXPECT_TRUE(parsed_schema["properties"].contains("location"));
}

// Test AnthropicToolChoice with auto type
TEST_F(AnthropicProtocolTest, AnthropicToolChoiceAuto) {
  AnthropicToolChoice tool_choice;
  tool_choice.set_type("auto");

  EXPECT_EQ(tool_choice.type(), "auto");
  EXPECT_FALSE(tool_choice.has_name());
}

// Test AnthropicToolChoice with specific tool
TEST_F(AnthropicProtocolTest, AnthropicToolChoiceSpecific) {
  AnthropicToolChoice tool_choice;
  tool_choice.set_type("tool");
  tool_choice.set_name("get_weather");

  EXPECT_EQ(tool_choice.type(), "tool");
  EXPECT_EQ(tool_choice.name(), "get_weather");
}

// Test basic AnthropicMessagesRequest
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestBasic) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("What is the weather like today?");

  EXPECT_EQ(request.model(), "my_model");
  EXPECT_EQ(request.messages_size(), 1);
  EXPECT_EQ(request.messages(0).role(), "user");
  EXPECT_EQ(request.max_tokens(), 100);
  EXPECT_FALSE(request.has_stream());
  EXPECT_FALSE(request.has_system_string());
  EXPECT_FALSE(request.has_temperature());
}

// Test AnthropicMessagesRequest with system prompt
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestWithSystem) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);
  request.set_system_string("You are a helpful weather assistant.");

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("What is the weather like today?");

  EXPECT_EQ(request.system_string(), "You are a helpful weather assistant.");
}

// Test AnthropicMessagesRequest with tools
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestWithTools) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("What is the weather in Paris?");

  // Add tool
  nlohmann::json tool_schema = {
      {"type", "object"},
      {"properties",
       {{"location", {{"type", "string"}}},
        {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
      {"required", {"location"}}};

  auto* tool = request.add_tools();
  tool->set_name("get_weather");
  tool->set_description("Get the current weather for a location");
  *tool->mutable_input_schema() = json_to_struct(tool_schema);

  // Set tool choice
  auto* tool_choice = request.mutable_tool_choice();
  tool_choice->set_type("auto");

  EXPECT_EQ(request.tools_size(), 1);
  EXPECT_EQ(request.tools(0).name(), "get_weather");
  EXPECT_EQ(request.tool_choice().type(), "auto");
}

// Test AnthropicMessagesRequest with streaming
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestStreaming) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(1000);
  request.set_stream(true);

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("Tell me a story.");

  EXPECT_TRUE(request.stream());
}

// Test AnthropicMessagesRequest with all parameters
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestAllParams) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);
  request.set_temperature(0.7f);
  request.set_top_p(0.9f);
  request.set_top_k(50);

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("Hello");

  EXPECT_FLOAT_EQ(request.temperature(), 0.7f);
  EXPECT_FLOAT_EQ(request.top_p(), 0.9f);
  EXPECT_EQ(request.top_k(), 50);
}

// Test AnthropicUsage model
TEST_F(AnthropicProtocolTest, AnthropicUsage) {
  AnthropicUsage usage;
  usage.set_input_tokens(50);
  usage.set_output_tokens(75);

  EXPECT_EQ(usage.input_tokens(), 50);
  EXPECT_EQ(usage.output_tokens(), 75);
  EXPECT_FALSE(usage.has_cache_creation_input_tokens());
  EXPECT_FALSE(usage.has_cache_read_input_tokens());
}

// Test AnthropicUsage model with cache tokens
TEST_F(AnthropicProtocolTest, AnthropicUsageWithCache) {
  AnthropicUsage usage;
  usage.set_input_tokens(50);
  usage.set_output_tokens(75);
  usage.set_cache_creation_input_tokens(25);
  usage.set_cache_read_input_tokens(15);

  EXPECT_EQ(usage.input_tokens(), 50);
  EXPECT_EQ(usage.output_tokens(), 75);
  EXPECT_EQ(usage.cache_creation_input_tokens(), 25);
  EXPECT_EQ(usage.cache_read_input_tokens(), 15);
}

// Test AnthropicMessagesResponse model
TEST_F(AnthropicProtocolTest, AnthropicMessagesResponse) {
  AnthropicMessagesResponse response;
  response.set_id("msg_123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("my_model");
  response.set_stop_reason("end_turn");

  // Add content block
  auto* content_block = response.add_content();
  content_block->set_type("text");
  content_block->set_text("Hello, user!");

  // Set usage
  auto* usage = response.mutable_usage();
  usage->set_input_tokens(10);
  usage->set_output_tokens(20);

  EXPECT_EQ(response.id(), "msg_123");
  EXPECT_EQ(response.type(), "message");
  EXPECT_EQ(response.role(), "assistant");
  EXPECT_EQ(response.content_size(), 1);
  EXPECT_EQ(response.content(0).type(), "text");
  EXPECT_EQ(response.content(0).text(), "Hello, user!");
  EXPECT_EQ(response.model(), "my_model");
  EXPECT_EQ(response.stop_reason(), "end_turn");
  EXPECT_EQ(response.usage().input_tokens(), 10);
  EXPECT_EQ(response.usage().output_tokens(), 20);
}

// Test AnthropicMessagesResponse with specific ID
TEST_F(AnthropicProtocolTest, AnthropicMessagesResponseAutoId) {
  AnthropicMessagesResponse response;
  response.set_id("msg_test123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("my_model");

  // Add content block
  auto* content_block = response.add_content();
  content_block->set_type("text");
  content_block->set_text("Hello!");

  // Set usage
  auto* usage = response.mutable_usage();
  usage->set_input_tokens(5);
  usage->set_output_tokens(10);

  EXPECT_EQ(response.id(), "msg_test123");
  // Check that ID starts with "msg_"
  EXPECT_EQ(response.id().substr(0, 4), "msg_");
}

// Test AnthropicTool with default input_schema type
TEST_F(AnthropicProtocolTest, AnthropicToolInputSchemaDefault) {
  nlohmann::json tool_schema = {
      {"properties", {{"param", {{"type", "string"}}}}}};

  AnthropicTool tool;
  tool.set_name("simple_tool");
  *tool.mutable_input_schema() = json_to_struct(tool_schema);

  auto parsed_schema = struct_to_json(tool.input_schema());
  EXPECT_TRUE(parsed_schema.contains("properties"));
}

// Test message with multiple content blocks
TEST_F(AnthropicProtocolTest, AnthropicMessageMultipleContentBlocks) {
  AnthropicMessage message;
  message.set_role("assistant");

  auto* content_blocks = message.mutable_content_blocks();

  // Add text block
  auto* text_block = content_blocks->add_blocks();
  text_block->set_type("text");
  text_block->set_text("I'll help you check the weather.");

  // Add tool use block
  auto* tool_block = content_blocks->add_blocks();
  tool_block->set_type("tool_use");
  tool_block->set_id("toolu_456");
  tool_block->set_name("get_weather");
  nlohmann::json input_json = {{"location", "Tokyo"}};
  *tool_block->mutable_input() = json_to_struct(input_json);

  EXPECT_EQ(message.content_blocks().blocks_size(), 2);
  EXPECT_EQ(message.content_blocks().blocks(0).type(), "text");
  EXPECT_EQ(message.content_blocks().blocks(1).type(), "tool_use");
  EXPECT_EQ(message.content_blocks().blocks(1).id(), "toolu_456");
}

// Test AnthropicMessagesRequest with stop_sequences
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestStopSequences) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);

  // Add stop sequences
  request.add_stop_sequences("END");
  request.add_stop_sequences("STOP");

  // Add message
  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("Generate text until END");

  EXPECT_EQ(request.stop_sequences_size(), 2);
  EXPECT_EQ(request.stop_sequences(0), "END");
  EXPECT_EQ(request.stop_sequences(1), "STOP");
}

// Test JSON serialization round-trip
TEST_F(AnthropicProtocolTest, JsonSerializationRoundTrip) {
  AnthropicMessagesRequest original;
  original.set_model("my_model");
  original.set_max_tokens(100);
  original.set_temperature(0.7f);

  auto* message = original.add_messages();
  message->set_role("user");
  message->set_content_string("Hello!");

  // Serialize to JSON
  std::string json_str;
  google::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  auto status =
      google::protobuf::util::MessageToJsonString(original, &json_str, options);
  ASSERT_TRUE(status.ok());

  // Deserialize back
  AnthropicMessagesRequest parsed;
  status = google::protobuf::util::JsonStringToMessage(json_str, &parsed);
  ASSERT_TRUE(status.ok());

  // Verify
  EXPECT_EQ(parsed.model(), original.model());
  EXPECT_EQ(parsed.max_tokens(), original.max_tokens());
  EXPECT_FLOAT_EQ(parsed.temperature(), original.temperature());
  EXPECT_EQ(parsed.messages_size(), 1);
  EXPECT_EQ(parsed.messages(0).content_string(), "Hello!");
}

// Test AnthropicContentBlock with image source
TEST_F(AnthropicProtocolTest, AnthropicContentBlockImage) {
  AnthropicContentBlock content_block;
  content_block.set_type("image");

  nlohmann::json source_json = {
      {"type", "base64"}, {"media_type", "image/png"}, {"data", "xxxxxx"}};
  *content_block.mutable_source() = json_to_struct(source_json);

  EXPECT_EQ(content_block.type(), "image");
  EXPECT_TRUE(content_block.has_source());

  auto parsed_source = struct_to_json(content_block.source());
  EXPECT_EQ(parsed_source["type"], "base64");
  EXPECT_EQ(parsed_source["media_type"], "image/png");
}

// Test tool result with list content
TEST_F(AnthropicProtocolTest, AnthropicContentBlockToolResultList) {
  AnthropicContentBlock content_block;
  content_block.set_type("tool_result");
  content_block.set_id("call_789");

  // Use content_list for complex content
  auto* content_list = content_block.mutable_content_list();
  nlohmann::json item_json = {{"type", "text"}, {"text", "Result item 1"}};
  *content_list->add_items() = json_to_struct(item_json);

  EXPECT_EQ(content_block.type(), "tool_result");
  EXPECT_TRUE(content_block.has_content_list());
  EXPECT_EQ(content_block.content_list().items_size(), 1);
}

// Test response with tool_use stop_reason
TEST_F(AnthropicProtocolTest, AnthropicMessagesResponseToolUse) {
  AnthropicMessagesResponse response;
  response.set_id("msg_tool_123");
  response.set_type("message");
  response.set_role("assistant");
  response.set_model("my_model");
  response.set_stop_reason("tool_use");

  // Add text content
  auto* text_block = response.add_content();
  text_block->set_type("text");
  text_block->set_text("Let me check the weather for you.");

  // Add tool_use content
  auto* tool_block = response.add_content();
  tool_block->set_type("tool_use");
  tool_block->set_id("toolu_weather_1");
  tool_block->set_name("get_weather");
  nlohmann::json input_json = {{"location", "San Francisco"},
                               {"unit", "fahrenheit"}};
  *tool_block->mutable_input() = json_to_struct(input_json);

  // Set usage
  auto* usage = response.mutable_usage();
  usage->set_input_tokens(30);
  usage->set_output_tokens(45);

  EXPECT_EQ(response.stop_reason(), "tool_use");
  EXPECT_EQ(response.content_size(), 2);
  EXPECT_EQ(response.content(0).type(), "text");
  EXPECT_EQ(response.content(1).type(), "tool_use");
  EXPECT_EQ(response.content(1).name(), "get_weather");

  auto parsed_input = struct_to_json(response.content(1).input());
  EXPECT_EQ(parsed_input["location"], "San Francisco");
}

// Test request with system blocks
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestSystemBlocks) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(100);

  // Use system_blocks instead of system_string
  auto* system_blocks = request.mutable_system_blocks();
  auto* block = system_blocks->add_blocks();
  block->set_type("text");
  block->set_text("You are a helpful assistant specialized in weather.");

  auto* message = request.add_messages();
  message->set_role("user");
  message->set_content_string("What's the weather?");

  EXPECT_TRUE(request.has_system_blocks());
  EXPECT_FALSE(request.has_system_string());
  EXPECT_EQ(request.system_blocks().blocks_size(), 1);
  EXPECT_EQ(request.system_blocks().blocks(0).text(),
            "You are a helpful assistant specialized in weather.");
}

// Test AnthropicToolChoice with "any" type
TEST_F(AnthropicProtocolTest, AnthropicToolChoiceAny) {
  AnthropicToolChoice tool_choice;
  tool_choice.set_type("any");

  EXPECT_EQ(tool_choice.type(), "any");
  EXPECT_FALSE(tool_choice.has_name());
}

// Test complex conversation with multiple turns
TEST_F(AnthropicProtocolTest, AnthropicMessagesRequestMultiTurn) {
  AnthropicMessagesRequest request;
  request.set_model("my_model");
  request.set_max_tokens(200);

  // User message 1
  auto* msg1 = request.add_messages();
  msg1->set_role("user");
  msg1->set_content_string("What's the weather in Paris?");

  // Assistant message with tool use
  auto* msg2 = request.add_messages();
  msg2->set_role("assistant");
  auto* content_blocks2 = msg2->mutable_content_blocks();
  auto* tool_use_block = content_blocks2->add_blocks();
  tool_use_block->set_type("tool_use");
  tool_use_block->set_id("toolu_123");
  tool_use_block->set_name("get_weather");
  nlohmann::json input_json = {{"location", "Paris"}};
  *tool_use_block->mutable_input() = json_to_struct(input_json);

  // User message with tool result
  auto* msg3 = request.add_messages();
  msg3->set_role("user");
  auto* content_blocks3 = msg3->mutable_content_blocks();
  auto* tool_result_block = content_blocks3->add_blocks();
  tool_result_block->set_type("tool_result");
  tool_result_block->set_id("toolu_123");
  tool_result_block->set_content_string("The weather in Paris is 22°C, sunny.");

  EXPECT_EQ(request.messages_size(), 3);
  EXPECT_EQ(request.messages(0).role(), "user");
  EXPECT_EQ(request.messages(1).role(), "assistant");
  EXPECT_EQ(request.messages(2).role(), "user");
  EXPECT_EQ(request.messages(1).content_blocks().blocks(0).type(), "tool_use");
  EXPECT_EQ(request.messages(2).content_blocks().blocks(0).type(),
            "tool_result");
}

}  // namespace
}  // namespace xllm
