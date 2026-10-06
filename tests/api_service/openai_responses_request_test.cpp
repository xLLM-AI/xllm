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

#include "api_service/openai_responses_request.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace xllm::api_service {
namespace {

using Json = nlohmann::json;

Json with_fields(const Json& fields) {
  Json body{{"input", "hello"}};
  body.update(fields);
  return body;
}

void expect_invalid(const Json& body, const std::string& expected_param) {
  SCOPED_TRACE(body.dump());
  std::string param = "stale";
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", &param);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_FALSE(status.message().empty());
  EXPECT_EQ(param, expected_param);
}

TEST(OpenAIResponsesRequestTest, DefaultsUseLoadedModelAndStatelessProfile) {
  std::string param = "stale";
  auto [status, request] = parse_responses_request(
      R"({"input":"Hello, 世界 👋"})", "loaded-model", &param);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(param.empty());
  EXPECT_EQ(request.model, "loaded-model");
  ASSERT_EQ(request.messages.size(), 1U);
  EXPECT_EQ(request.messages[0].role, "user");
  EXPECT_EQ(std::get<std::string>(request.messages[0].content),
            "Hello, 世界 👋");
  EXPECT_FALSE(request.params.streaming);
  EXPECT_EQ(request.params.n, 1U);
  EXPECT_EQ(request.params.max_tokens, 5120U);
  EXPECT_FLOAT_EQ(request.params.temperature, 1.0f);
  EXPECT_FLOAT_EQ(request.params.top_p, 1.0f);
  EXPECT_TRUE(request.params.responses_request);
  EXPECT_TRUE(request.params.request_id.starts_with("resp_"));
  EXPECT_EQ(request.params.tool_choice, "auto");
  EXPECT_EQ(request.params.response_format, ResponseFormatType::NONE);

  const Json response = responses_initial_response(request);
  EXPECT_EQ(response["object"], "response");
  EXPECT_EQ(response["id"], request.params.request_id);
  EXPECT_TRUE(response["created_at"].is_number_integer());
  EXPECT_EQ(response["status"], "in_progress");
  EXPECT_EQ(response["model"], "loaded-model");
  EXPECT_EQ(response["store"], false);
  EXPECT_EQ(response["background"], false);
  EXPECT_EQ(response["parallel_tool_calls"], true);
  EXPECT_EQ(response["truncation"], "disabled");
  EXPECT_EQ(response["output"], Json::array());
  EXPECT_EQ(response["tools"], Json::array());
  EXPECT_EQ(response["metadata"], Json::object());
  EXPECT_EQ(response["text"], Json({{"format", {{"type", "text"}}}}));
  for (const char* field : {"usage",
                            "error",
                            "incomplete_details",
                            "instructions",
                            "previous_response_id",
                            "max_output_tokens",
                            "completed_at",
                            "reasoning"}) {
    ASSERT_TRUE(response.contains(field)) << field;
    EXPECT_TRUE(response[field].is_null()) << field;
  }
  EXPECT_FALSE(response.contains("choices"));

  auto [next_status, next] =
      parse_responses_request(R"({"input":"hello"})", "loaded-model", nullptr);
  ASSERT_TRUE(next_status.ok());
  EXPECT_NE(next.params.request_id, request.params.request_id);
}

TEST(OpenAIResponsesRequestTest, MapsSamplingInstructionsAndMetadata) {
  const Json body{{"model", "selected-model"},
                  {"input", "answer briefly"},
                  {"instructions", "You are a helpful assistant."},
                  {"max_output_tokens", 128},
                  {"temperature", 0.25},
                  {"top_p", 0.75},
                  {"stream", true},
                  {"store", false},
                  {"background", false},
                  {"metadata", {{"purpose", "protocol-test"}}},
                  {"text", {{"format", {{"type", "json_object"}}}}}};
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(request.model, "selected-model");
  EXPECT_EQ(request.params.max_tokens, 128U);
  EXPECT_FLOAT_EQ(request.params.temperature, 0.25f);
  EXPECT_FLOAT_EQ(request.params.top_p, 0.75f);
  EXPECT_TRUE(request.params.streaming);
  EXPECT_EQ(request.params.response_format, ResponseFormatType::JSON_OBJECT);
  ASSERT_EQ(request.messages.size(), 2U);
  EXPECT_EQ(request.messages[0].role, "system");
  EXPECT_EQ(std::get<std::string>(request.messages[0].content),
            "You are a helpful assistant.");
  EXPECT_EQ(request.messages[1].role, "user");
  const Json response = responses_initial_response(request);
  EXPECT_EQ(response["max_output_tokens"], 128);
  EXPECT_EQ(response["instructions"], "You are a helpful assistant.");
  EXPECT_EQ(response["metadata"], Json({{"purpose", "protocol-test"}}));
  EXPECT_EQ(response["text"]["format"]["type"], "json_object");
}

TEST(OpenAIResponsesRequestTest, NullableControlsKeepDocumentedDefaults) {
  const Json body = with_fields({{"temperature", nullptr},
                                 {"top_p", nullptr},
                                 {"max_output_tokens", nullptr},
                                 {"instructions", nullptr},
                                 {"store", nullptr},
                                 {"stream", nullptr},
                                 {"background", nullptr},
                                 {"previous_response_id", nullptr},
                                 {"parallel_tool_calls", nullptr},
                                 {"reasoning", nullptr},
                                 {"include", nullptr},
                                 {"truncation", nullptr},
                                 {"metadata", nullptr}});
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FLOAT_EQ(request.params.temperature, 1.0f);
  EXPECT_FLOAT_EQ(request.params.top_p, 1.0f);
  EXPECT_EQ(request.params.max_tokens, 5120U);
  ASSERT_EQ(request.messages.size(), 1U);
  const Json response = responses_initial_response(request);
  EXPECT_EQ(response["store"], false);
  EXPECT_EQ(response["background"], false);
  EXPECT_EQ(response["metadata"], Json::object());
  EXPECT_TRUE(response["max_output_tokens"].is_null());
}

TEST(OpenAIResponsesRequestTest, AcceptsInclusiveSamplingAndIntegerBounds) {
  for (const Json& fields :
       {Json{{"temperature", 0}, {"top_p", 0}, {"max_output_tokens", 16}},
        Json{{"temperature", 2.0},
             {"top_p", 1.0},
             {"max_output_tokens", std::numeric_limits<uint32_t>::max()}}}) {
    SCOPED_TRACE(fields.dump());
    auto [status, request] = parse_responses_request(
        with_fields(fields).dump(), "loaded-model", nullptr);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(request.params.temperature,
                    fields["temperature"].get<float>());
    EXPECT_FLOAT_EQ(request.params.top_p, fields["top_p"].get<float>());
    EXPECT_EQ(request.params.max_tokens,
              fields["max_output_tokens"].get<uint32_t>());
  }
}

TEST(OpenAIResponsesRequestTest,
     ReplaysTextReasoningAndParallelFunctionHistory) {
  const Json body = Json::parse(R"({
    "instructions":"global instructions",
    "input":[
      {"role":"developer","content":"developer instructions"},
      {"role":"user","content":[
        {"type":"input_text","text":"weather in "},
        {"type":"input_text","text":"巴黎 👋"}]},
      {"type":"reasoning","id":"rs_prior","summary":[],"content":[
        {"type":"reasoning_text","text":"Check "},
        {"type":"reasoning_text","text":"both places."}]},
      {"type":"message","id":"msg_prior","status":"completed",
       "role":"assistant","content":[
         {"type":"output_text","text":"I will check.",
          "annotations":[],"logprobs":[]}]},
      {"type":"function_call","id":"fc_a","status":"completed",
       "call_id":"call_a","name":"weather","arguments":"{\"city\":\"Paris\"}"},
      {"type":"function_call","call_id":"call_b","name":"weather",
       "arguments":"{\"city\":\"London\"}"},
      {"type":"function_call_output","call_id":"call_b","output":[
        {"type":"input_text","text":"London: "},
        {"type":"input_text","text":"rainy"}]},
      {"type":"function_call_output","call_id":"call_a","output":"Paris: sunny"},
      {"role":"user","content":"Compare them."}
    ]
  })");
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(request.messages.size(), 7U);
  EXPECT_EQ(request.messages[0].role, "system");
  EXPECT_EQ(request.messages[1].role, "system");
  EXPECT_EQ(std::get<std::string>(request.messages[1].content),
            "developer instructions");
  EXPECT_EQ(request.messages[2].role, "user");
  EXPECT_EQ(std::get<std::string>(request.messages[2].content),
            "weather in 巴黎 👋");
  const Message& assistant = request.messages[3];
  EXPECT_EQ(assistant.role, "assistant");
  EXPECT_EQ(std::get<std::string>(assistant.content), "I will check.");
  ASSERT_TRUE(assistant.reasoning_content.has_value());
  EXPECT_EQ(*assistant.reasoning_content, "Check both places.");
  ASSERT_TRUE(assistant.tool_calls.has_value());
  ASSERT_EQ(assistant.tool_calls->size(), 2U);
  EXPECT_EQ((*assistant.tool_calls)[0].id, "call_a");
  EXPECT_EQ((*assistant.tool_calls)[0].type, "function");
  EXPECT_EQ((*assistant.tool_calls)[0].function.name, "weather");
  EXPECT_EQ((*assistant.tool_calls)[0].function.arguments,
            R"({"city":"Paris"})");
  EXPECT_EQ((*assistant.tool_calls)[1].id, "call_b");
  EXPECT_EQ(request.messages[4].role, "tool");
  EXPECT_EQ(request.messages[4].tool_call_id, "call_b");
  EXPECT_EQ(std::get<std::string>(request.messages[4].content),
            "London: rainy");
  EXPECT_EQ(request.messages[5].tool_call_id, "call_a");
  EXPECT_EQ(request.messages[6].role, "user");
  EXPECT_EQ(std::get<std::string>(request.messages[6].content),
            "Compare them.");
}

TEST(OpenAIResponsesRequestTest, FunctionItemsCreateTheirOwnAssistantMessage) {
  const Json body = Json::parse(R"({"input":[
    {"role":"user","content":"weather?"},
    {"type":"reasoning","summary":[],"content":[
      {"type":"reasoning_text","text":"Use the client function."}]},
    {"type":"function_call","call_id":"call_a","name":"weather","arguments":"{}"},
    {"type":"function_call_output","call_id":"call_a","output":"sunny"}
  ]})");
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(request.messages.size(), 3U);
  const Message& assistant = request.messages[1];
  EXPECT_EQ(assistant.role, "assistant");
  EXPECT_EQ(std::get<std::string>(assistant.content), "");
  EXPECT_EQ(assistant.reasoning_content, "Use the client function.");
  ASSERT_TRUE(assistant.tool_calls.has_value());
  ASSERT_EQ(assistant.tool_calls->size(), 1U);
  EXPECT_EQ((*assistant.tool_calls)[0].function.arguments, "{}");
}

TEST(OpenAIResponsesRequestTest,
     FlatFunctionToolsUseExplicitNonstrictDefaults) {
  const Json body = with_fields(Json::parse(R"({"tools":[
    {"type":"function","name":"weather","description":"City weather",
     "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                   "required":["city"]},"strict":false},
    {"type":"function","name":"no_args"},
    {"type":"function","name":"nullable","description":null,"parameters":null,"strict":null}
  ],"tool_choice":"auto","parallel_tool_calls":true})"));
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(request.params.tools.size(), 3U);
  EXPECT_EQ(request.params.tools[0].type, "function");
  EXPECT_EQ(request.params.tools[0].function.name, "weather");
  EXPECT_EQ(request.params.tools[0].function.description, "City weather");
  EXPECT_EQ(Json(request.params.tools[0].function.parameters),
            body["tools"][0]["parameters"]);
  const Json empty_schema{{"type", "object"}, {"properties", Json::object()}};
  EXPECT_EQ(Json(request.params.tools[1].function.parameters), empty_schema);
  EXPECT_EQ(Json(request.params.tools[2].function.parameters), empty_schema);
  ASSERT_EQ(request.tools.size(), 3U);
  for (const Json& tool : request.tools) {
    EXPECT_EQ(tool["strict"], false);
    EXPECT_FALSE(tool.contains("function"));
  }
}

TEST(OpenAIResponsesRequestTest,
     NoneDisablesNewToolsButPreservesTheirPublicEcho) {
  const Json body = with_fields(
      {{"tools", Json::array({{{"type", "function"}, {"name", "weather"}}})},
       {"tool_choice", "none"},
       {"text", {{"format", {{"type", "json_object"}}}}}});
  auto [status, request] =
      parse_responses_request(body.dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(request.params.tools.empty());
  EXPECT_EQ(request.params.tool_choice, "none");
  EXPECT_EQ(request.params.response_format, ResponseFormatType::JSON_OBJECT);
  const Json response = responses_initial_response(request);
  ASSERT_EQ(response["tools"].size(), 1U);
  EXPECT_EQ(response["tools"][0]["name"], "weather");
  EXPECT_EQ(response["tool_choice"], "none");
}

TEST(OpenAIResponsesRequestTest,
     EmptyReasoningControlsDoNotInventEffortOrSummary) {
  for (const Json& reasoning :
       {Json::object(), Json{{"effort", nullptr}, {"summary", nullptr}}}) {
    auto [status, request] =
        parse_responses_request(with_fields({{"reasoning", reasoning},
                                             {"include", Json::array()},
                                             {"truncation", "disabled"}})
                                    .dump(),
                                "loaded-model",
                                nullptr);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(responses_initial_response(request)["reasoning"], reasoning);
  }
}

TEST(OpenAIResponsesRequestTest, RejectsMalformedBodiesAndMissingModelOrInput) {
  for (const std::string& body : {std::string("{"),
                                  std::string("[]"),
                                  std::string("null"),
                                  std::string("42"),
                                  std::string("\"hello\"")}) {
    SCOPED_TRACE(body);
    std::string param;
    auto [status, request] =
        parse_responses_request(body, "loaded-model", &param);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(param.empty());
  }
  expect_invalid(Json::object(), "input");
  std::string param;
  auto [status, request] =
      parse_responses_request(R"({"input":"hello"})", "", &param);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(param, "model");
}

TEST(OpenAIResponsesRequestTest,
     RejectsWrongTypesRangesAndUnsupportedControls) {
  struct InvalidCase {
    Json fields;
    std::string param;
  };
  const std::vector<InvalidCase> cases{
      {{{"model", nullptr}}, "model"},
      {{{"model", ""}}, "model"},
      {{{"model", 7}}, "model"},
      {{{"input", nullptr}}, "input"},
      {{{"input", Json::array()}}, "input"},
      {{{"input", 7}}, "input"},
      {{{"temperature", -0.01}}, "temperature"},
      {{{"temperature", 2.01}}, "temperature"},
      {{{"temperature", "1"}}, "temperature"},
      {{{"temperature", true}}, "temperature"},
      {{{"top_p", -0.01}}, "top_p"},
      {{{"top_p", 1.01}}, "top_p"},
      {{{"top_p", "1"}}, "top_p"},
      {{{"max_output_tokens", 0}}, "max_output_tokens"},
      {{{"max_output_tokens", -1}}, "max_output_tokens"},
      {{{"max_output_tokens", 15}}, "max_output_tokens"},
      {{{"max_output_tokens", 16.5}}, "max_output_tokens"},
      {{{"max_output_tokens", true}}, "max_output_tokens"},
      {{{"max_output_tokens", "16"}}, "max_output_tokens"},
      {{{"max_output_tokens", uint64_t{4294967296}}}, "max_output_tokens"},
      {{{"stream", 1}}, "stream"},
      {{{"instructions", Json::array()}}, "instructions"},
      {{{"store", true}}, "store"},
      {{{"store", 0}}, "store"},
      {{{"background", true}}, "background"},
      {{{"background", "false"}}, "background"},
      {{{"previous_response_id", "resp_prior"}}, "previous_response_id"},
      {{{"tools", nullptr}}, "tools"},
      {{{"tool_choice", nullptr}}, "tool_choice"},
      {{{"tool_choice", "required"}}, "tool_choice"},
      {{{"tool_choice", {{"type", "function"}, {"name", "weather"}}}},
       "tool_choice"},
      {{{"parallel_tool_calls", false}}, "parallel_tool_calls"},
      {{{"parallel_tool_calls", "true"}}, "parallel_tool_calls"},
      {{{"text", nullptr}}, "text"},
      {{{"text", {{"format", nullptr}}}}, "text.format"},
      {{{"text", {{"format", {{"type", "json_schema"}}}}}}, "text.format.type"},
      {{{"text", {{"format", {{"type", "text"}, {"strict", false}}}}}},
       "text.format.strict"},
      {{{"text", {{"verbosity", "low"}}}}, "text.verbosity"},
      {{{"reasoning", "low"}}, "reasoning"},
      {{{"reasoning", {{"effort", "low"}}}}, "reasoning.effort"},
      {{{"reasoning", {{"summary", "auto"}}}}, "reasoning.summary"},
      {{{"reasoning", {{"generate_summary", "auto"}}}},
       "reasoning.generate_summary"},
      {{{"include", Json::array({"reasoning.encrypted_content"})}}, "include"},
      {{{"include", "output_text.logprobs"}}, "include"},
      {{{"truncation", "auto"}}, "truncation"},
      {{{"metadata", Json::array()}}, "metadata"},
      {{{"metadata", {{"tag", 42}}}}, "metadata.tag"},
      {{{"n", 1}}, "n"},
      {{{"max_tokens", 16}}, "max_tokens"},
      {{{"messages", Json::array()}}, "messages"},
      {{{"seed", 1}}, "seed"},
      {{{"service_tier", "auto"}}, "service_tier"},
      {{{"conversation", "conv_prior"}}, "conversation"},
  };
  for (const InvalidCase& item : cases) {
    expect_invalid(with_fields(item.fields), item.param);
  }
  for (const std::string& body :
       {std::string(R"({"input":"hello","temperature":1e999})"),
        std::string(R"({"input":"hello","top_p":1e999})")}) {
    auto [status, request] =
        parse_responses_request(body, "loaded-model", nullptr);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  }
}

TEST(OpenAIResponsesRequestTest, RejectsMalformedOrUnsupportedInputItems) {
  struct InvalidCase {
    Json input;
    std::string param;
  };
  const std::vector<InvalidCase> cases{
      {Json::array({"hello"}), "input[0]"},
      {Json::parse(R"([{"role":"tool","content":"result"}])"), "input[0].role"},
      {Json::parse(R"([{"role":"user"}])"), "input[0].content"},
      {Json::parse(R"([{"role":"user","content":null}])"), "input[0].content"},
      {Json::parse(
           R"([{"role":"user","content":[{"type":"output_text","text":"x"}]}])"),
       "input[0].content[0].type"},
      {Json::parse(R"([{"role":"user","content":[{"type":"input_image"}]}])"),
       "input[0].content[0].type"},
      {Json::parse(R"([{"role":"user","content":[{"type":"input_audio"}]}])"),
       "input[0].content[0].type"},
      {Json::parse(R"([{"role":"user","content":[{"type":"input_file"}]}])"),
       "input[0].content[0].type"},
      {Json::parse(
           R"([{"role":"user","content":[{"type":"input_text","text":7}]}])"),
       "input[0].content[0].text"},
      {Json::parse(
           R"([{"role":"user","content":[{"type":"input_text","text":"x","annotations":[]}]}])"),
       "input[0].content[0].annotations"},
      {Json::parse(
           R"([{"role":"assistant","content":[{"type":"output_text","text":"x","annotations":[{}]}]}])"),
       "input[0].content[0].annotations"},
      {Json::parse(
           R"([{"role":"assistant","content":"x","phase":"final_answer"}])"),
       "input[0].phase"},
      {Json::parse(R"([{"role":"assistant","content":"x","id":3}])"),
       "input[0].id"},
      {Json::parse(R"([{"type":"item_reference","id":"msg_prior"}])"),
       "input[0].type"},
      {Json::parse(
           R"([{"type":"reasoning","summary":[{"type":"summary_text","text":"x"}],"content":[]}])"),
       "input[0].summary"},
      {Json::parse(
           R"([{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"why"}]}])"),
       "input"},
      {Json::parse(
           R"([{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"why"}]},{"role":"user","content":"next"}])"),
       "input[1]"},
      {Json::parse(
           R"([{"type":"reasoning","summary":[],"encrypted_content":"opaque","content":[]}])"),
       "input[0].encrypted_content"},
  };
  for (const InvalidCase& item : cases) {
    expect_invalid(with_fields({{"input", item.input}}), item.param);
  }
}

TEST(OpenAIResponsesRequestTest,
     RejectsBrokenFunctionHistoryInsteadOfRepairingIt) {
  const Json call{{"type", "function_call"},
                  {"call_id", "call_a"},
                  {"name", "weather"},
                  {"arguments", "{}"}};
  const Json result{{"type", "function_call_output"},
                    {"call_id", "call_a"},
                    {"output", "sunny"}};
  expect_invalid(with_fields({{"input", Json::array({result})}}),
                 "input[0].call_id");
  expect_invalid(with_fields({{"input", Json::array({call})}}), "input");
  expect_invalid(with_fields({{"input", Json::array({call, call, result})}}),
                 "input[1].call_id");
  expect_invalid(with_fields({{"input", Json::array({call, result, result})}}),
                 "input[2].call_id");
  for (const std::string& arguments : {std::string(""),
                                       std::string("{"),
                                       std::string("[]"),
                                       std::string("null")}) {
    Json malformed_call = call;
    malformed_call["arguments"] = arguments;
    expect_invalid(
        with_fields({{"input", Json::array({malformed_call, result})}}),
        "input[0].arguments");
  }
  Json malformed_result = result;
  malformed_result["output"] = Json::array({{{"type", "input_image"}}});
  expect_invalid(
      with_fields({{"input", Json::array({call, malformed_result})}}),
      "input[1].output[0].type");
}

TEST(OpenAIResponsesRequestTest, RejectsInvalidHistoryStatusAndTurnOrdering) {
  const Json call{{"type", "function_call"},
                  {"call_id", "call_a"},
                  {"name", "weather"},
                  {"arguments", "{}"}};
  const Json result{{"type", "function_call_output"},
                    {"call_id", "call_a"},
                    {"output", "sunny"}};
  const Json user{{"role", "user"}, {"content", "next turn"}};
  const Json assistant{{"role", "assistant"}, {"content", "answer"}};
  for (Json item : {user, assistant}) {
    item["status"] = "invented";
    expect_invalid(with_fields({{"input", Json::array({item})}}),
                   "input[0].status");
  }
  Json invalid_call = call;
  invalid_call["status"] = "invented";
  expect_invalid(with_fields({{"input", Json::array({invalid_call, result})}}),
                 "input[0].status");
  invalid_call = call;
  invalid_call["name"] = "invalid name";
  expect_invalid(with_fields({{"input", Json::array({invalid_call, result})}}),
                 "input[0].name");
  expect_invalid(with_fields({{"input", Json::array({call, user, result})}}),
                 "input[1]");
  Json second_call = call;
  second_call["call_id"] = "call_b";
  Json second_result = result;
  second_result["call_id"] = "call_b";
  Json third_call = call;
  third_call["call_id"] = "call_c";
  Json third_result = result;
  third_result["call_id"] = "call_c";
  expect_invalid(with_fields({{"input",
                               Json::array({call,
                                            second_call,
                                            result,
                                            third_call,
                                            second_result,
                                            third_result})}}),
                 "input[3]");
  auto [status, request] =
      parse_responses_request(with_fields({{"input",
                                            Json::array({call,
                                                         second_call,
                                                         second_result,
                                                         result,
                                                         user,
                                                         third_call,
                                                         third_result,
                                                         assistant})}})
                                  .dump(),
                              "loaded-model",
                              nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(request.messages.size(), 7U);
  EXPECT_EQ(request.messages[3].role, "user");
  EXPECT_EQ(request.messages[6].role, "assistant");
}

TEST(OpenAIResponsesRequestTest,
     RejectsUnsupportedToolsAndUnenforcedConstraints) {
  struct InvalidCase {
    Json tools;
    std::string param;
  };
  const std::vector<InvalidCase> cases{
      {Json::array({{{"type", "web_search"}}}), "tools[0].type"},
      {Json::array({{{"type", "code_interpreter"}}}), "tools[0].type"},
      {Json::array(
           {{{"type", "function"}, {"function", {{"name", "weather"}}}}}),
       "tools[0].function"},
      {Json::array({{{"type", "function"}, {"name", ""}}}), "tools[0].name"},
      {Json::array({{{"type", "function"}, {"name", "with space"}}}),
       "tools[0].name"},
      {Json::array({{{"type", "function"}, {"name", std::string(65, 'a')}}}),
       "tools[0].name"},
      {Json::array({{{"type", "function"}, {"name", "f"}, {"strict", true}}}),
       "tools[0].strict"},
      {Json::array(
           {{{"type", "function"}, {"name", "f"}, {"strict", "false"}}}),
       "tools[0].strict"},
      {Json::array(
           {{{"type", "function"}, {"name", "f"}, {"parameters", "{}"}}}),
       "tools[0].parameters"},
      {Json::array(
           {{{"type", "function"}, {"name", "f"}, {"description", 42}}}),
       "tools[0].description"},
      {Json::array({{{"type", "function"}, {"name", "f"}},
                    {{"type", "function"}, {"name", "f"}}}),
       "tools[1].name"},
  };
  for (const InvalidCase& item : cases) {
    expect_invalid(with_fields({{"tools", item.tools}}), item.param);
  }
  expect_invalid(
      with_fields(
          {{"tools", Json::array({{{"type", "function"}, {"name", "f"}}})},
           {"text", {{"format", {{"type", "json_object"}}}}}}),
      "text.format");
}

TEST(OpenAIResponsesRequestTest, ValidatesMetadataLimits) {
  Json metadata = Json::object();
  for (int32_t i = 0; i < 16; ++i) {
    metadata["key" + std::to_string(i)] = std::string(512, 'v');
  }
  metadata.erase("key0");
  metadata[std::string(64, 'k')] = "value";
  auto [status, request] = parse_responses_request(
      with_fields({{"metadata", metadata}}).dump(), "loaded-model", nullptr);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(request.metadata, metadata);
  metadata["extra"] = "value";
  expect_invalid(with_fields({{"metadata", metadata}}), "metadata");
  expect_invalid(with_fields({{"metadata", {{std::string(65, 'k'), "value"}}}}),
                 "metadata." + std::string(65, 'k'));
  expect_invalid(with_fields({{"metadata", {{"key", std::string(513, 'v')}}}}),
                 "metadata.key");
}

}  // namespace
}  // namespace xllm::api_service
