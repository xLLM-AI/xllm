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

#include <brpc/callback.h>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "api_service/chat_response_builder.h"
#include "api_service/openai_json.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "core/disaggregation/pd_response.h"
#include "core/distributed_runtime/http_callback_channel_cache.h"
#include "core/distributed_runtime/xservice_client.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/request/request_output.h"
#include "disagg_pd.pb.h"
#include "xllm_service.pb.h"
#include "xservice.pb.h"

namespace xllm::api_service {
namespace {

struct ReceivedOutput {
  std::string body;
  std::string content_type;
  std::string sequence;
  std::string finished;
  std::string status;
};

class ScopedServiceProtocol final {
 public:
  explicit ScopedServiceProtocol(std::string protocol)
      : previous_(DistributedConfig::get_instance().xservice_rpc_protocol()) {
    DistributedConfig::get_instance().xservice_rpc_protocol(
        std::move(protocol));
  }
  ~ScopedServiceProtocol() {
    DistributedConfig::get_instance().xservice_rpc_protocol(previous_);
  }

 private:
  std::string previous_;
};

OutputCallback decode_callback(const proto::DisaggRequest& request,
                               const std::string& tools,
                               const std::string& reasoning,
                               std::string fingerprint) {
  auto formatter = make_decode_output_formatter(
      request, tools, reasoning, std::move(fingerprint));
  if (!formatter) {
    return {};
  }
  return [formatter,
          request_id = request.req_id(),
          service_request_id = request.service_req_id(),
          address = request.source_xservice_addr(),
          stream = request.stream()](RequestOutput output) {
    output.request_id = request_id;
    output.service_request_id = service_request_id;
    output.target_xservice_addr = address;
    output.stream = stream;
    output.response_formatter = formatter;
    return XServiceClient::get_instance()->generation(output) &&
           (!output.status.has_value() || output.status->ok());
  };
}

// Transport tests supply preformatted payloads through the same public entry.
std::vector<RequestOutput> transport_outputs(
    const std::vector<ServiceGenerationOutput>& outputs) {
  std::vector<RequestOutput> requests;
  requests.reserve(outputs.size());
  for (const auto& output : outputs) {
    RequestOutput request;
    request.service_request_id = output.service_request_id;
    request.target_xservice_addr = output.target_xservice_addr;
    request.finished = true;
    request.response_formatter = std::make_shared<ServiceOutputFormatter>(
        [output](const RequestOutput&,
                 bool,
                 std::optional<ServiceGenerationOutput>* result) {
          *result = output;
          return true;
        });
    requests.emplace_back(std::move(request));
  }
  return requests;
}

using ChatStreamCall = StreamCall<proto::ChatRequest, proto::ChatResponse>;

TEST(HttpCallbackChannelCacheTest, BoundsEndpointsWithoutClosingInflightCalls) {
  HttpCallbackChannelCache cache;
  auto in_flight = std::make_shared<brpc::Channel>();
  std::weak_ptr<brpc::Channel> lifetime = in_flight;
  EXPECT_EQ(cache.insert("first", in_flight), in_flight);
  EXPECT_EQ(cache.find("first"), in_flight);
  EXPECT_EQ(cache.insert("first", std::make_shared<brpc::Channel>()),
            in_flight);
  for (size_t index = 0; index < HttpCallbackChannelCache::kCapacity; ++index) {
    const std::string address = std::to_string(index);
    cache.insert(address, std::make_shared<brpc::Channel>());
  }
  EXPECT_EQ(cache.size(), HttpCallbackChannelCache::kCapacity);
  EXPECT_EQ(cache.find("first"), nullptr);
  EXPECT_FALSE(lifetime.expired());
  in_flight.reset();
  EXPECT_TRUE(lifetime.expired());

  cache.erase("0");
  EXPECT_EQ(cache.find("0"), nullptr);
  EXPECT_EQ(cache.size(), HttpCallbackChannelCache::kCapacity - 1);
  cache.insert("0", std::make_shared<brpc::Channel>());
  cache.insert("newest", std::make_shared<brpc::Channel>());
  EXPECT_EQ(cache.size(), HttpCallbackChannelCache::kCapacity);
  EXPECT_NE(cache.find("0"), nullptr);
  EXPECT_NE(cache.find("newest"), nullptr);
  EXPECT_EQ(cache.find("1"), nullptr);
}

TEST(StreamCallOutputTest, AsyncAdmissionReturnsJsonWithoutStartingSse) {
  for (bool stream : {false, true}) {
    proto::ChatRequest request;
    request.set_stream(stream);
    proto::ChatResponse response;
    brpc::Controller controller;
    bool completed = false;
    {
      ChatStreamCall call(&controller,
                          brpc::NewCallback(
                              +[](bool* done) { *done = true; }, &completed),
                          &request,
                          &response,
                          /*use_arena=*/true,
                          /*is_http_request=*/true);
      call.accept_async();
      EXPECT_FALSE(completed);
      EXPECT_EQ(controller.http_response().status_code(), 202);
      EXPECT_EQ(controller.http_response().content_type(), "application/json");
      EXPECT_EQ(nlohmann::json::parse(
                    controller.response_attachment().to_string())["accepted"],
                true);
    }
    EXPECT_TRUE(completed);
  }
}

TEST(OpenAIResponseEncodingTest, StreamingUsageAndFingerprintShareOneEncoder) {
  proto::ChatRequest request;
  request.set_stream(true);
  request.mutable_stream_options()->set_include_usage(true);
  request.mutable_stream_options()->set_continuous_usage_stats(true);
  proto::Usage usage;
  usage.set_prompt_tokens(3);
  usage.set_completion_tokens(1);
  usage.set_total_tokens(4);
  proto::ChatResponse response;
  response.set_object("chat.completion.chunk");
  response.add_choices()->mutable_delta()->set_content("A");
  const auto delta = nlohmann::json::parse(serialize_openai_response(
      request, response, usage, "fingerprint", /*delta=*/true));
  EXPECT_EQ(delta["usage"]["completion_tokens"], 1);
  EXPECT_FALSE(delta.contains("system_fingerprint"));
  response.clear_choices();
  *response.mutable_usage() = usage;
  const auto terminal = nlohmann::json::parse(serialize_openai_response(
      request, response, usage, "fingerprint", /*delta=*/true));
  EXPECT_EQ(terminal["system_fingerprint"], "fingerprint");
}

TEST(ChatResponseBuilderTest, EmitsTypedFramesAndStopsWhenConsumerRejects) {
  RequestOutput output;
  output.outputs.emplace_back();
  output.outputs.back().index = 0;
  output.outputs.back().text = "A";
  output.outputs.back().finish_reason = "stop";
  output.finished = true;
  proto::ChatResponse frame;
  std::unordered_set<size_t> first_messages;
  std::vector<proto::ChatResponse> frames;
  frames.reserve(3);
  EXPECT_TRUE(build_chat_delta(frame,
                               /*include_usage=*/false,
                               &first_messages,
                               "request",
                               /*created_time=*/1,
                               "model",
                               output,
                               /*stream_parser=*/nullptr,
                               [&frames](const proto::ChatResponse& response) {
                                 frames.emplace_back(response);
                                 return true;
                               }));
  ASSERT_EQ(frames.size(), 3U);
  EXPECT_EQ(frames[0].choices(0).delta().role(), "assistant");
  EXPECT_EQ(frames[1].choices(0).delta().content(), "A");
  EXPECT_EQ(frames[2].choices(0).finish_reason(), "stop");
  int32_t rejected_frames = 0;
  EXPECT_FALSE(build_chat_delta(
      frame,
      /*include_usage=*/false,
      &first_messages,
      "request",
      /*created_time=*/1,
      "model",
      output,
      /*stream_parser=*/nullptr,
      [&rejected_frames](const proto::ChatResponse& /*response*/) {
        ++rejected_frames;
        return false;
      }));
  EXPECT_EQ(rejected_frames, 1);
}

class ResultService final : public proto::XllmAPIService {
 public:
  void ChatCompletionsHttp(google::protobuf::RpcController* controller,
                           const proto::HttpRequest* /*request*/,
                           proto::HttpResponse* /*response*/,
                           google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto batch =
        nlohmann::json::parse(ctrl->request_attachment().to_string());
    batches_.emplace_back(batch);
    remote_ports_.emplace_back(ctrl->remote_side().port);
    nlohmann::json accepted = nlohmann::json::array();
    for (const auto& item : batch["results"]) {
      outputs_.emplace_back(
          ReceivedOutput{item["payload"].get<std::string>(),
                         item["content_type"].get<std::string>(),
                         std::to_string(item["sequence"].get<uint64_t>()),
                         item["finished"].get<bool>() ? "true" : "false",
                         std::to_string(item["status_code"].get<int32_t>())});
      accepted.emplace_back(item["attempt_id"] != "rejected");
    }
    ctrl->http_response().set_status_code(reject_ ? 409 : 200);
    ctrl->http_response().set_content_type("application/json");
    ctrl->response_attachment().append(
        invalid_ack_ ? "{\"accepted\":[]}"
                     : nlohmann::json{{"accepted", accepted}}.dump());
  }

  std::vector<ReceivedOutput> outputs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return outputs_;
  }

  void reject() {
    std::lock_guard<std::mutex> lock(mutex_);
    reject_ = true;
  }

  std::vector<nlohmann::json> batches() {
    std::lock_guard<std::mutex> lock(mutex_);
    return batches_;
  }

  std::vector<int32_t> remote_ports() {
    std::lock_guard<std::mutex> lock(mutex_);
    return remote_ports_;
  }

  void invalidate_ack() {
    std::lock_guard<std::mutex> lock(mutex_);
    invalid_ack_ = true;
  }

 private:
  std::mutex mutex_;
  std::vector<ReceivedOutput> outputs_;
  std::vector<nlohmann::json> batches_;
  std::vector<int32_t> remote_ports_;
  bool reject_ = false;
  bool invalid_ack_ = false;
};

class PDHttpResponseTest : public ::testing::Test {
 protected:
  ScopedServiceProtocol protocol_{"http"};
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/xllm/v1/results => ChatCompletionsHttp"),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    request_.set_model("model");
    request_.set_req_id("attempt");
    request_.set_service_req_id("attempt");
    request_.set_source_xservice_addr(
        "127.0.0.1:" + std::to_string(server_.listen_address().port));
  }

  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  OutputCallback callback(const std::string& tool_format = "",
                          const std::string& reasoning_format = "") {
    return decode_callback(request_,
                           tool_format,
                           reasoning_format,
                           /*system_fingerprint=*/"xllm-test-fingerprint");
  }

  ResultService service_;
  brpc::Server server_;
  proto::DisaggRequest request_;
};

RequestOutput output(std::string text, bool finished = false) {
  RequestOutput result;
  result.outputs.emplace_back();
  result.outputs.back().index = 0;
  result.outputs.back().text = std::move(text);
  if (finished) {
    result.outputs.back().finish_reason = "stop";
    result.usage = Usage{4, 2, 6, 2};
  }
  result.finished = finished;
  return result;
}

std::vector<nlohmann::json> chunks(const std::vector<ReceivedOutput>& outputs) {
  std::vector<nlohmann::json> result;
  for (const auto& message : outputs) {
    size_t offset = 0;
    while (offset < message.body.size()) {
      const size_t end = message.body.find("\n\n", offset);
      if (end == std::string::npos) {
        ADD_FAILURE() << "Incomplete SSE frame: " << message.body;
        return result;
      }
      const std::string frame = message.body.substr(offset, end - offset);
      offset = end + 2;
      if (frame == "data: [DONE]") {
        continue;
      }
      result.emplace_back(nlohmann::json::parse(frame.substr(6)));
    }
  }
  return result;
}

TEST_F(PDHttpResponseTest,
       StreamsFirstTokenExactlyOnceWithUsageAndOneTerminal) {
  request_.set_stream(true);
  request_.mutable_stream_options()->set_include_usage(true);
  auto emit = callback();
  ASSERT_TRUE(emit);
  ASSERT_TRUE(emit(output("A")));
  ASSERT_TRUE(emit(output("B", /*finished=*/true)));
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 2U);
  EXPECT_EQ(received[0].sequence, "0");
  EXPECT_EQ(received[1].sequence, "1");
  EXPECT_EQ(received[0].finished, "false");
  EXPECT_EQ(received[1].finished, "true");
  EXPECT_EQ(received[0].content_type, "text/event-stream");
  EXPECT_TRUE(received.back().body.ends_with("data: [DONE]\n\n"));
  std::string text;
  int32_t roles = 0;
  int32_t finishes = 0;
  for (const auto& chunk : chunks(received)) {
    EXPECT_EQ(chunk["object"], "chat.completion.chunk");
    EXPECT_EQ(chunk["id"], "attempt");
    for (const auto& choice : chunk["choices"]) {
      const auto& delta = choice["delta"];
      if (delta.contains("role")) {
        ++roles;
      }
      if (delta.contains("content") && delta["content"].is_string()) {
        text += delta["content"].get<std::string>();
      }
      if (!choice["finish_reason"].is_null()) {
        ++finishes;
      }
    }
  }
  EXPECT_EQ(text, "AB");
  EXPECT_EQ(roles, 1);
  EXPECT_EQ(finishes, 1);
  const auto parsed = chunks(received);
  EXPECT_EQ(parsed.back()["usage"]["completion_tokens"], 2);
  EXPECT_EQ(parsed.back()["usage"]["prompt_tokens_details"]["cached_tokens"],
            2);
  EXPECT_EQ(parsed.back()["system_fingerprint"], "xllm-test-fingerprint");
  for (size_t index = 0; index + 1 < parsed.size(); ++index) {
    EXPECT_FALSE(parsed[index].contains("system_fingerprint"));
  }
  EXPECT_FALSE(emit(output("duplicate", /*finished=*/true)));
  EXPECT_EQ(service_.outputs().size(), 2U);
}

TEST_F(PDHttpResponseTest, NonstreamPreservesChoicesLogprobsAndStopReason) {
  auto emit = callback();
  RequestOutput result = output("A", /*finished=*/true);
  result.outputs[0].stop_reason = int32_t{42};
  LogProb logprob;
  logprob.token = "A";
  logprob.token_id = 65;
  logprob.logprob = -0.25F;
  logprob.top_logprobs = std::vector<LogProbData>{{"A", 65, -0.25F, true}};
  result.outputs[0].logprobs = std::vector<LogProb>{logprob};
  result.outputs.emplace_back();
  result.outputs.back().index = 1;
  result.outputs.back().text = "B";
  result.outputs.back().finish_reason = "length";
  ASSERT_TRUE(emit(std::move(result)));
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 1U);
  EXPECT_EQ(received[0].content_type, "application/json");
  EXPECT_EQ(received[0].finished, "true");
  const auto json = nlohmann::json::parse(received[0].body);
  EXPECT_EQ(json["object"], "chat.completion");
  EXPECT_EQ(json["system_fingerprint"], "xllm-test-fingerprint");
  ASSERT_EQ(json["choices"].size(), 2U);
  EXPECT_EQ(json["choices"][0]["message"]["content"], "A");
  EXPECT_EQ(json["choices"][0]["stop_reason"], 42);
  EXPECT_EQ(json["choices"][0]["logprobs"]["content"][0]["token"], "A");
  EXPECT_EQ(json["choices"][1]["finish_reason"], "length");
}

TEST_F(PDHttpResponseTest,
       CompletionWireOptionsSelectTextResponseForAllChoices) {
  request_.set_response_type(proto::TEXT_COMPLETION);
  proto::DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(request_.SerializeAsString()));
  auto emit = decode_callback(decoded, "qwen25", "qwen3", "fingerprint");
  ASSERT_TRUE(emit);
  // Echoed prompt text and parser markers are already decoded by core.
  RequestOutput result = output("prompt<think>A", /*finished=*/true);
  LogProb logprob;
  logprob.token = "A";
  logprob.token_id = 65;
  logprob.logprob = -0.25F;
  logprob.top_logprobs = std::vector<LogProbData>{{"A", 65, -0.25F, true}};
  result.outputs.front().logprobs = std::vector<LogProb>{logprob};
  result.outputs.emplace_back();
  result.outputs.back().index = 1;
  result.outputs.back().text = "promptB";
  result.outputs.back().finish_reason = "length";
  ASSERT_TRUE(emit(std::move(result)));
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 1U);
  EXPECT_EQ(received.front().content_type, "application/json");
  EXPECT_EQ(received.front().finished, "true");
  const auto json = nlohmann::json::parse(received.front().body);
  EXPECT_EQ(json["object"], "text_completion");
  EXPECT_EQ(json["id"], "attempt");
  EXPECT_EQ(json["model"], "model");
  EXPECT_EQ(json["system_fingerprint"], "fingerprint");
  ASSERT_EQ(json["choices"].size(), 2U);
  EXPECT_EQ(json["choices"][0]["text"], "prompt<think>A");
  EXPECT_FALSE(json["choices"][0].contains("message"));
  EXPECT_EQ(json["choices"][0]["logprobs"]["tokens"][0], "A");
  EXPECT_EQ(json["choices"][0]["logprobs"]["top_logprobs"][0]["A"], -0.25F);
  EXPECT_EQ(json["choices"][1]["text"], "promptB");
  EXPECT_EQ(json["choices"][1]["finish_reason"], "length");
  EXPECT_EQ(json["usage"]["completion_tokens"], 2);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
}

TEST_F(PDHttpResponseTest, CompletionStreamsFirstTokenAndTracksLogprobOffsets) {
  request_.set_response_type(proto::TEXT_COMPLETION);
  request_.set_stream(true);
  request_.mutable_stream_options()->set_include_usage(true);
  auto emit = callback("qwen25", "qwen3");
  ASSERT_TRUE(emit);
  RequestOutput first = output("é");
  LogProb logprob;
  logprob.token = "é";
  logprob.token_id = 42;
  logprob.logprob = -0.5F;
  first.outputs.front().logprobs = std::vector<LogProb>{logprob};
  ASSERT_TRUE(emit(std::move(first)));
  RequestOutput last = output("B", /*finished=*/true);
  logprob.token = "B";
  last.outputs.front().logprobs = std::vector<LogProb>{logprob};
  ASSERT_TRUE(emit(std::move(last)));
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 2U);
  EXPECT_EQ(received[0].sequence, "0");
  EXPECT_EQ(received[0].finished, "false");
  EXPECT_EQ(received[1].sequence, "1");
  EXPECT_EQ(received[1].finished, "true");
  const auto parsed = chunks(received);
  ASSERT_EQ(parsed.size(), 4U);
  for (const auto& chunk : parsed) {
    EXPECT_EQ(chunk["object"], "text_completion");
  }
  EXPECT_EQ(parsed[0]["choices"][0]["text"], "é");
  EXPECT_EQ(parsed[0]["choices"][0]["logprobs"]["text_offset"][0], 0);
  EXPECT_EQ(parsed[1]["choices"][0]["text"], "B");
  EXPECT_EQ(parsed[1]["choices"][0]["logprobs"]["text_offset"][0], 1);
  EXPECT_EQ(parsed[2]["choices"][0]["finish_reason"], "stop");
  EXPECT_TRUE(parsed[3]["choices"].empty());
  EXPECT_EQ(parsed[3]["usage"]["completion_tokens"], 2);
  EXPECT_EQ(parsed[3]["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
  EXPECT_EQ(parsed[3]["system_fingerprint"], "xllm-test-fingerprint");
  EXPECT_TRUE(received.back().body.ends_with("data: [DONE]\n\n"));
  EXPECT_FALSE(emit(output("duplicate", /*finished=*/true)));
  EXPECT_EQ(service_.outputs().size(), 2U);
}

TEST_F(PDHttpResponseTest, RejectsUnknownResponseTypeWithoutSendingChatData) {
  request_.set_response_type(static_cast<proto::CompletionResponseType>(99));
  EXPECT_FALSE(callback());
  EXPECT_TRUE(service_.outputs().empty());
}

TEST_F(PDHttpResponseTest, StreamsFingerprintAtFinishWithoutUsageFrame) {
  request_.set_stream(true);
  auto emit = callback();
  ASSERT_TRUE(emit);
  ASSERT_TRUE(emit(output("A")));
  ASSERT_TRUE(emit(output("B", /*finished=*/true)));
  const auto parsed = chunks(service_.outputs());
  ASSERT_FALSE(parsed.empty());
  EXPECT_EQ(parsed.back()["choices"][0]["finish_reason"], "stop");
  EXPECT_EQ(parsed.back()["system_fingerprint"], "xllm-test-fingerprint");
  for (size_t index = 0; index + 1 < parsed.size(); ++index) {
    EXPECT_FALSE(parsed[index].contains("system_fingerprint"));
  }
}

TEST_F(PDHttpResponseTest, ParsesReasoningAcrossTheFirstTokenBoundary) {
  request_.set_stream(true);
  auto emit = callback(/*tool_format=*/"", /*reasoning_format=*/"qwen3");
  RequestOutput first = output("<think>");
  LogProb logprob;
  logprob.token = "<think>";
  logprob.token_id = 42;
  logprob.logprob = -0.25F;
  first.outputs.front().logprobs = std::vector<LogProb>{logprob};
  ASSERT_TRUE(emit(std::move(first)));
  ASSERT_TRUE(emit(output("why</think>answer", /*finished=*/true)));
  std::string reasoning;
  std::string content;
  int32_t logprob_tokens = 0;
  for (const auto& chunk : chunks(service_.outputs())) {
    for (const auto& choice : chunk["choices"]) {
      if (choice.contains("logprobs") && choice["logprobs"].is_object()) {
        logprob_tokens += choice["logprobs"]["content"].size();
      }
      const auto& delta = choice["delta"];
      if (delta.contains("reasoning") && delta["reasoning"].is_string()) {
        reasoning += delta["reasoning"].get<std::string>();
      }
      if (delta.contains("content") && delta["content"].is_string()) {
        content += delta["content"].get<std::string>();
      }
    }
  }
  EXPECT_EQ(reasoning, "why");
  EXPECT_EQ(content, "answer");
  EXPECT_EQ(logprob_tokens, 1);
}

TEST_F(PDHttpResponseTest, ParsesToolCallAndPreservesItsIdAcrossChunks) {
  request_.set_stream(true);
  auto* tool = request_.add_tools();
  tool->set_type("function");
  tool->mutable_function()->set_name("weather");
  request_.set_req_id("decode-output");
  request_.set_model("decode-model");
  request_.set_tool_choice("auto");
  (*request_.mutable_chat_template_kwargs()
        ->mutable_fields())["enable_thinking"]
      .set_bool_value(false);
  request_.mutable_stream_options()->set_include_usage(true);
  request_.mutable_stream_options()->set_continuous_usage_stats(true);
  proto::DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(request_.SerializeAsString()));
  auto emit = decode_callback(decoded, "qwen25", "qwen3", "decode-fingerprint");
  ASSERT_TRUE(emit);
  ASSERT_TRUE(emit(output("<tool_call>\n")));
  ASSERT_TRUE(emit(output(
      "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
      /*finished=*/true)));
  int32_t ids = 0;
  std::string name;
  std::string arguments;
  bool tool_finish = false;
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 2U);
  EXPECT_EQ(received.front().sequence, "0");
  EXPECT_EQ(received.back().finished, "true");
  const auto parsed = chunks(received);
  ASSERT_FALSE(parsed.empty());
  for (const auto& chunk : parsed) {
    EXPECT_EQ(chunk["id"], "decode-output");
    EXPECT_EQ(chunk["model"], "decode-model");
    EXPECT_TRUE(chunk.contains("usage"));
    for (const auto& choice : chunk["choices"]) {
      tool_finish |= choice["finish_reason"] == "tool_calls";
      const auto& delta = choice["delta"];
      if (!delta.contains("tool_calls")) {
        continue;
      }
      for (const auto& call : delta["tool_calls"]) {
        if (call.contains("id") && !call["id"].is_null()) {
          ++ids;
        }
        const auto& function = call["function"];
        if (function.contains("name")) {
          name += function["name"].get<std::string>();
        }
        if (function.contains("arguments")) {
          arguments += function["arguments"].get<std::string>();
        }
      }
    }
  }
  EXPECT_EQ(ids, 1);
  EXPECT_EQ(name, "weather");
  EXPECT_EQ(nlohmann::json::parse(arguments)["city"], "Paris");
  EXPECT_TRUE(tool_finish);
  EXPECT_EQ(parsed.back()["system_fingerprint"], "decode-fingerprint");
  EXPECT_EQ(parsed.back()["usage"]["completion_tokens"], 2);
}

TEST_F(PDHttpResponseTest, PrefillErrorsNeedNoTokenResponseSession) {
  const std::vector<std::pair<StatusCode, std::string>> errors = {
      {StatusCode::INVALID_ARGUMENT, "400"},
      {StatusCode::RATE_LIMITED, "429"},
      {StatusCode::RESOURCE_EXHAUSTED, "500"},
      {StatusCode::UNKNOWN, "500"}};
  for (bool stream : {false, true}) {
    for (const auto& [code, status] : errors) {
      RequestOutput failed(Status(code, "prefill failed"),
                           request_.service_req_id(),
                           request_.source_xservice_addr());
      failed.stream = stream;
      EXPECT_EQ(XServiceClient::get_instance()->generations({failed}),
                std::vector<bool>({true}));
      const auto received = service_.outputs();
      const auto& result = received.back();
      EXPECT_EQ(result.sequence, "0");
      EXPECT_EQ(result.finished, "true");
      EXPECT_EQ(result.status, status);
      if (stream) {
        EXPECT_EQ(chunks({result}).front()["error"]["message"],
                  "prefill failed");
        EXPECT_NE(result.body.find("data: [DONE]"), std::string::npos);
      } else {
        EXPECT_EQ(nlohmann::json::parse(result.body)["error"]["message"],
                  "prefill failed");
      }
      EXPECT_EQ(result.body.find("choices"), std::string::npos);
    }
  }
  RequestOutput invalid(Status(StatusCode::UNKNOWN, "failure"));
  EXPECT_EQ(XServiceClient::get_instance()->generations({invalid}),
            std::vector<bool>({false}));
  EXPECT_EQ(service_.outputs().size(), errors.size() * 2);
}

TEST_F(PDHttpResponseTest, ErrorsAndRejectedAcknowledgementsStopWithoutReplay) {
  auto emit = callback();
  ASSERT_TRUE(emit);
  RequestOutput error;
  error.status = Status(StatusCode::INVALID_ARGUMENT, "bad sampling parameter");
  EXPECT_FALSE(emit(std::move(error)));
  auto received = service_.outputs();
  ASSERT_EQ(received.size(), 1U);
  EXPECT_EQ(received[0].status, "400");
  EXPECT_EQ(received[0].finished, "true");
  EXPECT_EQ(nlohmann::json::parse(received[0].body)["error"]["message"],
            "bad sampling parameter");
  service_.reject();
  request_.set_stream(true);
  auto rejected = callback();
  EXPECT_FALSE(rejected(output("A")));
  EXPECT_EQ(service_.outputs().size(), 2U);
  EXPECT_FALSE(rejected(output("B")));
  EXPECT_EQ(service_.outputs().size(), 2U);
}

TEST_F(PDHttpResponseTest, RejectsMissingOutputOwnershipBeforeCreatingOutput) {
  EXPECT_FALSE(decode_callback(proto::DisaggRequest{}, "", "", ""));
  request_.clear_source_xservice_addr();
  EXPECT_FALSE(callback());
  request_.set_source_xservice_addr("127.0.0.1:1");
  request_.clear_service_req_id();
  EXPECT_FALSE(callback());
  EXPECT_TRUE(service_.outputs().empty());
}

TEST_F(PDHttpResponseTest,
       BatchesFormattedResponsesAndReusesTheHttpConnection) {
  request_.set_stream(true);
  auto first_formatter = make_decode_output_formatter(request_, "", "", "");
  request_.set_req_id("second");
  request_.set_service_req_id("second");
  request_.set_stream(false);
  auto second_formatter = make_decode_output_formatter(request_, "", "", "");
  ASSERT_TRUE(first_formatter);
  ASSERT_TRUE(second_formatter);
  RequestOutput first = output("汉字\n\"quoted\"");
  first.service_request_id = "attempt";
  first.target_xservice_addr = request_.source_xservice_addr();
  first.response_formatter = first_formatter;
  RequestOutput second = output("B", /*finished=*/true);
  second.service_request_id = "second";
  second.target_xservice_addr = request_.source_xservice_addr();
  second.response_formatter = second_formatter;
  EXPECT_EQ(XServiceClient::get_instance()->generations({first, second}),
            std::vector<bool>({true, true}));
  ASSERT_EQ(service_.batches().size(), 1U);
  const auto received = service_.outputs();
  ASSERT_EQ(received.size(), 2U);
  EXPECT_EQ(received[0].content_type, "text/event-stream");
  EXPECT_EQ(received[1].content_type, "application/json");
  EXPECT_EQ(nlohmann::json::parse(
                received[1].body)["choices"][0]["message"]["content"],
            "B");
  RequestOutput next = output("C", /*finished=*/true);
  next.service_request_id = "attempt";
  next.target_xservice_addr = request_.source_xservice_addr();
  next.response_formatter = first_formatter;
  EXPECT_EQ(XServiceClient::get_instance()->generations({next}),
            std::vector<bool>({true}));
  EXPECT_EQ(service_.outputs().back().sequence, "1");
  const auto ports = service_.remote_ports();
  ASSERT_EQ(ports.size(), 2U);
  EXPECT_EQ(ports[0], ports[1]);
}

TEST_F(PDHttpResponseTest, BatchAcknowledgementsArePerRequestAndNeverReplayed) {
  ServiceGenerationOutput healthy;
  healthy.service_request_id = "healthy";
  healthy.target_xservice_addr = request_.source_xservice_addr();
  healthy.payload = "data: healthy\n\n";
  healthy.stream = true;
  ServiceGenerationOutput rejected = healthy;
  rejected.service_request_id = "rejected";
  const std::vector<ServiceGenerationOutput> batch{healthy, rejected};
  EXPECT_EQ(
      XServiceClient::get_instance()->generations(transport_outputs(batch)),
      std::vector<bool>({true, false}));
  service_.invalidate_ack();
  EXPECT_EQ(
      XServiceClient::get_instance()->generations(transport_outputs(batch)),
      std::vector<bool>({false, false}));
  service_.reject();
  EXPECT_EQ(
      XServiceClient::get_instance()->generations(transport_outputs(batch)),
      std::vector<bool>({false, false}));
  EXPECT_EQ(service_.batches().size(), 3U);
  const std::vector<ServiceGenerationOutput> split_batch(257, healthy);
  const auto accepted = XServiceClient::get_instance()->generations(
      transport_outputs(split_batch));
  EXPECT_EQ(accepted, std::vector<bool>(split_batch.size(), false));
  // The first chunk's ambiguous failure must not issue the second chunk.
  EXPECT_EQ(service_.batches().size(), 4U);
}

TEST_F(PDHttpResponseTest,
       MapsBufferedAndRejectedOutputsAndStopsFailedContexts) {
  ServiceGenerationOutput healthy;
  healthy.service_request_id = "healthy";
  healthy.target_xservice_addr = request_.source_xservice_addr();
  healthy.payload = "data: A\n\n";
  ServiceGenerationOutput rejected = healthy;
  rejected.service_request_id = "rejected";
  auto outputs = transport_outputs({healthy, rejected});
  RequestOutput buffered = output("partial parser delimiter");
  buffered.response_formatter = std::make_shared<ServiceOutputFormatter>(
      [](const RequestOutput&, bool, std::optional<ServiceGenerationOutput>*) {
        return true;
      });
  outputs.insert(outputs.begin() + 1, std::move(buffered));
  EXPECT_EQ(XServiceClient::get_instance()->generations(outputs),
            std::vector<bool>({true, true, false}));
  EXPECT_EQ(service_.outputs().size(), 2U);
  EXPECT_FALSE(*outputs.back().response_formatter);
  EXPECT_EQ(XServiceClient::get_instance()->generations({outputs.back()}),
            std::vector<bool>({false}));
  EXPECT_EQ(service_.outputs().size(), 2U);
}

TEST_F(PDHttpResponseTest, SplitsLargeBatchesAndIsolatesInvalidItemsAndOwners) {
  ServiceGenerationOutput result;
  result.target_xservice_addr = request_.source_xservice_addr();
  result.payload = "data: A\n\n";
  result.stream = true;
  std::vector<ServiceGenerationOutput> outputs;
  outputs.reserve(260);
  for (int32_t index = 0; index < 257; ++index) {
    result.service_request_id = std::to_string(index);
    outputs.emplace_back(result);
  }
  result.payload.assign(1024 * 1024, 'x');
  result.service_request_id = "oversized";
  outputs.emplace_back(result);
  result.payload = "data: unreachable\n\n";
  result.target_xservice_addr = "127.0.0.1:1";
  outputs.emplace_back(result);
  outputs.emplace_back(ServiceGenerationOutput{});
  const auto accepted =
      XServiceClient::get_instance()->generations(transport_outputs(outputs));
  ASSERT_EQ(accepted.size(), 260U);
  for (size_t index = 0; index < accepted.size(); ++index) {
    EXPECT_EQ(accepted[index], index < 257);
  }
  const auto batches = service_.batches();
  ASSERT_EQ(batches.size(), 2U);
  EXPECT_EQ(batches[0]["results"].size(), 256U);
  EXPECT_EQ(batches[1]["results"].size(), 2U);
  const auto& error = batches[1]["results"][1];
  EXPECT_EQ(error["attempt_id"], "oversized");
  EXPECT_EQ(error["finished"], true);
  EXPECT_EQ(error["status_code"], 500);
  EXPECT_NE(
      error["payload"].get<std::string>().find("HTTP callback body limit"),
      std::string::npos);
  outputs.clear();
  result.target_xservice_addr = request_.source_xservice_addr();
  result.payload.assign(400 * 1024, 'x');
  for (int32_t index = 0; index < 3; ++index) {
    result.service_request_id = std::to_string(index);
    outputs.emplace_back(result);
  }
  EXPECT_EQ(
      XServiceClient::get_instance()->generations(transport_outputs(outputs)),
      std::vector<bool>(3, true));
  const auto byte_limited = service_.batches();
  ASSERT_EQ(byte_limited.size(), 4U);
  EXPECT_EQ(byte_limited[2]["results"].size(), 2U);
  EXPECT_EQ(byte_limited[3]["results"].size(), 1U);
  EXPECT_LE(byte_limited[2].dump().size(), 1024U * 1024U);
  EXPECT_LE(byte_limited[3].dump().size(), 1024U * 1024U);
}

TEST_F(PDHttpResponseTest,
       OversizeStopsPendingOutputAndKeepsOtherAttemptsAlive) {
  for (bool stream : {false, true}) {
    ServiceGenerationOutput result;
    result.target_xservice_addr = request_.source_xservice_addr();
    result.service_request_id = stream ? "large-stream" : "large-json";
    result.stream = stream;
    result.sequence = 7;
    // Escaping the outer envelope, rather than payload size alone, exceeds
    // the wire limit here.
    result.payload.assign(600 * 1024, '"');
    ServiceGenerationOutput healthy = result;
    healthy.service_request_id += "-healthy";
    healthy.payload = stream ? "data: A\n\n" : R"({"choices":[]})";
    auto outputs = transport_outputs({result, healthy});
    EXPECT_EQ(XServiceClient::get_instance()->generations(outputs),
              std::vector<bool>({false, true}));
    EXPECT_FALSE(*outputs[0].response_formatter);
    const auto batches = service_.batches();
    const auto& error = batches.back()["results"][0];
    EXPECT_EQ(error["attempt_id"], result.service_request_id);
    EXPECT_EQ(error["sequence"], 7);
    EXPECT_EQ(error["finished"], true);
    EXPECT_EQ(error["status_code"], 500);
    const std::string payload = error["payload"];
    EXPECT_NE(payload.find("HTTP callback body limit"), std::string::npos);
    if (stream) {
      EXPECT_TRUE(payload.starts_with("data: "));
      EXPECT_TRUE(payload.ends_with("data: [DONE]\n\n"));
    } else {
      EXPECT_TRUE(nlohmann::json::parse(payload).contains("error"));
    }
    const size_t delivered = service_.outputs().size();
    EXPECT_EQ(XServiceClient::get_instance()->generations({outputs[0]}),
              std::vector<bool>({false}));
    EXPECT_EQ(service_.outputs().size(), delivered);
  }
}

class NativeResultService final : public xllm_service::proto::XllmRpcService {
 public:
  void Generations(google::protobuf::RpcController* /*controller*/,
                   const proto::DisaggStreamGenerations* request,
                   proto::StatusSet* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    received_ = *request;
    for (int32_t i = 0; i < request->gens_size(); ++i) {
      response->add_all_status()->set_ok(true);
    }
  }

  proto::DisaggStreamGenerations received() {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_;
  }

 private:
  std::mutex mutex_;
  proto::DisaggStreamGenerations received_;
};

TEST(PDNativeResponseTest, KeepsLegacyBrpcInterfaceAndFirstTokenMetadata) {
  ScopedServiceProtocol protocol("baidu_std");
  NativeResultService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  RequestOutput result = output("AB", /*finished=*/true);
  result.request_id = "native-attempt";
  result.service_request_id = "native-service-request";
  result.target_xservice_addr =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  result.outputs.front().token_ids = {65, 66};
  result.outputs.front().stop_reason = int32_t{42};
  LogProb logprob;
  logprob.token = "A";
  logprob.token_id = 65;
  logprob.logprob = -0.25F;
  result.outputs.front().logprobs = std::vector<LogProb>{logprob};
  proto::DisaggRequest options;
  options.set_req_id(result.request_id);
  options.set_service_req_id(result.service_request_id);
  options.set_source_xservice_addr(result.target_xservice_addr);
  options.set_model("native-model");
  result.response_formatter = make_decode_output_formatter(options, "", "", "");
  ASSERT_TRUE(result.response_formatter);
  const auto accepted = XServiceClient::get_instance()->generations({result});
  ASSERT_EQ(accepted.size(), 1U);
  EXPECT_TRUE(accepted.front());
  const auto request = service.received();
  ASSERT_EQ(request.gens_size(), 1);
  const auto& generation = request.gens(0);
  EXPECT_EQ(generation.req_id(), "native-attempt");
  EXPECT_EQ(generation.service_req_id(), "native-service-request");
  EXPECT_TRUE(generation.finished());
  EXPECT_FALSE(generation.finished_on_prefill_instance());
  ASSERT_EQ(generation.outputs_size(), 1);
  EXPECT_EQ(generation.outputs(0).text(), "AB");
  ASSERT_EQ(generation.outputs(0).token_ids_size(), 2);
  EXPECT_EQ(generation.outputs(0).token_ids(0), 65);
  EXPECT_EQ(generation.outputs(0).logprobs(0).log_prob_data().token_id(), 65);
  EXPECT_EQ(generation.outputs(0).stop_reason().token_id(), 42);
  EXPECT_EQ(generation.usage().num_generated_tokens(), 2);
  ASSERT_TRUE(generation.has_chat_output());
  ASSERT_EQ(generation.chat_output().frames_size(), 1);
  const auto& frame = generation.chat_output().frames(0);
  EXPECT_EQ(frame.choices(0).message().content(), "AB");
  EXPECT_EQ(frame.choices(0).logprobs().content(0).token_id(), 65);
  EXPECT_EQ(frame.usage().completion_tokens(), 2);
  server.Stop(/*timeout_ms=*/0);
  server.Join();
}

TEST(PDNativeResponseTest, BuffersMarkersAndParsesReasoningOnEngine) {
  proto::DisaggRequest request;
  request.set_req_id("native");
  request.set_service_req_id("service");
  request.set_source_xservice_addr("127.0.0.1:1");
  request.set_stream(true);
  request.mutable_stream_options()->set_include_usage(true);
  auto formatter = make_decode_output_formatter(request, "", "qwen3", "");
  ASSERT_TRUE(formatter);
  std::optional<ServiceGenerationOutput> formatted;
  ASSERT_TRUE((*formatter)(output(""), /*http=*/false, &formatted));
  ASSERT_TRUE(formatted.has_value());
  ASSERT_TRUE(formatted->chat_output);
  EXPECT_EQ(formatted->chat_output->frames(0).choices(0).delta().role(),
            "assistant");
  ASSERT_TRUE((*formatter)(output("<thin"), /*http=*/false, &formatted));
  ASSERT_TRUE(formatted.has_value());
  ASSERT_TRUE(formatted->chat_output);
  EXPECT_TRUE(formatted->chat_output->frames().empty());
  ASSERT_TRUE((*formatter)(output("k>why</think>answer", /*finished=*/true),
                           /*http=*/false,
                           &formatted));
  ASSERT_TRUE(formatted.has_value());
  ASSERT_TRUE(formatted->chat_output);
  std::string reasoning;
  std::string text;
  for (const auto& frame : formatted->chat_output->frames()) {
    for (const auto& choice : frame.choices()) {
      reasoning += choice.delta().reasoning_content();
      text += choice.delta().content();
    }
  }
  EXPECT_EQ(reasoning, "why");
  EXPECT_EQ(text, "answer");
  EXPECT_TRUE(formatted->payload.empty());
  EXPECT_TRUE(formatted->finished);
  EXPECT_FALSE((*formatter)(output("late"), /*http=*/false, &formatted));
}

TEST(PDNativeResponseTest, ParsesToolCallsForRpcDecodeOutput) {
  proto::DisaggRequest request;
  request.set_req_id("native");
  request.set_service_req_id("service");
  request.set_source_xservice_addr("127.0.0.1:1");
  request.set_stream(true);
  auto* tool = request.add_tools();
  tool->set_type("function");
  tool->mutable_function()->set_name("weather");
  auto formatter = make_decode_output_formatter(request, "qwen25", "", "");
  ASSERT_TRUE(formatter);
  std::optional<ServiceGenerationOutput> formatted;
  ASSERT_TRUE(
      (*formatter)(output("<tool_call>\n"), /*http=*/false, &formatted));
  ASSERT_TRUE((*formatter)(output("{\"name\":\"weather\",\"arguments\":{"
                                  "\"city\":\"Paris\"}}\n</tool_call>",
                                  /*finished=*/true),
                           /*http=*/false,
                           &formatted));
  ASSERT_TRUE(formatted.has_value());
  ASSERT_TRUE(formatted->chat_output);
  int32_t ids = 0;
  std::string arguments;
  std::string name;
  bool tool_finish = false;
  for (const auto& frame : formatted->chat_output->frames()) {
    for (const auto& choice : frame.choices()) {
      tool_finish |= choice.finish_reason() == "tool_calls";
      for (const auto& tool : choice.delta().tool_calls()) {
        ids += !tool.id().empty();
        name += tool.function().name();
        arguments += tool.function().arguments();
      }
    }
  }
  EXPECT_EQ(ids, 1);
  EXPECT_EQ(name, "weather");
  EXPECT_EQ(nlohmann::json::parse(arguments)["city"], "Paris");
  EXPECT_TRUE(tool_finish);
}

}  // namespace
}  // namespace xllm::api_service
