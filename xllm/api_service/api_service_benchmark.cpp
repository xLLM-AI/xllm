/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Hop 1 of the request path: api_service.
//
// Micro-benchmarks for the CPU work the API layer performs on every request
// before it hands off to LLMMaster, and on every streamed token on the way
// back out. None of this touches the model or a device.
//
//   Ingress (APIService::CompletionsHttp / ChatCompletionsHttp):
//   * BM_ApiService_CompletionJsonToProto   - preprocess_completion_prompt +
//                                             json2pb::JsonToProtoMessage
//   * BM_ApiService_ChatJsonToProto         - LlmChatJsonParser::preprocess +
//                                             protobuf JsonStringToMessage
//   * BM_ApiService_RequestParamsFromCompletion - proto -> RequestParams
//   * BM_ApiService_RequestParamsFromChat       - proto -> RequestParams
//
//   Ingress cost attribution: the two *JsonToProto benchmarks above run the
//   preprocess pass and the proto parse back to back, which hides how the cost
//   splits between them. These break the two hops apart so that
//   *PreprocessOnly + *<parser>Only ~= *JsonToProto:
//   * BM_ApiService_CompletionPreprocessOnly    - preprocess pass alone
//   * BM_ApiService_CompletionJson2PbOnly       - json2pb parse alone
//   * BM_ApiService_ChatPreprocessOnly          - preprocess pass alone
//   * BM_ApiService_ChatJsonToMessageOnly       - protobuf util parse alone
//   * BM_ApiService_ChatJson2PbOnly             - json2pb parse of the same
//                                                 body, for comparison with
//                                                 the protobuf util one
//
//   Egress (send_delta_to_client_brpc / send_result_to_client_brpc):
//   * BM_ApiService_StreamChunkToJson - one SSE delta: build the
//                                       CompletionResponse chunk and serialize
//                                       it with json2pb the way StreamCall
//                                       does. This is the per-token cost of a
//                                       streaming request.
//   * BM_ApiService_ResultToJson      - one non-stream result with n generated
//                                       tokens (+ logprobs) -> JSON.
//
// Build & run (example):
//   python setup.py test --test-name api_service_benchmark
//   ./api_service_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>
#include <butil/iobuf.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api_service/chat_json_parser.h"
#include "api_service/completion_json_parser.h"
#include "api_service/serving_mode.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/framework/request/usage.h"

namespace xllm {
namespace {

// benchmark 1.8.x exposes only DoNotOptimize(Tp const&) -- deprecated -- and
// DoNotOptimize(Tp&); there is no rvalue overload, so a temporary or a const
// local resolves to the deprecated one. Sinking the value into a non-const
// parameter first selects the supported overload.
template <typename T>
inline BENCHMARK_ALWAYS_INLINE void do_not_optimize(T value) {
  benchmark::DoNotOptimize(value);
}

constexpr char kModel[] = "bench-model";
constexpr char kRequestId[] = "cmpl-0123456789abcdef0123456789abcdef";
constexpr uint32_t kCreatedTime = 1700000000;

std::string make_completion_json(size_t prompt_chars) {
  std::string json = R"({"model":"bench-model","prompt":")";
  json.append(prompt_chars, 'x');
  json += R"(","max_tokens":128,"temperature":0.7,"top_p":0.9,)";
  json += R"("stream":true,"stop":["</s>","<|im_end|>"]})";
  return json;
}

std::string make_chat_json(size_t num_messages) {
  std::string json = R"({"model":"bench-model","messages":[)";
  json += R"({"role":"system","content":"You are a helpful assistant."})";
  for (size_t i = 1; i < num_messages; ++i) {
    json += (i % 2 == 1) ? R"(,{"role":"user","content":")"
                         : R"(,{"role":"assistant","content":")";
    json.append(256, 'y');
    json += R"("})";
  }
  json += R"(],"max_tokens":128,"temperature":0.7,"stream":true})";
  return json;
}

proto::CompletionRequest make_completion_request(size_t prompt_chars) {
  proto::CompletionRequest request;
  request.set_model(kModel);
  request.set_prompt(std::string(prompt_chars, 'x'));
  request.set_max_tokens(128);
  request.set_temperature(0.7f);
  request.set_top_p(0.9f);
  request.set_stream(true);
  request.add_stop("</s>");
  request.add_stop("<|im_end|>");
  request.add_stop_token_ids(2);
  return request;
}

proto::ChatRequest make_chat_request(size_t num_messages) {
  proto::ChatRequest request;
  request.set_model(kModel);
  auto* system = request.add_messages();
  system->set_role("system");
  system->set_content("You are a helpful assistant.");
  for (size_t i = 1; i < num_messages; ++i) {
    auto* message = request.add_messages();
    message->set_role((i % 2 == 1) ? "user" : "assistant");
    message->set_content(std::string(256, 'y'));
  }
  request.set_max_tokens(128);
  request.set_temperature(0.7f);
  request.set_stream(true);
  request.add_stop("</s>");
  return request;
}

// Mirrors the Pb2JsonOptions StreamCall configures in its constructor.
json2pb::Pb2JsonOptions stream_call_json_options() {
  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = false;
  return options;
}

std::vector<LogProb> make_logprobs(size_t num_tokens) {
  std::vector<LogProb> logprobs;
  logprobs.reserve(num_tokens);
  for (size_t i = 0; i < num_tokens; ++i) {
    LogProb logprob;
    logprob.token = "tok";
    logprob.token_id = static_cast<int32_t>(1000 + i);
    logprob.logprob = -0.25f;
    logprobs.emplace_back(std::move(logprob));
  }
  return logprobs;
}

// Same field-by-field copy CompletionServiceImpl's set_logprobs performs.
void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs->empty()) {
    return;
  }
  auto* proto_logprobs = choice->mutable_logprobs();
  const int num_logprobs = static_cast<int>(logprobs->size());
  proto_logprobs->mutable_tokens()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_ids()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_logprobs()->Reserve(num_logprobs);
  for (const auto& logprob : *logprobs) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

// Same as StreamCall::write minus the brpc ProgressiveAttachment::Write.
bool write_sse_chunk(const proto::CompletionResponse& response,
                     const json2pb::Pb2JsonOptions& options,
                     butil::IOBuf* io_buf) {
  io_buf->clear();
  io_buf->append("data: ");
  butil::IOBufAsZeroCopyOutputStream json_output(io_buf);
  std::string err_msg;
  if (!json2pb::ProtoMessageToJson(response, &json_output, options, &err_msg)) {
    return false;
  }
  io_buf->append("\n\n");
  return true;
}

// --------------------------------------------------------------------------
// Ingress
// --------------------------------------------------------------------------

void BM_ApiService_CompletionJsonToProto(benchmark::State& state) {
  const std::string body =
      make_completion_json(static_cast<size_t>(state.range(0)));
  proto::CompletionRequest request;

  for (auto _ : state) {
    // preprocess_completion_prompt takes the body by value, exactly like the
    // HTTP handler hands it `request_attachment().to_string()`.
    auto [status, processed_json] = preprocess_completion_prompt(body);
    request.Clear();
    std::string error;
    json2pb::Json2PbOptions options;
    const bool ok =
        json2pb::JsonToProtoMessage(processed_json, &request, options, &error);
    do_not_optimize(ok);
    do_not_optimize(request.prompt().data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(body.size()));
}

void BM_ApiService_ChatJsonToProto(benchmark::State& state) {
  const std::string body = make_chat_json(static_cast<size_t>(state.range(0)));
  const ChatJsonParser& parser = ChatJsonParser::get(ServingMode::LLM);
  proto::ChatRequest request;

  for (auto _ : state) {
    auto [status, processed_json] = parser.preprocess(body);
    request.Clear();
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const auto parse_status = google::protobuf::util::JsonStringToMessage(
        processed_json, &request, options);
    do_not_optimize(parse_status.ok());
    do_not_optimize(request.messages_size());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(body.size()));
}

// --------------------------------------------------------------------------
// Ingress cost attribution
//
// The *PreprocessOnly benchmarks pass `body` as an lvalue, exactly like the
// combined benchmarks above, so the by-value parameter copy of the request body
// is charged to both and the subtraction against *JsonToProto holds. (The HTTP
// handlers hand over a freshly materialised string and therefore move it, so
// that copy is an artefact of reusing one body across iterations.)
//
// The parse-only benchmarks hoist the preprocess pass out of the loop and time
// the proto parse of its result on its own.
// --------------------------------------------------------------------------

void BM_ApiService_CompletionPreprocessOnly(benchmark::State& state) {
  const std::string body =
      make_completion_json(static_cast<size_t>(state.range(0)));

  for (auto _ : state) {
    auto [status, processed_json] = preprocess_completion_prompt(body);
    do_not_optimize(status.ok());
    do_not_optimize(processed_json.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(body.size()));
}

void BM_ApiService_CompletionJson2PbOnly(benchmark::State& state) {
  const std::string body =
      make_completion_json(static_cast<size_t>(state.range(0)));
  auto [status, processed_json] = preprocess_completion_prompt(body);
  if (!status.ok()) {
    state.SkipWithError("failed to preprocess the benchmark body");
    return;
  }
  proto::CompletionRequest request;

  for (auto _ : state) {
    request.Clear();
    std::string error;
    json2pb::Json2PbOptions options;
    const bool ok =
        json2pb::JsonToProtoMessage(processed_json, &request, options, &error);
    do_not_optimize(ok);
    do_not_optimize(request.prompt().data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(processed_json.size()));
}

void BM_ApiService_ChatPreprocessOnly(benchmark::State& state) {
  const std::string body = make_chat_json(static_cast<size_t>(state.range(0)));
  const ChatJsonParser& parser = ChatJsonParser::get(ServingMode::LLM);

  for (auto _ : state) {
    auto [status, processed_json] = parser.preprocess(body);
    do_not_optimize(status.ok());
    do_not_optimize(processed_json.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(body.size()));
}

void BM_ApiService_ChatJsonToMessageOnly(benchmark::State& state) {
  const std::string body = make_chat_json(static_cast<size_t>(state.range(0)));
  const ChatJsonParser& parser = ChatJsonParser::get(ServingMode::LLM);
  auto [status, processed_json] = parser.preprocess(body);
  if (!status.ok()) {
    state.SkipWithError("failed to preprocess the benchmark body");
    return;
  }
  proto::ChatRequest request;

  for (auto _ : state) {
    request.Clear();
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const auto parse_status = google::protobuf::util::JsonStringToMessage(
        processed_json, &request, options);
    do_not_optimize(parse_status.ok());
    do_not_optimize(request.messages_size());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(processed_json.size()));
}

// Same input as BM_ApiService_ChatJsonToMessageOnly, parsed with json2pb -- the
// parser every other endpoint in api_service.cpp already uses -- so the two are
// directly comparable. ChatRequest.chat_template_kwargs is a
// google.protobuf.Struct, a well-known type json2pb does not map canonically;
// the bodies here carry no such field, so this measures the fast path only.
void BM_ApiService_ChatJson2PbOnly(benchmark::State& state) {
  const std::string body = make_chat_json(static_cast<size_t>(state.range(0)));
  const ChatJsonParser& parser = ChatJsonParser::get(ServingMode::LLM);
  auto [status, processed_json] = parser.preprocess(body);
  if (!status.ok()) {
    state.SkipWithError("failed to preprocess the benchmark body");
    return;
  }
  proto::ChatRequest request;

  for (auto _ : state) {
    request.Clear();
    std::string error;
    json2pb::Json2PbOptions options;
    if (!json2pb::JsonToProtoMessage(
            processed_json, &request, options, &error)) {
      // A failed parse would make the comparison meaningless.
      state.SkipWithError("json2pb failed to parse the chat body");
      break;
    }
    do_not_optimize(request.messages_size());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(processed_json.size()));
}

void BM_ApiService_RequestParamsFromCompletion(benchmark::State& state) {
  const proto::CompletionRequest request =
      make_completion_request(static_cast<size_t>(state.range(0)));
  const std::string x_request_id = "x-rid";
  const std::string x_request_time = "x-rtime";

  for (auto _ : state) {
    RequestParams params(request, x_request_id, x_request_time);
    do_not_optimize(params.request_id.data());
    do_not_optimize(params.stop.has_value());
  }
}

void BM_ApiService_RequestParamsFromChat(benchmark::State& state) {
  const proto::ChatRequest request =
      make_chat_request(static_cast<size_t>(state.range(0)));
  const std::string x_request_id = "x-rid";
  const std::string x_request_time = "x-rtime";

  for (auto _ : state) {
    RequestParams params(request, x_request_id, x_request_time);
    do_not_optimize(params.request_id.data());
    do_not_optimize(params.stop.has_value());
  }
}

// --------------------------------------------------------------------------
// Egress
// --------------------------------------------------------------------------

// range(0): number of logprob entries carried by the delta (0 = logprobs off).
void BM_ApiService_StreamChunkToJson(benchmark::State& state) {
  const size_t num_logprobs = static_cast<size_t>(state.range(0));
  SequenceOutput seq_output;
  seq_output.index = 0;
  seq_output.text = "Hello";
  if (num_logprobs > 0) {
    seq_output.logprobs = make_logprobs(num_logprobs);
  }
  const json2pb::Pb2JsonOptions options = stream_call_json_options();
  proto::CompletionResponse response;
  butil::IOBuf io_buf;

  for (auto _ : state) {
    // Text delta chunk, as built by send_delta_to_client_brpc.
    response.Clear();
    response.set_object("text_completion");
    response.set_id(kRequestId);
    response.set_created(kCreatedTime);
    response.set_model(kModel);
    auto* choice = response.add_choices();
    choice->set_index(static_cast<uint32_t>(seq_output.index));
    choice->set_text(seq_output.text);
    set_logprobs(choice, seq_output.logprobs);
    const bool ok = write_sse_chunk(response, options, &io_buf);
    do_not_optimize(ok);
    do_not_optimize(io_buf.size());
  }
}

// range(0): number of generated tokens in the finished (non-stream) result.
void BM_ApiService_ResultToJson(benchmark::State& state) {
  const size_t num_tokens = static_cast<size_t>(state.range(0));
  RequestOutput req_output;
  SequenceOutput seq_output;
  seq_output.index = 0;
  seq_output.text = std::string(num_tokens * 4, 'z');
  seq_output.finish_reason = "stop";
  seq_output.logprobs = make_logprobs(num_tokens);
  req_output.outputs.emplace_back(std::move(seq_output));
  Usage usage;
  usage.num_prompt_tokens = 128;
  usage.num_generated_tokens = static_cast<int32_t>(num_tokens);
  usage.num_total_tokens = usage.num_prompt_tokens + usage.num_generated_tokens;
  req_output.usage = usage;
  const json2pb::Pb2JsonOptions options = stream_call_json_options();
  proto::CompletionResponse response;
  butil::IOBuf io_buf;

  for (auto _ : state) {
    // Result proto, as built by send_result_to_client_brpc ...
    response.Clear();
    response.set_object("text_completion");
    response.set_id(kRequestId);
    response.set_created(kCreatedTime);
    response.set_model(kModel);
    response.mutable_choices()->Reserve(
        static_cast<int32_t>(req_output.outputs.size()));
    for (const auto& output : req_output.outputs) {
      auto* choice = response.add_choices();
      choice->set_index(static_cast<uint32_t>(output.index));
      choice->set_text(output.text);
      set_logprobs(choice, output.logprobs);
      choice->set_finish_reason(output.finish_reason.value());
    }
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(req_output.usage->num_prompt_tokens);
    proto_usage->set_completion_tokens(req_output.usage->num_generated_tokens);
    proto_usage->set_total_tokens(req_output.usage->num_total_tokens);
    // ... then serialized as StreamCall::write_and_finish does.
    io_buf.clear();
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf);
    std::string err_msg;
    const bool ok =
        json2pb::ProtoMessageToJson(response, &json_output, options, &err_msg);
    do_not_optimize(ok);
    do_not_optimize(io_buf.size());
  }
}

// Prompt length in characters: 64 chars .. 64K chars.
BENCHMARK(BM_ApiService_CompletionJsonToProto)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);
// Conversation length in messages.
BENCHMARK(BM_ApiService_ChatJsonToProto)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
// Same ranges as the combined benchmarks above so the numbers line up.
BENCHMARK(BM_ApiService_CompletionPreprocessOnly)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ApiService_CompletionJson2PbOnly)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ApiService_ChatPreprocessOnly)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ApiService_ChatJsonToMessageOnly)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ApiService_ChatJson2PbOnly)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ApiService_RequestParamsFromCompletion)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ApiService_RequestParamsFromChat)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ApiService_StreamChunkToJson)
    ->Arg(0)
    ->Arg(1)
    ->Unit(benchmark::kNanosecond);
// Generated tokens in the final result.
BENCHMARK(BM_ApiService_ResultToJson)
    ->RangeMultiplier(8)
    ->Range(8, 4096)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm
