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

#include "framework/request/llm_request_factory.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/metrics.h"
#include "core/common/types.h"
#include "framework/chat_template/chat_template.h"
#include "framework/config/service_config.h"
#include "framework/model/model_args.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "tests/core/framework/request/request_factory_test_utils.h"

namespace xllm {
namespace {

using test::CallbackCapture;
using test::make_capture_callback;

constexpr int32_t kBareStopToken = 13;
constexpr int32_t kStopWithLfToken = 624;
constexpr int32_t kStopWithCrlfToken = 625;
constexpr int32_t kStopWithSpaceToken = 626;
constexpr int32_t kNonStopToken = 627;

struct EncodeCall {
  std::string text;
  bool add_special_tokens;
};

struct BatchEncodeCall {
  std::vector<std::string> texts;
  bool add_special_tokens;
};

struct ChatTemplateCall {
  std::vector<JsonTool> tools;
  nlohmann::ordered_json kwargs;
};

class BatchCallbackCapture final {
 public:
  explicit BatchCallbackCapture(size_t size)
      : outputs_(size), call_counts_(size, 0) {}

  BatchOutputCallback callback() {
    return [this](size_t index, RequestOutput output) {
      ++call_counts_.at(index);
      outputs_.at(index) = std::move(output);
      return false;
    };
  }

  int32_t call_count(size_t index) const { return call_counts_.at(index); }

  const std::optional<RequestOutput>& output(size_t index) const {
    return outputs_.at(index);
  }

 private:
  std::vector<std::optional<RequestOutput>> outputs_;
  std::vector<int32_t> call_counts_;
};

// Deterministic tokenizer used for factory tests. It encodes each character to
// a token id inside the vocabulary range, except for the stop strings used by
// the factory tests. Those strings model a byte-level tokenizer where the
// first generated token can contain the stop text and its line ending.
class StopSequenceTokenizer final : public Tokenizer {
 public:
  explicit StopSequenceTokenizer(int32_t vocab_size) : fallback_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    encode_calls_.emplace_back(
        EncodeCall{std::string(text), add_special_tokens});
    return encode_internal(text, ids, add_special_tokens);
  }

  bool batch_encode(const std::vector<std::string>& texts,
                    std::vector<std::vector<int32_t>>* ids,
                    bool add_special_tokens = true) const override {
    batch_encode_calls_.emplace_back(
        BatchEncodeCall{texts, add_special_tokens});
    ids->clear();
    ids->reserve(texts.size());
    for (const std::string& text : texts) {
      ids->emplace_back();
      if (!encode_internal(text, &ids->back(), add_special_tokens)) {
        ids->pop_back();
        return false;
      }
    }
    return true;
  }

  const std::vector<EncodeCall>& encode_calls() const { return encode_calls_; }

  const std::vector<BatchEncodeCall>& batch_encode_calls() const {
    return batch_encode_calls_;
  }

  const std::vector<std::string>& attempted_texts() const {
    return attempted_texts_;
  }

  void set_special_token_id(int32_t token_id) { special_token_id_ = token_id; }

  size_t vocab_size() const override { return fallback_.vocab_size(); }

  std::string id_to_token(int32_t id) const override {
    if (id == kBareStopToken) {
      return ".";
    }
    if (id == kStopWithLfToken) {
      return ".\\n";
    }
    if (id == kStopWithCrlfToken) {
      return ".\\r\\n";
    }
    if (id == kStopWithSpaceToken) {
      return ". ";
    }
    return std::string(1, static_cast<char>('a' + (id % 26)));
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<StopSequenceTokenizer>(*this);
  }

 private:
  bool encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids,
                       bool add_special_tokens) const {
    attempted_texts_.emplace_back(text);
    if (text == ".") {
      *ids = {kBareStopToken};
      return true;
    }
    if (text == ".\n") {
      *ids = {kStopWithLfToken};
      return true;
    }
    if (text == ".\r\n") {
      *ids = {kStopWithCrlfToken};
      return true;
    }
    if (text == ". ") {
      *ids = {kStopWithSpaceToken};
      return true;
    }
    if (!fallback_.encode(text, ids)) {
      return false;
    }
    if (add_special_tokens && special_token_id_.has_value()) {
      ids->insert(ids->begin(), *special_token_id_);
    }
    return true;
  }

  test::FakeTokenizer fallback_;
  std::optional<int32_t> special_token_id_;
  mutable std::vector<EncodeCall> encode_calls_;
  mutable std::vector<BatchEncodeCall> batch_encode_calls_;
  mutable std::vector<std::string> attempted_texts_;
};

// Chat template that renders to a fixed prompt, or reports failure when
// configured to, so the message overload's error path can be tested without a
// real Jinja template.
class FakeChatTemplate final : public ChatTemplate {
 public:
  std::optional<std::string> apply(
      const ChatMessages& messages) const override {
    return apply(messages, {}, nlohmann::ordered_json::object());
  }

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const override {
    calls_.emplace_back(ChatTemplateCall{json_tools, chat_template_kwargs});
    if (!succeed_) {
      return std::nullopt;
    }
    if (!messages.empty() &&
        std::holds_alternative<std::string>(messages.front().content)) {
      const auto it = rendered_prompts_.find(
          std::get<std::string>(messages.front().content));
      if (it != rendered_prompts_.end()) {
        return it->second;
      }
    }
    return rendered_prompt_;
  }

  void set_succeed(bool succeed) { succeed_ = succeed; }
  void set_rendered_prompt(std::string prompt) {
    rendered_prompt_ = std::move(prompt);
  }

  void set_rendered_prompt_for_message(std::string message,
                                       std::optional<std::string> prompt) {
    rendered_prompts_[std::move(message)] = std::move(prompt);
  }

  const std::vector<ChatTemplateCall>& calls() const { return calls_; }

 private:
  bool succeed_ = true;
  std::string rendered_prompt_ = "rendered prompt";
  std::unordered_map<std::string, std::optional<std::string>> rendered_prompts_;
  mutable std::vector<ChatTemplateCall> calls_;
};

class LLMRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_json_object_output_ =
        ServiceConfig::get_instance().enable_json_object_output();
    ServiceConfig::get_instance().enable_json_object_output(true);
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    ASSERT_TRUE(rate_limiter_.acquire().ok());
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  void TearDown() override {
    ServiceConfig::get_instance().enable_json_object_output(
        previous_json_object_output_);
  }

  std::unique_ptr<LLMRequestFactory> make_factory(
      int32_t vocab_size = 1000,
      int32_t max_position = 2048,
      std::string task = "generate") {
    tokenizer_ = std::make_unique<StopSequenceTokenizer>(vocab_size);
    chat_template_ = std::make_unique<FakeChatTemplate>();
    model_args_.vocab_size(vocab_size)
        .max_position_embeddings(max_position)
        .eos_token_id(-1);
    // enable_chunked_prefill defaults true; keep it so the prompt length limit
    // is exactly max_position_embeddings.
    options_.enable_service_routing(false);
    return std::make_unique<LLMRequestFactory>(
        tokenizer_.get(),
        chat_template_.get(),
        &model_args_,
        &options_,
        &rate_limiter_,
        std::move(task),
        [](const std::vector<RequestOutput>&) { return std::vector<bool>{}; });
  }

  void acquire_batch_slots(size_t num_requests) {
    for (size_t i = 1; i < num_requests; ++i) {
      ASSERT_TRUE(rate_limiter_.acquire().ok());
    }
  }

  std::unique_ptr<StopSequenceTokenizer> tokenizer_;
  std::unique_ptr<FakeChatTemplate> chat_template_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
  bool previous_json_object_output_ = true;
};

TEST_F(LLMRequestFactoryTest, RejectsEmptyPromptAndReleasesRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.called);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Prompt is empty");
  // The factory must release the rate-limit slot on every early return.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest,
       ServerTtftIncludesPreprocessingAndRecordsFirstTokenOnce) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams params;
  params.n = 2;
  const std::chrono::steady_clock::time_point arrival_time{};
  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 params,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture),
                                 /*generation_mode=*/std::nullopt,
                                 arrival_time);
  ASSERT_NE(request, nullptr);
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request->sequences().size(), 2u);
  const int64_t sample_count =
      HISTOGRAM_server_time_to_first_token_latency_milliseconds.count();
  const auto first_token_time = arrival_time + std::chrono::seconds(3);
  EXPECT_FALSE(request->record_server_first_token(first_token_time));
  EXPECT_FALSE(request->server_ttft_seconds().has_value());

  Sequence& sequence = *request->sequences().back();
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(kNonStopToken));
  EXPECT_TRUE(request->record_server_first_token(first_token_time));
  ASSERT_TRUE(request->server_ttft_seconds().has_value());
  EXPECT_DOUBLE_EQ(*request->server_ttft_seconds(), 3.0);
  EXPECT_DOUBLE_EQ(sequence.time_to_first_token_latency_seconds(), 0.0);
  EXPECT_EQ(HISTOGRAM_server_time_to_first_token_latency_milliseconds.count(),
            sample_count + 1);

  Sequence& other_sequence = *request->sequences().front();
  other_sequence.kv_state().set_kv_cache_tokens_num(
      other_sequence.num_prompt_tokens());
  other_sequence.append_token(Token(kNonStopToken));
  EXPECT_FALSE(request->record_server_first_token(first_token_time +
                                                  std::chrono::seconds(1)));
  EXPECT_DOUBLE_EQ(*request->server_ttft_seconds(), 3.0);
  EXPECT_EQ(HISTOGRAM_server_time_to_first_token_latency_milliseconds.count(),
            sample_count + 1);
}

TEST_F(LLMRequestFactoryTest,
       ChatPreservesAdmissionTimeBeforeTemplateRendering) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams params;
  const std::chrono::steady_clock::time_point arrival_time{};
  auto request = factory->create(
      std::vector<Message>{Message("user", std::string("hello"))},
      /*prompt_tokens=*/std::nullopt,
      params,
      /*call=*/std::nullopt,
      make_capture_callback(&capture),
      arrival_time);
  ASSERT_NE(request, nullptr);
  const std::optional<double> elapsed =
      request->server_elapsed_seconds(arrival_time + std::chrono::seconds(7));
  ASSERT_TRUE(elapsed.has_value());
  EXPECT_DOUBLE_EQ(*elapsed, 7.0);
}

TEST_F(LLMRequestFactoryTest, ServerTtftRequiresAdmissionAndUncancelledOutput) {
  auto factory = make_factory();
  CallbackCapture capture;
  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 RequestParams{},
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  Sequence& sequence = *request->sequences().front();
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(kNonStopToken));
  const int64_t sample_count =
      HISTOGRAM_server_time_to_first_token_latency_milliseconds.count();

  request->state().arrival_time.reset();
  EXPECT_FALSE(request->record_server_first_token());
  EXPECT_FALSE(request->server_ttft_seconds().has_value());
  request->state().arrival_time = std::chrono::steady_clock::now();
  request->set_cancel();
  EXPECT_FALSE(request->record_server_first_token());
  EXPECT_FALSE(request->server_ttft_seconds().has_value());
  EXPECT_EQ(HISTOGRAM_server_time_to_first_token_latency_milliseconds.count(),
            sample_count);
}

TEST_F(LLMRequestFactoryTest, PromptBatchPreservesSharedAdmissionTime) {
  auto factory = make_factory();
  const std::vector<std::string> prompts = {"hello", "world"};
  acquire_batch_slots(prompts.size());
  BatchCallbackCapture capture(prompts.size());
  const std::chrono::steady_clock::time_point arrival_time{};
  auto requests = factory->create_batch(
      prompts, {RequestParams{}}, capture.callback(), arrival_time);
  ASSERT_EQ(requests.size(), prompts.size());
  for (const auto& request : requests) {
    ASSERT_NE(request, nullptr);
    const std::optional<double> elapsed =
        request->server_elapsed_seconds(arrival_time + std::chrono::seconds(7));
    ASSERT_TRUE(elapsed.has_value());
    EXPECT_DOUBLE_EQ(*elapsed, 7.0);
  }
}

TEST_F(LLMRequestFactoryTest, ChatBatchPreservesSharedAdmissionTime) {
  auto factory = make_factory();
  const std::vector<std::vector<Message>> conversations = {
      {Message("user", std::string("hello"))},
      {Message("user", std::string("world"))}};
  acquire_batch_slots(conversations.size());
  BatchCallbackCapture capture(conversations.size());
  const std::chrono::steady_clock::time_point arrival_time{};
  auto requests = factory->create_batch(
      conversations, {RequestParams{}}, capture.callback(), arrival_time);
  ASSERT_EQ(requests.size(), conversations.size());
  for (const auto& request : requests) {
    ASSERT_NE(request, nullptr);
    const std::optional<double> elapsed =
        request->server_elapsed_seconds(arrival_time + std::chrono::seconds(7));
    ASSERT_TRUE(elapsed.has_value());
    EXPECT_DOUBLE_EQ(*elapsed, 7.0);
  }
}

TEST_F(LLMRequestFactoryTest, FailsWhenPromptEncodingFails) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"please FAIL here",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Failed to encode prompt");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, RejectsPromptTokensOutOfVocabulary) {
  auto factory = make_factory(/*vocab_size=*/100);
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
                                 /*prompt_tokens=*/std::vector<int>{5, 200},
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(capture.status->message().find("out of vocabulary range"),
            std::string::npos);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, RejectsPromptLongerThanContext) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8},
      sp,
      /*call=*/std::nullopt,
      make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Prompt is too long");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, FailsWhenStopSequenceEncodingFails) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.stop = std::vector<std::string>{"FAIL-STOP"};

  auto request = factory->create(/*prompt=*/"hello world",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to encode stop sequence");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, ContextErrorRetainsServiceCallbackOwnership) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  RequestParams params;
  params.service_request_id = "attempt-context";
  params.source_xservice_addr = "127.0.0.1:9000";
  params.min_tokens = 7;
  bool called = false;
  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::vector<int>{1, 2},
      params,
      /*call=*/std::nullopt,
      [&called, &params](RequestOutput output) {
        called = true;
        EXPECT_EQ(output.service_request_id, params.service_request_id);
        EXPECT_EQ(output.target_xservice_addr, params.source_xservice_addr);
        EXPECT_TRUE(output.status.has_value());
        if (output.status.has_value()) {
          EXPECT_EQ(output.status->message(),
                    "The remaining model context cannot satisfy min_tokens");
        }
        return false;
      });
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, OverlapAcceptsBadWords) {
  options_.enable_schedule_overlap(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams params;
  params.bad_words = {"bad"};
  auto request = factory->create("hello",
                                 std::nullopt,
                                 params,
                                 std::nullopt,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_FALSE(request->state().sampling_param.bad_words_token_ids.empty());
}

TEST_F(LLMRequestFactoryTest,
       CreatesRequestForValidPromptAndKeepsRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-1";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"hello world",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  // On success the slot stays held; it is later released when the request is
  // completed/destroyed by the scheduler, not by the factory.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  auto response_owner = request;
  request->set_cancel();
  request->set_cancel();
  request.reset();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  response_owner.reset();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
  EXPECT_TRUE(rate_limiter_.acquire().ok());
  rate_limiter_.decrease_one_request();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, ModelStopTokenCanEndGenerationPrompt) {
  auto factory = make_factory();
  model_args_.stop_token_ids() = {kNonStopToken};
  CallbackCapture capture;
  RequestParams sp;
  auto request = factory->create(
      /*prompt=*/"",
      /*prompt_tokens=*/std::vector<int32_t>{1, kNonStopToken},
      sp,
      /*call=*/std::nullopt,
      make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(request->finished());
  const StoppingChecker& checker = request->state().stopping_checker;
  const std::vector<int32_t> generated = {1, kNonStopToken, kNonStopToken};
  EXPECT_EQ(checker.check(generated, /*num_prompt_tokens=*/2),
            FinishReason::STOP);
}

TEST_F(LLMRequestFactoryTest, FactoryAddsStopSequenceVariants) {
  options_.enable_schedule_overlap(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.stop = std::vector<std::string>{"."};

  auto request = factory->create(/*prompt=*/".\n",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  ASSERT_TRUE(request->state().enable_schedule_overlap);
  EXPECT_FALSE(request->finished());
  const StoppingChecker& checker = request->state().stopping_checker;
  const std::vector<int32_t>& prompt_tokens = request->state().prompt_tokens;
  EXPECT_EQ(prompt_tokens, std::vector<int32_t>({kStopWithLfToken}));
  const std::vector<int32_t> pending_tokens = {kStopWithLfToken, -1};
  EXPECT_EQ(checker.check(pending_tokens, prompt_tokens.size()),
            FinishReason::NONE);
  const std::vector<int32_t> bare_stop_prompt = {kBareStopToken};
  EXPECT_EQ(checker.check(bare_stop_prompt, bare_stop_prompt.size()),
            FinishReason::STOP);
  const auto check_generated_token = [&](int32_t token_id,
                                         FinishReason expected) {
    std::vector<int32_t> token_ids(prompt_tokens.begin(), prompt_tokens.end());
    token_ids.push_back(token_id);
    EXPECT_EQ(checker.check(token_ids, prompt_tokens.size()), expected);
    token_ids.push_back(-1);
    StopReason stop_reason;
    EXPECT_EQ(
        checker.check(token_ids, prompt_tokens.size(), nullptr, &stop_reason),
        expected);
    if (expected == FinishReason::STOP) {
      ASSERT_TRUE(std::holds_alternative<std::string>(stop_reason));
      EXPECT_EQ(std::get<std::string>(stop_reason), ".");
    }
  };

  check_generated_token(kBareStopToken, FinishReason::STOP);
  check_generated_token(kStopWithLfToken, FinishReason::STOP);
  check_generated_token(kStopWithCrlfToken, FinishReason::STOP);
  check_generated_token(kStopWithSpaceToken, FinishReason::STOP);
  check_generated_token(kNonStopToken, FinishReason::NONE);
}

TEST_F(LLMRequestFactoryTest, MtpCreatesJsonObjectChatRequest) {
  options_.num_speculative_tokens(3)
      .draft_model_path(std::string("/fake/draft"))
      .speculative_algorithm("MTP");
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.response_format = ResponseFormatType::JSON_OBJECT;
  const std::vector<Message> messages = {Message("user", std::string("hi"))};

  chat_template_->set_rendered_prompt("rendered prompt </think>");
  auto request = factory->create(messages,
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_NE(request->state().json_object_grammar, nullptr);
  EXPECT_FALSE(request->state().json_reasoning_enabled);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(LLMRequestFactoryTest, MtpMinimumLengthIsIndependentOfOverlap) {
  for (const bool overlap : {false, true}) {
    SCOPED_TRACE(overlap);
    options_.enable_schedule_overlap(overlap)
        .num_speculative_tokens(4)
        .speculative_algorithm("MTP");
    auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/65536);
    CallbackCapture capture;
    RequestParams params;
    params.max_tokens = 2048;
    params.min_tokens = 2048;
    auto request = factory->create("hello",
                                   std::nullopt,
                                   params,
                                   std::nullopt,
                                   make_capture_callback(&capture));
    ASSERT_NE(request, nullptr);
    EXPECT_FALSE(capture.called);
    EXPECT_EQ(request->state().sampling_param.min_tokens, 2048U);
    EXPECT_TRUE(request->state().sampling_param.device_min_tokens());
  }
}

TEST_F(LLMRequestFactoryTest, MtpSamplingControlsAreIndependentOfOverlap) {
  for (const bool overlap : {false, true}) {
    SCOPED_TRACE(overlap);
    options_.enable_schedule_overlap(overlap)
        .num_speculative_tokens(4)
        .speculative_algorithm("MTP");
    auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/65536);
    CallbackCapture capture;
    RequestParams params;
    params.max_tokens = 64;
    params.min_tokens = 16;
    params.seed = 42;
    params.bad_words = {"bad"};
    params.logit_bias = {{1, 2.0F}};
    params.allowed_token_ids = std::vector<int32_t>{1, 2, 3};
    params.frequency_penalty = 0.5F;
    params.presence_penalty = 0.5F;
    params.repetition_penalty = 1.1F;
    params.top_k = 10;
    params.top_p = 0.9F;
    params.min_p = 0.1F;
    params.logprobs = true;
    params.top_logprobs = 5;
    auto request = factory->create("hello",
                                   std::nullopt,
                                   params,
                                   std::nullopt,
                                   make_capture_callback(&capture));
    ASSERT_NE(request, nullptr);
    EXPECT_FALSE(capture.called);
    const auto& sampling = request->state().sampling_param;
    EXPECT_EQ(sampling.seed, 42);
    EXPECT_TRUE(sampling.device_min_tokens());
    EXPECT_FALSE(sampling.bad_words_token_ids.empty());
  }
}

TEST_F(LLMRequestFactoryTest, JsonObjectReachesGrammarValidation) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.response_format = ResponseFormatType::JSON_OBJECT;

  auto request = factory->create(
      /*prompt=*/"hello world",
      /*prompt_tokens=*/std::nullopt,
      sp,
      /*call=*/std::nullopt,
      make_capture_callback(&capture),
      ChatTemplateGenerationMode::UNKNOWN);

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(
      capture.status->message(),
      "JSON object constraint requires a recognizable chat generation mode");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

// verify_params only knows the model-agnostic 2000 cap; the factory knows the
// vocabulary. A top-k above it would throw inside the sampler and, since the
// batch uses max(top_logprobs), take every request in the batch down with it.
TEST_F(LLMRequestFactoryTest, RejectsTopLogprobsAboveVocabulary) {
  auto factory = make_factory(/*vocab_size=*/1000);
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 16;

  sp.logprobs = true;
  // Under the 2000 cap, so it passes verify_params, but above the vocabulary.
  sp.top_logprobs = 1500;

  auto request = factory->create(/*prompt=*/"hello world",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(capture.status->message().find("top_logprobs (1500)"),
            std::string::npos);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, RetainsHttpResponseOptionsForDecodeDispatch) {
  options_.enable_disagg_pd(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  sp.max_tokens = 16;
  sp.source_xservice_addr = "worker:1234";
  sp.response_model = "response-model";
  sp.response_tools.emplace_back().mutable_function()->set_name("weather");
  sp.response_tool_choice = "auto";
  (*sp.response_chat_template_kwargs.mutable_fields())["enable_thinking"]
      .set_bool_value(false);
  sp.response_stream_options.set_include_usage(true);
  sp.response_stream_options.set_continuous_usage_stats(true);
  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  const auto& state = request->state();
  EXPECT_FALSE(state.response_formatter);
  EXPECT_EQ(state.response_type, proto::CHAT_COMPLETION);
  EXPECT_EQ(state.response_model, "response-model");
  ASSERT_EQ(state.response_tools.size(), 1U);
  EXPECT_EQ(state.response_tools.front().function().name(), "weather");
  EXPECT_EQ(state.response_tool_choice, "auto");
  EXPECT_FALSE(state.response_chat_template_kwargs.fields()
                   .at("enable_thinking")
                   .bool_value());
  EXPECT_TRUE(state.response_stream_options.include_usage());
  EXPECT_TRUE(state.response_stream_options.continuous_usage_stats());

  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  EXPECT_FALSE(capture.called);
}

TEST_F(LLMRequestFactoryTest, RetainsCompletionResponseShapeForDecodeDispatch) {
  options_.enable_disagg_pd(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams params;
  params.source_xservice_addr = "worker:1234";
  params.response_type = proto::TEXT_COMPLETION;
  params.response_model = "completion-model";
  params.response_stream_options.set_include_usage(true);
  params.echo = true;
  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 params,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().response_type, proto::TEXT_COMPLETION);
  EXPECT_EQ(request->state().response_model, "completion-model");
  EXPECT_TRUE(request->state().response_stream_options.include_usage());
  EXPECT_TRUE(request->state().echo);
  EXPECT_FALSE(capture.called);
}

TEST_F(LLMRequestFactoryTest, MessageOverloadFailsWhenTemplateRejects) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("hi"));

  auto request = factory->create(messages,
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Failed to construct prompt from messages");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, MessageOverloadCreatesRequestOnSuccess) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-chat";
  sp.max_tokens = 16;
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("hi"));

  auto request = factory->create(messages,
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(LLMRequestFactoryTest, BatchGroupsPromptsBySpecialTokenPolicy) {
  auto factory = make_factory();
  constexpr int32_t kSpecialToken = 999;
  tokenizer_->set_special_token_id(kSpecialToken);
  const std::vector<std::string> prompts = {"a", "b", "c", "d"};
  acquire_batch_slots(prompts.size());
  std::vector<RequestParams> params(prompts.size());
  params[0].add_special_tokens = false;
  params[1].add_special_tokens = true;
  params[2].add_special_tokens = false;
  params[3].add_special_tokens = true;
  BatchCallbackCapture capture(prompts.size());

  auto requests = factory->create_batch(prompts, params, capture.callback());

  ASSERT_EQ(requests.size(), prompts.size());
  const std::vector<std::vector<int32_t>> expected_tokens = {
      {97}, {kSpecialToken, 98}, {99}, {kSpecialToken, 100}};
  for (size_t i = 0; i < requests.size(); ++i) {
    ASSERT_NE(requests[i], nullptr);
    EXPECT_EQ(requests[i]->state().prompt, prompts[i]);
    EXPECT_EQ(requests[i]->state().prompt_tokens, expected_tokens[i]);
    EXPECT_EQ(capture.call_count(i), 0);
    EXPECT_FALSE(requests[i]->state().output_func(RequestOutput{}));
    EXPECT_EQ(capture.call_count(i), 1);
  }
  EXPECT_TRUE(tokenizer_->encode_calls().empty());
  ASSERT_EQ(tokenizer_->batch_encode_calls().size(), 2U);
  EXPECT_EQ(tokenizer_->batch_encode_calls()[0].texts,
            std::vector<std::string>({"a", "c"}));
  EXPECT_FALSE(tokenizer_->batch_encode_calls()[0].add_special_tokens);
  EXPECT_EQ(tokenizer_->batch_encode_calls()[1].texts,
            std::vector<std::string>({"b", "d"}));
  EXPECT_TRUE(tokenizer_->batch_encode_calls()[1].add_special_tokens);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 4);
  auto response_owner = requests[1];
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  response_owner.reset();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, SingletonBatchUsesSingleEncodeWithSharedParams) {
  auto factory = make_factory();
  tokenizer_->set_special_token_id(/*token_id=*/999);
  RequestParams params;
  params.add_special_tokens = false;
  params.temperature = 0.5;
  BatchCallbackCapture capture(/*size=*/1);

  auto requests = factory->create_batch(
      std::vector<std::string>{"a"}, {params}, capture.callback());

  ASSERT_EQ(requests.size(), 1U);
  ASSERT_NE(requests.front(), nullptr);
  EXPECT_EQ(requests.front()->state().prompt_tokens,
            std::vector<int32_t>({97}));
  EXPECT_EQ(requests.front()->state().sampling_param.temperature, 0.5);
  EXPECT_EQ(capture.call_count(/*index=*/0), 0);
  EXPECT_TRUE(tokenizer_->batch_encode_calls().empty());
  ASSERT_EQ(tokenizer_->encode_calls().size(), 1U);
  EXPECT_EQ(tokenizer_->encode_calls().front().text, "a");
  EXPECT_FALSE(tokenizer_->encode_calls().front().add_special_tokens);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, BatchAppliesSharedParamsToEveryPrompt) {
  auto factory = make_factory();
  const std::vector<std::string> prompts = {"a", "b", "c"};
  acquire_batch_slots(prompts.size());
  RequestParams params;
  params.max_tokens = 7;
  params.temperature = 0.5;
  params.add_special_tokens = false;
  BatchCallbackCapture capture(prompts.size());

  auto requests = factory->create_batch(prompts, {params}, capture.callback());

  ASSERT_EQ(requests.size(), prompts.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    ASSERT_NE(requests[i], nullptr);
    EXPECT_EQ(requests[i]->state().prompt, prompts[i]);
    EXPECT_EQ(requests[i]->state().sampling_param.temperature, 0.5);
    EXPECT_EQ(capture.call_count(i), 0);
  }
  ASSERT_EQ(tokenizer_->batch_encode_calls().size(), 1U);
  EXPECT_EQ(tokenizer_->batch_encode_calls().front().texts, prompts);
  EXPECT_FALSE(tokenizer_->batch_encode_calls().front().add_special_tokens);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 3);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest,
       BatchEncodingRetainsPrefixAndContinuesAfterEachFailedInput) {
  auto factory = make_factory();
  const std::vector<std::string> prompts = {
      "a", "FAIL-first", "b", "c", "FAIL-second", "d"};
  acquire_batch_slots(prompts.size());
  std::vector<RequestParams> params(prompts.size());
  for (size_t i = 0; i < params.size(); ++i) {
    params[i].service_request_id = "attempt-" + std::to_string(i);
    params[i].source_xservice_addr = "worker:1234";
  }
  BatchCallbackCapture capture(prompts.size());

  auto requests = factory->create_batch(prompts, params, capture.callback());

  ASSERT_EQ(requests.size(), prompts.size());
  for (const size_t i : {0U, 2U, 3U, 5U}) {
    ASSERT_NE(requests[i], nullptr);
    EXPECT_EQ(requests[i]->state().prompt, prompts[i]);
    EXPECT_EQ(requests[i]->state().prompt_tokens,
              std::vector<int32_t>({static_cast<int32_t>(prompts[i].front())}));
    EXPECT_EQ(capture.call_count(i), 0);
  }
  for (const size_t i : {1U, 4U}) {
    EXPECT_EQ(requests[i], nullptr);
    EXPECT_EQ(capture.call_count(i), 1);
    ASSERT_TRUE(capture.output(i).has_value());
    ASSERT_TRUE(capture.output(i)->status.has_value());
    EXPECT_EQ(capture.output(i)->status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capture.output(i)->status->message(), "Failed to encode prompt");
    EXPECT_EQ(capture.output(i)->service_request_id,
              params[i].service_request_id);
    EXPECT_EQ(capture.output(i)->target_xservice_addr,
              params[i].source_xservice_addr);
  }
  // The successful prefix, failed inputs, and later inputs are each attempted
  // exactly once, even when more than one batch operation reports failure.
  EXPECT_EQ(tokenizer_->attempted_texts(), prompts);
  ASSERT_EQ(tokenizer_->batch_encode_calls().size(), 2U);
  EXPECT_EQ(tokenizer_->batch_encode_calls()[0].texts, prompts);
  EXPECT_EQ(tokenizer_->batch_encode_calls()[1].texts,
            std::vector<std::string>({"b", "c", "FAIL-second", "d"}));
  ASSERT_EQ(tokenizer_->encode_calls().size(), 1U);
  EXPECT_EQ(tokenizer_->encode_calls().front().text, "d");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 4);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest,
       BatchEncodingHandlesFirstConsecutiveAndLastFailuresWithoutRetry) {
  auto factory = make_factory();
  const std::vector<std::string> prompts = {
      "FAIL-first", "FAIL-consecutive", "ok", "FAIL-last"};
  acquire_batch_slots(prompts.size());
  BatchCallbackCapture capture(prompts.size());

  auto requests =
      factory->create_batch(prompts, {RequestParams{}}, capture.callback());

  ASSERT_EQ(requests.size(), prompts.size());
  ASSERT_NE(requests[2], nullptr);
  EXPECT_EQ(requests[2]->state().prompt, "ok");
  EXPECT_EQ(requests[2]->state().prompt_tokens,
            std::vector<int32_t>({111, 107}));
  EXPECT_EQ(capture.call_count(/*index=*/2), 0);
  for (const size_t i : {0U, 1U, 3U}) {
    EXPECT_EQ(requests[i], nullptr);
    EXPECT_EQ(capture.call_count(i), 1);
    ASSERT_TRUE(capture.output(i).has_value());
    ASSERT_TRUE(capture.output(i)->status.has_value());
    EXPECT_EQ(capture.output(i)->status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capture.output(i)->status->message(), "Failed to encode prompt");
  }
  EXPECT_EQ(tokenizer_->attempted_texts(), prompts);
  EXPECT_EQ(tokenizer_->batch_encode_calls().size(), 3U);
  EXPECT_TRUE(tokenizer_->encode_calls().empty());
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, BatchPreservesValidationErrorPrecedence) {
  options_.enable_disagg_pd(true);
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  const std::vector<std::string> prompts = {"", "", "12345678", "ok"};
  acquire_batch_slots(prompts.size());
  std::vector<RequestParams> params(prompts.size());
  params[0].temperature = -1;

  BatchCallbackCapture capture(prompts.size());

  auto requests = factory->create_batch(prompts, params, capture.callback());

  ASSERT_EQ(requests.size(), prompts.size());
  const std::vector<std::string> expected_errors = {
      "temperature must be finite and non-negative",
      "Prompt is empty",
      "Prompt is too long"};
  for (size_t i = 0; i < expected_errors.size(); ++i) {
    EXPECT_EQ(requests[i], nullptr);
    EXPECT_EQ(capture.call_count(i), 1);
    ASSERT_TRUE(capture.output(i).has_value());
    ASSERT_TRUE(capture.output(i)->status.has_value());
    EXPECT_EQ(capture.output(i)->status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(capture.output(i)->status->message(), expected_errors[i]);
  }
  ASSERT_NE(requests.back(), nullptr);
  EXPECT_EQ(capture.call_count(/*index=*/3), 0);
  EXPECT_EQ(tokenizer_->attempted_texts(),
            std::vector<std::string>({"12345678", "ok"}));
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, ChatBatchRejectsRenderingFailureIndependently) {
  auto factory = make_factory();
  const std::vector<std::vector<Message>> conversations = {
      {Message("user", std::string("left"))},
      {Message("user", std::string("reject"))},
      {Message("user", std::string("right"))},
      {Message("user", std::string("reject"))}};
  chat_template_->set_rendered_prompt_for_message("left", "left prompt");
  chat_template_->set_rendered_prompt_for_message("reject", std::nullopt);
  chat_template_->set_rendered_prompt_for_message("right", "right prompt");
  acquire_batch_slots(conversations.size());
  std::vector<RequestParams> params(conversations.size());
  JsonTool left_tool;
  left_tool.function.name = "left_tool";
  left_tool.function.parameters = {{"type", "object"}};
  params[0].tools = {left_tool};
  params[0].chat_template_kwargs = {{"enable_thinking", false},
                                    {"marker", "left"}};
  JsonTool right_tool;
  right_tool.function.name = "right_tool";
  right_tool.function.parameters = {{"type", "object"},
                                    {"required", {"value"}}};
  params[2].tools = {right_tool};
  params[2].chat_template_kwargs = {{"enable_thinking", true},
                                    {"marker", "right"}};
  params[3].temperature = -1;
  BatchCallbackCapture capture(conversations.size());

  auto requests =
      factory->create_batch(conversations, params, capture.callback());

  ASSERT_EQ(requests.size(), conversations.size());
  ASSERT_NE(requests[0], nullptr);
  EXPECT_EQ(requests[0]->state().prompt, "left prompt");
  EXPECT_EQ(capture.call_count(/*index=*/0), 0);
  ASSERT_NE(requests[2], nullptr);
  EXPECT_EQ(requests[2]->state().prompt, "right prompt");
  EXPECT_EQ(capture.call_count(/*index=*/2), 0);
  for (const size_t i : {1U, 3U}) {
    EXPECT_EQ(requests[i], nullptr);
    EXPECT_EQ(capture.call_count(i), 1);
    ASSERT_TRUE(capture.output(i).has_value());
    ASSERT_TRUE(capture.output(i)->status.has_value());
    EXPECT_EQ(capture.output(i)->status->code(), StatusCode::INVALID_ARGUMENT);
  }
  EXPECT_EQ(capture.output(/*index=*/1)->status->message(),
            "Failed to construct prompt from messages");
  EXPECT_EQ(capture.output(/*index=*/3)->status->message(),
            "temperature must be finite and non-negative");
  ASSERT_EQ(chat_template_->calls().size(), 3U);
  for (const size_t i : {0U, 2U}) {
    const ChatTemplateCall& call = chat_template_->calls()[i];
    ASSERT_EQ(call.tools.size(), 1U);
    EXPECT_EQ(call.tools.front().function.name,
              params[i].tools.front().function.name);
    EXPECT_EQ(call.tools.front().function.parameters,
              params[i].tools.front().function.parameters);
    EXPECT_EQ(call.kwargs,
              nlohmann::ordered_json(params[i].chat_template_kwargs));
  }
  ASSERT_EQ(tokenizer_->batch_encode_calls().size(), 1U);
  EXPECT_EQ(tokenizer_->batch_encode_calls().front().texts,
            std::vector<std::string>({"left prompt", "right prompt"}));
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 2);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest,
       ChatBatchRetainsGenerationModesForJsonConstraints) {
  auto factory = make_factory();
  const std::vector<std::vector<Message>> conversations = {
      {Message("user", std::string("thinking"))},
      {Message("user", std::string("chat"))},
      {Message("user", std::string("unknown"))}};
  chat_template_->set_rendered_prompt_for_message("thinking",
                                                  "prompt <think>\n");
  chat_template_->set_rendered_prompt_for_message("chat", "prompt </think>\n");
  chat_template_->set_rendered_prompt_for_message("unknown", "plain prompt");
  acquire_batch_slots(conversations.size());
  std::vector<RequestParams> params(conversations.size());
  for (RequestParams& param : params) {
    param.response_format = ResponseFormatType::JSON_OBJECT;
  }
  // The mode inferred from the rendered prompt overrides template kwargs.
  params[0].chat_template_kwargs = {{"enable_thinking", false}};
  params[1].chat_template_kwargs = {{"enable_thinking", true}};
  BatchCallbackCapture capture(conversations.size());

  auto requests =
      factory->create_batch(conversations, params, capture.callback());

  ASSERT_EQ(requests.size(), conversations.size());
  ASSERT_NE(requests[0], nullptr);
  EXPECT_NE(requests[0]->state().json_object_grammar, nullptr);
  EXPECT_TRUE(requests[0]->state().json_reasoning_enabled);
  EXPECT_TRUE(requests[0]->state().sampling_param.json_object);
  EXPECT_EQ(capture.call_count(/*index=*/0), 0);
  ASSERT_NE(requests[1], nullptr);
  EXPECT_NE(requests[1]->state().json_object_grammar, nullptr);
  EXPECT_FALSE(requests[1]->state().json_reasoning_enabled);
  EXPECT_TRUE(requests[1]->state().sampling_param.json_object);
  EXPECT_EQ(capture.call_count(/*index=*/1), 0);
  EXPECT_EQ(requests[2], nullptr);
  EXPECT_EQ(capture.call_count(/*index=*/2), 1);
  ASSERT_TRUE(capture.output(/*index=*/2).has_value());
  ASSERT_TRUE(capture.output(/*index=*/2)->status.has_value());
  EXPECT_EQ(capture.output(/*index=*/2)->status->code(),
            StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(
      capture.output(/*index=*/2)->status->message(),
      "JSON object constraint requires a recognizable chat generation mode");
  ASSERT_EQ(tokenizer_->batch_encode_calls().size(), 1U);
  EXPECT_EQ(tokenizer_->batch_encode_calls().front().texts,
            std::vector<std::string>(
                {"prompt <think>\n", "prompt </think>\n", "plain prompt"}));
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 2);
  requests.clear();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, EmptyBatchesDoNotEncodeOrChangeRateLimitSlots) {
  auto factory = make_factory();
  rate_limiter_.decrease_one_request();
  bool callback_called = false;
  const auto callback = [&callback_called](size_t /*index*/,
                                           RequestOutput /*output*/) {
    callback_called = true;
    return false;
  };

  EXPECT_TRUE(
      factory->create_batch(std::vector<std::string>{}, {}, callback).empty());
  EXPECT_TRUE(factory
                  ->create_batch(std::vector<std::vector<Message>>{},
                                 {RequestParams{}},
                                 callback)
                  .empty());

  EXPECT_FALSE(callback_called);
  EXPECT_TRUE(tokenizer_->encode_calls().empty());
  EXPECT_TRUE(tokenizer_->batch_encode_calls().empty());
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

}  // namespace
}  // namespace xllm
