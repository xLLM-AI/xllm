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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
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

// Deterministic tokenizer used for factory tests. It encodes each character to
// a token id inside the vocabulary range, except for the stop strings used by
// the factory tests. Those strings model a byte-level tokenizer where the
// first generated token can contain the stop text and its line ending.
class StopSequenceTokenizer final : public Tokenizer {
 public:
  explicit StopSequenceTokenizer(int32_t vocab_size) : fallback_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
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
    return fallback_.encode(text, ids);
  }

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
  test::FakeTokenizer fallback_;
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
      const ChatMessages& /*messages*/,
      const std::vector<xllm::JsonTool>& /*json_tools*/,
      const nlohmann::ordered_json& /*chat_template_kwargs*/) const override {
    if (!succeed_) {
      return std::nullopt;
    }
    return rendered_prompt_;
  }

  void set_succeed(bool succeed) { succeed_ = succeed; }
  void set_rendered_prompt(std::string prompt) {
    rendered_prompt_ = std::move(prompt);
  }

 private:
  bool succeed_ = true;
  std::string rendered_prompt_ = "rendered prompt";
};

class LLMRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_config_ = ExecutionConfig::get_instance();
    model_config_ = ModelConfig::get_instance();
    previous_json_object_output_ =
        ServiceConfig::get_instance().enable_json_object_output();
    ServiceConfig::get_instance().enable_json_object_output(true);
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    ASSERT_TRUE(rate_limiter_.acquire().ok());
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  void TearDown() override {
    ExecutionConfig::get_instance() = execution_config_;
    ModelConfig::get_instance() = model_config_;
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

  std::unique_ptr<StopSequenceTokenizer> tokenizer_;
  std::unique_ptr<FakeChatTemplate> chat_template_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
  bool previous_json_object_output_ = true;
  ExecutionConfig execution_config_;
  ModelConfig model_config_;
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

TEST_F(LLMRequestFactoryTest, OverlapErrorRetainsServiceCallbackOwnership) {
  options_.enable_schedule_overlap(true);
  auto factory = make_factory();
  RequestParams params;
  params.service_request_id = "attempt-overlap";
  params.source_xservice_addr = "127.0.0.1:9000";
  params.bad_words = {"bad"};
  bool called = false;
  auto request = factory->create(
      /*prompt=*/"hello",
      /*prompt_tokens=*/std::nullopt,
      params,
      /*call=*/std::nullopt,
      [&called, &params](RequestOutput output) {
        called = true;
        EXPECT_EQ(output.service_request_id, params.service_request_id);
        EXPECT_EQ(output.target_xservice_addr, params.source_xservice_addr);
        EXPECT_TRUE(output.status.has_value());
        if (output.status.has_value()) {
          EXPECT_EQ(output.status->message(),
                    "bad_words requires enable_schedule_overlap=false");
        }
        return false;
      });
  EXPECT_EQ(request, nullptr);
  EXPECT_TRUE(called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
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

TEST_F(LLMRequestFactoryTest, TaskPipelineRejectsJsonObjectBeforeGrammarSetup) {
  options_.enable_task_pipeline(true);
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
  EXPECT_EQ(capture.status->message(),
            "response_format=json_object is not supported with "
            "enable_task_pipeline");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

#if defined(USE_NPU)
TEST_F(LLMRequestFactoryTest,
       UnifiedTaskPipelineKeepsJsonObjectChatForLegacyCompatibilityFallback) {
  options_.enable_task_pipeline(true)
      .num_speculative_tokens(3)
      .draft_model_path(std::string("/fake/draft"))
      .speculative_algorithm("MTP");
  ModelConfig::get_instance().model_impl("python");
  ExecutionConfig::get_instance().enable_unified_mtp_graph(true);
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

TEST_F(LLMRequestFactoryTest, DisabledUnifiedTaskMtpRejectsJsonObject) {
  options_.enable_task_pipeline(true)
      .num_speculative_tokens(3)
      .draft_model_path(std::string("/fake/draft"))
      .speculative_algorithm("MTP");
  ModelConfig::get_instance().model_impl("python");
  ExecutionConfig::get_instance().enable_unified_mtp_graph(false);
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
  EXPECT_EQ(capture.status->message(),
            "response_format=json_object is not supported with "
            "enable_task_pipeline");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}
#endif

TEST_F(LLMRequestFactoryTest, TaskPipelineAcceptsOrdinaryRequest) {
  options_.enable_task_pipeline(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"hello world",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(LLMRequestFactoryTest, LegacyJsonObjectReachesGrammarValidation) {
  options_.enable_task_pipeline(false);
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
  sp.beam_width = 4;
  sp.logprobs = false;
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

TEST_F(LLMRequestFactoryTest,
       DisaggregatedPdRejectsBeamSearchAndReleasesRateLimitSlot) {
  options_.enable_disagg_pd(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.beam_width = 2;
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(capture.status->message().find("beam_width > 1"),
            std::string::npos);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(LLMRequestFactoryTest, NonDisaggregatedBeamSearchRemainsSupported) {
  options_.enable_disagg_pd(false);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.beam_width = 2;
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"hello",
                                 /*prompt_tokens=*/std::nullopt,
                                 sp,
                                 /*call=*/std::nullopt,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(request->state().sampling_param.beam_width, 2);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(LLMRequestFactoryTest, RetainsHttpResponseOptionsForDecodeDispatch) {
  options_.enable_disagg_pd(true);
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.beam_width = 1;
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
  EXPECT_EQ(state.sampling_param.beam_width, 1);
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

}  // namespace
}  // namespace xllm
