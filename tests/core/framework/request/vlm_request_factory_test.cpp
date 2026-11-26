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

#include "framework/request/vlm_request_factory.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "api_service/call.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/config/model_config.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "framework/chat_template/chat_template_factory.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request_params.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/text_prompt_processor.h"
#include "tests/core/framework/request/request_factory_test_utils.h"
#include "util/scope_guard.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {
namespace {

using test::CallbackCapture;
using test::FakeTokenizer;
using test::make_capture_callback;

// Fake multimodal processor that lets tests drive process_prompt success and
// the tokens it emits, without pulling in any real vision/audio pipeline.
class FakeProcessor final : public MultimodalProcessorBase {
 public:
  FakeProcessor() = default;

  bool process_prompt(std::string& /*prompt*/,
                      MMData& mm_data,
                      std::vector<int32_t>& token_ids,
                      bool add_special_tokens = true) override {
    last_add_special_tokens_ = add_special_tokens;
    last_multimodal_type_ = mm_data.type();
    if (!process_prompt_succeed_) {
      return false;
    }
    token_ids = prompt_tokens_;
    return true;
  }

  bool process_prompt_tokens(MMData& /*mm_data*/,
                             std::vector<int32_t>& /*token_ids*/) override {
    return process_prompt_succeed_;
  }

  bool process_multimodal(const MMInput& /*inputs*/,
                          MMData& data) const override {
    data = processed_data_;
    return process_multimodal_succeed_;
  }

  void set_process_prompt_succeed(bool succeed) {
    process_prompt_succeed_ = succeed;
  }

  void set_prompt_tokens(std::vector<int32_t> tokens) {
    prompt_tokens_ = std::move(tokens);
  }

  void set_processed_data(MMData data) { processed_data_ = std::move(data); }
  bool last_add_special_tokens() const { return last_add_special_tokens_; }
  uint32_t last_multimodal_type() const { return last_multimodal_type_; }

 private:
  bool process_prompt_succeed_ = true;
  bool process_multimodal_succeed_ = true;
  std::vector<int32_t> prompt_tokens_{1, 2, 3};
  MMData processed_data_;
  bool last_add_special_tokens_ = true;
  uint32_t last_multimodal_type_ = static_cast<uint32_t>(MMType::NONE);
};

class DisconnectableCall final : public Call {
 public:
  explicit DisconnectableCall(brpc::Controller* controller)
      : Call(controller) {}
  bool is_disconnected() const override { return disconnected_; }
  void disconnect() { disconnected_ = true; }

 private:
  bool disconnected_ = false;
};

// Jinja chat template stub. The base constructor parses a trivial literal
// template (no Jinja tags, so it never fails to parse), while apply is
// overridden to return a fixed prompt or report failure on demand.
class FakeJinjaChatTemplate final : public JinjaChatTemplate {
 public:
  FakeJinjaChatTemplate() : JinjaChatTemplate(make_template_args()) {}

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

 private:
  static TokenizerArgs make_template_args() {
    TokenizerArgs args;
    args.chat_template("hello");
    return args;
  }

  bool succeed_ = true;
  std::string rendered_prompt_ = "rendered prompt";
};

class VLMRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    rate_limiter_.is_limited();
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  std::unique_ptr<VLMRequestFactory> make_factory(int32_t vocab_size = 1000,
                                                  int32_t max_position = 2048) {
    processor_ = std::make_unique<FakeProcessor>();
    chat_template_ = std::make_unique<FakeJinjaChatTemplate>();
    tokenizer_ = std::make_unique<FakeTokenizer>(vocab_size);
    model_args_.vocab_size(vocab_size)
        .max_position_embeddings(max_position)
        .eos_token_id(-1);
    // enable_chunked_prefill defaults true; keep it so the prompt length limit
    // is exactly max_position_embeddings.
    options_.enable_service_routing(false).num_speculative_tokens(0);
    return std::make_unique<VLMRequestFactory>(processor_.get(),
                                               chat_template_.get(),
                                               tokenizer_.get(),
                                               &model_args_,
                                               &options_,
                                               &rate_limiter_);
  }

  std::unique_ptr<FakeProcessor> processor_;
  std::unique_ptr<FakeJinjaChatTemplate> chat_template_;
  std::unique_ptr<FakeTokenizer> tokenizer_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
};

TEST_F(VLMRequestFactoryTest,
       RejectsEmptyPromptAndMmDataReleasesRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.called);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Prompt and multimodal data are both empty.");
  // The factory must release the rate-limit slot on every early return.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, FailsWhenProcessPromptFails) {
  auto factory = make_factory();
  processor_->set_process_prompt_succeed(false);
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"describe this",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(), "Failed to process prompt.");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, RejectsPromptLongerThanContext) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  processor_->set_prompt_tokens(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8});
  CallbackCapture capture;
  RequestParams sp;

  auto request = factory->create(/*prompt=*/"describe this",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Prompt is too long");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, FailsWhenStopSequenceEncodingFails) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.stop = std::vector<std::string>{"FAIL-STOP"};

  auto request = factory->create(/*prompt=*/"describe this",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Failed to encode stop sequence");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest,
       CreatesRequestForValidPromptAndKeepsRateLimitSlot) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-1";
  sp.max_tokens = 16;

  auto request = factory->create(/*prompt=*/"describe this",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  // On success the slot stays held; it is later released when the request is
  // completed/destroyed by the scheduler, not by the factory.
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, MtpAcceptsMinimumLengthWithOverlap) {
  auto factory = make_factory();
  options_.enable_schedule_overlap(true)
      .num_speculative_tokens(4)
      .speculative_algorithm("MTP");
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 64;
  sp.min_tokens = 32;

  auto request = factory->create(
      "describe this", MMData{}, sp, make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(request->state().sampling_param.min_tokens, 32U);
  EXPECT_TRUE(request->state().sampling_param.device_min_tokens);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, MtpAcceptsSamplingControlsWithOverlap) {
  auto factory = make_factory();
  options_.enable_schedule_overlap(true)
      .num_speculative_tokens(4)
      .speculative_algorithm("MTP");
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 64;
  sp.min_tokens = 16;
  sp.seed = 42;
  sp.bad_words = {"bad"};
  sp.logit_bias = {{1, 2.0F}};
  sp.allowed_token_ids = std::vector<int32_t>{1, 2, 3};

  auto request = factory->create(
      "describe this", MMData{}, sp, make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  const auto& sampling = request->state().sampling_param;
  EXPECT_EQ(sampling.seed, 42);
  EXPECT_TRUE(sampling.device_min_tokens);
  EXPECT_FALSE(sampling.bad_words_token_ids.empty());
}

TEST_F(VLMRequestFactoryTest, SamplingControlsAreIndependentOfRuntimeOptions) {
  auto factory = make_factory();
  options_.enable_schedule_overlap(true)
      .num_speculative_tokens(4)
      .speculative_algorithm("DFlash");
  CallbackCapture capture;
  RequestParams sp;
  sp.max_tokens = 64;
  sp.min_tokens = 16;
  sp.seed = 42;
  sp.bad_words = {"bad"};
  sp.logit_bias = {{1, 2.0F}};
  sp.allowed_token_ids = std::vector<int32_t>{1, 2, 3};
  sp.logprobs = true;
  sp.top_logprobs = -1;

  auto request = factory->create(
      "describe this", MMData{}, sp, make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  const auto& sampling = request->state().sampling_param;
  EXPECT_EQ(sampling.seed, 42);
  EXPECT_EQ(sampling.min_tokens, 16U);
  EXPECT_TRUE(sampling.device_min_tokens);
  EXPECT_FALSE(sampling.bad_words_token_ids.empty());
  EXPECT_EQ(sampling.logit_bias.at(1), 2.0F);
  EXPECT_EQ(sampling.allowed_token_ids, sp.allowed_token_ids);
  EXPECT_EQ(sampling.top_logprobs, 1000);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, RuntimeOptionsDoNotBypassVocabularyValidation) {
  auto factory = make_factory();
  options_.enable_schedule_overlap(true).num_speculative_tokens(4);
  CallbackCapture capture;
  RequestParams sp;
  sp.seed = 42;
  sp.logit_bias = {{1000, 2.0F}};

  auto request = factory->create(
      "describe this", MMData{}, sp, make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "logit_bias token id is outside the model vocabulary");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, MessageOverloadFailsWhenTemplateRejects) {
  auto factory = make_factory();
  chat_template_->set_succeed(false);
  CallbackCapture capture;
  RequestParams sp;
  // MMInputTransfer requires MMContentVec content; a text part is enough.
  MMContentVec content;
  content.emplace_back("text", std::string("hi"));
  std::vector<Message> messages;
  messages.emplace_back("user", content);

  auto request =
      factory->create(messages, sp, MMInput{}, make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Failed to construct prompt from messages");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest,
       CompletionPreservesAdmissionTimeBeforePromptProcessing) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  const std::chrono::steady_clock::time_point arrival_time{};
  auto request = factory->create(/*prompt=*/"describe this",
                                 /*mm_data=*/MMData{},
                                 sp,
                                 make_capture_callback(&capture),
                                 arrival_time);

  ASSERT_NE(request, nullptr);
  const std::optional<double> elapsed =
      request->server_elapsed_seconds(arrival_time + std::chrono::seconds(7));
  ASSERT_TRUE(elapsed.has_value());
  EXPECT_DOUBLE_EQ(*elapsed, 7.0);
}

TEST_F(VLMRequestFactoryTest,
       ChatPreservesAdmissionTimeBeforeMultimodalPreprocessing) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams sp;
  sp.request_id = "req-chat";
  sp.max_tokens = 16;
  // MMInputTransfer requires MMContentVec content; a text part is enough.
  MMContentVec content;
  content.emplace_back("text", std::string("hi"));
  std::vector<Message> messages;
  messages.emplace_back("user", content);

  const std::chrono::steady_clock::time_point arrival_time{};
  auto request = factory->create(messages,
                                 sp,
                                 MMInput{},
                                 make_capture_callback(&capture),
                                 /*use_prompt_as_is=*/false,
                                 arrival_time);

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  const std::optional<double> elapsed =
      request->server_elapsed_seconds(arrival_time + std::chrono::seconds(7));
  ASSERT_TRUE(elapsed.has_value());
  EXPECT_DOUBLE_EQ(*elapsed, 7.0);
}

TEST_F(VLMRequestFactoryTest, SuppliedTokensDoNotRequirePromptText) {
  auto factory = make_factory();
  CallbackCapture capture;
  RequestParams params;
  auto request = factory->create("",
                                 std::vector<int32_t>{7, 8, 9},
                                 MMData{},
                                 params,
                                 std::nullopt,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().prompt_tokens, (std::vector<int32_t>{7, 8, 9}));
  EXPECT_FALSE(capture.called);
  request.reset();
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, RejectsSuppliedTokensOutsideVocabulary) {
  auto factory = make_factory();
  RequestParams params;
  params.service_request_id = "routed-request";
  params.source_xservice_addr = "source-service";
  RequestOutput failure;
  auto request = factory->create("",
                                 std::vector<int32_t>{1000},
                                 MMData{},
                                 params,
                                 std::nullopt,
                                 [&failure](const RequestOutput& output) {
                                   failure = output;
                                   return false;
                                 });
  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(failure.status.has_value());
  EXPECT_EQ(failure.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(failure.service_request_id, params.service_request_id);
  EXPECT_EQ(failure.target_xservice_addr, params.source_xservice_addr);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, RequestObservesClientDisconnection) {
  auto factory = make_factory();
  brpc::Controller controller;
  DisconnectableCall call(&controller);
  CallbackCapture capture;
  auto request = factory->create("",
                                 std::vector<int32_t>{7},
                                 MMData{},
                                 RequestParams{},
                                 &call,
                                 make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  request->update_connection_status();
  EXPECT_FALSE(request->cancelled());
  call.disconnect();
  request->update_connection_status();
  EXPECT_TRUE(request->cancelled());
}

TEST_F(VLMRequestFactoryTest, TokenCountAndGenerationUseSamePreparedPrompt) {
  auto factory = make_factory();
  RequestParams params;
  params.add_special_tokens = false;
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("hello"));
  int32_t count = 0;
  ASSERT_TRUE(
      factory->count_chat_tokens(messages, params, MMInput{}, count).ok());
  EXPECT_FALSE(processor_->last_add_special_tokens());
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  CallbackCapture capture;
  auto request = factory->create(
      std::move(messages), params, MMInput{}, make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(count, request->state().prompt_tokens.size());
  EXPECT_EQ(count, 3);
}

TEST_F(VLMRequestFactoryTest, TokenCountIncludesMultimodalProcessorExpansion) {
  auto factory = make_factory();
  processor_->set_processed_data(
      MMData(MMType::IMAGE, MMItemVec{MMDataItem(MMType::IMAGE)}));
  processor_->set_prompt_tokens({1, 2, 3, 4, 5, 6});
  MMInputItem image;
  image.type = MMType::IMAGE;
  image.decode_image = torch::ones({3, 2, 2});
  int32_t count = 0;
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("image question"));
  ASSERT_TRUE(
      factory
          ->count_chat_tokens(
              std::move(messages), RequestParams{}, MMInput({image}), count)
          .ok());
  EXPECT_EQ(count, 6);
  EXPECT_EQ(processor_->last_multimodal_type(),
            static_cast<uint32_t>(MMType::IMAGE));
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, CountingDoesNotApplyGenerationContextLimit) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  processor_->set_prompt_tokens(std::vector<int32_t>(12, 1));
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("long prompt"));
  int32_t count = 0;
  ASSERT_TRUE(factory
                  ->count_chat_tokens(
                      std::move(messages), RequestParams{}, MMInput{}, count)
                  .ok());
  EXPECT_EQ(count, 12);
  CallbackCapture capture;
  EXPECT_EQ(factory->create("long prompt",
                            MMData{},
                            RequestParams{},
                            make_capture_callback(&capture)),
            nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->message(), "Prompt is too long");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, CountingFailureDoesNotReleaseGenerationSlot) {
  auto factory = make_factory();
  processor_->set_process_prompt_succeed(false);
  int32_t count = 0;
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("hello"));
  const Status status = factory->count_chat_tokens(
      std::move(messages), RequestParams{}, MMInput{}, count);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

TEST_F(VLMRequestFactoryTest, SpeculativeTokensReserveModelContext) {
  auto factory = make_factory(/*vocab_size=*/1000, /*max_position=*/8);
  options_.num_speculative_tokens(4);
  RequestParams params;
  params.min_tokens = 2;
  CallbackCapture capture;
  EXPECT_EQ(factory->create(
                "hello", MMData{}, params, make_capture_callback(&capture)),
            nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest,
       RoutedRequestKeepsResponseMetadataAndBatchCallback) {
  auto initial_factory = make_factory();
  initial_factory.reset();
  options_.enable_service_routing(true);
  size_t forwarded = 0;
  VLMRequestFactory factory(
      processor_.get(),
      chat_template_.get(),
      tokenizer_.get(),
      &model_args_,
      &options_,
      &rate_limiter_,
      [&forwarded](const std::vector<RequestOutput>& outputs) {
        forwarded += outputs.size();
        return std::vector<bool>(outputs.size(), true);
      });
  RequestParams params;
  params.source_xservice_addr = "source-service";
  params.service_request_id = "service-request";
  params.response_model = "response-model";
  params.decode_address = "decode-address";
  params.ttft_slo_ms = 123;
  CallbackCapture capture;
  auto request = factory.create(
      "hello", MMData{}, params, make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->state().response_model, params.response_model);
  EXPECT_EQ(request->state().decode_address, params.decode_address);
  EXPECT_EQ(request->state().scheduler_param.ttft_slo_ms, params.ttft_slo_ms);
  ASSERT_TRUE(request->state().outputs_func);
  EXPECT_EQ(request->state().outputs_func({RequestOutput{}}),
            (std::vector<bool>{true}));
  EXPECT_EQ(forwarded, 1);
}

class VLMChatTemplateTest
    : public ::testing::TestWithParam<std::tuple<std::string, bool, bool>> {};

TEST_P(VLMChatTemplateTest, GenerationAndCountingUseSelectedTemplate) {
  const auto& [model_type, use_cpp_template, thinking] = GetParam();
  auto& model_config = ModelConfig::get_instance();
  const bool previous_use_cpp = model_config.use_cpp_chat_template();
  ScopeGuard restore_config([previous_use_cpp] {
    ModelConfig::get_instance().use_cpp_chat_template(previous_use_cpp);
  });
  model_config.use_cpp_chat_template(use_cpp_template);

  TokenizerArgs tokenizer_args;
  tokenizer_args.bos_token("<｜begin▁of▁sentence｜>")
      .chat_template(
          "jinja:{{ messages[0].content }}"
          "{% if thinking %}<think>{% else %}</think>{% endif %}");
  auto chat_template = ChatTemplateFactory::create(tokenizer_args, model_type);
  auto tokenizer = std::make_shared<FakeTokenizer>(/*vocab_size=*/1000);
  ModelArgs model_args;
  model_args.model_type(model_type)
      .vocab_size(1000)
      .max_position_embeddings(2048)
      .eos_token_id(-1);
  MultimodalProcessor<TextPromptProcessor> processor(
      model_args, tokenizer, tokenizer_args);
  Options options;
  RateLimiter rate_limiter;
  ASSERT_TRUE(rate_limiter.acquire().ok());
  VLMRequestFactory factory(&processor,
                            chat_template.get(),
                            tokenizer.get(),
                            &model_args,
                            &options,
                            &rate_limiter);
  RequestParams params;
  params.max_tokens = 16;
  params.chat_template_kwargs = {{"thinking", thinking},
                                 {"reasoning_effort", "low"}};
  std::vector<Message> messages;
  messages.emplace_back("user", std::string("hello"));

  std::string expected =
      use_cpp_template ? "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>"
                       : "jinja:hello";
  expected += thinking ? "<think>" : "</think>";
  std::vector<int32_t> expected_tokens;
  ASSERT_TRUE(tokenizer->encode(expected, &expected_tokens));

  int32_t count = 0;
  ASSERT_TRUE(
      factory.count_chat_tokens(messages, params, MMInput{}, count).ok());
  EXPECT_EQ(count, expected_tokens.size());
  CallbackCapture capture;
  auto request = factory.create(
      std::move(messages), params, MMInput{}, make_capture_callback(&capture));
  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(request->state().prompt, expected);
  EXPECT_EQ(request->state().prompt_tokens, expected_tokens);
}

INSTANTIATE_TEST_SUITE_P(Deepseek,
                         VLMChatTemplateTest,
                         ::testing::Combine(::testing::Values("deepseek_v32",
                                                              "deepseek_v4",
                                                              "deepseek_v41"),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace xllm
