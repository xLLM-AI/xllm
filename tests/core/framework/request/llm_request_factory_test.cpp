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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "framework/chat_template/chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

// Deterministic tokenizer used for factory tests. It encodes each character to
// a token id inside the vocabulary range, and can be told to fail encoding of
// any text containing the marker "FAIL" so stop-sequence encode errors can be
// exercised independently of prompt encoding.
class FakeTokenizer final : public Tokenizer {
 public:
  explicit FakeTokenizer(int32_t vocab_size) : vocab_size_(vocab_size) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    if (text.find("FAIL") != std::string_view::npos) {
      return false;
    }
    ids->clear();
    for (const char c : text) {
      ids->push_back(static_cast<int32_t>(static_cast<unsigned char>(c)) %
                     vocab_size_);
    }
    if (ids->empty()) {
      ids->push_back(1);
    }
    return true;
  }

  size_t vocab_size() const override {
    return static_cast<size_t>(vocab_size_);
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }

 private:
  int32_t vocab_size_ = 1000;
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

 private:
  bool succeed_ = true;
  std::string rendered_prompt_ = "rendered prompt";
};

// Records the last error surfaced through the OutputCallback.
struct CallbackCapture {
  bool called = false;
  std::optional<Status> status;
};

OutputCallback make_capture_callback(CallbackCapture* capture) {
  return [capture](RequestOutput output) {
    capture->called = true;
    capture->status = output.status;
    return false;
  };
}

class LLMRequestFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Simulate the caller (service entry) having acquired a rate-limit slot.
    rate_limiter_.is_limited();
    ASSERT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
  }

  std::unique_ptr<LLMRequestFactory> make_factory(
      int32_t vocab_size = 1000,
      int32_t max_position = 2048,
      std::string task = "generate") {
    tokenizer_ = std::make_unique<FakeTokenizer>(vocab_size);
    chat_template_ = std::make_unique<FakeChatTemplate>();
    model_args_.vocab_size(vocab_size)
        .max_position_embeddings(max_position)
        .eos_token_id(-1);
    // enable_chunked_prefill defaults true; keep it so the prompt length limit
    // is exactly max_position_embeddings.
    options_.enable_service_routing(false).num_speculative_tokens(0);
    return std::make_unique<LLMRequestFactory>(
        tokenizer_.get(),
        chat_template_.get(),
        &model_args_,
        &options_,
        &rate_limiter_,
        std::move(task),
        [](const std::vector<RequestOutput>&) { return std::vector<bool>{}; });
  }

  std::unique_ptr<FakeTokenizer> tokenizer_;
  std::unique_ptr<FakeChatTemplate> chat_template_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
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
