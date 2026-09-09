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

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {
namespace {

// Deterministic tokenizer used for factory tests. It only participates in stop
// sequence encoding here; encoding fails for any text containing "FAIL" so the
// stop-sequence error path can be exercised independently.
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

// Fake multimodal processor that lets tests drive process_prompt success and
// the tokens it emits, without pulling in any real vision/audio pipeline.
class FakeProcessor final : public MultimodalProcessorBase {
 public:
  FakeProcessor() : MultimodalProcessorBase(/*tokenizer=*/nullptr) {}

  bool process_prompt(std::string& /*prompt*/,
                      MMData& /*mm_data*/,
                      std::vector<int32_t>& token_ids) override {
    if (!process_prompt_succeed_) {
      return false;
    }
    token_ids = prompt_tokens_;
    return true;
  }

  bool process_multimodal(const MMInput& /*inputs*/,
                          MMData& /*data*/) const override {
    return process_multimodal_succeed_;
  }

  void set_process_prompt_succeed(bool succeed) {
    process_prompt_succeed_ = succeed;
  }

  void set_prompt_tokens(std::vector<int32_t> tokens) {
    prompt_tokens_ = std::move(tokens);
  }

 private:
  bool process_prompt_succeed_ = true;
  bool process_multimodal_succeed_ = true;
  std::vector<int32_t> prompt_tokens_{1, 2, 3};
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

  auto request = factory->create(messages,
                                 sp,
                                 /*payload=*/std::string{},
                                 make_capture_callback(&capture));

  EXPECT_EQ(request, nullptr);
  ASSERT_TRUE(capture.status.has_value());
  EXPECT_EQ(capture.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(capture.status->message(),
            "Failed to construct prompt from messages");
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(VLMRequestFactoryTest, MessageOverloadCreatesRequestOnSuccess) {
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

  auto request = factory->create(messages,
                                 sp,
                                 /*payload=*/std::string{},
                                 make_capture_callback(&capture));

  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(capture.called);
  EXPECT_EQ(rate_limiter_.get_num_concurrent_requests(), 1);
}

}  // namespace
}  // namespace xllm
