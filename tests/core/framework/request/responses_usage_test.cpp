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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "framework/block/block_manager_pool.h"
#include "framework/chat_template/chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/llm_request_factory.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "parser/reasoning_parser.h"

namespace xllm {
namespace {

constexpr int32_t kReasoningStart = 300;
constexpr int32_t kReasoningEnd = 301;

class UsageTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    ids->clear();
    ids->reserve(text.size());
    if (text == "<think>") {
      ids->push_back(kReasoningStart);
    } else if (text == "</think>") {
      ids->push_back(kReasoningEnd);
    } else {
      for (unsigned char value : text) {
        ids->push_back(value);
      }
    }
    return true;
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string text;
    for (int32_t id : ids) {
      if (id == kReasoningStart) {
        text += "<think>";
      } else if (id == kReasoningEnd) {
        text += "</think>";
      } else if (id >= 0 && id < 256) {
        text.push_back(static_cast<char>(id));
      }
    }
    return text;
  }
};

class UsageChatTemplate final : public ChatTemplate {
 public:
  std::optional<std::string> apply(
      const ChatMessages& /*messages*/) const override {
    return "prompt<think>";
  }

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<JsonTool>& /*json_tools*/,
      const nlohmann::ordered_json& /*chat_template_kwargs*/) const override {
    return apply(messages);
  }
};

std::shared_ptr<Request> make_usage_request(std::vector<int32_t> prompt_tokens,
                                            bool force_reasoning = false,
                                            bool overlap = false,
                                            bool responses_request = true) {
  RequestState state(
      "prompt",
      std::move(prompt_tokens),
      RequestSamplingParam{},
      SchedulerParam{},
      StoppingChecker(32, 64, -1, false, {}, {}),
      /*seq_capacity=*/64,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/false,
      overlap,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  state.responses_request = responses_request;
  state.force_reasoning = force_reasoning;
  return std::make_shared<Request>("usage-test", "", "", std::move(state));
}

TEST(ResponsesUsageTest, ExcludesOverlapPlaceholdersFromGeneratedUsage) {
  auto request = make_usage_request({1, 2, 3}, true, true);
  Sequence& seq = *request->sequences().front();
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  seq.append_token(Token(-1));
  EXPECT_EQ(seq.num_valid_generated_tokens(), 0u);
  seq.update_last_step_token(Token(10));
  EXPECT_EQ(seq.num_valid_generated_tokens(), 1u);
  seq.append_token(Token(-1));
  seq.update_last_step_token(Token(kReasoningEnd));
  seq.append_token(Token(-1));
  EXPECT_EQ(seq.num_valid_generated_tokens(), 2u);
}

TEST(ResponsesUsageTest, CountsRetainedEosAndPropagatesInitialTextState) {
  UsageTokenizer tokenizer;
  auto request = make_usage_request({1, 2, 3}, true);
  request->state().stopping_checker.set_eos_token(11);
  Sequence& seq = *request->sequences().front();
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  seq.append_token(Token(10));
  seq.append_token(Token(11));
  RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_prompt_tokens, 3);
  EXPECT_EQ(output.usage->num_generated_tokens, 2);
  EXPECT_EQ(output.usage->num_total_tokens, 5);
  EXPECT_EQ(output.force_reasoning, true);
  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_EQ(output.outputs.front().text, std::string(1, '\n'));
}

class ResponsesCacheUsageTest : public ::testing::Test {
 protected:
  BlockManagerPool::Options options(bool enabled) {
    BlockManagerPool::Options value;
    value.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
        enabled);
    value.max_seqs_per_batch(1024);
    return value;
  }

  void evict(BlockManagerPool& pool) {
    int32_t rank = 0;
    // All sequence owners have been released before evicting the cache.
    auto blocks = pool.allocate(15 * 4, rank);
    ASSERT_EQ(blocks.size(), 15u);
  }
};

TEST_F(ResponsesCacheUsageTest, CountsRetainedMtpIdsWithoutPlaceholders) {
  BlockManagerPool pool(options(/*enabled=*/false), /*dp_size=*/1);
  auto request = make_usage_request({1, 2, 3},
                                    /*force_reasoning=*/false,
                                    /*overlap=*/true);
  Sequence& seq = *request->sequences().front();
  ASSERT_TRUE(pool.allocate(&seq, /*num_tokens=*/12));
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  seq.append_token(Token(-1));
  seq.append_token(Token(-1));
  seq.update_last_step_token(Token(kReasoningStart), /*token_offset=*/0);
  for (int32_t id : {10, 11, kReasoningEnd, 12}) {
    seq.update_last_step_token(Token(id), /*token_offset=*/1);
  }
  const std::vector<int32_t> retained(seq.tokens().begin(), seq.tokens().end());
  EXPECT_EQ(retained,
            (std::vector<int32_t>{
                1, 2, 3, kReasoningStart, 10, 11, kReasoningEnd, 12, -1}));
  EXPECT_EQ(seq.num_valid_generated_tokens(), 5u);
  UsageTokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_generated_tokens, 5);
  EXPECT_EQ(output.usage->num_total_tokens, 8);
  pool.deallocate(&seq);
}

TEST_F(ResponsesCacheUsageTest, PreservesCachedInputHighWaterOnPreemption) {
  BlockManagerPool pool(options(/*enabled=*/true), /*dp_size=*/1);
  auto request = make_usage_request({1, 2, 3, 4, 5, 6, 7, 8, 9});
  Sequence& seq = *request->sequences().front();
  ASSERT_TRUE(pool.allocate(&seq, seq.num_prompt_tokens()));
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  pool.deallocate(&seq);
  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::KV), 0u);
  ASSERT_TRUE(pool.allocate(&seq, seq.num_prompt_tokens()));
  request->record_num_prefix_cache_tokens();
  EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  pool.deallocate(&seq);

  UsageTokenizer tokenizer;
  RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_cached_tokens, 8);
  evict(pool);
}

TEST_F(ResponsesCacheUsageTest, CountsInitialCachedInputHits) {
  BlockManagerPool pool(options(/*enabled=*/true), /*dp_size=*/1);
  auto first = make_usage_request({1, 2, 3, 4, 5, 6, 7, 8, 9});
  Sequence& seed = *first->sequences().front();
  ASSERT_TRUE(pool.allocate(&seed, seed.num_prompt_tokens()));
  seed.kv_state().set_kv_cache_tokens_num(seed.num_prompt_tokens());
  pool.deallocate(&seed);

  auto hit = make_usage_request({1, 2, 3, 4, 5, 6, 7, 8, 11});
  Sequence& seq = *hit->sequences().front();
  ASSERT_TRUE(pool.allocate(&seq, seq.num_prompt_tokens()));
  hit->record_num_prefix_cache_tokens();
  EXPECT_EQ(hit->num_prefix_cache_tokens(), 8u);
  seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
  pool.deallocate(&seq);
  evict(pool);
}

TEST_F(ResponsesCacheUsageTest, DisabledCacheAndLegacyRequestsPreserveUsage) {
  for (bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    BlockManagerPool pool(options(enabled), /*dp_size=*/1);
    auto request = make_usage_request({1, 2, 3, 4, 5}, false, false, !enabled);
    Sequence& seq = *request->sequences().front();
    ASSERT_TRUE(pool.allocate(&seq, seq.num_prompt_tokens()));
    seq.kv_state().set_kv_cache_tokens_num(seq.num_prompt_tokens());
    pool.deallocate(&seq);
    UsageTokenizer tokenizer;
    const RequestOutput output = request->generate_output(tokenizer);
    ASSERT_TRUE(output.usage.has_value());
    EXPECT_EQ(output.usage->num_cached_tokens, 0);
    EXPECT_EQ(output.force_reasoning.has_value(), !enabled);
    if (enabled) {
      evict(pool);
    }
  }
}

class ResponsesUsageFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    args_.vocab_size(512).max_position_embeddings(128).eos_token_id(-1);
    options_.enable_schedule_overlap(false).num_speculative_tokens(0);
  }

  std::shared_ptr<Request> create(std::string prompt,
                                  RequestParams params,
                                  std::optional<Status>* status) {
    EXPECT_TRUE(limiter_.acquire().ok());
    LLMRequestFactory factory(&tokenizer_,
                              &chat_template_,
                              &args_,
                              &options_,
                              &limiter_,
                              "generate",
                              LLMRequestFactory::RpcResponseHandler{});
    return factory.create(std::move(prompt),
                          std::nullopt,
                          params,
                          std::nullopt,
                          [status](RequestOutput output) {
                            *status = output.status;
                            return true;
                          });
  }

  UsageTokenizer tokenizer_;
  UsageChatTemplate chat_template_;
  ModelArgs args_;
  Options options_;
  RateLimiter limiter_;
};

TEST_F(ResponsesUsageFactoryTest, LargeOutputBudgetReservesOnlyModelContext) {
  args_.max_position_embeddings(8);
  options_.enable_schedule_overlap(true).num_speculative_tokens(2);
  RequestParams params;
  params.max_tokens = std::numeric_limits<uint32_t>::max();
  params.responses_request = true;
  params.responses_reasoning_parser = "qwen3";
  std::optional<Status> status;
  auto request = create("abc", params, &status);
  ASSERT_NE(request, nullptr);
  EXPECT_FALSE(status.has_value());
  EXPECT_EQ(request->state().seq_capacity, 14u);
  const StoppingChecker& checker = request->state().stopping_checker;
  EXPECT_EQ(checker.get_max_generated_tokens(),
            std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(checker.get_max_context_len(), 6u);
  EXPECT_EQ(
      checker.check(request->state().prompt_tokens, /*num_prompt_tokens=*/3),
      FinishReason::NONE);
  const std::vector<int32_t> context_limit = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(checker.check(context_limit, /*num_prompt_tokens=*/3),
            FinishReason::LENGTH);
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 1);
  request.reset();
  EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
}

TEST_F(ResponsesUsageFactoryTest,
       ResolvesSelectedParserAndRenderedPromptState) {
  for (const auto& [parser, prompt, forced] :
       std::vector<std::tuple<std::string, std::string, bool>>{
           {"qwen3", "prompt<think> \n", true},
           {"qwen3-thinking", "prompt</think>\n", false},
           {"qwen3-thinking", "prompt", true},
           {"glm5", "prompt", false}}) {
    SCOPED_TRACE(parser + ":" + prompt);
    RequestParams params;
    params.responses_request = true;
    params.responses_reasoning_parser = parser;
    std::optional<Status> status;
    auto request = create(prompt, params, &status);
    ASSERT_NE(request, nullptr);
    EXPECT_FALSE(status.has_value());
    EXPECT_TRUE(request->state().responses_request);
    EXPECT_EQ(request->state().force_reasoning, forced);
    ReasoningParser text_parser(parser,
                                /*stream_reasoning=*/true,
                                /*force_reasoning=*/false,
                                request->state().force_reasoning,
                                /*lossless=*/true);
    ReasoningResult text = text_parser.parse_stream_chunk("w");
    EXPECT_EQ(text.reasoning_text.has_value(), forced);
    EXPECT_EQ(text.normal_text.has_value(), !forced);
    request.reset();
    EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
  }
}

TEST_F(ResponsesUsageFactoryTest, RejectsUnsupportedConfigurationsAndReleases) {
  for (int32_t variant = 0; variant < 5; ++variant) {
    SCOPED_TRACE(variant);
    RequestParams params;
    params.responses_request = true;
    params.responses_reasoning_parser = "qwen3";
    options_ = Options{};
    options_.enable_schedule_overlap(false).num_speculative_tokens(0);
    switch (variant) {
      case 0:
        params.responses_reasoning_parser = "unsupported";
        break;
      case 1:
        options_.enable_disagg_pd(true);
        break;
      case 2:
        options_.enable_kvcache_store(true);
        break;
      case 3:
        options_.host_blocks_factor(2.0);
        break;
      case 4:
        params.n = 2;
        break;
    }
    std::optional<Status> status;
    EXPECT_EQ(create("prompt", params, &status), nullptr);
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(status->code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(limiter_.get_num_concurrent_requests(), 0);
  }
}

}  // namespace
}  // namespace xllm
