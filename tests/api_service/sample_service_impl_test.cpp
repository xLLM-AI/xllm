/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "sample_service_impl.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

class CharTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true,
              int32_t max_sequence_length = 0) const override {
    if (ids == nullptr) {
      return false;
    }
    ids->clear();
    if (add_special_tokens) {
      ids->push_back(kBosTokenId);
    }
    size_t pos = 0;
    while (pos < text.size()) {
      if (text.size() - pos >= kEmbTokenLen &&
          std::memcmp(text.data() + pos, kEmbToken, kEmbTokenLen) == 0) {
        ids->push_back(kEmbTokenId);
        pos += kEmbTokenLen;
        continue;
      }
      ids->push_back(static_cast<unsigned char>(text[pos]));
      ++pos;
    }
    return true;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    if (token == std::string_view(kEmbToken, kEmbTokenLen)) {
      return kEmbTokenId;
    }
    return std::nullopt;
  }

  std::string id_to_token(int32_t id) const override {
    if (id == kBosTokenId) {
      return "<bos>";
    }
    if (id == kEmbTokenId) {
      return std::string(kEmbToken, kEmbTokenLen);
    }
    return std::string(1, static_cast<char>(id));
  }

 private:
  static constexpr int32_t kBosTokenId = 1;
  static constexpr int32_t kEmbTokenId = 100000;
  static constexpr char kEmbToken[] = "<emb_0>";
  static constexpr size_t kEmbTokenLen = sizeof(kEmbToken) - 1;
};

class UnstableLiteralTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true,
              int32_t max_sequence_length = 0) const override {
    if (ids == nullptr) {
      return false;
    }
    ids->clear();
    if (add_special_tokens) {
      ids->push_back(1);
    }
    for (char ch : text) {
      ids->push_back(static_cast<unsigned char>(ch));
    }
    return true;
  }
};

proto::SampleRequest make_valid_request() {
  proto::SampleRequest request;
  request.set_model("mtp");
  request.set_prompt("plain text");
  auto* selector = request.mutable_selector();
  selector->set_type("literal");
  selector->set_value("<emb_0>");
  return request;
}

TEST(SampleServiceImplTest, ValidateRequestRejectsUnsupportedSelectorType) {
  auto request = make_valid_request();
  request.mutable_selector()->set_type("regex");

  const Status status = sample_service_internal::validate_request(request);

  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find("literal"), std::string::npos);
}

TEST(SampleServiceImplTest, ValidateRequestRejectsOutOfRangeLogprobs) {
  auto request = make_valid_request();
  request.set_logprobs(0);

  Status status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find("between 1 and 5"), std::string::npos);

  request.set_logprobs(6);
  status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find("between 1 and 5"), std::string::npos);
}

TEST(SampleServiceImplTest, ValidateRequestRejectsMissingRequiredFields) {
  auto request = make_valid_request();
  request.clear_model();

  Status status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "model is required");

  request = make_valid_request();
  request.clear_prompt();
  status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "prompt is required");

  request = make_valid_request();
  request.clear_selector();
  status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "selector is required");

  request = make_valid_request();
  request.mutable_selector()->clear_value();
  status = sample_service_internal::validate_request(request);
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "selector.value is required");
}

TEST(SampleServiceImplTest, ValidateRuntimeConfigRejectsScheduleOverlap) {
  Status status = sample_service_internal::validate_runtime_config(true);
  EXPECT_EQ(status.code(), StatusCode::UNAVAILABLE);
  EXPECT_NE(status.message().find("does not support async scheduling"),
            std::string::npos);

  status = sample_service_internal::validate_runtime_config(false);
  EXPECT_TRUE(status.ok());
}

TEST(SampleServiceImplTest, BuildRequestParamsAndEmptyResponseForSelectorMiss) {
  CharTokenizer tokenizer;
  auto request = make_valid_request();

  RequestParams params;
  ASSERT_TRUE(sample_service_internal::build_request_params(
      request, tokenizer, &params));
  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs,
            sample_service_internal::kDefaultSampleLogprobs);
  EXPECT_TRUE(params.is_sample_request);
  EXPECT_TRUE(params.sample_slots.empty());
  EXPECT_FALSE(params.request_id.empty());

  proto::SampleResponse response;
  ASSERT_TRUE(sample_service_internal::build_empty_response(
      request, tokenizer, params.request_id, &response));

  EXPECT_EQ(response.id(), params.request_id);
  EXPECT_EQ(response.object(), "sample_completion");
  EXPECT_EQ(response.model(), request.model());
  EXPECT_EQ(response.choices_size(), 0);
  EXPECT_GT(response.created(), 0U);
  ASSERT_TRUE(response.has_usage());
  const int32_t expected_prompt_tokens =
      static_cast<int32_t>(request.prompt().size() + 1);
  EXPECT_EQ(response.usage().prompt_tokens(), expected_prompt_tokens);
  EXPECT_EQ(response.usage().completion_tokens(), 0);
  EXPECT_EQ(response.usage().total_tokens(), expected_prompt_tokens);
}

TEST(SampleServiceImplTest,
     BuildRequestParamsKeepsExplicitRequestIdAndMatchedSampleSlots) {
  CharTokenizer tokenizer;
  auto request = make_valid_request();
  request.set_prompt("A<emb_0>B<emb_0>C");
  request.set_request_id("sample-explicit");
  request.set_logprobs(4);

  RequestParams params;
  ASSERT_TRUE(sample_service_internal::build_request_params(
      request, tokenizer, &params));

  EXPECT_EQ(params.request_id, "sample-explicit");
  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 4);
  EXPECT_TRUE(params.is_sample_request);
  ASSERT_EQ(params.sample_slots.size(), 2);
  EXPECT_EQ(params.sample_slots[0].request_id, "sample-explicit");
  EXPECT_EQ(params.sample_slots[0].sample_id, 0);
  EXPECT_EQ(params.sample_slots[0].token_position, 1);
  EXPECT_EQ(params.sample_slots[1].request_id, "sample-explicit");
  EXPECT_EQ(params.sample_slots[1].sample_id, 1);
  EXPECT_EQ(params.sample_slots[1].token_position, 3);
}

TEST(SampleServiceImplTest, BuildRequestParamsRejectsUnstableLiteralToken) {
  UnstableLiteralTokenizer tokenizer;
  auto request = make_valid_request();
  request.set_prompt("A<emb_0>B");
  request.set_request_id("sample-explicit");

  RequestParams params;
  EXPECT_FALSE(sample_service_internal::build_request_params(
      request, tokenizer, &params));
}

TEST(SampleServiceImplTest,
     BuildResponseSortsBySampleIdAndSerializesTopLogprobs) {
  RequestOutput req_output;
  Usage usage;
  usage.num_prompt_tokens = 8;
  usage.num_generated_tokens = 2;
  usage.num_total_tokens = 10;
  req_output.usage = usage;

  SequenceOutput missing_output;
  missing_output.index = 1;
  missing_output.finish_reason = "empty_logprobs";

  SequenceOutput sampled_output;
  sampled_output.index = 0;
  sampled_output.text = "stale";
  LogProb sampled_logprob;
  sampled_logprob.token = "True";
  sampled_logprob.token_id = 101;
  sampled_logprob.logprob = -0.10f;
  std::vector<LogProbData> top_logprobs;
  LogProbData top1;
  top1.token = "True";
  top1.token_id = 101;
  top1.logprob = -0.10f;
  top_logprobs.push_back(top1);
  LogProbData top2;
  top2.token = "False";
  top2.token_id = 102;
  top2.logprob = -2.30f;
  top_logprobs.push_back(top2);
  sampled_logprob.top_logprobs = top_logprobs;
  sampled_output.logprobs = std::vector<LogProb>{sampled_logprob};

  req_output.outputs = {missing_output, sampled_output};

  proto::SampleResponse response;
  ASSERT_TRUE(sample_service_internal::build_response(
      "sample-123", "mtp", 1773369600U, req_output, &response));

  EXPECT_EQ(response.id(), "sample-123");
  EXPECT_EQ(response.object(), "sample_completion");
  EXPECT_EQ(response.created(), 1773369600U);
  EXPECT_EQ(response.model(), "mtp");
  ASSERT_EQ(response.choices_size(), 2);

  EXPECT_EQ(response.choices(0).index(), 0);
  EXPECT_EQ(response.choices(0).text(), "True");
  EXPECT_EQ(response.choices(0).finish_reason(), "selector_match");
  ASSERT_TRUE(response.choices(0).has_logprobs());
  EXPECT_EQ(response.choices(0).logprobs().tokens_size(), 2);
  EXPECT_EQ(response.choices(0).logprobs().tokens(0), "True");
  EXPECT_EQ(response.choices(0).logprobs().tokens(1), "False");
  EXPECT_EQ(response.choices(0).logprobs().token_ids(0), 101);
  EXPECT_EQ(response.choices(0).logprobs().token_ids(1), 102);

  EXPECT_EQ(response.choices(1).index(), 1);
  EXPECT_TRUE(response.choices(1).text().empty());
  EXPECT_EQ(response.choices(1).finish_reason(), "empty_logprobs");
  ASSERT_TRUE(response.choices(1).has_logprobs());
  EXPECT_EQ(response.choices(1).logprobs().tokens_size(), 0);
  EXPECT_EQ(response.choices(1).logprobs().token_ids_size(), 0);
  EXPECT_EQ(response.choices(1).logprobs().token_logprobs_size(), 0);

  ASSERT_TRUE(response.has_usage());
  EXPECT_EQ(response.usage().prompt_tokens(), 8);
  EXPECT_EQ(response.usage().completion_tokens(), 2);
  EXPECT_EQ(response.usage().total_tokens(), 10);
}

TEST(SampleServiceImplTest, BuildResponseFallsBackToSelectedTokenLogprob) {
  RequestOutput req_output;

  SequenceOutput sampled_output;
  sampled_output.index = 0;
  LogProb sampled_logprob;
  sampled_logprob.token = "Maybe";
  sampled_logprob.token_id = 201;
  sampled_logprob.logprob = -0.25f;
  sampled_output.logprobs = std::vector<LogProb>{sampled_logprob};
  req_output.outputs = {sampled_output};

  proto::SampleResponse response;
  ASSERT_TRUE(sample_service_internal::build_response(
      "sample-456", "mtp", 1773369601U, req_output, &response));

  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_EQ(response.choices(0).text(), "Maybe");
  EXPECT_EQ(response.choices(0).finish_reason(), "selector_match");
  ASSERT_TRUE(response.choices(0).has_logprobs());
  EXPECT_EQ(response.choices(0).logprobs().tokens_size(), 1);
  EXPECT_EQ(response.choices(0).logprobs().tokens(0), "Maybe");
  EXPECT_EQ(response.choices(0).logprobs().token_ids(0), 201);
}

}  // namespace
}  // namespace xllm
