/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/rec_sequence.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/rec_config.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_type.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/onerec_sequence.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/sequence_factory.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

constexpr int32_t kBosTokenId = 7;

// Restores the process-wide RecConfig flags this test touches.
class ScopedRecConfig {
 public:
  ScopedRecConfig()
      : sku_logprobs_(RecConfig::get_instance().enable_output_sku_logprobs()),
        convert_items_(
            RecConfig::get_instance().enable_convert_tokens_to_item()),
        extended_info_(RecConfig::get_instance().enable_extended_item_info()),
        threshold_(RecConfig::get_instance().each_conversion_threshold()) {}

  ~ScopedRecConfig() {
    RecConfig::get_instance()
        .enable_output_sku_logprobs(sku_logprobs_)
        .enable_convert_tokens_to_item(convert_items_)
        .enable_extended_item_info(extended_info_)
        .each_conversion_threshold(threshold_);
  }

 private:
  bool sku_logprobs_;
  bool convert_items_;
  bool extended_info_;
  int32_t threshold_;
};

// Decodes any REC token triple to two fixed item ids.
class FakeItemTokenizer final : public Tokenizer {
 public:
  using Tokenizer::decode;

  bool decode(const Slice<int32_t>& /*token_ids*/,
              bool /*skip_special_tokens*/,
              std::vector<int64_t>* item_ids) const override {
    *item_ids = {1001, 1002, 1001};
    return true;
  }
};

class SequenceFixture {
 public:
  SequenceFixture() {
    params_.seq_capacity = 16;
    params_.echo = false;
    params_.skip_special_tokens = true;
    params_.streaming = false;
    params_.enable_schedule_overlap = false;
    params_.bos_token_id = kBosTokenId;
    params_.request_id = "rec_sequence_test";
    params_.sampling_param = &sampling_param_;
    params_.stopping_checker = &stopping_checker_;
  }

  SequenceParams& params() { return params_; }
  RequestSamplingParam& sampling_param() { return sampling_param_; }

  std::unique_ptr<Sequence> create(RecType rec_type,
                                   const std::vector<int32_t>& prompt,
                                   const MMData& mm_data = MMData()) {
    params_.rec_type = rec_type;
    IncrementalDecoder decoder(/*prompt=*/"prompt",
                               prompt.size(),
                               params_.echo,
                               params_.skip_special_tokens);
    return create_sequence(/*index=*/0,
                           prompt,
                           /*input_embedding=*/torch::Tensor(),
                           mm_data,
                           decoder,
                           params_);
  }

 private:
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  SequenceParams params_;
};

MMData embedding_mm_data(const std::string& key, int64_t rows) {
  MMData mm_data;
  mm_data.add(MMType::EMBEDDING, key, torch::zeros({rows, 4}));
  return mm_data;
}

void append_generated(Sequence& sequence,
                      const std::vector<int32_t>& ids,
                      float logprob = -0.5f) {
  for (const int32_t id : ids) {
    Token token(id);
    token.logprob = logprob;
    sequence.append_token(token);
  }
}

// ---------------------------------------------------------------------------
// Factory selection
// ---------------------------------------------------------------------------

TEST(SequenceFactoryTest, LlmRequestGetsPlainSequence) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kNone, {1, 2, 3});
  EXPECT_EQ(dynamic_cast<RecSequence*>(sequence.get()), nullptr);
  EXPECT_EQ(sequence->num_prompt_tokens(), 3u);
}

TEST(SequenceFactoryTest, LlmRecRequestGetsLlmRecSequence) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kLlmRec, {1, 2, 3});
  EXPECT_NE(dynamic_cast<LlmRecSequence*>(sequence.get()), nullptr);
  EXPECT_EQ(dynamic_cast<OneRecSequence*>(sequence.get()), nullptr);
  // LlmRec drives the decoder like an LLM: the prompt seeds the token buffer.
  EXPECT_EQ(sequence->num_prompt_tokens(), 3u);
  EXPECT_EQ(sequence->tokens()[0], 1);
}

TEST(SequenceFactoryTest, OneRecRequestGetsOneRecSequence) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {1, 2, 3});
  EXPECT_NE(dynamic_cast<OneRecSequence*>(sequence.get()), nullptr);
}

// ---------------------------------------------------------------------------
// Typed access
// ---------------------------------------------------------------------------

TEST(RecSequenceTest, FromAcceptsRecSequences) {
  SequenceFixture fixture;
  auto llm_rec = fixture.create(RecType::kLlmRec, {1, 2, 3});
  auto onerec = fixture.create(RecType::kOneRec, {1, 2, 3});
  EXPECT_EQ(&RecSequence::from(*llm_rec), llm_rec.get());
  EXPECT_EQ(&RecSequence::from(*onerec), onerec.get());
  EXPECT_EQ(&OneRecSequence::from(*onerec), onerec.get());
}

TEST(RecSequenceDeathTest, FromRejectsPlainSequence) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kNone, {1, 2, 3});
  EXPECT_DEATH(RecSequence::from(*sequence), "not a REC sequence");
}

TEST(RecSequenceDeathTest, OneRecFromRejectsLlmRecSequence) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kLlmRec, {1, 2, 3});
  EXPECT_DEATH(OneRecSequence::from(*sequence), "not a OneRec sequence");
}

// ---------------------------------------------------------------------------
// Beam result
// ---------------------------------------------------------------------------

TEST(RecBeamSearchResultTest, DefaultIsNotReady) {
  const RecBeamSearchResult result;
  EXPECT_FALSE(result.ready());
  EXPECT_EQ(result.beam_width(), 0);
  EXPECT_EQ(result.total_rounds(), 0);
  EXPECT_TRUE(result.beams().empty());
  EXPECT_TRUE(result.last_logprobs().empty());
}

TEST(RecBeamSearchResultTest, ReadyRequiresWidthRoundsAndBeams) {
  EXPECT_FALSE(RecBeamSearchResult(/*beam_width=*/2,
                                   /*total_rounds=*/3,
                                   /*beams=*/{},
                                   /*last_logprobs=*/{})
                   .ready());
  EXPECT_FALSE(RecBeamSearchResult(/*beam_width=*/0,
                                   /*total_rounds=*/3,
                                   /*beams=*/{{1, 2, 3}},
                                   /*last_logprobs=*/{})
                   .ready());
  EXPECT_FALSE(RecBeamSearchResult(/*beam_width=*/1,
                                   /*total_rounds=*/0,
                                   /*beams=*/{{1, 2, 3}},
                                   /*last_logprobs=*/{})
                   .ready());

  const RecBeamSearchResult result(/*beam_width=*/2,
                                   /*total_rounds=*/1,
                                   /*beams=*/{{1, 2, 3}, {4, 5, 6}},
                                   /*last_logprobs=*/{-0.1f, -0.2f});
  EXPECT_TRUE(result.ready());
  EXPECT_EQ(result.beam_width(), 2);
  EXPECT_EQ(result.total_rounds(), 1);
  EXPECT_EQ(result.beams().size(), 2u);
  EXPECT_EQ(result.last_logprobs(), (std::vector<float>{-0.1f, -0.2f}));
}

TEST(RecSequenceTest, ForkStartsWithEmptyBeamResult) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kLlmRec, {1, 2, 3});
  RecSequence& rec_sequence = RecSequence::from(*sequence);
  rec_sequence.set_beam_search_result(
      RecBeamSearchResult(/*beam_width=*/2,
                          /*total_rounds=*/1,
                          /*beams=*/{{1, 2, 3}, {4, 5, 6}},
                          /*last_logprobs=*/{-0.1f, -0.2f}));
  ASSERT_TRUE(rec_sequence.beam_search_result().ready());

  std::unique_ptr<Sequence> fork = sequence->fork(/*index=*/1);
  EXPECT_NE(dynamic_cast<LlmRecSequence*>(fork.get()), nullptr);
  EXPECT_EQ(fork->index(), 1u);
  EXPECT_FALSE(RecSequence::from(*fork).beam_search_result().ready());
  // The source keeps its result.
  EXPECT_TRUE(rec_sequence.beam_search_result().ready());
}

TEST(SequenceTest, PlainForkStaysPlain) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kNone, {1, 2, 3});
  std::unique_ptr<Sequence> fork = sequence->fork(/*index=*/2);
  EXPECT_EQ(dynamic_cast<RecSequence*>(fork.get()), nullptr);
  EXPECT_EQ(fork->index(), 2u);
  EXPECT_EQ(fork->num_prompt_tokens(), 3u);
}

// ---------------------------------------------------------------------------
// OneRec layout
// ---------------------------------------------------------------------------

TEST(OneRecSequenceTest, PromptTokensFeedEncoderAndBosSeedsDecoder) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13, 14});
  const OneRecSequence& onerec = OneRecSequence::from(*sequence);

  EXPECT_EQ(onerec.encoder_tokens(), (std::vector<int32_t>{11, 12, 13, 14}));
  EXPECT_EQ(onerec.encoder_seq_len(), 4u);
  EXPECT_EQ(onerec.num_decoder_embeddings(), 0u);

  // The decoder holds a single BOS token, not the prompt.
  EXPECT_EQ(sequence->num_prompt_tokens(), 1u);
  EXPECT_EQ(sequence->num_tokens(), 1u);
  EXPECT_EQ(sequence->tokens()[0], kBosTokenId);
  EXPECT_EQ(sequence->tokens().size(), 1u);
}

TEST(OneRecSequenceTest, SparseEmbeddingFeedsEncoder) {
  SequenceFixture fixture;
  auto sequence = fixture.create(
      RecType::kOneRec,
      /*prompt=*/{},
      embedding_mm_data(OneRecSequence::kEncoderSparseEmbeddingName, 6));
  const OneRecSequence& onerec = OneRecSequence::from(*sequence);

  EXPECT_TRUE(onerec.encoder_tokens().empty());
  EXPECT_EQ(onerec.encoder_seq_len(), 6u);
  EXPECT_EQ(sequence->num_prompt_tokens(), 1u);
  EXPECT_EQ(sequence->tokens()[0], kBosTokenId);
}

TEST(OneRecSequenceTest, DecoderContextEmbeddingReplacesBos) {
  SequenceFixture fixture;
  MMData mm_data =
      embedding_mm_data(OneRecSequence::kEncoderSparseEmbeddingName, 6);
  mm_data.add(MMType::EMBEDDING,
              OneRecSequence::kDecoderContextEmbeddingName,
              torch::zeros({5, 4}));
  auto sequence = fixture.create(RecType::kOneRec, /*prompt=*/{}, mm_data);
  const OneRecSequence& onerec = OneRecSequence::from(*sequence);

  EXPECT_EQ(onerec.num_decoder_embeddings(), 5u);
  EXPECT_EQ(sequence->num_prompt_tokens(), 0u);
  EXPECT_EQ(sequence->num_tokens(), 0u);
}

TEST(OneRecSequenceDeathTest, EmptyPromptRequiresSparseEmbedding) {
  SequenceFixture fixture;
  EXPECT_DEATH(fixture.create(RecType::kOneRec, /*prompt=*/{}),
               "encoder sparse embedding not found");
}

TEST(OneRecSequenceTest, ForkKeepsLayoutAndType) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  RecSequence::from(*sequence).set_beam_search_result(
      RecBeamSearchResult(/*beam_width=*/1,
                          /*total_rounds=*/1,
                          /*beams=*/{{1, 2, 3}},
                          /*last_logprobs=*/{}));

  std::unique_ptr<Sequence> fork = sequence->fork(/*index=*/1);
  const auto* onerec_fork = dynamic_cast<const OneRecSequence*>(fork.get());
  ASSERT_NE(onerec_fork, nullptr);
  EXPECT_EQ(onerec_fork->encoder_tokens(), (std::vector<int32_t>{11, 12, 13}));
  EXPECT_EQ(onerec_fork->encoder_seq_len(), 3u);
  EXPECT_FALSE(onerec_fork->beam_search_result().ready());
}

// ---------------------------------------------------------------------------
// OneRec behavior hooks
// ---------------------------------------------------------------------------

TEST(OneRecSequenceTest, AppendsBeforePrefillAndNeedsTokenToFinish) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});

  // No KV cache filled yet and no generated token: not finished, and appending
  // is allowed (an LLM sequence would CHECK-fail here).
  EXPECT_FALSE(sequence->finished());
  append_generated(*sequence, {21});
  EXPECT_EQ(sequence->num_generated_tokens(), 1u);
}

TEST(OneRecSequenceTest, StreamingOutputCarriesGeneratedIdsOnly) {
  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22});

  FakeItemTokenizer tokenizer;
  auto output =
      sequence->generate_streaming_output(sequence->num_tokens(), tokenizer);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->index, 0u);
  EXPECT_EQ(output->token_ids, (std::vector<int32_t>{21, 22}));
  EXPECT_TRUE(output->text.empty());
}

TEST(OneRecSequenceTest, OutputSkipsTrailingPlaceholderTokens) {
  SequenceFixture fixture;
  fixture.params().enable_schedule_overlap = true;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22});
  sequence->append_token(Token(-1));

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  EXPECT_EQ(output.token_ids, (std::vector<int32_t>{21, 22}));
}

TEST(OneRecSequenceTest, SkuLogprobsAreEmittedWhenEnabled) {
  ScopedRecConfig scoped;
  RecConfig::get_instance().enable_output_sku_logprobs(true);

  SequenceFixture fixture;
  fixture.sampling_param().logprobs = true;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22, 23}, /*logprob=*/-0.25f);

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  ASSERT_EQ(output.token_ids_logprobs.size(), 3u);
  for (const auto& logprob : output.token_ids_logprobs) {
    ASSERT_TRUE(logprob.has_value());
    EXPECT_FLOAT_EQ(logprob.value(), -0.25f);
  }
}

TEST(OneRecSequenceTest, NoSkuLogprobsWhenDisabled) {
  ScopedRecConfig scoped;
  RecConfig::get_instance().enable_output_sku_logprobs(false);

  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22, 23});

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  EXPECT_TRUE(output.token_ids_logprobs.empty());
}

TEST(OneRecSequenceTest, ConvertsCompleteTripleToDedupedItems) {
  ScopedRecConfig scoped;
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(true)
      .enable_extended_item_info(false)
      .each_conversion_threshold(50);

  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22, 23});

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  EXPECT_EQ(output.item_ids_list, (std::vector<int64_t>{1001, 1002}));
  ASSERT_TRUE(output.item_ids.has_value());
  EXPECT_EQ(output.item_ids.value(), 1001);
}

TEST(OneRecSequenceTest, DoesNotConvertIncompleteTriple) {
  ScopedRecConfig scoped;
  RecConfig::get_instance().enable_convert_tokens_to_item(true);

  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22});

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  EXPECT_TRUE(output.item_ids_list.empty());
  EXPECT_FALSE(output.item_ids.has_value());
}

TEST(OneRecSequenceTest, CapsConvertedItemsAtThreshold) {
  ScopedRecConfig scoped;
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(true)
      .enable_extended_item_info(false)
      .each_conversion_threshold(1);

  SequenceFixture fixture;
  auto sequence = fixture.create(RecType::kOneRec, {11, 12, 13});
  append_generated(*sequence, {21, 22, 23});

  FakeItemTokenizer tokenizer;
  SequenceOutput output = sequence->generate_output(tokenizer);
  EXPECT_EQ(output.item_ids_list.size(), 1u);
  ASSERT_TRUE(output.item_ids.has_value());
  EXPECT_EQ(output.item_ids.value(), output.item_ids_list.front());
}

}  // namespace
}  // namespace xllm
