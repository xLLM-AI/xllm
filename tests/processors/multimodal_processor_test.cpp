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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/hash_util.h"
#include "processors/cacheable_multimodal_processor.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/qwen3_vl_prompt_processor.h"
#include "processors/text_prompt_processor.h"

namespace xllm {
namespace {

class FixedTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/) const override {
    *ids = {17, 18};
    return true;
  }
  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    return token == "<pad>" ? std::optional<int32_t>(0) : std::nullopt;
  }
};

TEST(MultimodalProcessorTest, PaddingIsSelectedPerProcessor) {
  ModelArgs args;
  args.vision_start_token_id(10);
  args.vision_end_token_id(11);
  args.image_token_id(12);
  args.video_token_id(13);
  TokenizerArgs tokenizer_args;
  tokenizer_args.pad_token("<pad>");
  auto tokenizer = std::make_shared<FixedTokenizer>();
  MultimodalProcessor<Qwen3VLPromptProcessor> unpadded(
      args, tokenizer, tokenizer_args);
  MultimodalProcessor<Qwen3VLPromptProcessor> padded(
      args, tokenizer, tokenizer_args, 4);
  MultimodalProcessor<Qwen3VLPromptProcessor> shorter(
      args, tokenizer, tokenizer_args, 1);
  std::string prompt = "hello";
  MMData data(MMType::NONE, MMItemVec{});
  std::vector<int32_t> ids;
  ASSERT_TRUE(padded.process_prompt(prompt, data, ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{0, 0, 17, 18}));
  ASSERT_TRUE(unpadded.process_prompt(prompt, data, ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{17, 18}));
  ASSERT_TRUE(shorter.process_prompt(prompt, data, ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{17, 18}));
}

TEST(MultimodalProcessorTest, TextModelUsesTokenizerWithoutVisualSpans) {
  MultimodalProcessor<TextPromptProcessor> processor(
      ModelArgs{}, std::make_shared<FixedTokenizer>(), TokenizerArgs{});
  std::string prompt = "hello";
  MMData data(MMType::NONE, MMItemVec{});
  std::vector<int32_t> ids;
  ASSERT_TRUE(processor.process_prompt(prompt, data, ids));
  EXPECT_EQ(prompt, "hello");
  EXPECT_EQ(ids, (std::vector<int32_t>{17, 18}));

  MMData image_data(MMType::IMAGE, MMItemVec{MMDataItem(MMType::IMAGE)});
  EXPECT_FALSE(processor.process_prompt(prompt, image_data, ids));
}

TEST(MultimodalProcessorTest, SuppliedTextTokensAreNotEncodedAgain) {
  MultimodalProcessor<TextPromptProcessor> processor(
      ModelArgs{}, std::make_shared<FixedTokenizer>(), TokenizerArgs{});
  MMData data;
  std::vector<int32_t> tokens{29, 30};
  ASSERT_TRUE(processor.process_prompt_tokens(data, tokens));
  EXPECT_EQ(tokens, (std::vector<int32_t>{29, 30}));
}

TEST(MultimodalProcessorTest, SuppliedVisualTokensPopulateMatchingSpans) {
  ModelArgs args;
  args.vision_start_token_id(10)
      .vision_end_token_id(11)
      .image_token_id(12)
      .video_token_id(13)
      .mm_image_merge_size(1);
  MultimodalProcessor<Qwen3VLPromptProcessor> processor(
      args, std::make_shared<FixedTokenizer>(), TokenizerArgs{});
  MMData data(MMType::IMAGE,
              MMItemVec{MMDataItem(
                  MMType::IMAGE,
                  MMDict{{"image_grid_thw", torch::tensor({{1, 2, 2}})}})});
  std::vector<int32_t> tokens{42, 10, 12, 12, 12, 12, 11, 43};
  ASSERT_TRUE(processor.process_prompt_tokens(data, tokens));
  const auto& state = data.items<MMItemVec>()[0].state();
  EXPECT_EQ(state.token_pos().offset, 2);
  EXPECT_EQ(state.token_pos().length, 4);
  EXPECT_EQ(state.mm_token_num(), 4);
}

TEST(MultimodalProcessorTest,
     SuppliedVisualTokensRejectMissingMarkersAndWrongGrid) {
  ModelArgs args;
  args.vision_start_token_id(10)
      .vision_end_token_id(11)
      .image_token_id(12)
      .video_token_id(13)
      .mm_image_merge_size(1);
  MultimodalProcessor<Qwen3VLPromptProcessor> processor(
      args, std::make_shared<FixedTokenizer>(), TokenizerArgs{});
  const std::vector<std::vector<int32_t>> invalid_inputs{
      {10, 12, 12},
      {10, 12, 11},
      {12, 10, 12, 12, 12, 12, 11},
      {10, 13, 13, 13, 13, 11}};
  for (std::vector<int32_t> tokens : invalid_inputs) {
    MMData data(MMType::IMAGE,
                MMItemVec{MMDataItem(
                    MMType::IMAGE,
                    MMDict{{"image_grid_thw", torch::tensor({{1, 2, 2}})}})});
    EXPECT_FALSE(processor.process_prompt_tokens(data, tokens));
  }
}

class SpecialTokenTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* tokens,
              bool add_special_tokens) const override {
    *tokens = add_special_tokens ? std::vector<int32_t>{1, 2, 3}
                                 : std::vector<int32_t>{2, 3};
    return true;
  }
};

TEST(MultimodalProcessorTest, PromptEncodingHonorsSpecialTokenOption) {
  MultimodalProcessor<TextPromptProcessor> processor(
      ModelArgs{}, std::make_shared<SpecialTokenTokenizer>(), TokenizerArgs{});
  std::string prompt = "hello";
  MMData data;
  std::vector<int32_t> tokens;
  ASSERT_TRUE(processor.process_prompt(prompt,
                                       data,
                                       tokens,
                                       /*add_special_tokens=*/false));
  EXPECT_EQ(tokens, (std::vector<int32_t>{2, 3}));
  ASSERT_TRUE(processor.process_prompt(prompt,
                                       data,
                                       tokens,
                                       /*add_special_tokens=*/true));
  EXPECT_EQ(tokens, (std::vector<int32_t>{1, 2, 3}));
}

class CountingImageProcessor final : public ImageProcessor {
 public:
  explicit CountingImageProcessor(const ModelArgs&) {}
  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override {
    processed_images += images.size();
    output_items.reserve(images.size());
    for (const torch::Tensor& image : images) {
      output_items.emplace_back(MMType::IMAGE, MMDict{{"pixel_values", image}});
    }
    return true;
  }
  static inline size_t processed_images = 0;
};

TEST(MultimodalProcessorTest, CacheRestoresMixedHitsAndMissesInInputOrder) {
  using Pipeline =
      MultimodalProcessor<Qwen3VLPromptProcessor, CountingImageProcessor>;
  CountingImageProcessor::processed_images = 0;
  CacheableMultimodalProcessor processor(
      std::make_unique<Pipeline>(
          ModelArgs{}, std::make_shared<FixedTokenizer>(), TokenizerArgs{}),
      4);
  MMInputItem first;
  first.type = MMType::IMAGE;
  first.hash_key = hash_string(/*value=*/"first", /*seed=*/0);
  first.decode_image = torch::ones({3, 2, 2});
  MMInputItem second;
  second.type = MMType::IMAGE;
  second.hash_key = hash_string(/*value=*/"second", /*seed=*/0);
  second.decode_image = torch::zeros({3, 2, 2});
  MMData output;
  ASSERT_TRUE(processor.process_multimodal(MMInput({first}), output));
  EXPECT_EQ(CountingImageProcessor::processed_images, 1u);
  ASSERT_TRUE(processor.process_multimodal(MMInput({second, first}), output));
  EXPECT_EQ(CountingImageProcessor::processed_images, 2u);
  const MMItemVec& items = output.items<MMItemVec>();
  ASSERT_EQ(items.size(), 2u);
  EXPECT_EQ(items[0].state().schedule_data().key, second.hash_key.value());
  EXPECT_EQ(items[1].state().schedule_data().key, first.hash_key.value());
  EXPECT_TRUE(
      torch::equal(std::get<torch::Tensor>(items[0].data().at("pixel_values")),
                   second.decode_image));
  EXPECT_TRUE(
      torch::equal(std::get<torch::Tensor>(items[1].data().at("pixel_values")),
                   first.decode_image));
  ASSERT_TRUE(processor.process_multimodal(MMInput({first, second}), output));
  EXPECT_EQ(CountingImageProcessor::processed_images, 2u);
}

}  // namespace
}  // namespace xllm
