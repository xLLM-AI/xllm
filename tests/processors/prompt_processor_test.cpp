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

#include <string>
#include <vector>

#include "processors/clip_prompt_processor.h"
#include "processors/glm4v_prompt_processor.h"
#include "processors/kimi25_image_processor.h"
#include "processors/qwen2_vl_prompt_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"

namespace xllm {
namespace {

TEST(PromptProcessorTest, ClipExpandsMixedModalitiesInOrder) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  CLIPVLPromptProcessor processor(args);
  MMData data(MMType::IMAGE | MMType::VIDEO,
              MMDict{{"image_grid_thw", torch::tensor({{1, 1, 2}, {1, 1, 1}})},
                     {"video_grid_thw", torch::tensor({{2, 1, 1}})}});
  std::string prompt = "a<|image_pad|>b<|video_pad|>c<|image_pad|>d";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(
      prompt,
      "a<|image_pad|><|image_pad|>b<|video_pad|><|video_pad|>c<|image_pad|>d");
}

TEST(PromptProcessorTest, KimiExpandsMixedModalitiesWithoutChangingWrappers) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  KimiK25PromptProcessor processor(args);
  MMData data(MMType::IMAGE | MMType::VIDEO,
              MMDict{{"image_grid_thw", torch::tensor({{1, 1, 2}, {1, 1, 1}})},
                     {"video_grid_thw", torch::tensor({{2, 1, 1}})}});
  std::string prompt =
      "a<|media_begin|>image<|media_content|><|media_pad|><|media_end|>b"
      "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>c"
      "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>d";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "a<|media_begin|>image<|media_content|><|media_pad|><|media_pad|><|"
            "media_end|>b"
            "<|media_begin|>video<|media_content|><|media_pad|><|media_pad|><|"
            "media_end|>c"
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>d");
}

TEST(PromptProcessorTest, ClipAndKimiLeaveTextOnlyPromptsUnchanged) {
  ModelArgs args;
  CLIPVLPromptProcessor clip(args);
  KimiK25PromptProcessor kimi(args);
  MMData data;
  std::string prompt = "plain text";
  ASSERT_TRUE(clip.process(prompt, data));
  EXPECT_EQ(prompt, "plain text");
  ASSERT_TRUE(kimi.process(prompt, data));
  EXPECT_EQ(prompt, "plain text");
}

TEST(PromptProcessorTest, QwenPreservesMixedImageAndVideoExpansionOrder) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  Qwen2VLPromptProcessor processor(args);
  MMData data(MMType::IMAGE | MMType::VIDEO,
              MMDict{{"image_grid_thw", torch::tensor({{1, 1, 2}, {1, 1, 1}})},
                     {"video_grid_thw", torch::tensor({{2, 1, 1}})}});
  std::string prompt = "a<|image_pad|>b<|video_pad|>c<|image_pad|>d";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(
      prompt,
      "a<|image_pad|><|image_pad|>b<|video_pad|><|video_pad|>c<|image_pad|>d");
}

TEST(PromptProcessorTest, QwenTimestampsTruncateUnusedGroups) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  args.mm_temporal_patch_size(3);
  Qwen3VLPromptProcessor processor(args);
  VideoMetadata metadata;
  metadata.timestamps = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  MMData data(MMType::VIDEO,
              MMItemVec{MMDataItem(
                  MMType::VIDEO,
                  MMDict{{"video_grid_thw", torch::tensor({{2, 1, 1}})}},
                  metadata)});
  std::string prompt = "<|video_pad|>";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "<1.0 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<4.0 seconds><|vision_start|><|video_pad|><|vision_end|>");
}

TEST(PromptProcessorTest, QwenVideoTimestampsPadGroupsAndRepeatLastFrame) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  args.mm_temporal_patch_size(2);
  Qwen3VLPromptProcessor processor(args);
  VideoMetadata metadata;
  metadata.timestamps = {0.0, 2.0, 4.0};
  MMData data(MMType::VIDEO,
              MMItemVec{MMDataItem(
                  MMType::VIDEO,
                  MMDict{{"video_grid_thw", torch::tensor({{4, 1, 1}})}},
                  metadata)});
  std::string prompt = "before<|vision_start|><|video_pad|><|vision_end|>after";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "before<1.0 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<4.0 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<4.0 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<4.0 seconds><|vision_start|><|video_pad|><|vision_end|>after");
}

TEST(PromptProcessorTest, QwenVideoMaskIncludesGapsAndCountsOnlyVideoTokens) {
  ModelArgs args;
  args.vision_start_token_id(10);
  args.vision_end_token_id(11);
  args.image_token_id(12);
  args.video_token_id(13);
  Qwen3VLPromptProcessor processor(args);
  MMData data(MMType::VIDEO,
              MMItemVec{MMDataItem(
                  MMType::VIDEO,
                  MMDict{{"video_grid_thw", torch::tensor({{2, 1, 2}})}})});
  ASSERT_TRUE(
      processor.find_mm_spans({10, 13, 13, 11, 99, 10, 13, 7, 11}, data));
  const auto& state = data.items<MMItemVec>()[0].state();
  EXPECT_EQ(state.token_pos().offset, 1);
  EXPECT_EQ(state.token_pos().length, 7);
  EXPECT_EQ(state.mm_token_num(), 3);
  EXPECT_TRUE(
      torch::equal(state.mm_token_mask(),
                   torch::tensor({true, true, false, false, false, true, false},
                                 torch::kBool)));
}

TEST(PromptProcessorTest, GlmPreservesMixedExpansionAndVideoTimestamps) {
  ModelArgs args;
  args.mm_image_merge_size(1);
  GLM4VPromptProcessor processor(args);
  VideoMetadata metadata;
  metadata.timestamps = {0.0, 1.0, 2.0};
  MMData data(
      MMType::IMAGE | MMType::VIDEO,
      MMItemVec{
          MMDataItem(MMType::IMAGE,
                     MMDict{{"image_grid_thw", torch::tensor({{1, 1, 2}})}}),
          MMDataItem(MMType::VIDEO,
                     MMDict{{"video_grid_thw", torch::tensor({{2, 1, 1}})}},
                     metadata)});
  std::string prompt = "a<|image|>b<|video|>c";
  ASSERT_TRUE(processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "a<|image|><|image|>b<|begin_of_image|><|image|><|end_of_image|>0."
            "0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 secondsc");
}

TEST(PromptProcessorTest, GlmVideoMaskCountResetsBetweenVideos) {
  ModelArgs args;
  args.video_start_token_id(10);
  args.video_end_token_id(11);
  args.image_token_id(12);
  GLM4VPromptProcessor processor(args);
  MMData data(MMType::VIDEO,
              MMItemVec{MMDataItem(MMType::VIDEO), MMDataItem(MMType::VIDEO)});
  ASSERT_TRUE(
      processor.find_mm_spans({10, 12, 7, 12, 11, 10, 7, 12, 11}, data));
  const auto& items = data.items<MMItemVec>();
  EXPECT_EQ(items[0].state().mm_token_num(), 2);
  EXPECT_EQ(items[1].state().mm_token_num(), 1);
  EXPECT_TRUE(torch::equal(items[0].state().mm_token_mask(),
                           torch::tensor({true, false, true}, torch::kBool)));
  EXPECT_TRUE(torch::equal(items[1].state().mm_token_mask(),
                           torch::tensor({false, true}, torch::kBool)));
}

}  // namespace
}  // namespace xllm
