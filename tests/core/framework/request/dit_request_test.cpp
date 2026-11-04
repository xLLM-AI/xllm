/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/dit_request.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

DiTRequest create_request(DiTRequestKind request_kind,
                          DiTGenerationParams generation_params = {}) {
  DiTInputParams input_params;
  DiTOutputFunc output_func;
  DiTRequestState state(
      input_params, generation_params, output_func, request_kind);
  return DiTRequest("request", "x-request", "time", state);
}

DiTEncodedMedia encoded_video() {
  return {.data = std::string("A\0B", 3),
          .mime_type = "video/mp4",
          .container = "mp4",
          .width = 2,
          .height = 2,
          .num_frames = 1,
          .fps = 25.0,
          .audio_sample_rate = 32000,
          .audio_channels = 2};
}

TEST(DiTRequestTest, EncodedVideoPreservesOwnedBytesAndActualMetadata) {
  DiTGenerationParams params;
  params.width = 512;
  params.height = 512;
  params.video_fps = 8.0;
  params.seed = 17;
  params.seed_is_set = true;
  DiTRequest request = create_request(DiTRequestKind::kVideo, params);
  {
    std::vector<DiTEncodedMedia> media = {encoded_video()};
    request.handle_forward_encoded_output(std::move(media));
  }
  // A\0B is not an MP4: success here proves no second encoder is invoked.
  const DiTRequestOutput output = request.generate_output();
  ASSERT_EQ(output.outputs.size(), 1u);
  const DiTGenerationOutput& item = output.outputs.front();
  EXPECT_EQ(item.video, std::string("A\0B", 3));
  EXPECT_EQ(item.width, 2);
  EXPECT_EQ(item.height, 2);
  EXPECT_EQ(item.num_frames, 1);
  EXPECT_DOUBLE_EQ(item.video_fps, 25.0);
  EXPECT_EQ(item.mime_type, "video/mp4");
  EXPECT_EQ(item.container, "mp4");
  EXPECT_EQ(item.audio_sample_rate, 32000);
  EXPECT_EQ(item.audio_channels, 2);
  EXPECT_EQ(item.seed, 17);
  EXPECT_TRUE(item.seed_is_set);
}

TEST(DiTRequestTest, NewEncodedOutputReplacesPreviousPayload) {
  DiTRequest request = create_request(DiTRequestKind::kVideo);
  request.handle_forward_output(torch::zeros({1, 1, 3, 2, 2}));
  request.handle_forward_encoded_output({encoded_video()});
  EXPECT_EQ(request.generate_output().outputs.front().video,
            std::string("A\0B", 3));
  DiTEncodedMedia next = encoded_video();
  next.data = "next";
  request.handle_forward_encoded_output({next});
  EXPECT_EQ(request.generate_output().outputs.front().video, "next");
}

TEST(DiTRequestTest, GeneratesImageOutputByRequestKind) {
  DiTRequest request = create_request(DiTRequestKind::kImage);
  request.handle_forward_output(torch::zeros({1, 3, 2, 2}));

  DiTRequestOutput output = request.generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_FALSE(output.outputs[0].image.empty());
  EXPECT_TRUE(output.outputs[0].video.empty());
  EXPECT_TRUE(output.outputs[0].audio.empty());
}

TEST(DiTRequestTest, GeneratesVideoOutputByRequestKind) {
  DiTGenerationParams params;
  params.num_videos_per_prompt = 1;
  params.video_fps = 8.0;
  DiTRequest request = create_request(DiTRequestKind::kVideo, params);
  request.handle_forward_output(torch::zeros({1, 1, 3, 2, 2}));

  DiTRequestOutput output = request.generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_FALSE(output.outputs[0].video.empty());
  EXPECT_TRUE(output.outputs[0].image.empty());
  EXPECT_TRUE(output.outputs[0].audio.empty());
  EXPECT_EQ(output.outputs[0].num_frames, 1);
  EXPECT_DOUBLE_EQ(output.outputs[0].video_fps, 8.0);
}

TEST(DiTRequestTest, GeneratesAudioOutputByRequestKind) {
  DiTGenerationParams params;
  params.audio_sampling_rate = 24000;
  DiTRequest request = create_request(DiTRequestKind::kAudio, params);
  request.handle_forward_output(torch::zeros({1, 32}));

  DiTRequestOutput output = request.generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  ASSERT_GE(output.outputs[0].audio.size(), 12u);
  EXPECT_EQ(output.outputs[0].audio.substr(0, 4), "RIFF");
  EXPECT_EQ(output.outputs[0].audio.substr(8, 4), "WAVE");
  EXPECT_TRUE(output.outputs[0].image.empty());
  EXPECT_TRUE(output.outputs[0].video.empty());
}

TEST(DiTRequestTest, GeneratesTextOutputByRequestKind) {
  DiTRequest request = create_request(DiTRequestKind::kText);
  request.handle_forward_text_output("generated text");

  DiTRequestOutput output = request.generate_output();

  ASSERT_EQ(output.outputs.size(), 1u);
  EXPECT_EQ(output.outputs[0].text, "generated text");
  EXPECT_TRUE(output.outputs[0].image.empty());
  EXPECT_TRUE(output.outputs[0].video.empty());
  EXPECT_TRUE(output.outputs[0].audio.empty());
}

}  // namespace
}  // namespace xllm
