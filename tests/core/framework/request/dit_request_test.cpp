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

namespace xllm {
namespace {

DiTRequest create_request(DiTRequestKind request_kind,
                          DiTGenerationParams generation_params = {}) {
  DiTInputParams input_params;
  DiTOutputFunc output_func;
  DiTOutputsFunc outputs_func;
  DiTRequestState state(
      input_params, generation_params, output_func, outputs_func, request_kind);
  return DiTRequest("request", "x-request", "time", state);
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
