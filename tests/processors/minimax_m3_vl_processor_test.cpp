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

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "processors/minimax_m3_vl_image_processor.h"
#include "processors/minimax_m3_vl_prompt_processor.h"

namespace xllm {
namespace {

ModelArgs make_processor_args() {
  ModelArgs args;
  args.mm_image_normalize_mean({0.0, 0.0, 0.0});
  args.mm_image_normalize_std({1.0, 1.0, 1.0});
  args.mm_image_max_pixels(1000000);
  args.mm_image_min_pixels(1);
  args.mm_image_merge_size(2);
  args.mm_image_patch_size(14);
  args.mm_image_temporal_patch_size(2);
  return args;
}

MMData make_mm_data(const torch::Tensor& image_grid_thw) {
  MMData mm_data;
  mm_data.add(MMType::IMAGE, "image_grid_thw", image_grid_thw);
  return mm_data;
}

}  // namespace

TEST(MiniMaxM3VLProcessorTest, ImageResizeUsesPythonTieEvenRounding) {
  MiniMaxM3VLImageProcessor processor(make_processor_args());
  const torch::Tensor image =
      torch::ones({3, 70, 70}, torch::TensorOptions().dtype(torch::kFloat32));

  std::vector<MMDataItem> output_items;
  ASSERT_TRUE(processor.process({image}, output_items));
  ASSERT_EQ(output_items.size(), 1u);

  std::optional<torch::Tensor> image_grid_thw =
      output_items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(image_grid_thw.has_value());
  const torch::Tensor expected =
      torch::tensor({1, 4, 4}, torch::TensorOptions().dtype(torch::kLong));
  EXPECT_TRUE(torch::equal(image_grid_thw->reshape({3}), expected));
}

TEST(MiniMaxM3VLProcessorTest, PromptProcessorExpandsImagePlaceholder) {
  MiniMaxM3VLPromptProcessor processor(make_processor_args());
  MMData mm_data = make_mm_data(
      torch::tensor({{1, 4, 4}}, torch::TensorOptions().dtype(torch::kLong)));
  std::string prompt = "prefix ]<]image[>[ suffix";

  processor.process(prompt, mm_data);

  const std::string expected =
      "prefix ]<]start of "
      "image[>[]<]image[>[]<]image[>[]<]image[>[]<]image[>[]<]"
      "end of image[>[ suffix";
  EXPECT_EQ(prompt, expected);
}

TEST(MiniMaxM3VLProcessorTest, PromptProcessorRejectsMissingImagePlaceholder) {
  MiniMaxM3VLPromptProcessor processor(make_processor_args());
  MMData mm_data = make_mm_data(torch::tensor(
      {{1, 4, 4}, {1, 4, 4}}, torch::TensorOptions().dtype(torch::kLong)));
  std::string prompt = "prefix ]<]image[>[ suffix";

  EXPECT_DEATH(processor.process(prompt, mm_data),
               "Fewer MiniMax-M3 image tokens than processed images");
}

}  // namespace xllm
