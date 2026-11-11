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

#include "core/framework/multimodal/mm_batch_data.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {
namespace {

TEST(MMBatchDataTest, HostGridPolicySurvivesPerItemRegather) {
  const torch::Tensor grid = torch::tensor({{1, 4, 4}}, torch::kInt32);
  const torch::Tensor pixels = torch::ones({16, 8});
  MMItemVec items;
  items.reserve(2);
  items.emplace_back(
      MMType::IMAGE,
      MMDict{{"image_grid_thw", grid}, {"pixel_values", pixels}});
  items.emplace_back(
      MMType::VIDEO,
      MMDict{{"video_grid_thw", grid}, {"pixel_values_videos", pixels}});
  std::vector<MMData> requests;
  requests.reserve(1);
  requests.emplace_back(MMType::IMAGE | MMType::VIDEO, std::move(items));
  const MMBatchData original(std::move(requests));

  auto moved = MMBatchData::to(original,
                               torch::Device(torch::kMeta),
                               /*keep_vision_grid_on_cpu=*/true);
  EncoderInputGatherVisitor gather;
  ASSERT_TRUE(moved.foreach (gather));
  ASSERT_TRUE(gather.finish(moved));

  for (const std::string key : {"image_grid_thw", "video_grid_thw"}) {
    const torch::Tensor actual = moved.get<torch::Tensor>(key).value();
    EXPECT_TRUE(actual.device().is_cpu());
    EXPECT_TRUE(torch::equal(actual, grid));
  }
  EXPECT_EQ(moved.get<torch::Tensor>("pixel_values")->device().type(),
            torch::kMeta);
  EXPECT_EQ(moved.get<torch::Tensor>("pixel_values_videos")->device().type(),
            torch::kMeta);
  EXPECT_TRUE(original.get<torch::Tensor>("pixel_values")->device().is_cpu());

  auto legacy = MMBatchData::to(original, torch::Device(torch::kMeta));
  EncoderInputGatherVisitor legacy_gather;
  ASSERT_TRUE(legacy.foreach (legacy_gather));
  ASSERT_TRUE(legacy_gather.finish(legacy));
  EXPECT_EQ(legacy.get<torch::Tensor>("image_grid_thw")->device().type(),
            torch::kMeta);
  EXPECT_EQ(legacy.get<torch::Tensor>("video_grid_thw")->device().type(),
            torch::kMeta);
}

TEST(MMBatchDataTest, HostGridPolicyPreservesDictionaryTensorVectors) {
  const torch::Tensor grid = torch::tensor({{2, 4, 6}}, torch::kInt32);
  const std::vector<torch::Tensor> grids{grid, grid.clone()};
  const MMDict data{{"video_grid_thw", grids},
                    {"pixel_values_videos",
                     std::vector<torch::Tensor>{torch::ones({48, 8})}}};
  MMBatchData direct(MMType::VIDEO, data);
  direct.to(torch::Device(torch::kMeta), /*keep_vision_grid_on_cpu=*/true);
  const auto direct_grids =
      direct.get<std::vector<torch::Tensor>>("video_grid_thw").value();
  for (const auto& actual : direct_grids) {
    EXPECT_TRUE(actual.device().is_cpu());
    EXPECT_TRUE(torch::equal(actual, grid));
  }
  const auto pixels =
      direct.get<std::vector<torch::Tensor>>("pixel_values_videos").value();
  ASSERT_EQ(pixels.size(), 1);
  EXPECT_EQ(pixels.front().device().type(), torch::kMeta);

  std::vector<MMData> requests;
  requests.reserve(1);
  requests.emplace_back(MMType::VIDEO, data);
  MMBatchData batch(std::move(requests));
  batch.to(torch::Device(torch::kMeta), /*keep_vision_grid_on_cpu=*/true);
  const auto& per_request = batch.mm_data_vec().front().items<MMDict>();
  const auto& actual_grids =
      std::get<std::vector<torch::Tensor>>(per_request.at("video_grid_thw"));
  EXPECT_TRUE(torch::equal(actual_grids.front(), grid));
  const auto& actual_pixels = std::get<std::vector<torch::Tensor>>(
      per_request.at("pixel_values_videos"));
  EXPECT_EQ(actual_pixels.front().device().type(), torch::kMeta);
}

}  // namespace
}  // namespace xllm
