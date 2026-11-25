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

#include <future>
#include <utility>
#include <vector>

namespace xllm {
namespace {

MMData make_image(torch::Tensor pixels) {
  return MMData(MMType::IMAGE,
                MMItemVec{MMDataItem(MMType::IMAGE,
                                     {{"pixel_values", std::move(pixels)}})});
}

TEST(MMBatchDataTest, SingleImageSurvivesSourceRelease) {
  MMBatchData batch;
  {
    auto pixels = torch::arange(12, torch::kFloat32).reshape({3, 4});
    batch.batch({make_image(std::move(pixels))});
  }
  const auto pixels = batch.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(pixels.has_value());
  EXPECT_TRUE(torch::equal(*pixels,
                           torch::arange(12, torch::kFloat32).reshape({3, 4})));
}

TEST(MMBatchDataTest, SingleNoncontiguousImageKeepsContiguousBatchLayout) {
  const auto pixels =
      torch::arange(12, torch::kFloat32).reshape({3, 4}).transpose(0, 1);
  ASSERT_FALSE(pixels.is_contiguous());
  const MMBatchData batch({make_image(pixels)});
  const auto gathered = batch.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(gathered.has_value());
  EXPECT_TRUE(gathered->is_contiguous());
  EXPECT_TRUE(torch::equal(*gathered, pixels));
}

TEST(MMBatchDataTest, MultipleImagesKeepTheirInputOrder) {
  const auto first = torch::full({2, 3}, 1.0);
  const auto second = torch::full({1, 3}, 2.0);
  const MMBatchData batch({make_image(first), make_image(second)});
  const auto gathered = batch.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(gathered.has_value());
  EXPECT_TRUE(torch::equal(*gathered, torch::cat({first, second})));
}

TEST(MMBatchDataTest, MultipleImagesRemainAvailableAfterMoveAndSourceRelease) {
  MMBatchData moved;
  {
    const MMBatchData source({make_image(torch::full({2, 3}, 1.0)),
                              make_image(torch::full({1, 3}, 2.0))});
    // Reading the derived dictionary before moving must not discard the items.
    ASSERT_TRUE(source.get<torch::Tensor>("pixel_values").has_value());
    moved = MMBatchData::to(source, torch::Device(torch::kCPU));
  }
  const auto pixels = moved.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(pixels.has_value());
  EXPECT_TRUE(torch::equal(
      *pixels, torch::tensor({{1., 1., 1.}, {1., 1., 1.}, {2., 2., 2.}})));
  ASSERT_EQ(moved.mm_data_vec().size(), 2);
}

TEST(MMBatchDataTest, EncoderReplacementDiscardsPendingRawInputs) {
  MMBatchData batch(
      {make_image(torch::ones({2, 3})), make_image(torch::zeros({1, 3}))});
  const auto embeddings = torch::full({3, 4}, 2.0);
  batch.replace({{"image|embedding", embeddings}});
  batch.to(torch::Device(torch::kCPU));
  EXPECT_FALSE(batch.has("pixel_values"));
  const auto gathered = batch.get<torch::Tensor>("image|embedding");
  ASSERT_TRUE(gathered.has_value());
  EXPECT_TRUE(torch::equal(*gathered, embeddings));
}

TEST(MMBatchDataTest, BatchCopiesSupportConcurrentAggregation) {
  const MMBatchData batch(
      {make_image(torch::ones({2, 3})), make_image(torch::zeros({1, 3}))});
  auto read = [batch] { return batch.get<torch::Tensor>("pixel_values"); };
  auto first = std::async(std::launch::async, read);
  auto second = std::async(std::launch::async, read);
  const auto expected = torch::cat({torch::ones({2, 3}), torch::zeros({1, 3})});
  const auto first_pixels = first.get();
  const auto second_pixels = second.get();
  ASSERT_TRUE(first_pixels.has_value());
  ASSERT_TRUE(second_pixels.has_value());
  EXPECT_TRUE(torch::equal(*first_pixels, expected));
  EXPECT_TRUE(torch::equal(*second_pixels, expected));
}

TEST(MMBatchDataTest, ScalarAndIncompatibleInputsRemainTensorLists) {
  MMDataItem item(MMType::IMAGE,
                  {{"scalar", torch::tensor(3.0)},
                   {"crops",
                    std::vector<torch::Tensor>{torch::ones({2, 3}),
                                               torch::zeros({1, 4})}}});
  const MMBatchData batch({MMData(MMType::IMAGE, MMItemVec{std::move(item)})});
  const auto scalar = batch.get<std::vector<torch::Tensor>>("scalar");
  ASSERT_TRUE(scalar.has_value());
  ASSERT_EQ(scalar->size(), 1);
  EXPECT_FLOAT_EQ(scalar->front().item<float>(), 3.0);
  const auto crops = batch.get<std::vector<torch::Tensor>>("crops");
  ASSERT_TRUE(crops.has_value());
  ASSERT_EQ(crops->size(), 2);
  EXPECT_TRUE(torch::equal(crops->front(), torch::ones({2, 3})));
  EXPECT_TRUE(torch::equal(crops->back(), torch::zeros({1, 4})));
}

}  // namespace
}  // namespace xllm
