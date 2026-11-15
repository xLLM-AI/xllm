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

#include <utility>
#include <vector>

namespace xllm {
namespace {

MMData make_image(torch::Tensor pixels) {
  return MMData(MMType::IMAGE,
                MMItemVec{MMDataItem(MMType::IMAGE,
                                     {{"pixel_values", std::move(pixels)}})});
}

TEST(MMBatchDataTest, SingleImageSurvivesSourceReleaseAndProtoRoundTrip) {
  MMBatchData batch;
  {
    auto pixels = torch::arange(12, torch::kFloat32).reshape({3, 4});
    batch.batch({make_image(std::move(pixels))});
  }
  const auto pixels = batch.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(pixels.has_value());
  EXPECT_TRUE(torch::equal(*pixels,
                           torch::arange(12, torch::kFloat32).reshape({3, 4})));
  MMBatchData decoded;
  {
    proto::MMData wire;
    ASSERT_TRUE(mmdata_to_proto(batch, &wire));
    ASSERT_TRUE(proto_to_mmdata(wire, &decoded));
  }
  const auto restored = decoded.get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(restored.has_value());
  EXPECT_TRUE(torch::equal(*restored, *pixels));
  ASSERT_EQ(decoded.mm_data_vec().size(), 1);
  const auto item_pixels = decoded.mm_data_vec()
                               .front()
                               .items<MMItemVec>()
                               .front()
                               .get<torch::Tensor>("pixel_values");
  ASSERT_TRUE(item_pixels.has_value());
  EXPECT_TRUE(torch::equal(*item_pixels, *pixels));
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
