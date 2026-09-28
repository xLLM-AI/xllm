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

#include <gtest/gtest.h>

#include "core/framework/request/dit_input_sources.h"

namespace xllm {
namespace {

TEST(DiTImageSourcesTest, PreservesDuplicateNamesAndOrder) {
  DiTImageSources sources;
  sources.add("", torch::tensor({1}));
  sources.add("unknown", torch::tensor({2}));
  sources.add("mask_image", torch::tensor({3}));

  ASSERT_EQ(sources.size(), 3u);
  EXPECT_EQ(sources.at(0).name, "unknown");
  EXPECT_EQ(sources.at(1).name, "unknown");
  std::vector<torch::Tensor> tensors = sources.get({"unknown", "unknown"});
  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].item<int64_t>(), 1);
}

TEST(DiTImageSourcesTest, GetFallsBackByIndexForRequestedCount) {
  DiTImageSources sources;
  sources.add("last_image", torch::tensor({2}));
  sources.add("image", torch::tensor({1}));

  std::vector<torch::Tensor> named = sources.get({"image", "last_image"});
  ASSERT_EQ(named.size(), 2u);
  EXPECT_EQ(named[0].item<int64_t>(), 1);
  EXPECT_EQ(named[1].item<int64_t>(), 2);

  sources.add("extra", torch::tensor({3}));
  std::vector<torch::Tensor> fallback = sources.get({"image", "mask_image"});
  ASSERT_EQ(fallback.size(), 2u);
  EXPECT_EQ(fallback[0].item<int64_t>(), 2);
  EXPECT_EQ(fallback[1].item<int64_t>(), 1);
}

TEST(DiTImageSourcesTest, BatchSignatureChecksNameShapeAndDtypeByPosition) {
  DiTImageSources reference;
  reference.add("unknown", torch::zeros({3, 4, 4}, torch::kUInt8));
  reference.add("mask_image", torch::zeros({1, 4, 4}, torch::kUInt8));

  DiTImageSources same;
  same.add("unknown", torch::ones({3, 4, 4}, torch::kUInt8));
  same.add("mask_image", torch::ones({1, 4, 4}, torch::kUInt8));
  EXPECT_TRUE(reference.batch_signature_matches(same));

  DiTImageSources wrong_name;
  wrong_name.add("image", torch::ones({3, 4, 4}, torch::kUInt8));
  wrong_name.add("mask_image", torch::ones({1, 4, 4}, torch::kUInt8));
  EXPECT_FALSE(reference.batch_signature_matches(wrong_name));

  DiTImageSources wrong_shape;
  wrong_shape.add("unknown", torch::ones({3, 8, 8}, torch::kUInt8));
  wrong_shape.add("mask_image", torch::ones({1, 4, 4}, torch::kUInt8));
  EXPECT_FALSE(reference.batch_signature_matches(wrong_shape));
}

}  // namespace
}  // namespace xllm
