/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>
#include <vector>

#include "layers/npu/deepseek_v4_rotary_embedding.h"

namespace xllm {
namespace layer {
namespace {

DeepseekV4RotaryEmbedding create_test_rotary_embedding() {
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return DeepseekV4RotaryEmbedding(/*rotary_dim=*/8,
                                   /*max_position_embeddings=*/128,
                                   /*interleaved=*/false,
                                   /*rope_theta=*/10000.0f,
                                   /*compress_rope_theta=*/40000.0f,
                                   /*scaling_factor=*/4.0f,
                                   /*extrapolation_factor=*/1.0f,
                                   /*beta_fast=*/32,
                                   /*beta_slow=*/1,
                                   /*attn_factor=*/1.0f,
                                   /*mscale=*/1.0f,
                                   /*mscale_all_dim=*/1.0f,
                                   /*original_max_position_embeddings=*/128,
                                   options);
}

TEST(DeepseekV4RotaryEmbeddingTest, RegisteredGroupsAreExpected) {
  auto rope = create_test_rotary_embedding();
  auto groups = rope.registered_groups();

  ASSERT_EQ(groups.size(), 3);
  EXPECT_EQ(groups[0], "c128");
  EXPECT_EQ(groups[1], "c4");
  EXPECT_EQ(groups[2], "default");
}

TEST(DeepseekV4RotaryEmbeddingTest, BuildReturnsExpectedShapes) {
  auto rope = create_test_rotary_embedding();
  std::unordered_map<std::string, torch::Tensor> positions_map = {
      {"default", torch::tensor({0, 1, 2}, torch::kInt64)},
      {"c4", torch::tensor({0, 4}, torch::kInt64)},
      {"c128", torch::tensor({0}, torch::kInt64)}};

  auto result = rope.build(positions_map);

  ASSERT_EQ(result.size(), 3);
  ASSERT_TRUE(result.find("default") != result.end());
  ASSERT_TRUE(result.find("c4") != result.end());
  ASSERT_TRUE(result.find("c128") != result.end());

  EXPECT_EQ(result["default"].first.sizes(), torch::IntArrayRef({3, 8}));
  EXPECT_EQ(result["default"].second.sizes(), torch::IntArrayRef({3, 8}));
  EXPECT_EQ(result["c4"].first.sizes(), torch::IntArrayRef({2, 8}));
  EXPECT_EQ(result["c4"].second.sizes(), torch::IntArrayRef({2, 8}));
  EXPECT_EQ(result["c128"].first.sizes(), torch::IntArrayRef({1, 8}));
  EXPECT_EQ(result["c128"].second.sizes(), torch::IntArrayRef({1, 8}));
}

TEST(DeepseekV4RotaryEmbeddingTest, SelectLayerGroupsUsesRegistration) {
  auto rope = create_test_rotary_embedding();
  rope.register_layer("layers.0", {"default", "c4"});

  std::unordered_map<std::string, torch::Tensor> positions_map = {
      {"default", torch::tensor({0, 1}, torch::kInt64)},
      {"c4", torch::tensor({0, 1}, torch::kInt64)},
      {"c128", torch::tensor({0, 0}, torch::kInt64)}};

  auto group_cos_sin = rope.build(positions_map);
  auto selected = rope.select_layer_groups("layers.0", group_cos_sin);

  ASSERT_EQ(selected.size(), 2);
  EXPECT_TRUE(selected.find("default") != selected.end());
  EXPECT_TRUE(selected.find("c4") != selected.end());
  EXPECT_TRUE(selected.find("c128") == selected.end());
}

TEST(DeepseekV4RotaryEmbeddingTest, SelectLayerGroupsFallsBackToDefault) {
  auto rope = create_test_rotary_embedding();
  std::unordered_map<std::string, torch::Tensor> positions_map = {
      {"default", torch::tensor({0, 1}, torch::kInt64)},
      {"c4", torch::tensor({0, 1}, torch::kInt64)}};

  auto group_cos_sin = rope.build(positions_map);
  auto selected = rope.select_layer_groups("layers.unknown", group_cos_sin);

  ASSERT_EQ(selected.size(), 1);
  EXPECT_TRUE(selected.find("default") != selected.end());
}

}  // namespace
}  // namespace layer
}  // namespace xllm
