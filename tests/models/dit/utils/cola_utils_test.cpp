/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <limits>
#include <vector>

#include "models/dit/utils/cola_block_causal_mask.h"
#include "models/dit/utils/cola_weight_loader.h"

namespace xllm {
namespace {

// ===========================================================================
// cola_checkpoint_key_to_cpp_key
// ===========================================================================

TEST(ColaCheckpointKeyTest, DotDigitBecomesUnderscore) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("layers.0.proj"), "layers_0.proj");
}

TEST(ColaCheckpointKeyTest, MultipleDotDigits) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("blocks.12.attn.qkv"),
            "blocks_12.attn.qkv");
}

TEST(ColaCheckpointKeyTest, DotBeforeNonDigitIsPreserved) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("encoder.weight"), "encoder.weight");
}

TEST(ColaCheckpointKeyTest, NoDotsUnchanged) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("weight"), "weight");
}

TEST(ColaCheckpointKeyTest, EmptyString) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key(""), "");
}

TEST(ColaCheckpointKeyTest, DotAtEndUnchanged) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("layers."), "layers.");
}

TEST(ColaCheckpointKeyTest, DotDigitAtStart) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key(".0.weight"), "_0.weight");
}

// ===========================================================================
// is_cola_ignored_checkpoint_key
// ===========================================================================

TEST(ColaIgnoredKeyTest, RopeFreqsSuffixIgnored) {
  EXPECT_TRUE(is_cola_ignored_checkpoint_key("model.rope.rope.freqs"));
}

TEST(ColaIgnoredKeyTest, BareRopeFreqsNotIgnoredWithoutLeadingDot) {
  // Suffix is ".rope.rope.freqs"; bare "rope.rope.freqs" lacks the leading '.'.
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("rope.rope.freqs"));
}

TEST(ColaIgnoredKeyTest, MissingMiddleRopeNotIgnored) {
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("model.rope.freqs"));
}

TEST(ColaIgnoredKeyTest, NormalWeightNotIgnored) {
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("model.weight"));
}

TEST(ColaIgnoredKeyTest, ShortStringNotIgnored) {
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("freqs"));
}

TEST(ColaIgnoredKeyTest, EmptyStringNotIgnored) {
  EXPECT_FALSE(is_cola_ignored_checkpoint_key(""));
}

TEST(ColaIgnoredKeyTest, RopeFreqsMiddleNotIgnored) {
  // "rope.rope.freqs" in the middle but not at the end
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("rope.rope.freqs.extra"));
}

// ===========================================================================
// create_block_causal_mask
// ===========================================================================

TEST(BlockCausalMaskTest, SingleSampleShape) {
  std::vector<int64_t> k_lens = {8};
  std::vector<int64_t> q_lens = {4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kFloat32, torch::kCPU);

  EXPECT_EQ(mask.dim(), 4);
  EXPECT_EQ(mask.size(0), 1);
  EXPECT_EQ(mask.size(1), 1);
  EXPECT_EQ(mask.size(2), 4);  // L_q
  EXPECT_EQ(mask.size(3), 8);  // L_k
}

TEST(BlockCausalMaskTest, SingleSampleAllAllowed) {
  // Single sample, q covers last 4 of 8 K positions.
  // Block 0 (K positions 0-3) and Block 1 (K positions 4-7).
  // Q positions 4-7 are in block 1, can attend to blocks 0 and 1.
  std::vector<int64_t> k_lens = {8};
  std::vector<int64_t> q_lens = {4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kFloat32, torch::kCPU);

  // All positions should be allowed (0.0) since Q block 1 >= K blocks 0,1.
  float min_val = mask.min().item<float>();
  EXPECT_EQ(min_val, 0.0f);
}

TEST(BlockCausalMaskTest, SingleSampleBlockCausality) {
  // 16 K positions, 4 Q positions at the tail.
  // Q is in block 3 (positions 12-15), can attend to blocks 0,1,2,3.
  // But let's test with Q in block 1 (positions 4-7) and K=16.
  // Actually: Q positions are the LAST q_lens positions of K.
  // So k_lens=16, q_lens=4 → Q positions are 12-15 (block 3).
  // Q block 3 can attend to K blocks 0,1,2,3 → all 16 positions allowed.
  std::vector<int64_t> k_lens = {16};
  std::vector<int64_t> q_lens = {4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kFloat32, torch::kCPU);

  float min_val = mask.min().item<float>();
  EXPECT_EQ(min_val, 0.0f);
}

TEST(BlockCausalMaskTest, TwoSamplesCrossIsolation) {
  // Two samples, each with 8 K and 4 Q.
  std::vector<int64_t> k_lens = {8, 8};
  std::vector<int64_t> q_lens = {4, 4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kFloat32, torch::kCPU);

  // L_q = 4+4 = 8, L_k = 8+8 = 16
  EXPECT_EQ(mask.size(2), 8);
  EXPECT_EQ(mask.size(3), 16);

  // Cross-sample positions should be masked (min value).
  // Sample 0 Q (rows 0-3) attending to sample 1 K (cols 8-15) → masked.
  auto cross_block = mask.slice(2, 0, 4).slice(3, 8, 16);
  float cross_min = cross_block.min().item<float>();
  EXPECT_LT(cross_min, 0.0f);

  // Same-sample positions should be allowed (0.0).
  auto same_block = mask.slice(2, 0, 4).slice(3, 0, 8);
  float same_min = same_block.min().item<float>();
  EXPECT_EQ(same_min, 0.0f);
}

TEST(BlockCausalMaskTest, MaskDtypeMatchesTarget) {
  std::vector<int64_t> k_lens = {4};
  std::vector<int64_t> q_lens = {4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kBFloat16, torch::kCPU);

  EXPECT_EQ(mask.scalar_type(), torch::kBFloat16);
}

TEST(BlockCausalMaskTest, SingleBlockAllAllowed) {
  // 4 K, 4 Q, block_size=4 → single block, all positions allowed.
  std::vector<int64_t> k_lens = {4};
  std::vector<int64_t> q_lens = {4};
  int64_t block_size = 4;

  auto mask = create_block_causal_mask(
      k_lens, q_lens, block_size, torch::kFloat32, torch::kCPU);

  float min_val = mask.min().item<float>();
  float max_val = mask.max().item<float>();
  EXPECT_EQ(min_val, 0.0f);
  EXPECT_EQ(max_val, 0.0f);
}

}  // namespace
}  // namespace xllm
