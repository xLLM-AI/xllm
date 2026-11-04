/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
*/
#include "models/dit/utils/minimax_h3_qkv_layout.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

TEST(QualifiedQkvLayout, PreservesProjectionRowsAndRegisteredStorage) {
  torch::Tensor weight = torch::arange(12, torch::kFloat32).view({12, 1});
  const void* storage = weight.data_ptr();
  const torch::Tensor expected =
      torch::tensor({0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11}, torch::kFloat32)
          .view({12, 1});
  weight.copy_(minimax_h3_split_grouped_qkv_shard(weight, 2, 2));
  EXPECT_EQ(weight.data_ptr(), storage);
  EXPECT_EQ(weight.numel() * weight.element_size(), 48);
  EXPECT_TRUE(torch::equal(weight, expected));
  const torch::Tensor input = torch::full({1, 1}, 3.0F);
  EXPECT_TRUE(torch::equal(torch::matmul(input, weight.narrow(0, 0, 4).t()),
                           torch::tensor({{0, 3, 18, 21}}, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(torch::matmul(input, weight.narrow(0, 4, 4).t()),
                           torch::tensor({{6, 9, 24, 27}}, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(torch::matmul(input, weight.narrow(0, 8, 4).t()),
                           torch::tensor({{12, 15, 30, 33}}, torch::kFloat32)));
}

TEST(QualifiedQkvLayout, FullRankBfloatShardRoundTripsWithoutChangingBytes) {
  const torch::Tensor grouped = torch::arange(3 * 28 * 128 * 4, torch::kFloat32)
                                    .remainder(127)
                                    .view({3 * 28 * 128, 4})
                                    .to(torch::kBFloat16);
  const torch::Tensor split =
      minimax_h3_split_grouped_qkv_shard(grouped, 28, 128);
  const torch::Tensor restored = split.view({3, 28, 128, 4})
                                     .permute({1, 0, 2, 3})
                                     .reshape(grouped.sizes())
                                     .contiguous();
  EXPECT_EQ(split.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::equal(grouped, restored));
}

TEST(QualifiedQkvLayout, RejectsInvalidMetadataBeforeReordering) {
  EXPECT_THROW(minimax_h3_split_grouped_qkv_shard(torch::Tensor(), 2, 2),
               std::invalid_argument);
  EXPECT_THROW(minimax_h3_split_grouped_qkv_shard(torch::zeros({11, 1}), 2, 2),
               std::invalid_argument);
  EXPECT_THROW(minimax_h3_split_grouped_qkv_shard(torch::zeros({12, 1}), 0, 2),
               std::invalid_argument);
}

}  // namespace
}  // namespace xllm
