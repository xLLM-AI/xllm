/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/parallel_state/flash_comm1_context.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace parallel_state {

// --- FlashComm1Context / FlashComm1Guard ------------------------------------
// These exercise the thread_local RAII scoping only and need no device.

TEST(FlashComm1ContextTest, InactiveByDefault) {
  EXPECT_FALSE(flash_comm1_active());
  EXPECT_FALSE(current_flash_comm1_context().enabled);
}

TEST(FlashComm1ContextTest, GuardPublishesAndRestores) {
  ASSERT_FALSE(flash_comm1_active());
  {
    FlashComm1Guard guard(/*enabled=*/true,
                          /*num_tokens=*/17,
                          /*tp_rank=*/2,
                          /*tp_world_size=*/4,
                          /*tp_group=*/nullptr);
    EXPECT_TRUE(flash_comm1_active());
    const auto& ctx = current_flash_comm1_context();
    EXPECT_TRUE(ctx.enabled);
    EXPECT_EQ(ctx.num_tokens, 17);
    EXPECT_EQ(ctx.tp_rank, 2);
    EXPECT_EQ(ctx.tp_world_size, 4);
    EXPECT_EQ(ctx.tp_group, nullptr);
  }
  // Restored to inactive on scope exit.
  EXPECT_FALSE(flash_comm1_active());
  EXPECT_FALSE(current_flash_comm1_context().enabled);
}

TEST(FlashComm1ContextTest, GuardsNestAndRestoreOuter) {
  FlashComm1Guard outer(/*enabled=*/true,
                        /*num_tokens=*/8,
                        /*tp_rank=*/0,
                        /*tp_world_size=*/2,
                        /*tp_group=*/nullptr);
  EXPECT_EQ(current_flash_comm1_context().num_tokens, 8);
  {
    FlashComm1Guard inner(/*enabled=*/true,
                          /*num_tokens=*/32,
                          /*tp_rank=*/1,
                          /*tp_world_size=*/2,
                          /*tp_group=*/nullptr);
    EXPECT_EQ(current_flash_comm1_context().num_tokens, 32);
    EXPECT_EQ(current_flash_comm1_context().tp_rank, 1);
  }
  // Inner guard restores the outer context, not the inactive default.
  EXPECT_TRUE(flash_comm1_active());
  EXPECT_EQ(current_flash_comm1_context().num_tokens, 8);
  EXPECT_EQ(current_flash_comm1_context().tp_rank, 0);
}

TEST(FlashComm1ContextTest, DisabledGuardStillScopes) {
  FlashComm1Guard guard(/*enabled=*/false,
                        /*num_tokens=*/5,
                        /*tp_rank=*/0,
                        /*tp_world_size=*/1,
                        /*tp_group=*/nullptr);
  // A disabled guard reports inactive but its fields are still published, so a
  // consumer that checks enabled first sees a consistent (off) state.
  EXPECT_FALSE(flash_comm1_active());
  EXPECT_FALSE(current_flash_comm1_context().enabled);
}

// --- shard_dim0_padded ------------------------------------------------------
// No process group required: the sharding math runs on plain CPU tensors.

TEST(ShardDim0PaddedTest, WorldSizeOneIsNoOp) {
  auto x = torch::arange(6).reshape({3, 2});
  auto shard = shard_dim0_padded(x, /*rank=*/0, /*world_size=*/1);
  EXPECT_TRUE(torch::equal(shard, x));
}

TEST(ShardDim0PaddedTest, DivisibleSplitsEvenly) {
  // 8 tokens, hidden=2, world=4 -> each rank owns 2 rows, no padding.
  auto x = torch::arange(16).reshape({8, 2}).to(torch::kFloat32);
  for (int32_t rank = 0; rank < 4; ++rank) {
    auto shard = shard_dim0_padded(x, rank, /*world_size=*/4);
    ASSERT_EQ(shard.size(0), 2);
    ASSERT_EQ(shard.size(1), 2);
    EXPECT_TRUE(torch::equal(shard, x.slice(0, rank * 2, rank * 2 + 2)));
  }
}

TEST(ShardDim0PaddedTest, NonDivisiblePadsOnTrailingShards) {
  // 5 tokens, world=2 -> padded to 6, chunk=3. rank0 = rows [0,3),
  // rank1 = rows [3,6) where row 5 is zero padding.
  auto x = torch::arange(5).reshape({5, 1}).to(torch::kFloat32);
  auto shard0 = shard_dim0_padded(x, /*rank=*/0, /*world_size=*/2);
  auto shard1 = shard_dim0_padded(x, /*rank=*/1, /*world_size=*/2);
  ASSERT_EQ(shard0.size(0), 3);
  ASSERT_EQ(shard1.size(0), 3);

  EXPECT_TRUE(torch::equal(shard0, x.slice(0, 0, 3)));
  // shard1 rows: [3, 4, pad(0)].
  EXPECT_FLOAT_EQ(shard1[0].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(shard1[1].item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(shard1[2].item<float>(), 0.0f);
}

TEST(ShardDim0PaddedTest, ConcatenatedShardsReconstructPaddedFull) {
  // Concatenating every rank's shard in order == the padded full tensor. This
  // mirrors what all_gather_dim0_unpad reconstructs before unpadding.
  auto x = torch::arange(7).reshape({7, 1}).to(torch::kFloat32);
  const int32_t world = 4;  // padded to 8, chunk = 2.
  std::vector<torch::Tensor> shards;
  for (int32_t rank = 0; rank < world; ++rank) {
    shards.push_back(shard_dim0_padded(x, rank, world));
  }
  auto full = torch::cat(shards, /*dim=*/0);
  ASSERT_EQ(full.size(0), 8);
  // First 7 rows equal the original; the trailing row is zero padding.
  EXPECT_TRUE(torch::equal(full.slice(0, 0, 7), x));
  EXPECT_FLOAT_EQ(full[7].item<float>(), 0.0f);
}

TEST(ShardDim0PaddedTest, PreservesTrailingDims) {
  // 3 tokens, shape [3, 2, 4], world=2 -> padded to 4, chunk=2, trailing dims
  // (2, 4) untouched.
  auto x = torch::randn({3, 2, 4});
  auto shard = shard_dim0_padded(x, /*rank=*/1, /*world_size=*/2);
  ASSERT_EQ(shard.dim(), 3);
  EXPECT_EQ(shard.size(0), 2);
  EXPECT_EQ(shard.size(1), 2);
  EXPECT_EQ(shard.size(2), 4);
  // rank1 row 0 == original row 2; row 1 is zero padding.
  EXPECT_TRUE(torch::equal(shard[0], x[2]));
  EXPECT_TRUE(torch::equal(shard[1], torch::zeros({2, 4})));
}

}  // namespace parallel_state
}  // namespace xllm
