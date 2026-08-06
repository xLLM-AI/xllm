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

#include "core/common/flash_comm1_context.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

FlashComm1Context make_context(int32_t rank,
                               int32_t world_size,
                               int32_t original_num_tokens,
                               int32_t padded_num_tokens) {
  FlashComm1Context context;
  context.enabled = true;
  context.tp_rank = rank;
  context.tp_world_size = world_size;
  context.original_num_tokens = original_num_tokens;
  context.padded_num_tokens = padded_num_tokens;
  context.padded_local_num_tokens = padded_num_tokens / world_size;
  context.pad_size = padded_num_tokens - original_num_tokens;
  return context;
}

ParallelArgs make_parallel_args(int32_t world_size,
                                int32_t dp_size,
                                int32_t cp_size) {
  return ParallelArgs(/*rank=*/0,
                      world_size,
                      dp_size,
                      cp_size,
                      /*process_group=*/nullptr,
                      /*ep_size=*/1);
}

FlashComm1Options enabled_options(int32_t min_prefill_tokens = 1000) {
  return FlashComm1Options{
      .enable_flashcomm1 = true,
      .min_prefill_tokens = min_prefill_tokens,
  };
}

TEST(FlashComm1ContextTest, EligibilityIsIndependentOfDpAndTpSize) {
  // dp4/tp4 must stay eligible: the #1890 dp_size==1 and tp_size>=8 limits were
  // tuned for Qwen3.5-27B only.
  const ParallelArgs dp4_tp4 = make_parallel_args(/*world_size=*/16,
                                                  /*dp_size=*/4,
                                                  /*cp_size=*/1);

  EXPECT_TRUE(is_flash_comm1_eligible(/*num_tokens=*/1000,
                                      /*is_prefill=*/true,
                                      dp4_tp4,
                                      enabled_options()));
  EXPECT_FALSE(is_flash_comm1_eligible(/*num_tokens=*/999,
                                       /*is_prefill=*/true,
                                       dp4_tp4,
                                       enabled_options()));
  EXPECT_FALSE(is_flash_comm1_eligible(/*num_tokens=*/1000,
                                       /*is_prefill=*/false,
                                       dp4_tp4,
                                       enabled_options()));
  EXPECT_FALSE(is_flash_comm1_eligible(/*num_tokens=*/1000,
                                       /*is_prefill=*/true,
                                       dp4_tp4,
                                       FlashComm1Options{}));
  EXPECT_FALSE(is_flash_comm1_eligible(/*num_tokens=*/1000,
                                       /*is_prefill=*/true,
                                       dp4_tp4,
                                       enabled_options(1200)));
}

TEST(FlashComm1ContextTest, ContextParallelRemainsIneligible) {
  EXPECT_FALSE(is_flash_comm1_eligible(/*num_tokens=*/1000,
                                       /*is_prefill=*/true,
                                       make_parallel_args(/*world_size=*/32,
                                                          /*dp_size=*/4,
                                                          /*cp_size=*/2),
                                       enabled_options()));
}

TEST(FlashComm1ContextTest, ScopePublishesAndRestoresContext) {
  EXPECT_EQ(get_current_flash_comm1_context(), nullptr);
  FlashComm1Context outer = make_context(
      /*rank=*/0,
      /*world_size=*/2,
      /*original_num_tokens=*/17,
      /*padded_num_tokens=*/32);
  FlashComm1Context inner = make_context(
      /*rank=*/1,
      /*world_size=*/2,
      /*original_num_tokens=*/32,
      /*padded_num_tokens=*/32);

  {
    FlashComm1ContextScope outer_scope(&outer);
    EXPECT_EQ(get_current_flash_comm1_context(), &outer);
    {
      FlashComm1ContextScope inner_scope(&inner);
      EXPECT_EQ(get_current_flash_comm1_context(), &inner);
    }
    EXPECT_EQ(get_current_flash_comm1_context(), &outer);
  }
  EXPECT_EQ(get_current_flash_comm1_context(), nullptr);
}

TEST(FlashComm1ContextTest, SequenceShardingRequiresEnabledTpContext) {
  FlashComm1Context context;
  EXPECT_FALSE(is_sequence_sharded(context));

  context.enabled = true;
  EXPECT_FALSE(is_sequence_sharded(context));

  context.tp_world_size = 2;
  EXPECT_TRUE(is_sequence_sharded(context));
}

TEST(FlashComm1ContextTest, ReduceModeTracksMmrsOption) {
  FlashComm1Context context;
  EXPECT_EQ(row_parallel_reduce_mode_for_fc1(context),
            RowParallelReduceMode::REDUCE_SCATTER);

  context.enable_mmrs_fusion = true;
  EXPECT_EQ(row_parallel_reduce_mode_for_fc1(context),
            RowParallelReduceMode::MATMUL_REDUCE_SCATTER);
}

TEST(FlashComm1ContextTest, ShardSequenceSplitsDivisibleInput) {
  torch::Tensor input = torch::arange(64).reshape({32, 2}).to(torch::kFloat32);
  for (int32_t rank = 0; rank < 2; ++rank) {
    FlashComm1Context context = make_context(rank,
                                             /*world_size=*/2,
                                             /*original_num_tokens=*/32,
                                             /*padded_num_tokens=*/32);
    torch::Tensor shard = shard_sequence(input, context);
    EXPECT_TRUE(torch::equal(
        shard, input.slice(0, rank * 16, static_cast<int64_t>(rank + 1) * 16)));
  }
}

TEST(FlashComm1ContextTest, ShardSequencePadsOnlyTailRank) {
  torch::Tensor input = torch::arange(17).reshape({17, 1}).to(torch::kFloat32);
  FlashComm1Context context = make_context(
      /*rank=*/1,
      /*world_size=*/2,
      /*original_num_tokens=*/17,
      /*padded_num_tokens=*/32);

  torch::Tensor shard = shard_sequence(input, context);

  ASSERT_EQ(shard.size(0), 16);
  EXPECT_FLOAT_EQ(shard[0].item<float>(), 16.0f);
  EXPECT_TRUE(
      torch::equal(shard.slice(0, 1), torch::zeros({15, 1}, input.options())));
}

TEST(FlashComm1ContextTest, PadRowsPreservesValuesAndZerosTail) {
  torch::Tensor input = torch::arange(6).reshape({3, 2}).to(torch::kFloat32);

  torch::Tensor padded = pad_rows_by_copy(input, /*padded_rows=*/5);

  ASSERT_EQ(padded.size(0), 5);
  EXPECT_TRUE(torch::equal(padded.slice(0, 0, 3), input));
  EXPECT_TRUE(
      torch::equal(padded.slice(0, 3), torch::zeros({2, 2}, input.options())));
}

}  // namespace
}  // namespace xllm
