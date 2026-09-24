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

#include <gtest/gtest.h>

#include "tests/core/runtime/task_pipeline_test_peer.h"

namespace xllm {
namespace {

TEST(MtpParallelPlanTest, EveryRankAgreesOnDecodeDespiteIdleShards) {
  ParallelInput input;
  input.dp_global_token_nums = {2, 0, 3};
  input.raw_dp_global_token_nums = input.dp_global_token_nums;
  input.dp_is_decode = {1, 0, 1};
  input.dp_global_kv_max_seq_lens = {20, 0, 10};
  input.dp_global_batch_generations = {11, 2, 9};
  for (uint32_t rank = 0; rank < 3; ++rank) {
    ParallelInput output;
    bool run_models = false;
    bool decode = false;
    ASSERT_TRUE(
        TaskPipelineTestPeer::plan_parallel(input,
                                            3,
                                            rank,
                                            input.dp_global_token_nums[rank],
                                            input.dp_is_decode[rank] == 1,
                                            output,
                                            run_models,
                                            decode)
            .ok());
    EXPECT_TRUE(run_models);
    EXPECT_TRUE(decode);
    EXPECT_EQ(output.dp_global_token_nums, input.dp_global_token_nums);
    EXPECT_EQ(output.raw_dp_global_token_nums, input.raw_dp_global_token_nums);
    EXPECT_EQ(output.dp_global_batch_generations,
              input.dp_global_batch_generations);
  }
}

TEST(MtpParallelPlanTest, MixedPhaseAndAllEmptyChooseCommonExecution) {
  ParallelInput input;
  input.dp_global_token_nums = {1, 13, 0};
  input.raw_dp_global_token_nums = input.dp_global_token_nums;
  input.dp_is_decode = {1, 0, 1};
  for (uint32_t rank = 0; rank < 3; ++rank) {
    ParallelInput output;
    bool run_models = false;
    bool decode = true;
    ASSERT_TRUE(
        TaskPipelineTestPeer::plan_parallel(input,
                                            3,
                                            rank,
                                            input.dp_global_token_nums[rank],
                                            input.dp_is_decode[rank] == 1,
                                            output,
                                            run_models,
                                            decode)
            .ok());
    EXPECT_TRUE(run_models);
    EXPECT_FALSE(decode);
  }
  input.dp_global_token_nums = {0, 0, 0};
  input.raw_dp_global_token_nums = input.dp_global_token_nums;
  for (uint32_t rank = 0; rank < 3; ++rank) {
    ParallelInput output;
    bool run_models = true;
    bool decode = true;
    ASSERT_TRUE(TaskPipelineTestPeer::plan_parallel(
                    input, 3, rank, 0, false, output, run_models, decode)
                    .ok());
    EXPECT_FALSE(run_models);
    EXPECT_FALSE(decode);
  }
}

TEST(MtpParallelPlanTest, InvalidTransportLeavesPreparedMetadataUnchanged) {
  ParallelInput valid;
  valid.dp_global_token_nums = {1, 2};
  valid.raw_dp_global_token_nums = valid.dp_global_token_nums;
  valid.dp_is_decode = {1, 1};
  std::vector<ParallelInput> invalid(4, valid);
  invalid[0].raw_dp_global_token_nums[1] = 1;
  invalid[1].dp_global_token_nums[1] = -1;
  invalid[2].dp_is_decode[1] = 2;
  invalid[3].dp_is_decode.clear();
  ParallelInput output;
  output.dp_global_token_nums = {77};
  bool run_models = true;
  bool decode = false;
  for (const auto& input : invalid) {
    EXPECT_FALSE(TaskPipelineTestPeer::plan_parallel(
                     input, 2, 0, 1, true, output, run_models, decode)
                     .ok());
    EXPECT_EQ(output.dp_global_token_nums, (std::vector<int32_t>{77}));
    EXPECT_TRUE(run_models);
    EXPECT_FALSE(decode);
  }
  EXPECT_FALSE(TaskPipelineTestPeer::plan_parallel(
                   valid, 2, 0, 2, true, output, run_models, decode)
                   .ok());
  EXPECT_FALSE(TaskPipelineTestPeer::plan_parallel(
                   valid, 2, 0, 1, false, output, run_models, decode)
                   .ok());
}

TEST(MtpParallelPlanTest, PreparedMetadataOwnsCallerValues) {
  ParallelInput input;
  input.dp_global_token_nums = {1, 2};
  input.raw_dp_global_token_nums = input.dp_global_token_nums;
  input.dp_is_decode = {1, 1};
  ParallelInput output;
  bool run_models = false;
  bool decode = false;
  ASSERT_TRUE(TaskPipelineTestPeer::plan_parallel(
                  input, 2, 0, 1, true, output, run_models, decode)
                  .ok());
  input.dp_global_token_nums.assign(2, 0);
  input.dp_is_decode.assign(2, 0);
  EXPECT_EQ(output.dp_global_token_nums, (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(output.dp_is_decode, (std::vector<int32_t>{1, 1}));
}

TEST(MtpParallelPlanTest, SingleRankCanOmitSchedulerSummaries) {
  ParallelInput output;
  bool run_models = false;
  bool decode = false;
  ASSERT_TRUE(TaskPipelineTestPeer::plan_parallel(
                  {}, 1, 0, 1, true, output, run_models, decode)
                  .ok());
  EXPECT_TRUE(run_models);
  EXPECT_TRUE(decode);
  EXPECT_EQ(output.dp_global_token_nums, (std::vector<int32_t>{1}));
  EXPECT_FALSE(TaskPipelineTestPeer::plan_parallel(
                   {}, 2, 0, 1, true, output, run_models, decode)
                   .ok());
}

}  // namespace
}  // namespace xllm
