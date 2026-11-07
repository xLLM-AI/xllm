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

#include "core/runtime/mlu_graph_planner.h"

#include <gtest/gtest.h>

#include "core/framework/model/rec_model_params.h"

namespace xllm::mlu {
namespace {
class MluGraphPlannerTokenBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.options.is_draft_engine(false);
    config_.options.max_seqs_per_batch(33);
    config_.options.enable_graph_mode_decode_no_padding(false);
    config_.max_tokens = 33;
    params_.meta.batch_forward_type = BatchForwardType::DECODE;
    params_.meta.q_max_seq_len = 1;
  }
  GraphDecision plan(int64_t tokens) {
    params_.meta.num_sequences = tokens;
    params_.parallel.dp_global_token_nums = {static_cast<int32_t>(tokens)};
    params_.parallel.dp_is_decode = {1};
    return MluGraphPlanner(config_).plan(tokens, params_);
  }
  GraphPlannerConfig config_;
  RecModelParams params_;
};
TEST_F(MluGraphPlannerTokenBudgetTest, RecRoundedBucketOverBudgetRunsEager) {
  const auto decision = plan(33);
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), EagerReason::TOKEN_LIMIT);
}
TEST_F(MluGraphPlannerTokenBudgetTest, RecWithinBudgetUsesGraph) {
  const auto decision = plan(32);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  EXPECT_EQ(std::get<GraphPlan>(decision).layout.padded_num_tokens, 32);
}
TEST_F(MluGraphPlannerTokenBudgetTest, RecExactBucketBudgetUsesGraph) {
  config_.max_tokens = 48;
  const auto decision = plan(33);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  EXPECT_EQ(std::get<GraphPlan>(decision).layout.padded_num_tokens, 48);
}
TEST_F(MluGraphPlannerTokenBudgetTest, RecNoPaddingUsesExactTokenBudget) {
  config_.options.enable_graph_mode_decode_no_padding(true);
  const auto decision = plan(33);
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  EXPECT_EQ(std::get<GraphPlan>(decision).layout.padded_num_tokens, 33);
}
}  // namespace
}  // namespace xllm::mlu
