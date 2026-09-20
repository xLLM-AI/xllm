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

#include "core/runtime/task_execution_pipeline.h"

namespace xllm {
class TaskExecutionPipelineInputTest : public ::testing::Test {
 protected:
  static Status validate_input(const ForwardInput& input) {
    return TaskExecutionPipeline::validate_input(input);
  }
};
namespace {

ForwardInput ordinary_input() {
  ForwardInput input;
  input.token_ids = torch::tensor({11, 12, 13}, torch::kInt32);
  input.positions = torch::tensor({0, 1, 2}, torch::kInt32);
  input.input_params.meta = {BatchForwardType::PREFILL, 1, 0, 3, 3, 42, false};
  auto& host = input.input_params.attention.host;
  host.q_seq_lens = {3};
  host.kv_seq_lens = {3};
  host.q_cu_seq_lens = {3};
  host.block_tables = torch::tensor({{0}}, torch::kInt32);
  input.input_params.attention.device.new_cache_slots =
      torch::tensor({0, 1, 2}, torch::kInt32);
  input.sampling_params.selected_token_idxes =
      torch::tensor({2}, torch::kInt32);
  return input;
}

TEST_F(TaskExecutionPipelineInputTest,
       AcceptsOrdinaryBuilderInputAndWarmupMarker) {
  auto source = ordinary_input();
  source.input_params.embedding.extra_token_ids = {-1};
  source.input_params.embedding.embedding_ids = {-1};
  source.input_params.embedding.linear_state_ids = {-1};
  source.input_params.embedding.linear_state_indices =
      torch::tensor({-1}, torch::kInt32);
  source.input_params.parallel.dp_global_token_nums = {3};
  source.input_params.meta.is_graph_warmup = true;
  ASSERT_TRUE(validate_input(source).ok());
  source.input_params.attention.host.new_cache_slots = {0, 1, 2};
  ASSERT_TRUE(validate_input(source).ok());
}

TEST_F(TaskExecutionPipelineInputTest, EmptyInputAndAbsentSamplingAreValid) {
  ASSERT_TRUE(validate_input(ForwardInput{}).ok());
  auto source = ordinary_input();
  source.sampling_params = {};
  ASSERT_TRUE(validate_input(source).ok());
}

TEST_F(TaskExecutionPipelineInputTest,
       RejectsInvalidTransportAndAlgorithmFields) {
  const auto rejected = [](ForwardInput invalid) {
    EXPECT_FALSE(validate_input(invalid).ok());
  };
  auto invalid = ordinary_input();
  invalid.token_ids = invalid.token_ids.to(torch::kInt64);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.positions = torch::zeros({6}, torch::kInt32)
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.attention.host.block_tables =
      torch::zeros({2, 1}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.meta.actual_num_sequences = 2;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_host_buffer_has_layout = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.device_tensors_ready = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.is_spec_verify = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.block_copy.src_block_indices =
      torch::tensor({0}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.embedding.linear_state_ids = {0};
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.embedding.linear_state_indices =
      torch::tensor({0}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.parallel.dp_global_token_nums = {3, 3};
  rejected(invalid);
  invalid = ordinary_input();
  invalid.skip_sampling_for_logits_only = true;
  rejected(invalid);
}

}  // namespace
}  // namespace xllm
