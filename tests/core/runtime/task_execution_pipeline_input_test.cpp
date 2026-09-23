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

#include <numeric>

#include "core/runtime/task_execution_pipeline.h"

namespace xllm {
class TaskExecutionPipelineInputTest : public ::testing::Test {
 protected:
  static Status validate_input(const ForwardInput& input,
                               const LlmTaskCapacity& capacity = {}) {
    return TaskExecutionPipeline::validate_input(input, capacity);
  }
  static uint32_t padded_batch_size(const ForwardInput& input,
                                    std::span<const int64_t> sizes) {
    return TaskExecutionPipeline::graph_batch_size(input, sizes);
  }
  static std::vector<int64_t> graph_batch_sizes(
      const LlmTaskCapacity& capacity) {
    return TaskExecutionPipeline::graph_batch_sizes(capacity);
  }
};
namespace {

ForwardInput ordinary_input(int32_t num_tokens = 3,
                            BatchForwardType type = BatchForwardType::PREFILL) {
  ForwardInput input;
  input.input_params.meta.batch_forward_type = type;
  if (num_tokens == 0) {
    return input;
  }
  const bool decode = type.is_decode();
  const int32_t rows = decode ? num_tokens : 1;
  const int32_t length = decode ? 1 : num_tokens;
  input.token_ids = torch::arange(num_tokens, torch::kInt32) + 11;
  input.positions = decode ? torch::zeros({num_tokens}, torch::kInt32)
                           : torch::arange(num_tokens, torch::kInt32);
  input.input_params.meta = {type, rows, 0, length, length, 42, false};
  auto& host = input.input_params.attention.host;
  host.q_seq_lens.assign(rows, length);
  host.kv_seq_lens = host.q_seq_lens;
  host.q_cu_seq_lens.resize(rows);
  std::partial_sum(host.q_seq_lens.begin(),
                   host.q_seq_lens.end(),
                   host.q_cu_seq_lens.begin());
  host.block_tables = torch::arange(rows, torch::kInt32).reshape({rows, 1});
  input.input_params.attention.device.new_cache_slots =
      decode ? torch::arange(rows, torch::kInt32) * 8
             : torch::arange(num_tokens, torch::kInt32);
  input.sampling_params.selected_token_idxes =
      torch::tensor({num_tokens - 1}, torch::kInt32);
  return input;
}

TEST_F(TaskExecutionPipelineInputTest,
       GraphBucketsRespectCapacityAndGraphLimit) {
  LlmTaskCapacity capacity;
  capacity.model.max_sequences = 40;
  capacity.model.max_tokens = 40;
  EXPECT_TRUE(graph_batch_sizes(capacity).empty());
  capacity.max_graph_batch_size = 35;
  EXPECT_EQ(graph_batch_sizes(capacity),
            (std::vector<int64_t>{1, 2, 4, 8, 16, 32, 35}));
  capacity.model.max_tokens = 5;
  EXPECT_EQ(graph_batch_sizes(capacity), (std::vector<int64_t>{1, 2, 4, 5}));
  capacity.model.max_sequences = 3;
  EXPECT_EQ(graph_batch_sizes(capacity), (std::vector<int64_t>{1, 2, 3}));
  capacity.max_graph_batch_size = 1;
  EXPECT_EQ(graph_batch_sizes(capacity), (std::vector<int64_t>{1}));
}

TEST_F(TaskExecutionPipelineInputTest, GraphBucketsPadSingleRankDecodeOnly) {
  const std::vector<int64_t> buckets{1, 2, 4, 5};
  EXPECT_EQ(
      padded_batch_size(ordinary_input(3, BatchForwardType::DECODE), buckets),
      4);
  EXPECT_EQ(
      padded_batch_size(ordinary_input(5, BatchForwardType::DECODE), buckets),
      5);
  EXPECT_EQ(
      padded_batch_size(ordinary_input(6, BatchForwardType::DECODE), buckets),
      0);
  EXPECT_EQ(padded_batch_size(ordinary_input(), buckets), 0);
  EXPECT_EQ(padded_batch_size(ForwardInput{}, buckets), 0);
  EXPECT_EQ(padded_batch_size(ordinary_input(1, BatchForwardType::DECODE), {}),
            0);
}

TEST_F(TaskExecutionPipelineInputTest, GraphBucketsUseAllActiveDpRanks) {
  const std::vector<int64_t> buckets{1, 2, 4};
  auto input = ordinary_input(1, BatchForwardType::DECODE);
  auto& parallel = input.input_params.parallel;
  parallel.dp_global_token_nums = {1, 3};
  parallel.dp_is_decode = {1, 1};
  EXPECT_EQ(padded_batch_size(input, buckets), 4);
  parallel.dp_global_token_nums = {1, 5};
  EXPECT_EQ(padded_batch_size(input, buckets), 0);
  parallel.dp_global_token_nums = {1, 2};
  parallel.dp_is_decode = {1, 0};
  EXPECT_EQ(padded_batch_size(input, buckets), 0);
  input = ordinary_input(0, BatchForwardType::DECODE);
  input.input_params.parallel.dp_global_token_nums = {0, 3};
  input.input_params.parallel.dp_is_decode = {0, 1};
  EXPECT_EQ(padded_batch_size(input, buckets), 4);
  input.input_params.parallel.dp_global_token_nums = {0, 0};
  EXPECT_EQ(padded_batch_size(input, buckets), 0);
}

TEST_F(TaskExecutionPipelineInputTest,
       ServingUsesCapturedBucketsAcrossUnevenAndEmptyDpRanks) {
  // The completed warmup can cover fewer buckets than configured capacity.
  const std::vector<int64_t> buckets{2, 4};
  for (int32_t local_count : {0, 1, 3}) {
    auto input = ordinary_input(local_count, BatchForwardType::DECODE);
    input.input_params.parallel.dp_global_token_nums = {0, 1, 3};
    input.input_params.parallel.dp_is_decode = {0, 1, 1};
    EXPECT_EQ(padded_batch_size(input, buckets), 4);
    EXPECT_EQ(padded_batch_size(input, {}), 0);
    input.input_params.parallel.dp_global_token_nums = {0, 1, 5};
    EXPECT_EQ(padded_batch_size(input, buckets), 0);
  }
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
  source.input_params.parallel.dp_is_decode = {0};
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
  invalid.skip_sampling_for_logits_only = true;
  rejected(invalid);
}

TEST_F(TaskExecutionPipelineInputTest,
       AcceptsDpSummariesAndEmptyPeerTransport) {
  auto input = ordinary_input();
  auto& parallel = input.input_params.parallel;
  parallel.dp_global_token_nums = {3, 0};
  parallel.raw_dp_global_token_nums = {3, 0};
  parallel.dp_is_decode = {0, 0};
  parallel.dp_global_sequence_nums = {1, 0};
  LlmTaskCapacity capacity;
  capacity.dp_size = 2;
  ASSERT_TRUE(validate_input(input, capacity).ok());
  ForwardInput empty;
  empty.input_params.parallel = parallel;
  // The engine gives empty peers the active shard's forward type.
  empty.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  capacity.dp_rank = 1;
  EXPECT_TRUE(validate_input(empty, capacity).ok());
}

TEST_F(TaskExecutionPipelineInputTest, ValidatesUnequalIdleAndEmptyDpInputs) {
  ParallelInput parallel;
  parallel.dp_global_token_nums = {2, 0, 7};
  LlmTaskCapacity capacity;
  capacity.dp_size = 3;
  for (const std::vector<int32_t>& phases :
       {std::vector<int32_t>{1, 0, 1}, std::vector<int32_t>{1, 0, 0}}) {
    parallel.dp_is_decode = phases;
    for (uint32_t rank = 0; rank < 3; ++rank) {
      auto input =
          ordinary_input(parallel.dp_global_token_nums[rank],
                         phases[rank] == 1 ? BatchForwardType::DECODE
                                           : BatchForwardType::PREFILL);
      input.token_ids_host = input.token_ids;
      input.token_ids = torch::Tensor();
      input.input_params.parallel = parallel;
      capacity.dp_rank = rank;
      EXPECT_TRUE(validate_input(input, capacity).ok());
    }
  }
  ForwardInput empty;
  empty.input_params.parallel.dp_global_token_nums = {0, 0, 0};
  empty.input_params.parallel.dp_is_decode = {1, 1, 1};
  capacity.dp_rank = 0;
  EXPECT_TRUE(validate_input(empty, capacity).ok());
  EXPECT_TRUE(validate_input(ordinary_input()).ok());
  capacity.dp_size = 2;
  EXPECT_FALSE(validate_input(ordinary_input(), capacity).ok());
}

TEST_F(TaskExecutionPipelineInputTest, RejectsMisalignedDpInputs) {
  auto input = ordinary_input(/*num_tokens=*/1, BatchForwardType::DECODE);
  input.input_params.parallel.dp_global_token_nums = {1, 2};
  input.input_params.parallel.dp_is_decode = {1, 1};
  LlmTaskCapacity capacity;
  capacity.dp_size = 2;
  const auto rejected = [&capacity](const ForwardInput& value) {
    EXPECT_FALSE(validate_input(value, capacity).ok());
  };
  for (const auto member :
       {&ParallelInput::dp_global_token_nums, &ParallelInput::dp_is_decode}) {
    auto invalid = input;
    (invalid.input_params.parallel.*member).pop_back();
    rejected(invalid);
  }
  auto invalid = input;
  invalid.input_params.parallel.dp_global_token_nums[1] = -1;
  rejected(invalid);
  invalid = input;
  invalid.input_params.parallel.dp_is_decode[1] = 2;
  rejected(invalid);
  invalid = input;
  invalid.input_params.parallel.dp_global_token_nums[0] = 2;
  rejected(invalid);
  invalid = input;
  invalid.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  rejected(invalid);
}
}  // namespace
}  // namespace xllm
