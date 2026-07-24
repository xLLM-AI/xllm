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

#include "framework/parallel_state/npu_cp_plan.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "framework/parallel_state/process_group.h"

namespace xllm {
namespace {

torch::Tensor int32_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values, torch::dtype(torch::kInt32));
}

torch::Tensor int64_tensor(const std::vector<int64_t>& values) {
  return torch::tensor(values, torch::dtype(torch::kInt64));
}

void expect_tensor_bytes_equal(const torch::Tensor& actual,
                               const torch::Tensor& expected) {
  ASSERT_TRUE(actual.defined());
  ASSERT_TRUE(expected.defined());
  ASSERT_EQ(actual.scalar_type(), expected.scalar_type());
  ASSERT_EQ(actual.device(), expected.device());
  ASSERT_EQ(actual.layout(), expected.layout());
  ASSERT_EQ(actual.sizes(), expected.sizes());
  ASSERT_EQ(actual.strides(), expected.strides());
  ASSERT_EQ(actual.storage_offset(), expected.storage_offset());
  ASSERT_EQ(actual.is_contiguous(), expected.is_contiguous());
  torch::Tensor actual_cpu = actual.cpu().contiguous();
  torch::Tensor expected_cpu = expected.cpu().contiguous();
  const size_t byte_count =
      static_cast<size_t>(actual_cpu.numel()) * actual_cpu.element_size();
  ASSERT_EQ(
      byte_count,
      static_cast<size_t>(expected_cpu.numel()) * expected_cpu.element_size());
  EXPECT_EQ(
      std::memcmp(actual_cpu.data_ptr(), expected_cpu.data_ptr(), byte_count),
      0);
}

float legacy_cp_ep_buffer_factor(int64_t length, int32_t attention_cp_size) {
  length *= attention_cp_size;
  const std::vector<std::pair<int64_t, float>> thresholds = {{1048576, 1.32f},
                                                             {524288, 1.4f},
                                                             {262144, 1.53f},
                                                             {131072, 1.8f},
                                                             {32768, 3.0f},
                                                             {8192, 5.2f},
                                                             {0, 8.0f}};
  for (const auto& threshold : thresholds) {
    if (length >= threshold.first) {
      return threshold.second;
    }
  }
  return 8.0f;
}

CpEpMeta build_legacy_cp_ep_meta(int64_t local_padded_token_count,
                                 const CpPlanConfig& config) {
  const int64_t input_length = std::max<int64_t>(local_padded_token_count, 1);
  const int64_t padding_length =
      (config.attention_tp_size - input_length % config.attention_tp_size) %
      config.attention_tp_size;
  const int64_t padded_group_length = input_length + padding_length;
  const int64_t padded_rank_length =
      padded_group_length / config.attention_tp_size;

  CpEpMeta meta;
  meta.attention_tp_padding_indices =
      torch::cat({torch::arange(input_length, torch::kInt32),
                  torch::zeros({padding_length}, torch::kInt32)});
  meta.prenorm_gather_indices = meta.attention_tp_padding_indices.slice(
      /*dim=*/0,
      config.attention_tp_rank * padded_rank_length,
      (config.attention_tp_rank + 1) * padded_rank_length);

  std::vector<torch::Tensor> skip_padding_parts;
  skip_padding_parts.reserve(config.attention_cp_group_size);
  for (int32_t cp_rank = 0; cp_rank < config.attention_cp_group_size;
       ++cp_rank) {
    skip_padding_parts.emplace_back(torch::arange(input_length, torch::kInt32) +
                                    cp_rank * padded_group_length);
  }
  torch::Tensor skip_padding_indices = torch::cat(skip_padding_parts, 0);

  const bool dynamic_ep =
      config.moe_ep_size > 1 && (config.expert_parallel_degree == 2 ||
                                 config.expert_parallel_degree == 3);
  if (dynamic_ep) {
    meta.attention_tp_unpadding_indices =
        torch::arange(padded_rank_length, torch::kInt32);
    meta.ffn_padding_indices = meta.attention_tp_unpadding_indices;
  } else {
    meta.attention_tp_unpadding_indices = skip_padding_indices;
    std::vector<torch::Tensor> ffn_padding_parts;
    ffn_padding_parts.reserve(config.attention_cp_group_size);
    for (int32_t cp_rank = 0; cp_rank < config.attention_cp_group_size;
         ++cp_rank) {
      ffn_padding_parts.emplace_back(
          torch::cat({torch::arange(input_length * cp_rank,
                                    input_length * (cp_rank + 1),
                                    torch::kInt32),
                      torch::zeros({padding_length}, torch::kInt32)}));
    }
    meta.ffn_padding_indices = torch::cat(ffn_padding_parts, 0);
  }

  meta.attention_padding_indices = meta.attention_tp_padding_indices;
  meta.attention_unpadding_indices = torch::zeros({1}, torch::kInt32);
  meta.ffn_unpadding_indices = torch::arange(input_length, torch::kInt32);
  meta.lm_head_skip_padding_indices = skip_padding_indices;

  if (!dynamic_ep) {
    meta.dynamic_ep_indices = torch::zeros({1}, torch::kInt32);
    meta.moe_indices = torch::zeros({1}, torch::kInt32);
    meta.expert_array = torch::tensor({0});
    return meta;
  }

  const int64_t dynamic_ep_length =
      (config.attention_tp_size == 1 ? input_length : padded_rank_length) *
      config.num_experts_per_token;
  meta.dynamic_ep_indices = torch::arange(dynamic_ep_length, torch::kInt32);
  const float buffer_factor =
      legacy_cp_ep_buffer_factor(dynamic_ep_length, config.attention_cp_size);
  int32_t ep_input_length =
      static_cast<int32_t>(dynamic_ep_length * buffer_factor);
  const int32_t all_to_all_padding = ep_input_length % config.moe_ep_size;
  if (all_to_all_padding != 0) {
    ep_input_length += config.moe_ep_size - all_to_all_padding;
  }
  std::vector<int32_t> moe_indices;
  moe_indices.reserve(ep_input_length);
  for (int32_t i = 1; i <= ep_input_length; ++i) {
    moe_indices.push_back(i);
  }
  meta.moe_indices = torch::tensor(moe_indices, torch::kInt32);
  meta.expert_array =
      torch::ones({ep_input_length}, config.dtype).view({-1, 1});
  return meta;
}

torch::Tensor prepare_cache_slots_reference(
    const torch::Tensor& global_logical_slots,
    const NpuCpPlan& plan,
    const CpPlanConfig& config) {
  torch::Tensor gathered_slots = torch::full(
      {plan.recovered_token_count()}, -1, global_logical_slots.options());
  gathered_slots.index_put_({plan.output_merge_meta().output_restore_indices},
                            global_logical_slots);
  torch::Tensor recovered_logical_slots = gathered_slots.index_select(
      /*dim=*/0, plan.attention_meta().kv_reorder_indices.to(torch::kLong));

  const int32_t logical_block_size = config.block_size * config.kv_split_size;
  torch::Tensor row_indices =
      torch::arange(recovered_logical_slots.numel(), torch::kCPU);
  torch::Tensor logical_block_offsets = row_indices % logical_block_size;
  torch::Tensor row_kv_split_ranks =
      torch::floor_divide(logical_block_offsets, config.block_size);
  torch::Tensor local_row_indices =
      torch::nonzero(row_kv_split_ranks == config.kv_split_rank).flatten();

  torch::Tensor physical_slots = torch::full_like(recovered_logical_slots, -1);
  if (local_row_indices.numel() == 0) {
    return physical_slots;
  }
  torch::Tensor logical_slots =
      recovered_logical_slots.index_select(/*dim=*/0, local_row_indices)
          .to(torch::kInt32);
  torch::Tensor logical_block_ids =
      torch::floor_divide(logical_slots, logical_block_size);
  torch::Tensor physical_block_offsets = logical_slots % config.block_size;
  torch::Tensor local_physical_slots =
      logical_block_ids * config.block_size + physical_block_offsets;
  physical_slots.index_put_({local_row_indices},
                            local_physical_slots.to(physical_slots.dtype()));
  return physical_slots;
}

void expect_cp_ep_meta_bytes_equal(const CpEpMeta& actual,
                                   const CpEpMeta& expected) {
  expect_tensor_bytes_equal(actual.attention_tp_padding_indices,
                            expected.attention_tp_padding_indices);
  expect_tensor_bytes_equal(actual.attention_tp_unpadding_indices,
                            expected.attention_tp_unpadding_indices);
  expect_tensor_bytes_equal(actual.ffn_padding_indices,
                            expected.ffn_padding_indices);
  expect_tensor_bytes_equal(actual.ffn_unpadding_indices,
                            expected.ffn_unpadding_indices);
  expect_tensor_bytes_equal(actual.lm_head_skip_padding_indices,
                            expected.lm_head_skip_padding_indices);
  expect_tensor_bytes_equal(actual.prenorm_gather_indices,
                            expected.prenorm_gather_indices);
  expect_tensor_bytes_equal(actual.attention_padding_indices,
                            expected.attention_padding_indices);
  expect_tensor_bytes_equal(actual.attention_unpadding_indices,
                            expected.attention_unpadding_indices);
  expect_tensor_bytes_equal(actual.dynamic_ep_indices,
                            expected.dynamic_ep_indices);
  expect_tensor_bytes_equal(actual.moe_indices, expected.moe_indices);
  expect_tensor_bytes_equal(actual.expert_array, expected.expert_array);
}

CpPlanInput make_plan_input(const std::vector<int32_t>& q_seq_lens,
                            const std::vector<int32_t>& position_starts) {
  CHECK_EQ(q_seq_lens.size(), position_starts.size());
  CpPlanInput input;
  input.q_seq_lens = q_seq_lens;
  std::vector<int32_t> positions;
  for (size_t i = 0; i < q_seq_lens.size(); ++i) {
    for (int32_t token = 0; token < q_seq_lens[i]; ++token) {
      positions.push_back(position_starts[i] + token);
    }
  }
  input.position_ids = int32_tensor(positions);
  input.prefix_token_counts.resize(q_seq_lens.size(), 0);
  return input;
}

CpPlanInput aligned_input() { return make_plan_input({8, 12}, {0, 0}); }

CpPlanConfig cp2_rank0_config() {
  CpPlanConfig config;
  config.cp_size = 2;
  config.cp_rank = 0;
  config.kv_split_size = 2;
  config.block_size = 128;
  config.attention_tp_size = 1;
  config.attention_tp_rank = 0;
  config.attention_cp_size = 2;
  config.attention_cp_group_size = 2;
  config.moe_ep_size = 1;
  config.expert_parallel_degree = 1;
  config.num_experts_per_token = 8;
  config.device = torch::kCPU;
  config.dtype = torch::kBFloat16;
  return config;
}

TEST(NpuCpPlanTest, GraphMetadataMatchesLegacyBytes) {
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), cp2_rank0_config());
  const CpInputShardMeta& shard_meta = plan.input_shard_meta();
  const CpOutputMergeMeta& merge_meta = plan.output_merge_meta();
  const CpAttentionMeta& attention = plan.attention_meta();
  const CpEpMeta& cp_ep = plan.cp_ep_meta();

  EXPECT_EQ(shard_meta.global_real_token_count, 20);
  EXPECT_EQ(merge_meta.global_padded_token_count, 20);
  EXPECT_EQ(shard_meta.local_real_token_count, 10);
  EXPECT_EQ(shard_meta.local_padded_token_count, 10);
  EXPECT_EQ(shard_meta.local_real_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(shard_meta.local_padded_seq_lens, std::vector<int32_t>({4, 6}));
  expect_tensor_bytes_equal(shard_meta.input_source_indices,
                            int64_tensor({0, 1, 6, 7, 8, 9, 10, 17, 18, 19}));
  expect_tensor_bytes_equal(shard_meta.input_destination_indices,
                            int64_tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  expect_tensor_bytes_equal(shard_meta.local_position_ids,
                            int32_tensor({0, 1, 6, 7, 0, 1, 2, 9, 10, 11}));
  expect_tensor_bytes_equal(merge_meta.output_restore_indices,
                            int64_tensor({0, 1,  10, 11, 12, 13, 2,  3, 4, 5,
                                          6, 14, 15, 16, 17, 18, 19, 7, 8, 9}));

  EXPECT_EQ(attention.host_q_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(attention.host_kv_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(attention.host_q_cu_seq_lens, std::vector<int32_t>({4, 10}));
  EXPECT_EQ(attention.q_max_seq_len, 6);
  EXPECT_EQ(attention.kv_max_seq_len, 6);
  expect_tensor_bytes_equal(attention.q_seq_lens, int32_tensor({4, 6}));
  expect_tensor_bytes_equal(attention.kv_seq_lens, int32_tensor({4, 6}));
  expect_tensor_bytes_equal(attention.q_cu_seq_lens, int32_tensor({4, 10}));
  expect_tensor_bytes_equal(attention.query_balance_indices,
                            int32_tensor({0, 1, 4, 5, 6, 2, 3, 7, 8, 9}));
  expect_tensor_bytes_equal(attention.attention_output_reorder_indices,
                            int32_tensor({0, 1, 5, 6, 2, 3, 4, 7, 8, 9}));
  expect_tensor_bytes_equal(attention.kv_reorder_indices,
                            int32_tensor({0, 1,  10, 11, 12, 13, 2,  3, 4, 5,
                                          6, 14, 15, 16, 17, 18, 19, 7, 8, 9}));
  expect_tensor_bytes_equal(attention.prev_kv_gather_indices,
                            int32_tensor({0, 1, 8, 9, 10}));
  expect_tensor_bytes_equal(
      attention.next_kv_gather_indices,
      int32_tensor({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
  expect_tensor_bytes_equal(attention.prev_query_cu_seq_lens,
                            int32_tensor({2, 5}));
  expect_tensor_bytes_equal(attention.next_query_cu_seq_lens,
                            int32_tensor({2, 5}));
  expect_tensor_bytes_equal(attention.prev_key_cu_seq_lens,
                            int32_tensor({2, 5}));
  expect_tensor_bytes_equal(attention.next_key_cu_seq_lens,
                            int32_tensor({8, 20}));

  expect_tensor_bytes_equal(cp_ep.attention_tp_padding_indices,
                            torch::arange(10, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.attention_tp_unpadding_indices,
                            torch::arange(20, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.ffn_padding_indices,
                            torch::arange(20, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.ffn_unpadding_indices,
                            torch::arange(10, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.lm_head_skip_padding_indices,
                            torch::arange(20, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.prenorm_gather_indices,
                            torch::arange(10, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.attention_padding_indices,
                            torch::arange(10, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.attention_unpadding_indices,
                            int32_tensor({0}));
  expect_tensor_bytes_equal(cp_ep.dynamic_ep_indices, int32_tensor({0}));
  expect_tensor_bytes_equal(cp_ep.moe_indices, int32_tensor({0}));
  expect_tensor_bytes_equal(cp_ep.expert_array, int64_tensor({0}));
}

TEST(NpuCpPlanTest, ShardsModelInputAndAppliesAttentionMeta) {
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), cp2_rank0_config());
  torch::Tensor hidden = torch::arange(80, torch::kFloat).view({20, 4});
  CpInputShard sharded =
      plan.shard_model_input(hidden, aligned_input().position_ids);

  expect_tensor_bytes_equal(
      sharded.hidden_states,
      hidden.index_select(
          /*dim=*/0, int64_tensor({0, 1, 6, 7, 8, 9, 10, 17, 18, 19})));
  expect_tensor_bytes_equal(sharded.position_ids,
                            int32_tensor({0, 1, 6, 7, 0, 1, 2, 9, 10, 11}));

  ModelInputParams params;
  plan.apply_attention_meta(params);
  EXPECT_EQ(params.attention.host.q_seq_lens,
            plan.attention_meta().host_q_seq_lens);
  EXPECT_EQ(params.attention.host.kv_seq_lens,
            plan.attention_meta().host_kv_seq_lens);
  EXPECT_EQ(params.attention.host.q_cu_seq_lens,
            plan.attention_meta().host_q_cu_seq_lens);
  expect_tensor_bytes_equal(params.attention.device.q_seq_lens,
                            plan.attention_meta().q_seq_lens);
  expect_tensor_bytes_equal(params.attention.device.kv_seq_lens,
                            plan.attention_meta().kv_seq_lens);
  expect_tensor_bytes_equal(params.attention.device.q_cu_seq_lens,
                            plan.attention_meta().q_cu_seq_lens);
  EXPECT_EQ(params.meta.q_max_seq_len, plan.attention_meta().q_max_seq_len);
  EXPECT_EQ(params.meta.kv_max_seq_len, plan.attention_meta().kv_max_seq_len);
}

TEST(NpuCpPlanTest, NonAlignedInputUsesVirtualPadding) {
  CpPlanInput input;
  input.q_seq_lens = {5, 7};
  input.position_ids = int32_tensor({0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6});
  input.prefix_token_counts = {0, 0};
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());

  EXPECT_EQ(plan.input_shard_meta().global_real_token_count, 12);
  EXPECT_EQ(plan.output_merge_meta().global_padded_token_count, 16);
  EXPECT_EQ(plan.input_shard_meta().local_real_token_count, 5);
  EXPECT_EQ(plan.input_shard_meta().local_padded_token_count, 8);
  EXPECT_EQ(plan.input_shard_meta().local_real_seq_lens,
            std::vector<int32_t>({2, 3}));
  EXPECT_EQ(plan.input_shard_meta().local_padded_seq_lens,
            std::vector<int32_t>({4, 4}));

  torch::Tensor hidden = torch::arange(12, torch::kFloat).view({12, 1});
  CpInputShard sharded = plan.shard_model_input(hidden, input.position_ids);
  expect_tensor_bytes_equal(
      sharded.hidden_states.flatten(),
      torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 6.0f, 11.0f, 0.0f}));

  const CpAttentionMeta& attention = plan.attention_meta();
  EXPECT_EQ(attention.host_q_seq_lens, std::vector<int32_t>({4, 4}));
  EXPECT_EQ(attention.host_kv_seq_lens, std::vector<int32_t>({4, 4}));
  EXPECT_EQ(attention.host_q_cu_seq_lens, std::vector<int32_t>({4, 8}));
  EXPECT_EQ(attention.q_max_seq_len, 4);
  EXPECT_EQ(attention.kv_max_seq_len, 4);
  expect_tensor_bytes_equal(attention.q_seq_lens, int32_tensor({4, 4}));
  expect_tensor_bytes_equal(attention.kv_seq_lens, int32_tensor({4, 4}));
  expect_tensor_bytes_equal(attention.q_cu_seq_lens, int32_tensor({4, 8}));
  expect_tensor_bytes_equal(attention.query_balance_indices,
                            int32_tensor({0, 1, 4, 5, 2, 3, 6, 7}));
  expect_tensor_bytes_equal(attention.attention_output_reorder_indices,
                            int32_tensor({0, 1, 4, 5, 2, 3, 6, 7}));
  expect_tensor_bytes_equal(
      attention.kv_reorder_indices,
      int32_tensor({0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 12, 13, 14, 15, 6, 7}));
  expect_tensor_bytes_equal(attention.prev_kv_gather_indices,
                            int32_tensor({0, 1, 8, 9}));
  expect_tensor_bytes_equal(
      attention.next_kv_gather_indices,
      int32_tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  expect_tensor_bytes_equal(attention.prev_query_cu_seq_lens,
                            int32_tensor({2, 4}));
  expect_tensor_bytes_equal(attention.next_query_cu_seq_lens,
                            int32_tensor({2, 4}));
  expect_tensor_bytes_equal(attention.prev_key_cu_seq_lens,
                            int32_tensor({2, 4}));
  expect_tensor_bytes_equal(attention.next_key_cu_seq_lens,
                            int32_tensor({8, 16}));
}

TEST(NpuCpPlanTest, InputShardAndOutputMergeRoundTripAcrossRanks) {
  struct TestCase {
    std::vector<int32_t> q_seq_lens;
    int32_t cp_size;
  };
  const std::vector<TestCase> cases = {
      {{8, 12}, 2}, {{5, 7, 1, 3}, 2}, {{16, 8, 24}, 4}, {{1}, 4}};

  for (const TestCase& test_case : cases) {
    const std::vector<int32_t> position_starts(test_case.q_seq_lens.size(), 0);
    const CpPlanInput input =
        make_plan_input(test_case.q_seq_lens, position_starts);
    torch::Tensor global_hidden =
        torch::arange(input.position_ids.numel(), torch::kInt32).view({-1, 1});
    std::vector<torch::Tensor> rank_shards;
    rank_shards.reserve(test_case.cp_size);
    NpuCpPlan rank0_plan;
    for (int32_t cp_rank = 0; cp_rank < test_case.cp_size; ++cp_rank) {
      CpPlanConfig config = cp2_rank0_config();
      config.cp_size = test_case.cp_size;
      config.cp_rank = cp_rank;
      config.kv_split_size = test_case.cp_size;
      config.attention_cp_size = test_case.cp_size;
      config.attention_cp_group_size = test_case.cp_size;
      NpuCpPlan plan = NpuCpPlan::build(input, config);
      if (cp_rank == 0) {
        rank0_plan = plan;
      }
      rank_shards.push_back(
          plan.shard_model_input(global_hidden, input.position_ids)
              .hidden_states);
    }

    torch::Tensor rank_major_gathered = torch::cat(rank_shards, /*dim=*/0);
    torch::Tensor merged = rank_major_gathered.index_select(
        /*dim=*/0, rank0_plan.output_merge_meta().output_restore_indices);
    expect_tensor_bytes_equal(merged, global_hidden);
  }
}

TEST(NpuCpPlanTest, OutputMergeRejectsInvalidProcessGroup) {
#if GTEST_HAS_DEATH_TEST
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), cp2_rank0_config());
  const torch::Tensor local_hidden = torch::zeros({10, 1}, torch::kFloat);
  EXPECT_DEATH(plan.merge_model_output(local_hidden, nullptr), "process_group");

  ProcessGroup wrong_size_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  EXPECT_DEATH(plan.merge_model_output(local_hidden, &wrong_size_group),
               "size mismatch");

  ProcessGroup wrong_rank_group(
      /*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU));
  EXPECT_DEATH(plan.merge_model_output(local_hidden, &wrong_rank_group),
               "rank mismatch");
#endif
}

TEST(NpuCpPlanTest, PrefixAttentionMetadataMatchesLegacyBytes) {
  CpPlanInput input;
  input.q_seq_lens = {8};
  input.position_ids = int32_tensor({256, 257, 258, 259, 260, 261, 262, 263});
  input.prefix_token_counts = {256};
  input.block_tables = int32_tensor({5, 6}).view({1, 2});
  input.has_prefix_slots = true;
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());
  const CpAttentionMeta& attention = plan.attention_meta();

  expect_tensor_bytes_equal(attention.prev_kv_gather_indices,
                            torch::arange(258, torch::kInt32));
  expect_tensor_bytes_equal(attention.next_kv_gather_indices,
                            torch::arange(264, torch::kInt32));
  expect_tensor_bytes_equal(attention.prev_key_cu_seq_lens,
                            int32_tensor({258}));
  expect_tensor_bytes_equal(attention.next_key_cu_seq_lens,
                            int32_tensor({264}));
  expect_tensor_bytes_equal(attention.prefix_cache_slots,
                            torch::arange(640, 768, torch::kInt32));

  ModelInputParams params;
  plan.apply_attention_meta(params);
  expect_tensor_bytes_equal(params.attention.device.in_prefix_slots,
                            attention.prefix_cache_slots);
}

TEST(NpuCpPlanTest, MixedPrefixAttentionMetadataSkipsPaddingSlots) {
  CpPlanInput input;
  input.q_seq_lens = {8, 4};
  input.position_ids =
      int32_tensor({256, 257, 258, 259, 260, 261, 262, 263, 0, 1, 2, 3});
  input.prefix_token_counts = {256, 0};
  input.block_tables = int32_tensor({5, 6, 9, 10}).view({2, 2});
  input.has_prefix_slots = true;
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());
  const CpAttentionMeta& attention = plan.attention_meta();

  const torch::Tensor prefix_indices =
      torch::cat({torch::arange(0, 128, torch::kInt32),
                  torch::arange(129, 257, torch::kInt32)});
  expect_tensor_bytes_equal(attention.prev_kv_gather_indices,
                            torch::cat({prefix_indices,
                                        torch::arange(258, 260, torch::kInt32),
                                        int32_tensor({266})}));
  expect_tensor_bytes_equal(
      attention.next_kv_gather_indices,
      torch::cat({prefix_indices, torch::arange(258, 270, torch::kInt32)}));
  expect_tensor_bytes_equal(attention.prev_key_cu_seq_lens,
                            int32_tensor({258, 259}));
  expect_tensor_bytes_equal(attention.next_key_cu_seq_lens,
                            int32_tensor({264, 268}));
  expect_tensor_bytes_equal(
      attention.prefix_cache_slots,
      torch::cat({torch::arange(640, 768, torch::kInt32), int32_tensor({0})}));
}

TEST(NpuCpPlanTest, PrefixCacheSlotsMatchLegacyBytesAcrossKvSplit) {
  CpPlanInput input;
  input.q_seq_lens = {8};
  input.position_ids = int32_tensor({256, 257, 258, 259, 260, 261, 262, 263});
  input.prefix_token_counts = {256};
  input.block_tables = int32_tensor({5, 6}).view({1, 2});
  input.has_prefix_slots = true;

  CpPlanConfig config = cp2_rank0_config();
  config.kv_split_size = 1;
  const NpuCpPlan plan = NpuCpPlan::build(input, config);
  expect_tensor_bytes_equal(plan.attention_meta().prefix_cache_slots,
                            torch::arange(640, 896, torch::kInt32));

  CpPlanInput empty_input = make_plan_input({}, {});
  empty_input.has_prefix_slots = true;
  empty_input.block_tables = torch::empty({0, 2}, torch::kInt32);
  const NpuCpPlan empty_plan =
      NpuCpPlan::build(empty_input, cp2_rank0_config());
  expect_tensor_bytes_equal(empty_plan.attention_meta().prefix_cache_slots,
                            int32_tensor({0}));
}

TEST(NpuCpPlanTest, DynamicEpMetadataMatchesLegacyBytes) {
  CpPlanConfig config = cp2_rank0_config();
  config.attention_tp_size = 2;
  config.moe_ep_size = 4;
  config.expert_parallel_degree = 2;
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), config);
  const CpEpMeta& cp_ep = plan.cp_ep_meta();

  expect_tensor_bytes_equal(cp_ep.attention_tp_unpadding_indices,
                            torch::arange(5, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.ffn_padding_indices,
                            torch::arange(5, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.dynamic_ep_indices,
                            torch::arange(40, torch::kInt32));
  expect_tensor_bytes_equal(cp_ep.moe_indices,
                            torch::arange(1, 321, torch::kInt32));
  expect_tensor_bytes_equal(
      cp_ep.expert_array,
      torch::ones({320, 1}, torch::dtype(torch::kBFloat16)));
}

TEST(NpuCpPlanTest, CpEpMetadataMatchesLegacyBuilderForConfigMatrix) {
  struct TestCase {
    const char* name;
    CpPlanInput input;
    CpPlanConfig config;
  };

  CpPlanConfig tp4_rank0 = cp2_rank0_config();
  tp4_rank0.attention_tp_size = 4;
  CpPlanConfig tp4_rank1 = tp4_rank0;
  tp4_rank1.attention_tp_rank = 1;

  CpPlanConfig cp4_rank3 = cp2_rank0_config();
  cp4_rank3.cp_size = 4;
  cp4_rank3.cp_rank = 3;
  cp4_rank3.kv_split_size = 2;
  cp4_rank3.attention_cp_size = 4;
  cp4_rank3.attention_cp_group_size = 4;

  CpPlanConfig dynamic_tp1 = cp2_rank0_config();
  dynamic_tp1.moe_ep_size = 4;
  dynamic_tp1.expert_parallel_degree = 2;
  CpPlanConfig dynamic_tp2 = dynamic_tp1;
  dynamic_tp2.attention_tp_size = 2;
  dynamic_tp2.attention_tp_rank = 1;
  dynamic_tp2.expert_parallel_degree = 3;

  const std::vector<TestCase> cases = {
      {"cp2_tp1", aligned_input(), cp2_rank0_config()},
      {"cp2_tp4_rank0", aligned_input(), tp4_rank0},
      {"cp2_tp4_rank1", aligned_input(), tp4_rank1},
      {"cp4_rank3", make_plan_input({16, 8, 24}, {0, 100, 200}), cp4_rank3},
      {"dynamic_ep_tp1", aligned_input(), dynamic_tp1},
      {"dynamic_ep_tp2_rank1", aligned_input(), dynamic_tp2},
      {"empty_shard", make_plan_input({}, {}), cp2_rank0_config()},
  };

  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const NpuCpPlan plan = NpuCpPlan::build(test_case.input, test_case.config);
    const CpEpMeta expected = build_legacy_cp_ep_meta(
        plan.local_padded_token_count(), test_case.config);
    expect_cp_ep_meta_bytes_equal(plan.cp_ep_meta(), expected);
  }
}

TEST(NpuCpPlanTest, CacheSlotsMatchTwoStageReferenceAcrossRanks) {
  struct TestCase {
    const char* name;
    std::vector<int32_t> q_seq_lens;
    int32_t cp_size;
    int32_t kv_split_size;
  };
  const std::vector<TestCase> cases = {
      {"aligned_cp2_kv2", {8, 12}, 2, 2},
      {"aligned_cp2_kv1", {8, 12}, 2, 1},
      {"non_aligned_cp2_kv2", {5, 7, 1, 3}, 2, 2},
      {"aligned_cp4_kv2", {16, 8, 24}, 4, 2},
      {"short_cp4_kv1", {1}, 4, 1},
  };

  for (const TestCase& test_case : cases) {
    const CpPlanInput input =
        make_plan_input(test_case.q_seq_lens,
                        std::vector<int32_t>(test_case.q_seq_lens.size(), 0));
    torch::Tensor global_slots =
        torch::arange(input.position_ids.numel(), torch::kInt32) * 137 + 1000;
    for (int32_t cp_rank = 0; cp_rank < test_case.cp_size; ++cp_rank) {
      SCOPED_TRACE(test_case.name);
      SCOPED_TRACE(cp_rank);
      CpPlanConfig config = cp2_rank0_config();
      config.cp_size = test_case.cp_size;
      config.cp_rank = cp_rank;
      config.kv_split_size = test_case.kv_split_size;
      config.kv_split_rank = cp_rank % test_case.kv_split_size;
      config.attention_cp_size = test_case.cp_size;
      config.attention_cp_group_size = test_case.cp_size;
      const NpuCpPlan plan = NpuCpPlan::build(input, config);

      expect_tensor_bytes_equal(
          plan.prepare_cache_slots(global_slots),
          prepare_cache_slots_reference(global_slots, plan, config));
    }
  }
}

TEST(NpuCpPlanTest, MtpTargetAndDraftShardGlobalInputsExactlyOnce) {
  const CpPlanInput mtp_input = make_plan_input({5, 7}, {0, 0});
  const NpuCpPlan target_plan = NpuCpPlan::build(mtp_input, cp2_rank0_config());
  const NpuCpPlan draft_plan = NpuCpPlan::build(mtp_input, cp2_rank0_config());
  torch::Tensor global_slots = torch::arange(12, torch::kInt32) + 1000;
  torch::Tensor target_slots = target_plan.prepare_cache_slots(global_slots);
  torch::Tensor draft_slots = draft_plan.prepare_cache_slots(global_slots);
  EXPECT_EQ(target_slots.numel(), target_plan.recovered_token_count());
  expect_tensor_bytes_equal(target_slots,
                            int32_tensor({488,
                                          489,
                                          490,
                                          491,
                                          492,
                                          -1,
                                          -1,
                                          -1,
                                          493,
                                          494,
                                          495,
                                          496,
                                          497,
                                          498,
                                          499,
                                          -1}));
  expect_tensor_bytes_equal(draft_slots, target_slots);

  torch::Tensor target_hidden = torch::arange(96, torch::kFloat).view({12, 8});
  torch::Tensor draft_hidden = target_hidden + 1000;
  CpInputShard target_shard =
      target_plan.shard_model_input(target_hidden, mtp_input.position_ids);
  CpInputShard draft_shard =
      draft_plan.shard_model_input(draft_hidden, mtp_input.position_ids);
  EXPECT_EQ(target_shard.hidden_states.size(0),
            target_plan.local_padded_token_count());
  EXPECT_EQ(draft_shard.hidden_states.size(0),
            draft_plan.local_padded_token_count());
  const torch::Tensor& destination_indices =
      target_plan.input_shard_meta().input_destination_indices;
  expect_tensor_bytes_equal(
      draft_shard.hidden_states.index_select(/*dim=*/0, destination_indices),
      target_shard.hidden_states.index_select(/*dim=*/0, destination_indices) +
          1000);
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(draft_plan.prepare_cache_slots(draft_slots),
               "global-real logical layout");
  EXPECT_DEATH(draft_plan.shard_model_input(draft_shard.hidden_states,
                                            draft_shard.position_ids),
               "exactly once");
#endif
}

TEST(NpuCpPlanTest, CumulativeHostLayoutIsPreserved) {
  CpPlanInput input = aligned_input();
  input.q_seq_lens_are_cumulative = true;
  input.kv_seq_lens_are_cumulative = true;
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());
  EXPECT_EQ(plan.attention_meta().host_q_seq_lens,
            std::vector<int32_t>({0, 4, 10}));
  EXPECT_EQ(plan.attention_meta().host_kv_seq_lens,
            std::vector<int32_t>({0, 4, 10}));
  expect_tensor_bytes_equal(plan.attention_meta().q_seq_lens,
                            int32_tensor({0, 4, 10}));
  expect_tensor_bytes_equal(plan.attention_meta().kv_seq_lens,
                            int32_tensor({0, 4, 10}));
  expect_tensor_bytes_equal(plan.attention_meta().q_cu_seq_lens,
                            int32_tensor({4, 10}));
}

TEST(NpuCpPlanTest, EmptyPlanDropsWorkerFakeModelRow) {
  const NpuCpPlan plan =
      NpuCpPlan::build(make_plan_input({}, {}), cp2_rank0_config());
  const torch::Tensor fake_hidden = torch::ones({1, 8}, torch::kFloat);
  const torch::Tensor fake_position = int32_tensor({0});
  const CpInputShard sharded =
      plan.shard_model_input(fake_hidden, fake_position);

  EXPECT_EQ(sharded.hidden_states.sizes(), torch::IntArrayRef({0, 8}));
  EXPECT_EQ(sharded.position_ids.numel(), 0);
}

TEST(NpuCpPlanTest, DisabledPlanIsNoOp) {
  const NpuCpPlan plan;
  torch::Tensor hidden = torch::randn({4, 8});
  torch::Tensor positions = torch::arange(4, torch::kInt32);
  torch::Tensor slots = torch::arange(4, torch::kInt32);
  CpInputShard sharded = plan.shard_model_input(hidden, positions);
  EXPECT_TRUE(sharded.hidden_states.is_same(hidden));
  EXPECT_TRUE(sharded.position_ids.is_same(positions));
  EXPECT_TRUE(plan.prepare_cache_slots(slots).is_same(slots));
  EXPECT_TRUE(plan.merge_model_output(hidden, nullptr).is_same(hidden));
}

}  // namespace
}  // namespace xllm
