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

#include "layers/common/dp_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>

#include "layers/mlu/tests_utils.h"

namespace xllm {
namespace layer {
namespace test {

ParallelArgs make_tp_args(std::unique_ptr<xllm::ProcessGroup>& process_group,
                          int32_t rank,
                          int32_t tp_size) {
  process_group = std::make_unique<test::MockProcessGroup>(
      torch::Device(torch::kCPU), rank, tp_size);
  ParallelArgs args(rank, tp_size, process_group.get());
  args.tp_group_ = process_group.get();
  return args;
}

ParallelArgs make_world_args(std::unique_ptr<xllm::ProcessGroup>& process_group,
                             std::unique_ptr<xllm::ProcessGroup>& tp_group,
                             int32_t global_rank,
                             int32_t dp_size,
                             int32_t tp_size) {
  process_group = std::make_unique<test::MockProcessGroup>(
      torch::Device(torch::kCPU), global_rank, dp_size * tp_size);
  tp_group = std::make_unique<test::MockProcessGroup>(
      torch::Device(torch::kCPU), global_rank % tp_size, tp_size);
  ParallelArgs args(
      global_rank, dp_size * tp_size, dp_size, process_group.get());
  args.tp_group_ = tp_group.get();
  return args;
}

struct GatherShape {
  int64_t padded_dp_tokens = 0;
  int64_t shard_tokens = 0;
};

GatherShape make_gather_shape(const std::vector<int32_t>& dp_tokens,
                              int32_t tp_size) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, tp_size);
  int64_t padded_dp_tokens = get_dp_gather_tokens(dp_tokens, args);
  return {padded_dp_tokens, padded_dp_tokens / tp_size};
}

std::vector<torch::Tensor> make_global_shards(
    const std::vector<int32_t>& dp_tokens,
    int32_t tp_size) {
  GatherShape shape = make_gather_shape(dp_tokens, tp_size);
  std::vector<torch::Tensor> shards;
  for (size_t dp_rank = 0; dp_rank < dp_tokens.size(); ++dp_rank) {
    std::vector<float> padded_vals(shape.padded_dp_tokens, -1.0f);
    for (int64_t i = 0; i < dp_tokens[dp_rank]; ++i) {
      padded_vals[i] = static_cast<float>(dp_rank * 100 + i);
    }
    auto padded =
        torch::tensor(padded_vals).reshape({shape.padded_dp_tokens, 1});
    auto split = padded.split(shape.shard_tokens, 0);
    for (int32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
      shards.push_back(split[tp_rank].clone());
    }
  }
  return shards;
}

TEST(DpUtilsTest, RsAttnInputNoPadWhenTpOne) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 1);

  torch::Tensor x = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  torch::Tensor residual = torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}});

  auto rs_result =
      reduce_scatter_attn_input(x, residual, /*target_tokens=*/2, args);

  EXPECT_FALSE(rs_result.second.active);
  EXPECT_EQ(rs_result.second.original_tokens, 2);
  EXPECT_EQ(rs_result.second.padded_tokens, 2);
  test::verify_tensor_close(rs_result.first,
                            torch::tensor({{11.0f, 22.0f}, {33.0f, 44.0f}}));
}

TEST(DpUtilsTest, GetRsTokensKeepsTpOneShape) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 1);

  EXPECT_EQ(get_reduce_scatter_tokens(/*num_tokens=*/5, args), 5);
}

TEST(DpUtilsTest, GetRsTokensAlignsTpShard) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 4);

  EXPECT_EQ(get_reduce_scatter_tokens(/*num_tokens=*/2, args), 4);
  EXPECT_EQ(get_reduce_scatter_tokens(/*num_tokens=*/5, args), 8);
  EXPECT_EQ(get_reduce_scatter_tokens(/*num_tokens=*/8, args), 8);
}

TEST(DpUtilsTest, PadTokensPreservesTrailingDims) {
  auto x = torch::arange(12, torch::kFloat32).reshape({2, 2, 3});

  auto [padded, info] = pad_tokens(x, /*target_tokens=*/4);

  EXPECT_TRUE(info.active);
  EXPECT_EQ(info.original_tokens, 2);
  EXPECT_EQ(info.padded_tokens, 4);
  EXPECT_EQ(padded.sizes().vec(), std::vector<int64_t>({4, 2, 3}));
  test::verify_tensor_close(padded.slice(0, 0, 2), x);
  test::verify_tensor_close(padded.slice(0, 2, 4),
                            torch::zeros({2, 2, 3}, x.options()));
}

TEST(DpUtilsTest, RsAttnInputPadsWhenTokensLtTp) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 4);

  torch::Tensor x = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  torch::Tensor residual = torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}});

  auto rs_result =
      reduce_scatter_attn_input(x, residual, /*target_tokens=*/4, args);

  EXPECT_TRUE(rs_result.second.active);
  EXPECT_EQ(rs_result.second.original_tokens, 2);
  EXPECT_EQ(rs_result.second.padded_tokens, 4);
  test::verify_tensor_close(rs_result.first, torch::tensor({{11.0f, 22.0f}}));
}

TEST(DpUtilsTest, RsAttnInputPadsToAlignedTokens) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 1, 4);

  torch::Tensor x = torch::tensor(
      {{1.0f, 1.5f}, {2.0f, 2.5f}, {3.0f, 3.5f}, {4.0f, 4.5f}, {5.0f, 5.5f}});
  torch::Tensor residual = torch::full({5, 2}, 100.0f);

  auto rs_result =
      reduce_scatter_attn_input(x, residual, /*target_tokens=*/8, args);

  EXPECT_TRUE(rs_result.second.active);
  EXPECT_EQ(rs_result.second.original_tokens, 5);
  EXPECT_EQ(rs_result.second.padded_tokens, 8);
  test::verify_tensor_close(rs_result.first,
                            torch::tensor({{3.0f, 3.5f}, {4.0f, 4.5f}}));
}

TEST(DpUtilsTest, GetDpGatherTokensAlignsMixedDpTokens) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 4);

  EXPECT_EQ(get_dp_gather_tokens({513, 1, 128}, args), 516);
}

TEST(DpUtilsTest, GetDpGatherTokensKeepsShardTokensPositive) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 4);
  GatherShape shape = make_gather_shape({0, 1, 33}, /*tp_size=*/4);

  EXPECT_EQ(get_dp_gather_tokens({0, 1, 33}, args), 36);
  EXPECT_EQ(shape.shard_tokens, 9);
  EXPECT_GT(shape.shard_tokens, 0);
}

TEST(DpUtilsTest, GatherGlobalTokensTpOneHandlesEqualDpTokens) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  std::unique_ptr<xllm::ProcessGroup> tp_group;
  ParallelArgs args =
      make_world_args(process_group, tp_group, /*global_rank=*/2, 3, 1);
  const std::vector<int32_t> dp_tokens = {4, 4, 4};

  auto* mock_pg = dynamic_cast<test::MockProcessGroup*>(process_group.get());
  ASSERT_NE(mock_pg, nullptr);
  auto shards = make_global_shards(dp_tokens, /*tp_size=*/1);
  mock_pg->set_allgather_outputs(shards);

  auto output = gather_global_tokens(shards[2], dp_tokens, args);

  EXPECT_EQ(output.size(0), 12);
  test::verify_tensor_close(output,
                            torch::tensor({{0.0f},
                                           {1.0f},
                                           {2.0f},
                                           {3.0f},
                                           {100.0f},
                                           {101.0f},
                                           {102.0f},
                                           {103.0f},
                                           {200.0f},
                                           {201.0f},
                                           {202.0f},
                                           {203.0f}}));
}

TEST(DpUtilsTest, GatherGlobalTokensTpOneHandlesUnevenDpTokens) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  std::unique_ptr<xllm::ProcessGroup> tp_group;
  ParallelArgs args =
      make_world_args(process_group, tp_group, /*global_rank=*/1, 3, 1);
  const std::vector<int32_t> dp_tokens = {4, 0, 2};

  auto* mock_pg = dynamic_cast<test::MockProcessGroup*>(process_group.get());
  ASSERT_NE(mock_pg, nullptr);
  auto shards = make_global_shards(dp_tokens, /*tp_size=*/1);
  mock_pg->set_allgather_outputs(shards);

  auto output = gather_global_tokens(shards[1], dp_tokens, args);

  EXPECT_EQ(output.size(0), 6);
  test::verify_tensor_close(
      output,
      torch::tensor({{0.0f}, {1.0f}, {2.0f}, {3.0f}, {200.0f}, {201.0f}}));
}

TEST(DpUtilsTest, GatherGlobalTokensRestoresDpMajorOrder) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  std::unique_ptr<xllm::ProcessGroup> tp_group;
  ParallelArgs args =
      make_world_args(process_group, tp_group, /*global_rank=*/1, 3, 2);
  const std::vector<int32_t> dp_tokens = {5, 2, 4};

  auto* mock_pg = dynamic_cast<test::MockProcessGroup*>(process_group.get());
  ASSERT_NE(mock_pg, nullptr);
  auto shards = make_global_shards(dp_tokens, /*tp_size=*/2);
  mock_pg->set_allgather_outputs(shards);

  auto output = gather_global_tokens(shards[1], dp_tokens, args);

  test::verify_tensor_close(output,
                            torch::tensor({{0.0f},
                                           {1.0f},
                                           {2.0f},
                                           {3.0f},
                                           {4.0f},
                                           {100.0f},
                                           {101.0f},
                                           {200.0f},
                                           {201.0f},
                                           {202.0f},
                                           {203.0f}}));
}

TEST(DpUtilsTest, GatherGlobalTokensDropsZeroTokenDpRank) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  std::unique_ptr<xllm::ProcessGroup> tp_group;
  ParallelArgs args =
      make_world_args(process_group, tp_group, /*global_rank=*/6, 3, 4);
  const std::vector<int32_t> dp_tokens = {0, 1, 33};

  auto* mock_pg = dynamic_cast<test::MockProcessGroup*>(process_group.get());
  ASSERT_NE(mock_pg, nullptr);
  auto shards = make_global_shards(dp_tokens, /*tp_size=*/4);
  mock_pg->set_allgather_outputs(shards);

  auto output = gather_global_tokens(shards[6], dp_tokens, args);

  EXPECT_EQ(output.size(0), 34);
  test::verify_tensor_close(
      output.slice(0, 0, 5),
      torch::tensor({{100.0f}, {200.0f}, {201.0f}, {202.0f}, {203.0f}}));
  test::verify_tensor_close(
      output.slice(0, 30, 34),
      torch::tensor({{229.0f}, {230.0f}, {231.0f}, {232.0f}}));
}

TEST(DpUtilsTest, GatherGlobalTokensHandlesEqualDpTokens) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  std::unique_ptr<xllm::ProcessGroup> tp_group;
  ParallelArgs args =
      make_world_args(process_group, tp_group, /*global_rank=*/4, 3, 2);
  const std::vector<int32_t> dp_tokens = {7, 7, 7};

  auto* mock_pg = dynamic_cast<test::MockProcessGroup*>(process_group.get());
  ASSERT_NE(mock_pg, nullptr);
  auto shards = make_global_shards(dp_tokens, /*tp_size=*/2);
  mock_pg->set_allgather_outputs(shards);

  auto output = gather_global_tokens(shards[4], dp_tokens, args);

  EXPECT_EQ(output.size(0), 21);
  test::verify_tensor_close(output.slice(0, 0, 3),
                            torch::tensor({{0.0f}, {1.0f}, {2.0f}}));
  test::verify_tensor_close(output.slice(0, 7, 10),
                            torch::tensor({{100.0f}, {101.0f}, {102.0f}}));
  test::verify_tensor_close(output.slice(0, 18, 21),
                            torch::tensor({{204.0f}, {205.0f}, {206.0f}}));
}

TEST(DpUtilsTest, NeedDpMoEGatherSkipsAll2AllPath) {
  std::unique_ptr<xllm::ProcessGroup> process_group;
  ParallelArgs args = make_tp_args(process_group, 0, 1);
  args.dp_size_ = 2;
  args.ep_size_ = 2;

  EXPECT_TRUE(need_dp_moe_gather(args, /*enable_moe_all2all=*/false));
  EXPECT_FALSE(need_dp_moe_gather(args, /*enable_moe_all2all=*/true));
}

TEST(DpUtilsTest, UnpadTokensRestoresOriginalLength) {
  auto x = torch::arange(12, torch::kFloat32).reshape({4, 3});
  PaddingInfo info;
  info.original_tokens = 2;
  info.padded_tokens = 4;
  info.active = true;

  auto output = unpad_tokens(x, info);

  test::verify_tensor_close(output, x.slice(0, 0, 2));
}

TEST(DpUtilsTest, AllDpRanksAreDecodeNeedsEveryRankDecode) {
  ModelInputParams decode_params;
  decode_params.parallel.dp_is_decode = {1, 1, 1};
  EXPECT_TRUE(all_dp_ranks_are_decode(decode_params));

  ModelInputParams mixed_params;
  mixed_params.parallel.dp_is_decode = {1, 0, 1};
  EXPECT_FALSE(all_dp_ranks_are_decode(mixed_params));
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
