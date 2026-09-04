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

#include "layers/npu_torch/dcp_attention_utils.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace xllm::layer::test {
namespace {

std::pair<torch::Tensor, torch::Tensor> compute_attention_partial(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.sizes(), key.sizes());
  CHECK_EQ(query.size(1), key.size(1));
  CHECK_EQ(query.size(2), key.size(2));

  const double scale = 1.0 / std::sqrt(static_cast<double>(query.size(2)));
  const torch::Tensor scores =
      torch::einsum("qhd,khd->qhk", {query, key}) * scale;
  const torch::Tensor partial_lse = torch::logsumexp(scores, -1, true);
  const torch::Tensor partial_out =
      torch::einsum("qhk,khd->qhd", {torch::softmax(scores, -1), value});
  return {partial_out, partial_lse};
}

TEST(DcpAttentionUtilsTest, ShardedKvMergeMatchesFullAttentionInFloat64) {
  torch::manual_seed(7);
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor query = torch::randn({5, 3, 8}, options);
  const torch::Tensor key = torch::randn({12, 3, 8}, options);
  const torch::Tensor value = torch::randn({12, 3, 8}, options);
  const auto [reference_out, reference_lse] =
      compute_attention_partial(query, key, value);
  (void)reference_lse;

  const std::vector<torch::Tensor> key_shards = key.chunk(3, 0);
  const std::vector<torch::Tensor> value_shards = value.chunk(3, 0);
  ASSERT_EQ(key_shards.size(), value_shards.size());
  std::vector<torch::Tensor> partial_outputs;
  std::vector<torch::Tensor> partial_lses;
  partial_outputs.reserve(key_shards.size());
  partial_lses.reserve(key_shards.size());
  for (int64_t shard_index = 0;
       shard_index < static_cast<int64_t>(key_shards.size());
       ++shard_index) {
    const auto [partial_out, partial_lse] = compute_attention_partial(
        query, key_shards[shard_index], value_shards[shard_index]);
    partial_outputs.emplace_back(partial_out);
    partial_lses.emplace_back(partial_lse);
  }

  const torch::Tensor merged_out = detail::merge_dcp_partials(
      torch::stack(partial_outputs, 0), torch::stack(partial_lses, 0));
  EXPECT_TRUE(torch::allclose(
      merged_out, reference_out, /*rtol=*/1e-10, /*atol=*/1e-10));
}

TEST(DcpAttentionUtilsTest, InvalidLseShardIsIgnored) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor valid_out =
      torch::tensor({3.0, 7.0}, options).view({1, 2, 1});
  const torch::Tensor invalid_out =
      torch::full_like(valid_out, std::numeric_limits<double>::quiet_NaN());
  const torch::Tensor valid_lse = torch::zeros({1, 2, 1}, options);
  const torch::Tensor invalid_lse =
      torch::full_like(valid_lse, -std::numeric_limits<double>::infinity());

  const torch::Tensor merged_out =
      detail::merge_dcp_partials(torch::stack({invalid_out, valid_out}, 0),
                                 torch::stack({invalid_lse, valid_lse}, 0));
  EXPECT_TRUE(torch::equal(merged_out, valid_out));
}

TEST(DcpAttentionUtilsTest, AllInvalidLseShardsProduceZero) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
  const torch::Tensor partial_out = torch::full(
      {4, 2, 3, 5}, std::numeric_limits<double>::quiet_NaN(), options);
  const torch::Tensor partial_lse = torch::full(
      {4, 2, 3, 1}, -std::numeric_limits<double>::infinity(), options);

  const torch::Tensor merged_out =
      detail::merge_dcp_partials(partial_out, partial_lse);
  EXPECT_TRUE(torch::equal(merged_out, torch::zeros_like(merged_out)));
}

TEST(DcpAttentionUtilsTest, DistributesPartialTailAcrossFourRanks) {
  std::vector<int64_t> local_kv_seq_lens;
  local_kv_seq_lens.reserve(4);
  for (int32_t dcp_rank = 0; dcp_rank < 4; ++dcp_rank) {
    const std::vector<int64_t> rank_local_kv_seq_lens =
        detail::compute_dcp_local_kv_seq_lens(
            /*global_kv_seq_lens=*/{257},
            /*dcp_size=*/4,
            dcp_rank,
            /*block_size=*/128);
    ASSERT_EQ(rank_local_kv_seq_lens.size(), 1);
    local_kv_seq_lens.emplace_back(rank_local_kv_seq_lens.front());
  }

  EXPECT_EQ(local_kv_seq_lens, (std::vector<int64_t>{128, 128, 1, 0}));
}

TEST(DcpAttentionUtilsTest, ContextLenIsKvMinusCurrentChunkQuery) {
  // Two requests packed into one chunked batch: request 0 has 130 query tokens
  // over a 130 KV (no cached context, first chunk); request 1 has 6 query
  // tokens over a 262 KV (256 cached context + 6 current chunk).
  const std::vector<int64_t> context_lens = detail::compute_dcp_context_lens(
      /*q_cu_seq_lens=*/{130, 136},
      /*global_kv_seq_lens=*/{130, 262});
  EXPECT_EQ(context_lens, (std::vector<int64_t>{0, 256}));
}

TEST(DcpAttentionUtilsTest, ContextShardLenReadsOnlyCachedContext) {
  // A request with 256 cached context + a current chunk: the local context
  // shard length must be derived from context_len (256), not the full KV, so
  // the context part never reads the current chunk's own KV.
  std::vector<int64_t> context_shard_lens;
  context_shard_lens.reserve(2);
  for (int32_t dcp_rank = 0; dcp_rank < 2; ++dcp_rank) {
    const std::vector<int64_t> rank_shard =
        detail::compute_dcp_local_kv_seq_lens(
            /*global_kv_seq_lens=*/{256},
            /*dcp_size=*/2,
            dcp_rank,
            /*block_size=*/128);
    ASSERT_EQ(rank_shard.size(), 1);
    context_shard_lens.emplace_back(rank_shard.front());
  }
  EXPECT_EQ(context_shard_lens, (std::vector<int64_t>{128, 128}));
}

TEST(DcpAttentionUtilsTest, CoversPhase4LocalContextContracts) {
  const std::vector<int64_t> context_lens = {128, 256, 512};
  std::vector<std::vector<int64_t>> local_context_lens_by_rank;
  local_context_lens_by_rank.reserve(2);
  for (int32_t dcp_rank = 0; dcp_rank < 2; ++dcp_rank) {
    local_context_lens_by_rank.emplace_back(
        detail::compute_dcp_local_kv_seq_lens(context_lens,
                                              /*dcp_size=*/2,
                                              dcp_rank,
                                              /*block_size=*/128));
  }

  ASSERT_EQ(local_context_lens_by_rank.size(), 2);
  EXPECT_EQ(local_context_lens_by_rank[0],
            (std::vector<int64_t>{128, 128, 256}));
  EXPECT_EQ(local_context_lens_by_rank[1], (std::vector<int64_t>{0, 128, 256}));
}

TEST(DcpAttentionUtilsTest, GraphZeroShardMaskTracksTensorKvLengthChanges) {
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  torch::Tensor global_kv_seq_lens = torch::tensor(
      {128}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  torch::Tensor partial_out = torch::full({1, 2, 3}, 7.0, fp32_options);
  torch::Tensor partial_lse = torch::full({1, 2, 1}, 9.0, fp32_options);

  detail::normalize_zero_dcp_partials_for_graph(partial_out,
                                                partial_lse,
                                                global_kv_seq_lens,
                                                /*dcp_rank=*/1,
                                                /*block_size=*/128);
  EXPECT_TRUE(torch::equal(partial_out, torch::zeros_like(partial_out)));
  EXPECT_TRUE(torch::equal(
      partial_lse,
      torch::full_like(partial_lse, -std::numeric_limits<float>::infinity())));

  global_kv_seq_lens.fill_(129);
  partial_out.fill_(7.0);
  partial_lse.fill_(9.0);
  detail::normalize_zero_dcp_partials_for_graph(partial_out,
                                                partial_lse,
                                                global_kv_seq_lens,
                                                /*dcp_rank=*/1,
                                                /*block_size=*/128);
  EXPECT_TRUE(torch::equal(partial_out, torch::full_like(partial_out, 7.0)));
  EXPECT_TRUE(torch::equal(partial_lse, torch::full_like(partial_lse, 9.0)));
}

TEST(DcpAttentionUtilsTest, ValidateChunkedLengthsAcceptsMultiTokenRequests) {
  const std::vector<int64_t> normalized_q_cu_seq_lens =
      detail::validate_dcp_chunked_lengths(
          /*q_cu_seq_lens=*/{130, 136},
          /*global_kv_seq_lens=*/{130, 262},
          /*token_count=*/136);
  EXPECT_EQ(normalized_q_cu_seq_lens, (std::vector<int64_t>{130, 136}));
}

TEST(DcpAttentionUtilsTest, NormalizesLeadingZeroBeforeValidation) {
  const std::vector<int64_t> normalized_q_cu_seq_lens =
      detail::validate_dcp_chunked_lengths(
          /*q_cu_seq_lens=*/{0, 130, 136},
          /*global_kv_seq_lens=*/{130, 262},
          /*token_count=*/136);
  EXPECT_EQ(normalized_q_cu_seq_lens, (std::vector<int64_t>{130, 136}));
  EXPECT_EQ(detail::compute_dcp_context_lens(normalized_q_cu_seq_lens,
                                             /*global_kv_seq_lens=*/{130, 262}),
            (std::vector<int64_t>{0, 256}));
}

TEST(DcpAttentionUtilsTest, ValidateChunkedLengthsRejectsTokenCountMismatch) {
  EXPECT_DEATH(detail::validate_dcp_chunked_lengths(
                   /*q_cu_seq_lens=*/{130, 136},
                   /*global_kv_seq_lens=*/{130, 262},
                   /*token_count=*/135),
               "query tokens");
}

TEST(DcpAttentionUtilsTest, ValidateChunkedLengthsRejectsKvShorterThanQuery) {
  EXPECT_DEATH(detail::validate_dcp_chunked_lengths(
                   /*q_cu_seq_lens=*/{10},
                   /*global_kv_seq_lens=*/{4},
                   /*token_count=*/10),
               "cover the current chunk query");
}

TEST(DcpAttentionUtilsTest, ValidateChunkedLengthsRejectsZeroQueryRequest) {
  EXPECT_DEATH(detail::validate_dcp_chunked_lengths(
                   /*q_cu_seq_lens=*/{0, 0, 1},
                   /*global_kv_seq_lens=*/{0, 1},
                   /*token_count=*/1),
               "at least one query token");
}

TEST(DcpAttentionUtilsTest, ValidateChunkedLengthsRejectsQueryAboveMaskLimit) {
  EXPECT_DEATH(
      detail::validate_dcp_chunked_lengths(
          /*q_cu_seq_lens=*/{detail::kMaxDcpChunkedPrefillQueryLen + 1},
          /*global_kv_seq_lens=*/{detail::kMaxDcpChunkedPrefillQueryLen + 1},
          /*token_count=*/
          detail::kMaxDcpChunkedPrefillQueryLen + 1),
      "does not yet support chunked query length above");
}

TEST(DcpAttentionUtilsTest, BlockTableRowsMatchRequestsNotQueryTokens) {
  const torch::Tensor local_block_table = torch::tensor(
      {{37, 89}, {41, 73}}, torch::TensorOptions().dtype(torch::kInt64));
  detail::validate_dcp_chunked_block_table(local_block_table,
                                           /*request_count=*/2);
  EXPECT_DEATH(detail::validate_dcp_chunked_block_table(local_block_table,
                                                        /*request_count=*/136),
               "request count");
}

TEST(DcpAttentionUtilsTest, NormalizeChunkedZeroesEmptyContextTokenRange) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  // token_count=5: request 0 owns tokens [0,2) with empty context on this rank,
  // request 1 owns tokens [2,5) with a non-empty context shard.
  torch::Tensor partial_out = torch::ones({5, 2, 3}, options);
  torch::Tensor partial_lse = torch::ones({5, 2, 1}, options);

  detail::normalize_zero_dcp_chunked_partials(partial_out,
                                              partial_lse,
                                              /*local_context_lens=*/{0, 64},
                                              /*q_cu_seq_lens=*/{2, 5});

  EXPECT_TRUE(torch::equal(partial_out.narrow(0, 0, 2),
                           torch::zeros({2, 2, 3}, options)));
  EXPECT_TRUE(torch::equal(partial_out.narrow(0, 2, 3),
                           torch::ones({3, 2, 3}, options)));
  const torch::Tensor expected_neg_inf_lse =
      torch::full({2, 2, 1}, -std::numeric_limits<float>::infinity(), options);
  EXPECT_TRUE(torch::equal(partial_lse.narrow(0, 0, 2), expected_neg_inf_lse));
  EXPECT_TRUE(torch::equal(partial_lse.narrow(0, 2, 3),
                           torch::ones({3, 2, 1}, options)));
}

}  // namespace
}  // namespace xllm::layer::test
