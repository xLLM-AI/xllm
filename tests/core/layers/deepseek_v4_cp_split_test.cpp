/* Copyright 2025-2026 The xLLM Authors.

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

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "layers/npu_torch/deepseek_v4_cp_context.h"

namespace xllm::layer::v4_cp {
namespace {

int64_t total_tokens(const std::vector<int32_t>& q_seq_lens) {
  return std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0});
}

// Every global row must be owned by exactly one rank. A duplicated row would be
// written to the KV cache twice; a dropped one would leave a zero row in the
// gathered output, and neither shows up as a crash.
TEST(DeepseekV4CpSplit, PartitionsEveryRowExactlyOnce) {
  const std::vector<std::vector<int32_t>> cases = {
      {8}, {8, 4}, {1}, {7, 3, 11}, {128, 1, 1, 64}, {0, 5}};
  for (const auto& q_seq_lens : cases) {
    for (const int32_t cp_size : {2, 3, 4, 8}) {
      const auto rows_by_rank = compute_cp_rows_by_rank(cp_size, q_seq_lens);
      ASSERT_EQ(static_cast<int32_t>(rows_by_rank.size()), cp_size);

      std::vector<int64_t> seen;
      for (const auto& rows : rows_by_rank) {
        seen.insert(seen.end(), rows.begin(), rows.end());
      }
      std::sort(seen.begin(), seen.end());
      const int64_t expected = total_tokens(q_seq_lens);
      ASSERT_EQ(static_cast<int64_t>(seen.size()), expected)
          << "cp_size=" << cp_size;
      for (int64_t i = 0; i < expected; ++i) {
        EXPECT_EQ(seen[i], i) << "cp_size=" << cp_size << ", row " << i;
      }
    }
  }
}

// Each rank's rows must be contiguous per sequence -- that is the property that
// lets one sequence stay at a single metadata row.
TEST(DeepseekV4CpSplit, RowsAreContiguousWithinEachSequence) {
  const std::vector<int32_t> q_seq_lens = {10, 5, 7};
  const int32_t cp_size = 3;
  const auto rows_by_rank = compute_cp_rows_by_rank(cp_size, q_seq_lens);

  for (int32_t r = 0; r < cp_size; ++r) {
    int64_t seq_base = 0;
    size_t cursor = 0;
    for (const int32_t q_i : q_seq_lens) {
      const int32_t seg = (q_i + cp_size - 1) / cp_size;
      const int32_t start = std::min(r * seg, q_i);
      const int32_t end = std::min(start + seg, q_i);
      for (int32_t j = start; j < end; ++j) {
        ASSERT_LT(cursor, rows_by_rank[r].size());
        EXPECT_EQ(rows_by_rank[r][cursor], seq_base + j);
        ++cursor;
      }
      seq_base += q_i;
    }
    EXPECT_EQ(cursor, rows_by_rank[r].size());
  }
}

// Sequences shorter than cp_size leave the higher ranks empty. Chunked prefill
// schedules such chunks routinely, so an empty segment must stay legal.
TEST(DeepseekV4CpSplit, ShortSequenceLeavesTrailingRanksEmpty) {
  const std::vector<int32_t> q_seq_lens = {2};
  const int32_t cp_size = 4;
  const auto rows_by_rank = compute_cp_rows_by_rank(cp_size, q_seq_lens);

  ASSERT_EQ(rows_by_rank.size(), 4u);
  // seg = ceil(2/4) = 1, so ranks 0 and 1 take one row each.
  EXPECT_EQ(rows_by_rank[0], std::vector<int64_t>({0}));
  EXPECT_EQ(rows_by_rank[1], std::vector<int64_t>({1}));
  EXPECT_TRUE(rows_by_rank[2].empty());
  EXPECT_TRUE(rows_by_rank[3].empty());
}

// cp_size == 1 must be the identity split, so the non-CP path is unaffected.
TEST(DeepseekV4CpSplit, SingleRankKeepsAllRowsInOrder) {
  const std::vector<int32_t> q_seq_lens = {4, 6};
  const auto rows_by_rank = compute_cp_rows_by_rank(/*cp_size=*/1, q_seq_lens);

  ASSERT_EQ(rows_by_rank.size(), 1u);
  ASSERT_EQ(static_cast<int64_t>(rows_by_rank[0].size()),
            total_tokens(q_seq_lens));
  for (size_t i = 0; i < rows_by_rank[0].size(); ++i) {
    EXPECT_EQ(rows_by_rank[0][i], static_cast<int64_t>(i));
  }
}

// The rank-major gather order and the restore permutation must be mutual
// inverses, otherwise merged hidden states come back scrambled.
TEST(DeepseekV4CpSplit, RestorePermutationInvertsGatherOrder) {
  const std::vector<int32_t> q_seq_lens = {9, 4};
  const int32_t cp_size = 3;
  const auto rows_by_rank = compute_cp_rows_by_rank(cp_size, q_seq_lens);
  const int64_t expected = total_tokens(q_seq_lens);

  // Mirror build_deepseek_v4_cp_context's inversion.
  std::vector<int64_t> restore(static_cast<size_t>(expected), -1);
  int64_t gathered_pos = 0;
  for (int32_t r = 0; r < cp_size; ++r) {
    for (const int64_t global_row : rows_by_rank[r]) {
      restore[static_cast<size_t>(global_row)] = gathered_pos++;
    }
  }
  ASSERT_EQ(gathered_pos, expected);

  // restore must be a permutation of [0, expected).
  std::set<int64_t> unique(restore.begin(), restore.end());
  EXPECT_EQ(static_cast<int64_t>(unique.size()), expected);
  for (const int64_t v : restore) {
    EXPECT_GE(v, 0);
    EXPECT_LT(v, expected);
  }

  // gathered[restore[i]] == global row i, so applying restore to a gathered
  // buffer that holds its own global row ids yields the identity.
  std::vector<int64_t> gathered(static_cast<size_t>(expected), -1);
  gathered_pos = 0;
  for (int32_t r = 0; r < cp_size; ++r) {
    for (const int64_t global_row : rows_by_rank[r]) {
      gathered[static_cast<size_t>(gathered_pos++)] = global_row;
    }
  }
  for (int64_t i = 0; i < expected; ++i) {
    const auto slot = static_cast<size_t>(restore[static_cast<size_t>(i)]);
    EXPECT_EQ(gathered[slot], i);
  }
}

// The kv extent handed to attention must end where this rank's query rows end.
// A global kv length instead relocates every non-last rank's queries to the
// tail of the sequence, because sparse_attn_sharedkv aligns the query block to
// the end of the kv window. That is an accuracy bug with no crash.
TEST(DeepseekV4CpLocalSeqLens, KvExtentEndsAtThisRanksLastRow) {
  // One 145-token prompt, no cached prefix, cp=2 -> seg = 73.
  const std::vector<int32_t> q_seq_lens = {145};
  const std::vector<int32_t> kv_seq_lens = {145};
  const int32_t cp_size = 2;

  std::vector<int32_t> local_q;
  std::vector<int32_t> local_kv;

  compute_cp_local_seq_lens(
      cp_size, /*cp_rank=*/0, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
  EXPECT_EQ(local_q, std::vector<int32_t>({73}));
  EXPECT_EQ(local_kv, std::vector<int32_t>({73}));

  compute_cp_local_seq_lens(
      cp_size, /*cp_rank=*/1, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
  EXPECT_EQ(local_q, std::vector<int32_t>({72}));
  // The last rank alone sees the full sequence.
  EXPECT_EQ(local_kv, std::vector<int32_t>({145}));
}

// Chunked prefill and prefix cache put a cached prefix in front of the query
// rows. The prefix belongs to every rank's window; only the tail is split.
TEST(DeepseekV4CpLocalSeqLens, CachedPrefixStaysInEveryRanksWindow) {
  // 20 new query rows on top of an 80-token cached prefix, cp=2 -> seg = 10.
  const std::vector<int32_t> q_seq_lens = {20};
  const std::vector<int32_t> kv_seq_lens = {100};
  const int32_t cp_size = 2;

  std::vector<int32_t> local_q;
  std::vector<int32_t> local_kv;

  compute_cp_local_seq_lens(
      cp_size, /*cp_rank=*/0, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
  EXPECT_EQ(local_q, std::vector<int32_t>({10}));
  EXPECT_EQ(local_kv, std::vector<int32_t>({90}));

  compute_cp_local_seq_lens(
      cp_size, /*cp_rank=*/1, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
  EXPECT_EQ(local_q, std::vector<int32_t>({10}));
  EXPECT_EQ(local_kv, std::vector<int32_t>({100}));
}

// A rank whose segment is empty must still advertise a legal window: it closes
// where its rows would have started, and never reaches past the global length.
TEST(DeepseekV4CpLocalSeqLens, EmptySegmentKeepsWindowLegal) {
  const std::vector<int32_t> q_seq_lens = {2};
  const std::vector<int32_t> kv_seq_lens = {2};
  const int32_t cp_size = 4;  // seg = 1, so ranks 2 and 3 are empty.

  std::vector<int32_t> local_q;
  std::vector<int32_t> local_kv;
  for (int32_t rank = 0; rank < cp_size; ++rank) {
    compute_cp_local_seq_lens(
        cp_size, rank, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
    if (rank < 2) {
      EXPECT_EQ(local_q[0], 1) << "rank=" << rank;
      EXPECT_EQ(local_kv[0], rank + 1) << "rank=" << rank;
    } else {
      EXPECT_EQ(local_q[0], 0) << "rank=" << rank;
      // No rows, so the window closes right where the rows ran out.
      EXPECT_EQ(local_kv[0], 2) << "rank=" << rank;
    }
  }
}

// cp_size == 1 must leave both axes at their global values, so the non-CP path
// is bit-identical.
TEST(DeepseekV4CpLocalSeqLens, SingleRankIsIdentity) {
  const std::vector<int32_t> q_seq_lens = {4, 6};
  const std::vector<int32_t> kv_seq_lens = {4, 30};

  std::vector<int32_t> local_q;
  std::vector<int32_t> local_kv;
  compute_cp_local_seq_lens(
      /*cp_size=*/1,
      /*cp_rank=*/0,
      q_seq_lens,
      kv_seq_lens,
      &local_q,
      &local_kv);

  EXPECT_EQ(local_q, q_seq_lens);
  EXPECT_EQ(local_kv, kv_seq_lens);
}

// Multi-sequence batches must stay per-row consistent: local_kv[i] - local_q[i]
// is the position of this rank's first row, which is what start_pos-style
// reasoning downstream assumes.
TEST(DeepseekV4CpLocalSeqLens, WindowMinusQueriesIsThisRanksFirstPosition) {
  const std::vector<int32_t> q_seq_lens = {10, 5, 7};
  const std::vector<int32_t> kv_seq_lens = {10, 25, 7};
  const int32_t cp_size = 3;

  std::vector<int32_t> local_q;
  std::vector<int32_t> local_kv;
  for (int32_t rank = 0; rank < cp_size; ++rank) {
    compute_cp_local_seq_lens(
        cp_size, rank, q_seq_lens, kv_seq_lens, &local_q, &local_kv);
    ASSERT_EQ(local_q.size(), q_seq_lens.size());
    ASSERT_EQ(local_kv.size(), q_seq_lens.size());
    for (size_t i = 0; i < q_seq_lens.size(); ++i) {
      const int32_t q_i = q_seq_lens[i];
      const int32_t seg = (q_i + cp_size - 1) / cp_size;
      const int32_t start = std::min(rank * seg, q_i);
      const int32_t prefix = kv_seq_lens[i] - q_i;
      EXPECT_EQ(local_kv[i] - local_q[i], prefix + start)
          << "rank=" << rank << ", seq=" << i;
      EXPECT_LE(local_kv[i], kv_seq_lens[i]) << "rank=" << rank;
    }
  }
}

}  // namespace
}  // namespace xllm::layer::v4_cp
