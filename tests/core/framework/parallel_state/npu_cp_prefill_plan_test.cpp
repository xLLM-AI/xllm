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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "framework/parallel_state/npu_cp_prepare.h"

namespace xllm {
namespace {

// Faithful re-derivation of the legacy per-rank token gather (zigzag chunk
// selection), used as the oracle. Per seq, chunk_len = ceil(q_i / (2*cp_size));
// rank r owns chunk r and chunk (2*cp_size-1-r), each clamped to the q_i real
// tokens.
struct LegacyPartition {
  std::vector<int64_t> gather_indices;
  std::vector<int32_t> local_q_seq_lens;
  int64_t local_token_num = 0;
};

LegacyPartition legacy_partition(int cp_size,
                                 int cp_rank,
                                 const std::vector<int32_t>& q_seq_lens) {
  const int32_t num_chunks = cp_size * 2;
  LegacyPartition out;
  int64_t seq_start = 0;
  for (int32_t q_i_signed : q_seq_lens) {
    const int64_t q_i = std::max<int64_t>(0, q_i_signed);
    const int64_t chunk_len = (q_i + num_chunks - 1) / num_chunks;
    int64_t local_len = 0;
    auto append = [&](int64_t a, int64_t b) {
      a = std::max<int64_t>(0, a);
      b = std::max<int64_t>(0, b);
      a = std::min<int64_t>(a, q_i);
      b = std::min<int64_t>(b, q_i);
      const int64_t valid = std::max<int64_t>(0, b - a);
      for (int64_t i = 0; i < valid; ++i) {
        out.gather_indices.push_back(seq_start + a + i);
      }
      local_len += valid;
    };
    append(chunk_len * cp_rank, chunk_len * (cp_rank + 1));
    append(chunk_len * (num_chunks - 1 - cp_rank),
           chunk_len * (num_chunks - cp_rank));
    out.local_q_seq_lens.push_back(static_cast<int32_t>(local_len));
    out.local_token_num += local_len;
    seq_start += q_i;
  }
  return out;
}

torch::Tensor build_positions(const std::vector<int32_t>& q_seq_lens,
                              const std::vector<int32_t>& pos_starts) {
  std::vector<int32_t> pos;
  for (size_t i = 0; i < q_seq_lens.size(); ++i) {
    for (int32_t t = 0; t < std::max(0, q_seq_lens[i]); ++t) {
      pos.push_back(pos_starts[i] + t);
    }
  }
  return torch::tensor(pos, torch::dtype(torch::kInt32).device(torch::kCPU));
}

bool tensor_equal(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if (!a.defined() || !b.defined()) {
    return false;
  }
  if (a.sizes() != b.sizes()) {
    return false;
  }
  return a.to(torch::kCPU).eq(b.to(torch::kCPU)).all().item<bool>();
}

// For aligned cases the new plan must reproduce the legacy per-rank slice and
// the legacy CpPrefillInputs byte-for-byte.
void check_aligned_matches_legacy(int cp_size,
                                  const std::vector<int32_t>& q_seq_lens,
                                  const std::vector<int32_t>& pos_starts) {
  const int64_t n = std::accumulate(
      q_seq_lens.begin(), q_seq_lens.end(), 0LL, [](long long acc, int v) {
        return acc + std::max(0, v);
      });
  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);

  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    LegacyPartition legacy = legacy_partition(cp_size, cp_rank, q_seq_lens);

    NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
        cp_size,
        cp_rank,
        q_seq_lens,
        global_positions,
        /*have_prefix_slots=*/false,
        /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
        /*block_size=*/128,
        /*kv_split_size=*/cp_size);
    ASSERT_TRUE(plan.enabled);
    ASSERT_EQ(plan.cp_size, cp_size);
    ASSERT_EQ(plan.cp_rank, cp_rank);

    // Aligned: local_padded_token_num == legacy per-rank real token count.
    EXPECT_EQ(plan.local_padded_token_num, legacy.local_token_num)
        << "cp_rank=" << cp_rank;
    // Source indices match the legacy zigzag gather.
    torch::Tensor legacy_gather =
        torch::tensor(legacy.gather_indices, torch::dtype(torch::kInt64));
    EXPECT_TRUE(tensor_equal(plan.local_source_indices, legacy_gather))
        << "cp_rank=" << cp_rank;
    // Virtual positions match the legacy gathered positions.
    torch::Tensor legacy_positions =
        global_positions.index_select(0, legacy_gather);
    EXPECT_TRUE(tensor_equal(plan.local_virtual_positions, legacy_positions))
        << "cp_rank=" << cp_rank;
    // Local q_seq_lens match the legacy per-rank real lengths.
    EXPECT_EQ(plan.local_q_seq_lens, legacy.local_q_seq_lens)
        << "cp_rank=" << cp_rank;

    // `prepare_cp_prefill_inputs_from_plan` must run cleanly on the plan and
    // produce a fully-populated CpPrefillInputs (the legacy wrapper that took
    // a per-rank slice has been removed; from_plan is the sole entry point).
    CpPrefillInputs plan_inputs = prepare_cp_prefill_inputs_from_plan(
        plan,
        /*have_prefix_slots=*/false,
        /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
        /*block_size=*/128,
        /*kv_split_size=*/cp_size);
    EXPECT_TRUE(plan_inputs.cp_load_balance_idx.defined());
    EXPECT_TRUE(plan_inputs.cp_o_recover_idx.defined());
    EXPECT_TRUE(plan_inputs.cp_kv_recover_idx.defined());
    EXPECT_TRUE(plan_inputs.actual_seq_lengths_query_prev.defined());
    EXPECT_TRUE(plan_inputs.actual_seq_lengths_key_prev.defined());
  }
}

// End-to-end localize -> rank-major gather -> restore must reproduce the
// global-real hidden for both aligned and non-aligned (virtual pad) cases.
void check_roundtrip(int cp_size,
                     const std::vector<int32_t>& q_seq_lens,
                     const std::vector<int32_t>& pos_starts) {
  const int64_t n = std::accumulate(
      q_seq_lens.begin(), q_seq_lens.end(), 0LL, [](long long acc, int v) {
        return acc + std::max(0, v);
      });
  torch::Tensor global_hidden =
      torch::arange(n, torch::dtype(torch::kFloat32)).reshape({n, 1});
  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);

  int64_t local_padded_ref = -1;
  std::vector<torch::Tensor> local_padded_per_rank;
  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
        cp_size,
        cp_rank,
        q_seq_lens,
        global_positions,
        /*have_prefix_slots=*/false,
        /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
        /*block_size=*/128,
        /*kv_split_size=*/cp_size);
    if (local_padded_ref < 0) {
      local_padded_ref = plan.local_padded_token_num;
    } else {
      ASSERT_EQ(plan.local_padded_token_num, local_padded_ref)
          << "local_padded_token_num must be identical across ranks";
    }
    torch::Tensor local =
        torch::zeros({plan.local_padded_token_num, 1}, torch::kFloat32);
    torch::Tensor src =
        global_hidden.index_select(0, plan.local_source_indices);
    local.index_put_({plan.local_destination_indices}, src);
    local_padded_per_rank.push_back(local);
  }

  torch::Tensor gathered = torch::cat(local_padded_per_rank, 0);
  NpuCpPrefillPlan plan0 =
      build_npu_cp_prefill_plan(cp_size,
                                0,
                                q_seq_lens,
                                global_positions,
                                false,
                                std::vector<int32_t>(q_seq_lens.size(), 0),
                                128,
                                cp_size);
  ASSERT_EQ(plan0.restore_indices.numel(), n);
  torch::Tensor restored = gathered.index_select(0, plan0.restore_indices);
  ASSERT_EQ(restored.numel(), n);
  EXPECT_TRUE(tensor_equal(restored.reshape({n}), global_hidden.reshape({n})))
      << "restore must reproduce global-real hidden";
}

TEST(NpuCpPrefillPlanTest, AlignedCp2MatchesLegacy) {
  // All seqs divisible by 2*cp_size = 4.
  check_aligned_matches_legacy(/*cp_size=*/2, {8, 12, 4}, {0, 0, 0});
}

TEST(NpuCpPrefillPlanTest, AlignedCp4MatchesLegacy) {
  // All seqs divisible by 2*cp_size = 8.
  check_aligned_matches_legacy(/*cp_size=*/4, {16, 8, 24}, {0, 0, 0});
}

TEST(NpuCpPrefillPlanTest, AlignedCp2WithPrefixOffset) {
  // Non-zero position starts still must match legacy (positions contiguous
  // per seq).
  check_aligned_matches_legacy(/*cp_size=*/2, {8, 12}, {100, 200});
}

TEST(NpuCpPrefillPlanTest, NonAlignedCp2VirtualPadding) {
  // Odd / short seqs: legacy per-rank lengths differ, but the plan must keep
  // local_padded_token_num constant across ranks and still round-trip.
  const std::vector<int32_t> q_seq_lens = {5, 7, 1, 3};
  check_roundtrip(/*cp_size=*/2, q_seq_lens, {0, 0, 0, 0});
}

TEST(NpuCpPrefillPlanTest, NonAlignedCp4VirtualPadding) {
  const std::vector<int32_t> q_seq_lens = {9, 13, 1, 6, 2};
  check_roundtrip(/*cp_size=*/4, q_seq_lens, {0, 0, 0, 0, 0});
}

TEST(NpuCpPrefillPlanTest, AlignedRoundtripCp2) {
  check_roundtrip(/*cp_size=*/2, {8, 12, 4}, {0, 0, 0});
}

TEST(NpuCpPrefillPlanTest, AlignedRoundtripCp4) {
  check_roundtrip(/*cp_size=*/4, {16, 8, 24}, {0, 0, 0});
}

// Short / non-aligned single-seq cases mirror the remote ATB matrix (q=1, 2, 3,
// 13). The CPU oracle must round-trip for every length, including the q=1 case
// where 2*cp_size > q leaves one CP rank with an empty real shard.
TEST(NpuCpPrefillPlanTest, ShortSingleSeqCp2Roundtrip) {
  for (int32_t q : {1, 2, 3, 13}) {
    check_roundtrip(/*cp_size=*/2, {q}, {0});
  }
}

TEST(NpuCpPrefillPlanTest, ShortSingleSeqCp4Roundtrip) {
  for (int32_t q : {1, 2, 3, 13}) {
    check_roundtrip(/*cp_size=*/4, {q}, {0});
  }
}

// Multi-seq short cases from the ATB matrix.
TEST(NpuCpPrefillPlanTest, ShortMultiSeqCp2Roundtrip) {
  check_roundtrip(/*cp_size=*/2, {1, 5}, {0, 0});
  check_roundtrip(/*cp_size=*/2, {5, 7}, {0, 0});
}

// A seq shorter than 2*cp_size leaves some CP rank with local_real == 0; the
// plan must still emit local_padded_token_num rows (clamped to >= 1) and
// round-trip. This is the CPU oracle for the rank-local real=0 ATB case.
TEST(NpuCpPrefillPlanTest, RankLocalRealZeroStillRoundtrips) {
  // q=1 under cp_size=2: 2*cp_size=4 > 1, so one rank owns 0 real tokens.
  const std::vector<int32_t> q_seq_lens = {1};
  torch::Tensor global_positions = build_positions(q_seq_lens, {0});
  int64_t local_padded_ref = -1;
  for (int cp_rank = 0; cp_rank < 2; ++cp_rank) {
    NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
        /*cp_size=*/2,
        cp_rank,
        q_seq_lens,
        global_positions,
        /*have_prefix_slots=*/false,
        /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
        /*block_size=*/128,
        /*kv_split_size=*/2);
    ASSERT_TRUE(plan.enabled);
    ASSERT_GE(plan.local_padded_token_num, 1)
        << "local_padded must be clamped to >= 1 even for empty real shard";
    if (local_padded_ref < 0) {
      local_padded_ref = plan.local_padded_token_num;
    } else {
      ASSERT_EQ(plan.local_padded_token_num, local_padded_ref)
          << "local_padded must be identical across ranks";
    }
  }
  check_roundtrip(/*cp_size=*/2, q_seq_lens, {0});
}

}  // namespace
}  // namespace xllm
