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

#include "framework/parallel_state/npu_cp_closure.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "framework/parallel_state/process_group.h"

namespace xllm {
namespace {

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

// Simulate the rank-major all-gather that `npu_cp::gather_restore` would issue
// across `cp_size` ranks, by localizing each rank and concatenating in CP-rank
// order. Then restore global-real order with the rank-0 plan and assert the
// roundtrip reproduces the original global-real hidden (real rows only).
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
  std::vector<torch::Tensor> local_per_rank;
  local_per_rank.reserve(cp_size);
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
    ASSERT_TRUE(plan.enabled);
    if (local_padded_ref < 0) {
      local_padded_ref = plan.local_padded_token_num;
    } else {
      ASSERT_EQ(plan.local_padded_token_num, local_padded_ref)
          << "local_padded_token_num must be identical across ranks";
    }
    local_per_rank.push_back(npu_cp::localize(global_hidden, plan));
  }

  // Rank-major gather, matching parallel_state::gather(..., dim=0).
  torch::Tensor gathered = torch::cat(local_per_rank, /*dim=*/0);
  NpuCpPrefillPlan plan0 = build_npu_cp_prefill_plan(
      cp_size,
      /*cp_rank=*/0,
      q_seq_lens,
      global_positions,
      /*have_prefix_slots=*/false,
      /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
      /*block_size=*/128,
      /*kv_split_size=*/cp_size);
  torch::Tensor restored = npu_cp::restore_from_gathered(gathered, plan0);
  ASSERT_EQ(restored.size(0), n)
      << "restored must have exactly global_real_token_num rows";
  EXPECT_TRUE(tensor_equal(restored, global_hidden))
      << "localize -> gather -> restore must reproduce global-real hidden";
}

TEST(NpuCpClosureTest, LocalizeIsNoOpWhenDisabled) {
  NpuCpPrefillPlan plan;
  plan.enabled = false;
  torch::Tensor h =
      torch::arange(8, torch::dtype(torch::kFloat32)).reshape({8, 1});
  EXPECT_TRUE(npu_cp::localize(h, plan).is_same(h))
      << "localize must return input unchanged when plan disabled";
  EXPECT_TRUE(npu_cp::restore_from_gathered(h, plan).is_same(h))
      << "restore_from_gathered must return input unchanged when plan disabled";
  EXPECT_TRUE(npu_cp::localize_positions(plan, h).is_same(h))
      << "localize_positions must return input unchanged when plan disabled";
}

TEST(NpuCpClosureTest, GatherRestoreIsNoOpWhenDisabled) {
  // Disabled plan must short-circuit before any cp_group validation, so a null
  // group is fine (decode / cp_size==1 paths are inert).
  NpuCpPrefillPlan plan;
  plan.enabled = false;
  torch::Tensor h =
      torch::arange(4, torch::dtype(torch::kFloat32)).reshape({4, 1});
  EXPECT_TRUE(npu_cp::gather_restore(h, plan, /*cp_group=*/nullptr).is_same(h))
      << "gather_restore must return input unchanged when plan disabled";
}

TEST(NpuCpClosureTest, GatherRestoreFailsFastOnNullCpGroup) {
  // Enabled plan requires a non-null cp_group; a missing group is a setup bug
  // and must CHECK-fail rather than silently returning local-padded hidden.
  NpuCpPrefillPlan plan;
  plan.enabled = true;
  plan.cp_size = 2;
  plan.cp_rank = 0;
  torch::Tensor h =
      torch::arange(4, torch::dtype(torch::kFloat32)).reshape({4, 1});
  EXPECT_DEATH(npu_cp::gather_restore(h, plan, /*cp_group=*/nullptr),
               "non-null cp_group");
}

TEST(NpuCpClosureTest, GatherRestoreFailsFastOnMismatchedGroupSize) {
  NpuCpPrefillPlan plan;
  plan.enabled = true;
  plan.cp_size = 2;
  plan.cp_rank = 0;
  ProcessGroup pg(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  torch::Tensor h =
      torch::arange(4, torch::dtype(torch::kFloat32)).reshape({4, 1});
  EXPECT_DEATH(npu_cp::gather_restore(h, plan, &pg), "must match plan.cp_size");
}

TEST(NpuCpClosureTest, GatherRestoreFailsFastOnMismatchedGroupRank) {
  NpuCpPrefillPlan plan;
  plan.enabled = true;
  plan.cp_size = 2;
  plan.cp_rank = 0;
  ProcessGroup pg(/*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU));
  torch::Tensor h =
      torch::arange(4, torch::dtype(torch::kFloat32)).reshape({4, 1});
  EXPECT_DEATH(npu_cp::gather_restore(h, plan, &pg), "must match plan.cp_rank");
}

TEST(NpuCpClosureTest, AlignedRoundtripCp2) {
  // All seqs divisible by 2*cp_size = 4 -> no virtual pad.
  check_roundtrip(
      /*cp_size=*/2, /*q_seq_lens=*/{8, 12, 4}, /*pos_starts=*/{0, 100, 200});
}

TEST(NpuCpClosureTest, NonAlignedRoundtripCp2) {
  // Non-divisible seqs exercise virtual pad; roundtrip still recovers real
  // rows.
  check_roundtrip(
      /*cp_size=*/2, /*q_seq_lens=*/{7, 5, 3}, /*pos_starts=*/{0, 50, 100});
}

TEST(NpuCpClosureTest, NonAlignedRoundtripCp4) {
  check_roundtrip(/*cp_size=*/4,
                  /*q_seq_lens=*/{10, 7, 6, 9},
                  /*pos_starts=*/{0, 64, 128, 192});
}

TEST(NpuCpClosureTest, EmptyShardRoundtripCp2) {
  // A seq shorter than 2*cp_size leaves some rank with an empty real shard;
  // localize must still produce local_padded_token_num rows (clamped to >=1).
  check_roundtrip(/*cp_size=*/2, /*q_seq_lens=*/{1, 8}, /*pos_starts=*/{0, 32});
}

TEST(NpuCpClosureTest, LocalizePositionsReturnsVirtualPositions) {
  const std::vector<int32_t> q_seq_lens = {6, 4};
  torch::Tensor global_positions =
      build_positions(q_seq_lens, /*pos_starts=*/{0, 100});
  NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
      /*cp_size=*/2,
      /*cp_rank=*/0,
      q_seq_lens,
      global_positions,
      /*have_prefix_slots=*/false,
      /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
      /*block_size=*/128,
      /*kv_split_size=*/2);
  ASSERT_TRUE(plan.enabled);
  torch::Tensor local_positions =
      npu_cp::localize_positions(plan, global_positions);
  EXPECT_FALSE(local_positions.is_same(global_positions));
  EXPECT_EQ(local_positions.size(0), plan.local_padded_token_num);
  // Real rows keep their global position; virtual-pad rows are a contiguous
  // fill (harmless because their hidden is zero).
  torch::Tensor real_positions =
      global_positions.index_select(/*dim=*/0, plan.local_source_indices);
  torch::Tensor localized_real =
      local_positions.index_select(/*dim=*/0, plan.local_destination_indices);
  EXPECT_TRUE(tensor_equal(localized_real, real_positions));
}

TEST(NpuCpClosureTest, LocalizeSlotsIsNoOpWhenDisabled) {
  NpuCpPrefillPlan plan;
  plan.enabled = false;
  torch::Tensor slots = torch::arange(8, torch::dtype(torch::kInt32));
  EXPECT_TRUE(npu_cp::localize_slots(slots, plan).is_same(slots))
      << "localize_slots must return input unchanged when plan disabled";
}

TEST(NpuCpClosureTest, LocalizeSlotsMatchesPlanIndicesAndPadsWithNegOne) {
  const std::vector<int32_t> q_seq_lens = {
      7, 5, 3};  // non-aligned -> virtual pad
  torch::Tensor global_positions =
      build_positions(q_seq_lens, /*pos_starts=*/{0, 50, 100});
  NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
      /*cp_size=*/2,
      /*cp_rank=*/0,
      q_seq_lens,
      global_positions,
      /*have_prefix_slots=*/false,
      /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
      /*block_size=*/128,
      /*kv_split_size=*/2);
  ASSERT_TRUE(plan.enabled);

  const int64_t global_real = plan.global_real_token_num;
  ASSERT_EQ(global_real, 7 + 5 + 3);
  // Distinct, nonzero slot ids so a stray -1 or wrong row is detectable.
  torch::Tensor global_slots =
      torch::arange(global_real, torch::dtype(torch::kInt32)) * 2 + 1000;

  torch::Tensor local = npu_cp::localize_slots(global_slots, plan);
  ASSERT_EQ(local.size(0), plan.local_padded_token_num)
      << "local slots must have local_padded_token_num rows";
  ASSERT_EQ(local.scalar_type(), torch::kInt32);

  // Real rows: local[destination[i]] == global_slots[source[i]].
  torch::Tensor expected_real =
      global_slots.index_select(/*dim=*/0, plan.local_source_indices);
  torch::Tensor localized_real =
      local.index_select(/*dim=*/0, plan.local_destination_indices);
  EXPECT_TRUE(tensor_equal(localized_real, expected_real))
      << "real rows must scatter via plan source/destination indices";

  // Virtual-pad rows (not in destination_indices) must be -1.
  torch::Tensor is_pad =
      torch::ones({plan.local_padded_token_num}, torch::kBool);
  is_pad.index_put_({plan.local_destination_indices},
                    torch::zeros({plan.local_real_token_num}, torch::kBool));
  torch::Tensor pad_rows = local.masked_select(is_pad);
  EXPECT_TRUE(torch::all(pad_rows.eq(-1)).item<bool>())
      << "every virtual-pad row must carry -1";

  // No real slot value of -1 leaked into a real row (sanity: source values
  // >=1000).
  torch::Tensor real_rows = local.masked_select(~is_pad);
  EXPECT_FALSE(torch::any(real_rows.eq(-1)).item<bool>())
      << "no real row should carry -1";
}

TEST(NpuCpClosureTest, LocalizeSlotsRecoveredIsIdempotentOnRecoveredLayout) {
  // The MTP draft reuses the target's already-recovered slots (cp_size *
  // local_padded long, in cp_kv_recover_idx order). localize_slots_recovered
  // must accept that layout as a no-op instead of forcing global_real length,
  // so a draft re-entering worker prepare does not double-convert.
  const std::vector<int32_t> q_seq_lens = {
      7, 5, 3};  // non-aligned -> virtual pad
  torch::Tensor global_positions =
      build_positions(q_seq_lens, /*pos_starts=*/{0, 50, 100});
  NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(
      /*cp_size=*/2,
      /*cp_rank=*/0,
      q_seq_lens,
      global_positions,
      /*have_prefix_slots=*/false,
      /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
      /*block_size=*/128,
      /*kv_split_size=*/2);
  ASSERT_TRUE(plan.enabled);

  CpPrefillInputs inputs = prepare_cp_prefill_inputs_from_plan(
      plan,
      /*have_prefix_slots=*/false,
      /*kv_cache_tokens_per_seq=*/std::vector<int32_t>(q_seq_lens.size(), 0),
      /*block_size=*/128,
      /*kv_split_size=*/2);

  const int64_t global_real = plan.global_real_token_num;
  torch::Tensor global_slots =
      torch::arange(global_real, torch::dtype(torch::kInt32)) * 2 + 1000;

  torch::Tensor recovered = npu_cp::localize_slots_recovered(
      global_slots, plan, inputs.cp_kv_recover_idx);
  const int64_t ag_rows =
      static_cast<int64_t>(plan.cp_size) * plan.local_padded_token_num;
  ASSERT_EQ(recovered.size(0), ag_rows)
      << "first conversion must expand to cp_size*local_padded rows";

  // Re-running on the already-recovered tensor must return it unchanged (the
  // tolerant path: numel == cp_size*local_padded).
  torch::Tensor recovered_again = npu_cp::localize_slots_recovered(
      recovered, plan, inputs.cp_kv_recover_idx);
  EXPECT_TRUE(recovered_again.is_same(recovered))
      << "localize_slots_recovered must no-op on already-recovered slots";
}

}  // namespace
}  // namespace xllm
