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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "layers/npu_torch/deepseek_v4_eplb_load_utils.h"
#include "layers/npu_torch/fused_moe.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor remap_with_log2phy(const torch::Tensor& ids,
                                 const torch::Tensor& log2phy_map) {
  torch::Tensor flat_ids = ids.reshape({-1}).to(torch::kInt64);
  return log2phy_map.index_select(/*dim=*/0, flat_ids).reshape(ids.sizes());
}

}  // namespace

TEST(DeepseekV4EplbTest, DispatchFfnWorkspaceTracksActualTokenCapacity) {
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(
                /*local_tokens=*/40, /*topk=*/6, /*ep_world_size=*/8),
            1920);
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(
                /*local_tokens=*/256, /*topk=*/6, /*ep_world_size=*/8),
            12288);
}

TEST(DeepseekV4EplbTest, DispatchFfnWorkspaceRejectsInvalidDimensions) {
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(0, 6, 8), 0);
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(40, 0, 8), 0);
  EXPECT_EQ(dsv4_eplb::dispatch_ffn_max_output_size(40, 6, 0), 0);
}

TEST(DeepseekV4EplbTest, BuildsInitialExpertDistributionWithRedundantSlots) {
  const std::vector<int32_t> expert_ids =
      dsv4_eplb::build_initial_expert_ids(/*num_total_experts=*/8,
                                          /*ep_world_size=*/2,
                                          /*device_experts_num=*/5,
                                          /*redundant_experts_num=*/1);

  const std::vector<int32_t> expected = {0, 1, 2, 3, 3, 4, 5, 6, 7, 7};
  EXPECT_EQ(expert_ids, expected);

  const std::vector<int32_t> rank1_expert_ids =
      dsv4_eplb::slice_rank_expert_ids(expert_ids,
                                       /*ep_rank=*/1,
                                       /*device_experts_num=*/5);
  const std::vector<int32_t> expected_rank1 = {4, 5, 6, 7, 7};
  EXPECT_EQ(rank1_expert_ids, expected_rank1);
}

TEST(DeepseekV4EplbTest, DecodeMaskExcludesPrefillRowsInMixedBatch) {
  torch::Tensor valid_weights = torch::ones({6}, torch::kInt64);
  torch::Tensor global_mask =
      torch::tensor({false, false, true, true}, torch::kBool);
  torch::Tensor local_mask = dsv4_eplb::select_decode_token_mask(
      global_mask,
      /*routed_token_count=*/3,
      /*dp_token_counts=*/{1, 3},
      /*dp_rank=*/1,
      /*routed_tokens_are_dp_gathered=*/false);

  EXPECT_TRUE(torch::equal(local_mask,
                           torch::tensor({false, true, true}, torch::kBool)));
  EXPECT_TRUE(torch::equal(
      dsv4_eplb::apply_decode_token_mask(valid_weights, local_mask, /*topk=*/2),
      torch::tensor({0, 0, 1, 1, 1, 1}, torch::kInt64)));
}

TEST(DeepseekV4EplbTest, BuildsLog2PhyMapWithStableDuplicateChoice) {
  const std::vector<int32_t> expert_ids = {0, 1, 2, 3, 3, 4, 5, 6, 7, 7};

  const std::vector<int32_t> rank0_map =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/0);
  const std::vector<int32_t> expected_rank0 = {0, 1, 2, 3, 5, 6, 7, 8};
  EXPECT_EQ(rank0_map, expected_rank0);

  const std::vector<int32_t> rank1_map =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/1);
  const std::vector<int32_t> expected_rank1 = {0, 1, 2, 4, 5, 6, 7, 9};
  EXPECT_EQ(rank1_map, expected_rank1);
}

TEST(DeepseekV4EplbTest, MarksMissingLogicalExpertAsUnmapped) {
  const std::vector<int32_t> expert_ids = {0, 1, 1, 3};

  const std::vector<int32_t> log2phy_map =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/4,
                                   /*ep_rank=*/0);
  const std::vector<int32_t> expected_map = {0, 1, -1, 3};
  EXPECT_EQ(log2phy_map, expected_map);
}

// Ranks that share the same EP position but different moe_tp positions must
// land on different physical duplicates so hot experts do not stack every TP
// consumer onto the first copy. Mirrors the rotation trick vLLM-ascend uses.
TEST(DeepseekV4EplbTest, Log2PhyRotatesAcrossMoeTpRanksWithinSameEpRank) {
  const std::vector<int32_t> expert_ids = {0, 1, 2, 3, 3, 4, 5, 6, 7, 7};

  const std::vector<int32_t> ep0_tp0 =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/0,
                                   /*moe_tp_rank_in_group=*/0);
  const std::vector<int32_t> ep0_tp1 =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/0,
                                   /*moe_tp_rank_in_group=*/1);

  // expert 3 (phys slots {3, 4}) and expert 7 (phys slots {8, 9}) each have
  // two duplicates. ep0/tp0 lands on the first copy (slot 3 and slot 8);
  // ep0/tp1 rotates to the second copy (slot 4 and slot 9).
  EXPECT_EQ(ep0_tp0[3], 3);
  EXPECT_EQ(ep0_tp1[3], 4);
  EXPECT_EQ(ep0_tp0[7], 8);
  EXPECT_EQ(ep0_tp1[7], 9);
  // Singleton experts stay put regardless of moe_tp rank.
  for (int32_t expert_id : {0, 1, 2, 5, 6}) {
    EXPECT_EQ(ep0_tp0[expert_id], ep0_tp1[expert_id]);
  }
}

// Default moe_tp_rank_in_group = 0 must keep the historical
// `ep_rank % duplicate_count` behaviour so callers unaware of moe_tp keep
// their current pick.
TEST(DeepseekV4EplbTest, Log2PhyDefaultsToLegacyEpRankRotation) {
  const std::vector<int32_t> expert_ids = {0, 1, 2, 3, 3, 4, 5, 6, 7, 7};

  const std::vector<int32_t> rank1_default =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/1);
  const std::vector<int32_t> rank1_tp0 =
      dsv4_eplb::build_log2phy_map(expert_ids,
                                   /*num_total_experts=*/8,
                                   /*ep_rank=*/1,
                                   /*moe_tp_rank_in_group=*/0);
  EXPECT_EQ(rank1_default, rank1_tp0);
  const std::vector<int32_t> expected_rank1 = {0, 1, 2, 4, 5, 6, 7, 9};
  EXPECT_EQ(rank1_default, expected_rank1);
}

TEST(DeepseekV4EplbTest, Log2PhyMapCopyKeepsStorageAndChangesRemapResult) {
  torch::Tensor log2phy_map = torch::tensor({0, 1, 2, 3}, torch::kInt32);
  const void* original_data = log2phy_map.data_ptr();
  torch::Tensor ids = torch::tensor({{0, 3}, {1, 2}}, torch::kInt32);

  torch::Tensor first_remap = remap_with_log2phy(ids, log2phy_map);
  EXPECT_TRUE(torch::equal(first_remap,
                           torch::tensor({{0, 3}, {1, 2}}, torch::kInt32)));

  torch::Tensor new_map = torch::tensor({3, 2, 1, 0}, torch::kInt32);
  log2phy_map.copy_(new_map);

  EXPECT_EQ(log2phy_map.data_ptr(), original_data);
  torch::Tensor second_remap = remap_with_log2phy(ids, log2phy_map);
  EXPECT_TRUE(torch::equal(second_remap,
                           torch::tensor({{3, 0}, {2, 1}}, torch::kInt32)));
}

TEST(DeepseekV4EplbTest, FindsNonResidentSlotSources) {
  const std::vector<int32_t> active_expert_ids = {4, 5, 6, 7, 7};

  const std::vector<int32_t> source_slots =
      dsv4_eplb::find_slot_sources(active_expert_ids, {7, 8, 4});
  const std::vector<int32_t> expected_sources = {3, -1, 0};
  EXPECT_EQ(source_slots, expected_sources);

  const std::vector<int32_t> resident_sources =
      dsv4_eplb::find_slot_sources(active_expert_ids, {5, 7});
  EXPECT_EQ(resident_sources, (std::vector<int32_t>{1, 3}));
}

TEST(DeepseekV4EplbTest, CollectsOnlyChangedLocalSlots) {
  const std::vector<int32_t> active_expert_ids = {4, 5, 6, 7, 7};
  const std::vector<int32_t> pending_expert_ids = {4, 7, 6, 5, 7};

  const std::vector<int32_t> changed_slots =
      dsv4_eplb::collect_changed_slots(active_expert_ids, pending_expert_ids);

  const std::vector<int32_t> expected = {1, 3};
  EXPECT_EQ(changed_slots, expected);
}

// Locks the "first occurrence wins" contract that find_slot_sources must
// preserve across the reverse-index refactor.
TEST(DeepseekV4EplbTest, SlotSourcesReturnFirstOccurrenceForDuplicates) {
  // Expert 7 shows up at slots {2, 4, 5}; expert 4 shows up at {0, 3}.
  const std::vector<int32_t> active_expert_ids = {4, 5, 7, 4, 7, 7};

  const std::vector<int32_t> lenient_slots =
      dsv4_eplb::find_slot_sources(active_expert_ids, {7, 4, 7});
  EXPECT_EQ(lenient_slots, (std::vector<int32_t>{2, 0, 2}));
}

// find_slot_sources treats missing targets as sentinel -1 slot-by-slot
// instead of failing the whole call. Locks the divergence from the strict
// variant even when duplicates of an unrelated expert are present, so a
// future refactor that consolidates the two paths can't accidentally
// collapse them to the same failure mode.
TEST(DeepseekV4EplbTest, FindSlotSourcesMixesSentinelWithHits) {
  const std::vector<int32_t> active_expert_ids = {4, 5, 7, 7, 7};

  const std::vector<int32_t> source_slots =
      dsv4_eplb::find_slot_sources(active_expert_ids, {8, 7, 9, 4});
  const std::vector<int32_t> expected = {-1, 2, -1, 0};
  EXPECT_EQ(source_slots, expected);
}

TEST(DeepseekV4EplbTest, DispatchCountsRecordReceiverObservedPhysicalLoad) {
  torch::Tensor expert_load_data =
      torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor receiver_counts = torch::tensor({2, 1, 3, 4}, torch::kInt32);

  dsv4_eplb::record_dispatch_expert_load(
      receiver_counts, expert_load_data, /*layer_id=*/1);

  torch::Tensor expected = torch::tensor({2, 3, 6, 10}, torch::kInt64);
  EXPECT_TRUE(torch::equal(expert_load_data[1], expected));
  EXPECT_TRUE(torch::equal(expert_load_data[0], torch::zeros_like(expected)));
}

TEST(DeepseekV4EplbTest, ActiveSlotAndMapUpdateKeepsStorage) {
  torch::Tensor active_weight =
      torch::tensor({{10, 11}, {20, 21}, {30, 31}}, torch::kInt32);
  torch::Tensor active_map = torch::tensor({0, 1, 2}, torch::kInt32);
  const void* original_weight_data = active_weight.data_ptr();
  const void* original_map_data = active_map.data_ptr();

  torch::Tensor pending_weight =
      torch::tensor({{30, 31}, {90, 91}, {10, 11}}, torch::kInt32);
  torch::Tensor pending_map = torch::tensor({2, 0, 1}, torch::kInt32);

  active_weight.copy_(pending_weight);
  active_map.copy_(pending_map);

  EXPECT_EQ(active_weight.data_ptr(), original_weight_data);
  EXPECT_EQ(active_map.data_ptr(), original_map_data);
  EXPECT_TRUE(torch::equal(active_weight, pending_weight));
  EXPECT_TRUE(torch::equal(active_map, pending_map));
}

// Baseline: no cross-rank motion at all — every rank keeps its slots. Both
// recv and send lists must be empty. Guards against a regression where the
// planner would emit spurious ops for identity migrations.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanIdentityIsEmpty) {
  const std::vector<int32_t> global_active = {0, 1, 2, 3};
  const std::vector<int32_t> global_pending = {0, 1, 2, 3};
  std::vector<dsv4_eplb::P2POp> recv_ops;
  std::vector<dsv4_eplb::P2POp> send_ops;
  EXPECT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv_ops,
                                                   send_ops));
  EXPECT_TRUE(recv_ops.empty());
  EXPECT_TRUE(send_ops.empty());
}

// Regression for F-1 (asymmetric commit participation). In the observed
// deadlock scenario one rank has purely local hits (needs_cross_rank=false in
// the old code path) while another rank needs to recv. Under the old flag the
// local-only rank would skip commit and stall the batched P2P call. Verifies
// both ranks' plans are non-empty AND paired correctly.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanPairsCrossRankMotion) {
  // rank0 active = [100,101]; rank1 active = [102,103]
  // rank0 pending = [102,101]; rank1 pending = [102,103]  (rank0 needs 102)
  const std::vector<int32_t> global_active = {100, 101, 102, 103};
  const std::vector<int32_t> global_pending = {102, 101, 102, 103};

  std::vector<dsv4_eplb::P2POp> recv0;
  std::vector<dsv4_eplb::P2POp> send0;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv0,
                                                   send0));

  std::vector<dsv4_eplb::P2POp> recv1;
  std::vector<dsv4_eplb::P2POp> send1;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/1,
                                                   /*device_experts_num=*/2,
                                                   recv1,
                                                   send1));

  // rank0 receives expert 102 into its slot 0 from rank1's slot 0.
  ASSERT_EQ(recv0.size(), 1u);
  EXPECT_EQ(recv0[0].peer_rank, 1);
  EXPECT_EQ(recv0[0].local_slot, 0);
  EXPECT_EQ(recv0[0].peer_slot, 0);
  EXPECT_TRUE(send0.empty());

  // Paired on rank1: expected to send from its slot 0 to rank0's slot 0.
  ASSERT_EQ(send1.size(), 1u);
  EXPECT_EQ(send1[0].peer_rank, 0);
  EXPECT_EQ(send1[0].local_slot, 0);
  EXPECT_EQ(send1[0].peer_slot, 0);
  EXPECT_TRUE(recv1.empty());
}

// If both ranks need cross-rank motion (swap), both plans have one recv and
// one send with matched peer_slot / local_slot fields.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanHandlesSwap) {
  // rank0 hosts [0,1], rank1 hosts [2,3]; swap so rank0 needs 2 and rank1
  // needs 0 in their slot 0.
  const std::vector<int32_t> global_active = {0, 1, 2, 3};
  const std::vector<int32_t> global_pending = {2, 1, 0, 3};

  std::vector<dsv4_eplb::P2POp> recv0;
  std::vector<dsv4_eplb::P2POp> send0;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv0,
                                                   send0));
  std::vector<dsv4_eplb::P2POp> recv1;
  std::vector<dsv4_eplb::P2POp> send1;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/1,
                                                   /*device_experts_num=*/2,
                                                   recv1,
                                                   send1));

  ASSERT_EQ(recv0.size(), 1u);
  EXPECT_EQ(recv0[0].peer_rank, 1);
  EXPECT_EQ(recv0[0].local_slot, 0);
  ASSERT_EQ(send0.size(), 1u);
  EXPECT_EQ(send0[0].peer_rank, 1);
  EXPECT_EQ(send0[0].local_slot, 0);

  ASSERT_EQ(recv1.size(), 1u);
  EXPECT_EQ(recv1[0].peer_rank, 0);
  EXPECT_EQ(recv1[0].local_slot, 0);
  ASSERT_EQ(send1.size(), 1u);
  EXPECT_EQ(send1[0].peer_rank, 0);
  EXPECT_EQ(send1[0].local_slot, 0);
}

// A pending expert that lives on no rank means the plan cannot be built. Every
// rank must see the failure symmetrically — otherwise participation in the
// downstream batched P2P is asymmetric and would deadlock. Verify all ranks
// return false + cleared outputs.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanFailsWhenExpertMissing) {
  const std::vector<int32_t> global_active = {0, 1, 2, 3};
  const std::vector<int32_t> global_pending = {0, 42, 2, 3};

  for (int32_t ep_rank : {0, 1}) {
    std::vector<dsv4_eplb::P2POp> recv_ops = {{9, 9, 9}};
    std::vector<dsv4_eplb::P2POp> send_ops = {{9, 9, 9}};
    EXPECT_FALSE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                      global_pending,
                                                      ep_rank,
                                                      /*device_experts_num=*/2,
                                                      recv_ops,
                                                      send_ops));
    EXPECT_TRUE(recv_ops.empty());
    EXPECT_TRUE(send_ops.empty());
  }
}

// A rank whose slots are fully local-hit must still get a non-empty plan
// entry only when peer motion involves it. If nothing involves it, both lists
// stay empty — but the OTHER ranks' plans must not depend on it. This is the
// symmetry check that F-1 broke in the old boolean-flag implementation.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanRankSymmetry) {
  // 3-rank EP: rank0 [0,1], rank1 [2,3], rank2 [4,5].
  // Pending: rank0 [0,1] (all local), rank1 [4,3] (needs rank2's slot 0),
  //          rank2 [4,5] (all local).
  const std::vector<int32_t> global_active = {0, 1, 2, 3, 4, 5};
  const std::vector<int32_t> global_pending = {0, 1, 4, 3, 4, 5};

  std::vector<dsv4_eplb::P2POp> recv0;
  std::vector<dsv4_eplb::P2POp> send0;
  std::vector<dsv4_eplb::P2POp> recv1;
  std::vector<dsv4_eplb::P2POp> send1;
  std::vector<dsv4_eplb::P2POp> recv2;
  std::vector<dsv4_eplb::P2POp> send2;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv0,
                                                   send0));
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/1,
                                                   /*device_experts_num=*/2,
                                                   recv1,
                                                   send1));
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/2,
                                                   /*device_experts_num=*/2,
                                                   recv2,
                                                   send2));

  // rank0 has no motion of any kind — no one wants its experts and it needs
  // none from peers. Both lists empty.
  EXPECT_TRUE(recv0.empty());
  EXPECT_TRUE(send0.empty());

  // rank1 receives expert 4 from rank2's slot 0.
  ASSERT_EQ(recv1.size(), 1u);
  EXPECT_EQ(recv1[0].peer_rank, 2);
  EXPECT_EQ(recv1[0].local_slot, 0);
  EXPECT_EQ(recv1[0].peer_slot, 0);
  EXPECT_TRUE(send1.empty());

  // rank2 sends its slot 0 to rank1.
  EXPECT_TRUE(recv2.empty());
  ASSERT_EQ(send2.size(), 1u);
  EXPECT_EQ(send2[0].peer_rank, 1);
  EXPECT_EQ(send2[0].local_slot, 0);
  EXPECT_EQ(send2[0].peer_slot, 0);
}

// The planner must reject size mismatch between the active and pending views,
// because otherwise we would silently drop expert motion. Verify both
// same-length-invalid-shape and mismatched-length rejections.
TEST(DeepseekV4EplbTest, ComputeP2PTransferPlanRejectsBadShape) {
  std::vector<dsv4_eplb::P2POp> recv_ops;
  std::vector<dsv4_eplb::P2POp> send_ops;
  // Length not divisible by device_experts_num.
  EXPECT_FALSE(dsv4_eplb::compute_p2p_transfer_plan(
      {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, 0, 2, recv_ops, send_ops));
  // Active and pending have different sizes.
  EXPECT_FALSE(dsv4_eplb::compute_p2p_transfer_plan(
      {0, 1, 2, 3}, {0, 1, 2}, 0, 2, recv_ops, send_ops));
  // ep_rank out of range.
  EXPECT_FALSE(dsv4_eplb::compute_p2p_transfer_plan(
      {0, 1, 2, 3}, {0, 1, 2, 3}, /*ep_rank=*/5, 2, recv_ops, send_ops));
}

TEST(DeepseekV4EplbTest, StagesResidentSlotsWithoutTouchingMissingSlots) {
  torch::Tensor source = torch::arange(4 * 3, torch::kFloat32).reshape({4, 3});
  torch::Tensor pending = torch::zeros_like(source);
  const std::vector<int32_t> source_slots = {2, -1, 0, 1};

  dsv4_eplb::stage_resident_expert_slots(source, pending, source_slots);

  const torch::Tensor expected = torch::tensor({{6.0F, 7.0F, 8.0F},
                                                {0.0F, 0.0F, 0.0F},
                                                {0.0F, 1.0F, 2.0F},
                                                {3.0F, 4.0F, 5.0F}});
  EXPECT_TRUE(torch::equal(pending, expected));
}

TEST(DeepseekV4EplbTest, StagesAllResidentAndAllMissingSlots) {
  const int64_t num_slots = 4;
  const int64_t hidden = 3;
  torch::Tensor source = torch::arange(num_slots * hidden, torch::kFloat32)
                             .reshape({num_slots, hidden});

  {
    torch::Tensor out = torch::zeros_like(source);
    dsv4_eplb::stage_resident_expert_slots(source, out, {0, 1, 2, 3});
    EXPECT_TRUE(torch::equal(source, out));
  }

  {
    torch::Tensor out = torch::zeros_like(source);
    dsv4_eplb::stage_resident_expert_slots(source, out, {-1, -1, -1, -1});
    EXPECT_TRUE(torch::equal(out, torch::zeros_like(source)));
  }
}

TEST(DeepseekV4EplbTest, ChoosesFullActivationForHighSlotChurn) {
  EXPECT_FALSE(dsv4_eplb::should_activate_full_expert_tensor(0, 33));
  EXPECT_FALSE(dsv4_eplb::should_activate_full_expert_tensor(16, 33));
  EXPECT_TRUE(dsv4_eplb::should_activate_full_expert_tensor(17, 33));
  EXPECT_TRUE(dsv4_eplb::should_activate_full_expert_tensor(33, 33));
}

TEST(DeepseekV4EplbTest, ActivatesOnlyChangedExpertSlots) {
  torch::Tensor active = torch::zeros({4, 2}, torch::kFloat32);
  const torch::Tensor pending =
      torch::arange(8, torch::kFloat32).reshape({4, 2});

  dsv4_eplb::activate_expert_slots(
      active, pending, {1, 3}, /*activate_full_tensor=*/false);

  const torch::Tensor expected =
      torch::tensor({{0.0F, 0.0F}, {2.0F, 3.0F}, {0.0F, 0.0F}, {6.0F, 7.0F}});
  EXPECT_TRUE(torch::equal(active, expected));
}

TEST(DeepseekV4EplbTest, ActivatesFullExpertTensor) {
  torch::Tensor active = torch::zeros({4, 2}, torch::kFloat32);
  const torch::Tensor pending =
      torch::arange(8, torch::kFloat32).reshape({4, 2});

  dsv4_eplb::activate_expert_slots(
      active, pending, {1, 3}, /*activate_full_tensor=*/true);

  EXPECT_TRUE(torch::equal(active, pending));
}

TEST(DeepseekV4EplbTest, StagingReservationIncludesFormatCastSources) {
  constexpr int64_t kMiB = 1024 * 1024;
  const std::vector<dsv4_eplb::StagingTensorSpec> tensor_specs = {
      {/*storage_bytes=*/786 * kMiB, /*requires_format_cast=*/false},
      {/*storage_bytes=*/266 * kMiB, /*requires_format_cast=*/true},
      {/*storage_bytes=*/4 * kMiB, /*requires_format_cast=*/true}};

  EXPECT_EQ(dsv4_eplb::calculate_staging_reservation_bytes(tensor_specs),
            (786 + 2 * 266 + 2 * 4) * kMiB);
}

TEST(DeepseekV4EplbTest, StagingReservationKeepsDistinctModelBuffers) {
  const dsv4_eplb::StagingBufferKey main_weight = {/*numel=*/831 * 1024,
                                                   /*scalar_type=*/1,
                                                   /*npu_format=*/2};
  const dsv4_eplb::StagingBufferKey draft_weight = {/*numel=*/1660 * 1024,
                                                    /*scalar_type=*/2,
                                                    /*npu_format=*/2};
  std::vector<dsv4_eplb::StagingBufferKey> available = {main_weight,
                                                        main_weight};

  EXPECT_EQ(dsv4_eplb::missing_staging_buffer_count(
                available, main_weight, /*required_count=*/2),
            0);
  EXPECT_EQ(dsv4_eplb::missing_staging_buffer_count(
                available, draft_weight, /*required_count=*/1),
            1);
  available.push_back(draft_weight);
  EXPECT_EQ(dsv4_eplb::missing_staging_buffer_count(
                available, main_weight, /*required_count=*/2),
            0);
  EXPECT_EQ(dsv4_eplb::missing_staging_buffer_count(
                available, draft_weight, /*required_count=*/1),
            0);
}

TEST(DeepseekV4EplbTest, StagingReservationSkipsMtpModel) {
  EXPECT_TRUE(dsv4_eplb::should_reserve_staging_buffers(
      /*enable_eplb=*/true, "deepseek_v4"));
  EXPECT_FALSE(dsv4_eplb::should_reserve_staging_buffers(
      /*enable_eplb=*/true, "deepseek_v4_mtp"));
  EXPECT_FALSE(dsv4_eplb::should_reserve_staging_buffers(
      /*enable_eplb=*/false, "deepseek_v4"));
}

TEST(DeepseekV4EplbTest, UpdatesOnlyChangedDispatchScaleSlots) {
  torch::Tensor source = torch::tensor(
      {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}}, torch::kFloat32);
  torch::Tensor dispatch_scale = torch::full({3, 2}, -7, torch::kInt64);

  dsv4_eplb::update_dispatch_scale_slots(
      source, dispatch_scale, torch::tensor({1}, torch::kInt64));

  torch::Tensor expected = torch::tensor(
      {{-7, -7}, {1077936128, 1082130432}, {-7, -7}}, torch::kInt64);
  EXPECT_TRUE(torch::equal(dispatch_scale, expected));
}

TEST(DeepseekV4EplbTest, RefreshesWholeDispatchScaleWhenMostSlotsChange) {
  torch::Tensor source = torch::tensor(
      {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}}, torch::kFloat32);
  torch::Tensor dispatch_scale =
      source.view(torch::kInt32).to(torch::kInt64).clone();
  source[1].add_(1.0F);
  source[2].add_(2.0F);

  dsv4_eplb::update_dispatch_scale_slots(
      source, dispatch_scale, torch::tensor({1, 2}, torch::kInt64));

  torch::Tensor expected = source.view(torch::kInt32).to(torch::kInt64);
  EXPECT_TRUE(torch::equal(dispatch_scale, expected));
}

TEST(DeepseekV4EplbTest, LogicalChunkStagingPreservesLayout) {
  torch::Tensor source =
      torch::arange(3 * 2 * 4, torch::kFloat32).reshape({3, 2, 4});
  source = source.transpose(1, 2);
  torch::Tensor pending = torch::empty(source.sizes(), source.options());

  ASSERT_FALSE(source[0].is_contiguous());
  ASSERT_TRUE(source[0].is_non_overlapping_and_dense());
  ASSERT_TRUE(pending[0].is_contiguous());
  ASSERT_NE(source[0].strides().vec(), pending[0].strides().vec());

  torch::Tensor source_slot = source[2];
  torch::Tensor pending_slot = pending[1];
  torch::Tensor raw_source_storage = source_slot.as_strided(
      {source_slot.numel()}, {1}, source_slot.storage_offset());
  torch::Tensor raw_destination = torch::empty_like(pending_slot);
  raw_destination.reshape({-1}).copy_(raw_source_storage);
  ASSERT_FALSE(torch::equal(raw_destination, source_slot));

  torch::Tensor send_staging =
      torch::empty(source_slot.sizes(), source.options());
  send_staging.copy_(source_slot);
  torch::Tensor recv_staging = torch::empty_like(send_staging);
  torch::Tensor flat_send_staging = send_staging.reshape({-1});
  torch::Tensor flat_recv_staging = recv_staging.reshape({-1});
  constexpr int64_t kChunkElements = 3;
  for (int64_t offset = 0; offset < flat_send_staging.numel();
       offset += kChunkElements) {
    const int64_t chunk_elements =
        std::min(kChunkElements, flat_send_staging.numel() - offset);
    flat_recv_staging.narrow(/*dim=*/0, offset, chunk_elements)
        .copy_(flat_send_staging.narrow(/*dim=*/0, offset, chunk_elements));
  }
  pending_slot.copy_(recv_staging);
  EXPECT_TRUE(torch::equal(pending[1], source[2]));
}

TEST(DeepseekV4EplbTest, P2PBucketStatsIdentityAllUnchanged) {
  // Same active == pending on every rank: every slot lands in UNCHANGED and
  // no HCCS P2P op is emitted.
  std::vector<int32_t> global_active = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> global_pending = global_active;
  std::vector<dsv4_eplb::P2POp> recv;
  std::vector<dsv4_eplb::P2POp> send;
  dsv4_eplb::EplbP2PBucketStats stats;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/3,
                                                   recv,
                                                   send,
                                                   &stats));
  EXPECT_TRUE(recv.empty());
  EXPECT_TRUE(send.empty());
  EXPECT_EQ(stats.unchanged, 6);
  EXPECT_EQ(stats.same_gpu, 0);
  EXPECT_EQ(stats.hccs, 0);
}

TEST(DeepseekV4EplbTest, P2PBucketStatsCountsSameGpuLocalRewrite) {
  // Rank 0 rewrites its own two slots: slot 0 <-> slot 1. Different expert
  // ids in both slots so this counts as SAME_GPU.
  std::vector<int32_t> global_active = {10, 20, 30, 40};
  std::vector<int32_t> global_pending = {20, 10, 30, 40};
  std::vector<dsv4_eplb::P2POp> recv;
  std::vector<dsv4_eplb::P2POp> send;
  dsv4_eplb::EplbP2PBucketStats stats;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv,
                                                   send,
                                                   &stats));
  EXPECT_TRUE(recv.empty());
  EXPECT_TRUE(send.empty());
  EXPECT_EQ(stats.same_gpu, 2);
  EXPECT_EQ(stats.unchanged, 2);
  EXPECT_EQ(stats.hccs, 0);
}

TEST(DeepseekV4EplbTest, P2PBucketStatsTreatEveryRemoteRankAsHccs) {
  // The deployment is one HCCS super-node. Every remote-rank transfer has
  // the same transport cost regardless of host placement.
  // Active:   rank0=[10,11]  rank1=[20,21]  rank2=[30,31]  rank3=[40,41]
  // Pending:  rank0=[20,11]  rank1=[10,21]  rank2=[40,31]  rank3=[30,41]
  std::vector<int32_t> global_active = {10, 11, 20, 21, 30, 31, 40, 41};
  std::vector<int32_t> global_pending = {20, 11, 10, 21, 40, 31, 30, 41};
  std::vector<dsv4_eplb::P2POp> recv;
  std::vector<dsv4_eplb::P2POp> send;
  dsv4_eplb::EplbP2PBucketStats stats;
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv,
                                                   send,
                                                   &stats));
  EXPECT_EQ(stats.unchanged, 4);
  EXPECT_EQ(stats.hccs, 4);

  // Pairing rank0 with rank2 and rank1 with rank3 remains the same HCCS
  // bucket because host boundaries are irrelevant inside the super-node.
  // Pending:  rank0=[30,11]  rank1=[40,21]  rank2=[10,31]  rank3=[20,41]
  std::vector<int32_t> global_pending_remote = {30, 11, 40, 21, 10, 31, 20, 41};
  ASSERT_TRUE(dsv4_eplb::compute_p2p_transfer_plan(global_active,
                                                   global_pending_remote,
                                                   /*ep_rank=*/0,
                                                   /*device_experts_num=*/2,
                                                   recv,
                                                   send,
                                                   &stats));
  EXPECT_EQ(stats.unchanged, 4);
  EXPECT_EQ(stats.hccs, 4);
}

}  // namespace layer
}  // namespace xllm
