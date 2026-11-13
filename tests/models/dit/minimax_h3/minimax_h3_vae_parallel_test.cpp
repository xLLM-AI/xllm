/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "core/framework/parallel_state/rank_generator.h"
#include "models/dit/transformers/minimax_h3_tp_layout.h"
#include "models/dit/utils/minimax_h3_qualified_vae.h"

namespace xllm {
namespace {

RankGroup make_vae_rank_group(int32_t vae_size, int32_t rank) {
  std::vector<int32_t> sizes{vae_size};
  std::vector<std::string> order{"vae"};
  if (vae_size != 16) {
    sizes.emplace_back(16 / vae_size);
    order.emplace_back("place_holder");
  }
  return RankGenerator(16).get_rank_group("vae", sizes, order, rank);
}

TEST(QualifiedVaeContract, SerialDefaultKeepsAllOwnersOnGlobalRankZero) {
  int32_t reference_owners = 0;
  int32_t video_workers = 0;
  int32_t audio_owners = 0;
  int32_t output_owners = 0;
  for (int32_t rank = 0; rank < 16; ++rank) {
    const MiniMaxH3QualifiedVaePlan plan =
        minimax_h3_qualified_vae_plan(16, rank, 1);
    reference_owners += plan.reference_encode_owner;
    video_workers += plan.video_vae_worker;
    audio_owners += plan.audio_decode_owner;
    output_owners += plan.output_owner;
    EXPECT_FALSE(plan.collaborative_video_decode);
    EXPECT_EQ(plan.reference_encode_owner, rank == 0);
    EXPECT_EQ(plan.video_vae_worker, rank == 0);
    EXPECT_EQ(plan.audio_decode_owner, rank == 0);
    EXPECT_EQ(plan.output_owner, rank == 0);
  }
  EXPECT_EQ(reference_owners, 1);
  EXPECT_EQ(video_workers, 1);
  EXPECT_EQ(audio_owners, 1);
  EXPECT_EQ(output_owners, 1);
}

TEST(QualifiedVaeContract, ParallelModeChangesOnlyFinalVideoWorkers) {
  int32_t reference_owners = 0;
  int32_t video_workers = 0;
  int32_t audio_owners = 0;
  int32_t output_owners = 0;
  for (int32_t rank = 0; rank < 16; ++rank) {
    const MiniMaxH3QualifiedVaePlan plan =
        minimax_h3_qualified_vae_plan(16, rank, 16);
    reference_owners += plan.reference_encode_owner;
    video_workers += plan.video_vae_worker;
    audio_owners += plan.audio_decode_owner;
    output_owners += plan.output_owner;
    EXPECT_TRUE(plan.collaborative_video_decode);
    EXPECT_TRUE(plan.video_vae_worker);
    EXPECT_EQ(plan.reference_encode_owner, rank == 0);
    EXPECT_EQ(plan.audio_decode_owner, rank == 0);
    EXPECT_EQ(plan.output_owner, rank == 0);
  }
  EXPECT_EQ(reference_owners, 1);
  EXPECT_EQ(video_workers, 16);
  EXPECT_EQ(audio_owners, 1);
  EXPECT_EQ(output_owners, 1);
}

TEST(QualifiedVaeContract, GenericVaeTwoCreatesEightRepeatedPairs) {
  const std::vector<std::vector<int32_t>> expected = {
      {0, 1},
      {2, 3},
      {4, 5},
      {6, 7},
      {8, 9},
      {10, 11},
      {12, 13},
      {14, 15},
  };
  for (int32_t rank = 0; rank < 16; ++rank) {
    const RankGroup vae = make_vae_rank_group(/*vae_size=*/2, rank);
    EXPECT_EQ(vae.count, 8);
    EXPECT_EQ(vae.ranks, expected[rank / 2]);
    EXPECT_EQ(vae.index, rank / 2);
    EXPECT_EQ(vae.rank, rank % 2);
  }
}

TEST(QualifiedVaeContract, VaeSixteenCreatesExactlyOneWorldGroup) {
  std::vector<int32_t> expected(16);
  std::iota(expected.begin(), expected.end(), 0);
  for (int32_t rank = 0; rank < 16; ++rank) {
    const RankGroup vae = make_vae_rank_group(/*vae_size=*/16, rank);
    EXPECT_EQ(vae.ranks, expected);
    EXPECT_EQ(vae.count, 1);
    EXPECT_EQ(vae.index, 0);
    EXPECT_EQ(vae.rank, rank);
  }
}

TEST(QualifiedVaeContract, TwentyEightTilesPadAndRestoreFourBySevenRowMajor) {
  constexpr size_t kTileCount = 28;
  constexpr size_t kWorldSize = 16;
  constexpr size_t kPaddedLocalCount = 2;
  std::vector<std::vector<int64_t>> gathered(
      kWorldSize, std::vector<int64_t>(kPaddedLocalCount, -1));
  std::vector<size_t> local_counts;
  local_counts.reserve(kWorldSize);
  for (size_t rank = 0; rank < kWorldSize; ++rank) {
    const std::vector<size_t> local =
        minimax_h3_vae_local_tile_indices(kTileCount, kWorldSize, rank);
    local_counts.emplace_back(local.size());
    for (size_t slot = 0; slot < local.size(); ++slot) {
      gathered[rank][slot] = static_cast<int64_t>(local[slot]);
    }
  }
  EXPECT_EQ(std::count(local_counts.begin(), local_counts.end(), 2), 12);
  EXPECT_EQ(std::count(local_counts.begin(), local_counts.end(), 1), 4);
  for (size_t rank = 12; rank < kWorldSize; ++rank) {
    EXPECT_EQ(gathered[rank][1], -1);
  }

  std::vector<std::vector<int64_t>> row_major(4, std::vector<int64_t>(7));
  for (size_t tile = 0; tile < kTileCount; ++tile) {
    const MiniMaxH3VaeGatherCoordinate coordinate =
        minimax_h3_vae_gather_coordinate(tile, kWorldSize);
    row_major[tile / 7][tile % 7] =
        gathered[coordinate.rank][coordinate.local_slot];
  }
  for (size_t row = 0; row < row_major.size(); ++row) {
    for (size_t column = 0; column < row_major[row].size(); ++column) {
      EXPECT_EQ(row_major[row][column], static_cast<int64_t>(row * 7 + column));
    }
  }
}

TEST(QualifiedVaeContract, RejectsMalformedQualifiedTopologies) {
  EXPECT_TRUE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 1, 1}));
  EXPECT_TRUE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 16, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({8, 1, 2, 4, 1, 8, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 16, 1, 1, 16, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 2, 4, 1, 16, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 2, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 8, 1}));
  EXPECT_THROW(minimax_h3_qualified_vae_plan(16, 0, 2), std::invalid_argument);
  EXPECT_THROW(minimax_h3_qualified_vae_plan(16, 16, 16),
               std::invalid_argument);
}

TEST(QualifiedVaeContract, CollectiveStateRequiresAgreementBeforeAllgather) {
  MiniMaxH3VaeCollectiveState succeeded;
  EXPECT_EQ(succeeded.phase(), MiniMaxH3VaeCollectivePhase::LOCAL_PREPARATION);
  succeeded.local_preparation_finished(true);
  EXPECT_EQ(succeeded.phase(), MiniMaxH3VaeCollectivePhase::AGREEMENT);
  succeeded.agreement_finished(true);
  EXPECT_EQ(succeeded.phase(), MiniMaxH3VaeCollectivePhase::ALLGATHER);
  succeeded.allgather_finished();
  EXPECT_EQ(succeeded.phase(), MiniMaxH3VaeCollectivePhase::REORDER);
  succeeded.reorder_finished();
  EXPECT_EQ(succeeded.phase(), MiniMaxH3VaeCollectivePhase::COMPLETE);

  MiniMaxH3VaeCollectiveState failed;
  failed.local_preparation_finished(false);
  failed.agreement_finished(false);
  EXPECT_EQ(failed.phase(), MiniMaxH3VaeCollectivePhase::FAILED);
  EXPECT_THROW(failed.allgather_finished(), std::logic_error);

  MiniMaxH3VaeCollectiveState peer_failed;
  peer_failed.local_preparation_finished(true);
  peer_failed.agreement_finished(false);
  EXPECT_EQ(peer_failed.phase(), MiniMaxH3VaeCollectivePhase::FAILED);
  EXPECT_THROW(peer_failed.allgather_finished(), std::logic_error);
}

TEST(QualifiedVaeContract, HbmPreflightFailsClosed) {
  constexpr int64_t kWeights = int64_t{10} * 1024 * 1024 * 1024;
  const int64_t owner_reserve = kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes;
  const int64_t required = kWeights + owner_reserve;
  EXPECT_TRUE(minimax_h3_parallel_vae_hbm_preflight_passes(
      required, kWeights, owner_reserve));
  EXPECT_FALSE(minimax_h3_parallel_vae_hbm_preflight_passes(
      required - 1, kWeights, owner_reserve));
  EXPECT_FALSE(
      minimax_h3_parallel_vae_hbm_preflight_passes(0, kWeights, owner_reserve));
  EXPECT_FALSE(
      minimax_h3_parallel_vae_hbm_preflight_passes(required, 0, owner_reserve));
  EXPECT_FALSE(minimax_h3_parallel_vae_hbm_preflight_passes(
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(),
      owner_reserve));
}

TEST(QualifiedVaeContract, OwnerAndNonownerReservesAreOrderedAndBounded) {
  EXPECT_TRUE(minimax_h3_parallel_vae_decode_reserve_bytes(/*decode_owner=*/
                                                           true) ==
              kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes);
  EXPECT_TRUE(minimax_h3_parallel_vae_decode_reserve_bytes(/*decode_owner=*/
                                                           false) ==
              kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes);
  // Nonowner reserve must clear the worst CO decode-start free (2.84 GiB).
  EXPECT_LT(kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes,
            static_cast<int64_t>(2.84 * 1024 * 1024 * 1024));
  EXPECT_GT(kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes,
            kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes);
}

TEST(QualifiedVaeContract, DecodeEntryAdmissionHardFailsNonownersOnly) {
  const int64_t nonowner_reserve =
      kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes;
  const int64_t owner_reserve = kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes;
  // Nonowner: hard-fail strictly below the reserve; pass at/above it.
  EXPECT_TRUE(minimax_h3_parallel_vae_decode_entry_hard_fails(
      /*decode_owner=*/false, nonowner_reserve - 1, nonowner_reserve));
  EXPECT_FALSE(minimax_h3_parallel_vae_decode_entry_hard_fails(
      /*decode_owner=*/false, nonowner_reserve, nonowner_reserve));
  EXPECT_FALSE(minimax_h3_parallel_vae_decode_entry_hard_fails(
      /*decode_owner=*/false, nonowner_reserve + 1, nonowner_reserve));
  // Owner: log-only admission; never hard-fails, including the formal CO
  // rank0 sampled raw free ~3.1 GiB (65536-62383 MiB) below the 4.75 GiB
  // analytic reserve.
  const int64_t co_rank0_sampled_free_bytes =
      (int64_t{65536} - int64_t{62383}) * 1024 * 1024;
  EXPECT_LT(co_rank0_sampled_free_bytes, owner_reserve);
  EXPECT_FALSE(minimax_h3_parallel_vae_decode_entry_hard_fails(
      /*decode_owner=*/true, co_rank0_sampled_free_bytes, owner_reserve));
  EXPECT_FALSE(minimax_h3_parallel_vae_decode_entry_hard_fails(
      /*decode_owner=*/true, 0, owner_reserve));
}

TEST(QualifiedVaePixelAffine, InPlaceMatchesOutOfPlaceChainBitForBit) {
  const torch::Tensor mean =
      torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 3, 1, 1, 1});
  const torch::Tensor std_ =
      torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 3, 1, 1, 1});
  // Representative chunk (17 frames) and full (124 frames) temporal extents;
  // H,W scaled for CPU budget - the affine is elementwise with [1,3,1,1,1]
  // broadcast so bit-identity is independent of H,W. Frame counts 17/124 and
  // channel count 3 are the real geometry.
  for (const int64_t frames : {int64_t{1}, int64_t{17}, int64_t{124}}) {
    torch::manual_seed(1234);
    const torch::Tensor decoded =
        torch::randn({1, 3, frames, 32, 48}, torch::kFloat32);
    // Reference: the ORIGINAL out-of-place chain (not the new helper).
    const torch::Tensor reference =
        torch::clamp(decoded.to(torch::kFloat32) * std_ + mean, 0.0, 1.0)
            .permute({0, 2, 1, 3, 4})
            .contiguous();
    const torch::Tensor actual =
        minimax_h3_vae_affine_to_pixels(decoded.clone(), mean, std_);
    EXPECT_TRUE(torch::equal(actual, reference));
    ASSERT_EQ(actual.numel(), reference.numel());
    EXPECT_EQ(0,
              std::memcmp(actual.data_ptr<float>(),
                          reference.data_ptr<float>(),
                          static_cast<size_t>(actual.numel()) * sizeof(float)));
    // Shape contract: [1,3,T,H,W] -> [1,T,3,H,W].
    EXPECT_EQ(actual.sizes().vec(),
              (std::vector<int64_t>{1, frames, 3, 32, 48}));
  }
}

}  // namespace
}  // namespace xllm
