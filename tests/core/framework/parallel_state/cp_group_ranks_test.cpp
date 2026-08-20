/* Copyright 2025-2026 The xLLM Authors.

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

#include <memory>
#include <set>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "core/framework/sampling/sampling_params.h"
#include "framework/block/block_manager_pool.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "layers/npu_torch/dcp_attention_utils.h"

namespace xllm {
namespace parallel_state {
namespace {

// Re-derive the CP rank of a global rank from the documented layout:
//   rank = dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size +
//   tp_rank
// where attn_tp_size = world_size / (dp_size * cp_size).
int32_t expected_cp_rank(int32_t global_rank,
                         int32_t world_size,
                         int32_t dp_size,
                         int32_t cp_size) {
  const int32_t attn_tp_size = world_size / (dp_size * cp_size);
  return (global_rank % (cp_size * attn_tp_size)) / attn_tp_size;
}

int32_t expected_dcp_rank(int32_t global_rank,
                          int32_t world_size,
                          int32_t dp_size,
                          int32_t dcp_size) {
  const int32_t tp_size = world_size / dp_size;
  return (global_rank % tp_size) % dcp_size;
}

class PrefixTestSequence final {
 public:
  PrefixTestSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
    sampling_param_.beam_width = 0;
    sampling_param_.is_embeddings = false;

    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "dcp_prefix_contract_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"prompt",
        /*num_prompt_tokens=*/prompt_tokens.size(),
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);
    sequence_ = std::make_unique<Sequence>(index,
                                           prompt_tokens,
                                           /*input_embedding=*/torch::Tensor(),
                                           /*mm_data=*/MMData(),
                                           decoder,
                                           params);
  }

  Sequence* get() { return sequence_.get(); }

 private:
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  std::unique_ptr<Sequence> sequence_;
};

TEST(ComputeCpGroupRanks, CpSizeTwoTpFourDpOne) {
  const int32_t world_size = 8;
  const int32_t dp_size = 1;
  const int32_t cp_size = 2;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    ASSERT_EQ(ranks.size(), cp_size);
    // CP group spans ranks that differ only in the CP dimension: same dp_rank
    // and tp_rank, varying cp_rank.
    const int32_t attn_tp_size = world_size / (dp_size * cp_size);
    const int32_t tp_rank = rank % attn_tp_size;
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
      EXPECT_EQ(ranks[cp_rank],
                dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size +
                    tp_rank);
    }
    // The rank's own position in the vector is its CP rank.
    EXPECT_EQ(ranks[expected_cp_rank(rank, world_size, dp_size, cp_size)],
              rank);
  }
}

TEST(ComputeCpGroupRanks, CpSizeFourTpTwoDpTwo) {
  const int32_t world_size = 16;
  const int32_t dp_size = 2;
  const int32_t cp_size = 4;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    ASSERT_EQ(ranks.size(), cp_size);
    EXPECT_EQ(ranks[expected_cp_rank(rank, world_size, dp_size, cp_size)],
              rank);
    // Every member shares the same dp_rank and tp_rank as `rank`.
    const int32_t attn_tp_size = world_size / (dp_size * cp_size);
    const int32_t tp_rank = rank % attn_tp_size;
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    for (int32_t r : ranks) {
      EXPECT_EQ(r % attn_tp_size, tp_rank);
      EXPECT_EQ(r / (cp_size * attn_tp_size), dp_rank);
    }
  }
}

TEST(ComputeCpGroupRanks, GroupsPartitionWorldAndAreOrthogonalToTp) {
  const int32_t world_size = 16;
  const int32_t dp_size = 2;
  const int32_t cp_size = 4;
  const int32_t attn_tp_size = world_size / (dp_size * cp_size);

  // Every global rank must appear in exactly one CP group; collect all groups
  // via rank 0's perspective is insufficient, so iterate all ranks and verify
  // that two ranks share a CP group iff they share (dp_rank, tp_rank).
  auto same_cp_group = [&](int32_t a, int32_t b) {
    return compute_cp_group_ranks(a, world_size, dp_size, cp_size) ==
           compute_cp_group_ranks(b, world_size, dp_size, cp_size);
  };

  for (int32_t a = 0; a < world_size; ++a) {
    // Capture the group once: computing begin()/end() on two separate
    // temporaries would yield iterators into different containers (undefined
    // behavior).
    const std::vector<int32_t> a_group =
        compute_cp_group_ranks(a, world_size, dp_size, cp_size);
    std::set<int32_t> group_members(a_group.begin(), a_group.end());
    EXPECT_EQ(group_members.size(), cp_size);
    for (int32_t b = 0; b < world_size; ++b) {
      const bool same_dp =
          a / (cp_size * attn_tp_size) == b / (cp_size * attn_tp_size);
      const bool same_tp = a % attn_tp_size == b % attn_tp_size;
      EXPECT_EQ(same_cp_group(a, b), same_dp && same_tp);
    }
  }

  // Orthogonality to TP: a rank's TP group (same dp_rank, same cp_rank, varying
  // tp_rank) must intersect its CP group only at the rank itself.
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> cp_ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    const int32_t cp_rank =
        expected_cp_rank(rank, world_size, dp_size, cp_size);
    std::set<int32_t> cp_set(cp_ranks.begin(), cp_ranks.end());
    for (int32_t tp_rank = 0; tp_rank < attn_tp_size; ++tp_rank) {
      const int32_t tp_peer =
          dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size + tp_rank;
      if (tp_peer == rank) {
        EXPECT_NE(cp_set.find(tp_peer), cp_set.end());
      } else {
        EXPECT_EQ(cp_set.find(tp_peer), cp_set.end());
      }
    }
  }
}

TEST(ComputeCpGroupRanks, RejectsNonIntegralAttnTpSize) {
  // world_size=8, dp_size=2, cp_size=3 => 8 not divisible by 6.
  EXPECT_DEATH(
      compute_cp_group_ranks(0, /*world_size=*/8, /*dp_size=*/2, /*cp_size=*/3),
      "");
}

TEST(ComputeDcpGroupRanks, DcpSizeTwoTpEightDpOne) {
  const int32_t world_size = 8;
  const int32_t dp_size = 1;
  const int32_t dcp_size = 2;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_dcp_group_ranks(rank, world_size, dp_size, dcp_size);
    ASSERT_EQ(ranks.size(), dcp_size);
    EXPECT_EQ(ranks[expected_dcp_rank(rank, world_size, dp_size, dcp_size)],
              rank);

    const int32_t tp_rank = rank % (world_size / dp_size);
    const int32_t expected_base = (tp_rank / dcp_size) * dcp_size;
    for (int32_t dcp_rank = 0; dcp_rank < dcp_size; ++dcp_rank) {
      EXPECT_EQ(ranks[dcp_rank], expected_base + dcp_rank);
    }
  }
}

TEST(ComputeDcpGroupRanks, DcpSizeTwoTpFourDpTwo) {
  const int32_t world_size = 8;
  const int32_t dp_size = 2;
  const int32_t dcp_size = 2;
  const int32_t tp_size = world_size / dp_size;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_dcp_group_ranks(rank, world_size, dp_size, dcp_size);
    ASSERT_EQ(ranks.size(), dcp_size);
    EXPECT_EQ(ranks[expected_dcp_rank(rank, world_size, dp_size, dcp_size)],
              rank);

    const int32_t dp_rank = rank / tp_size;
    const int32_t dcp_group_base = ((rank % tp_size) / dcp_size) * dcp_size;
    for (int32_t member : ranks) {
      EXPECT_EQ(member / tp_size, dp_rank);
      EXPECT_GE(member % tp_size, dcp_group_base);
      EXPECT_LT(member % tp_size, dcp_group_base + dcp_size);
    }
  }
}

TEST(ComputeDcpGroupRanks, DocumentsContinuousGroupCounterexample) {
  const std::vector<int32_t> ranks = compute_dcp_group_ranks(
      /*global_rank=*/2, /*world_size=*/12, /*dp_size=*/1, /*dcp_size=*/2);
  ASSERT_EQ(ranks.size(), 2);
  EXPECT_EQ(ranks[0], 2);
  EXPECT_EQ(ranks[1], 3);
}

TEST(ComputeDcpGroupRanks, RejectsNonIntegralDcpGroups) {
  EXPECT_DEATH(compute_dcp_group_ranks(/*global_rank=*/0,
                                       /*world_size=*/10,
                                       /*dp_size=*/1,
                                       /*dcp_size=*/4),
               "");
}

TEST(ComputeDcpCacheSlot, PreservesOwnerPhysicalSlots) {
  const int32_t block_size = 4;
  const int32_t dcp_size = 2;
  const int32_t interleave_size = block_size;

  EXPECT_EQ(compute_dcp_cache_slot(/*logical_slot=*/151,
                                   /*position=*/0,
                                   block_size,
                                   dcp_size,
                                   /*dcp_rank=*/0,
                                   interleave_size),
            151);
  EXPECT_EQ(compute_dcp_cache_slot(/*logical_slot=*/23,
                                   /*position=*/4,
                                   block_size,
                                   dcp_size,
                                   /*dcp_rank=*/0,
                                   interleave_size),
            -1);
  EXPECT_EQ(compute_dcp_cache_slot(/*logical_slot=*/23,
                                   /*position=*/4,
                                   block_size,
                                   dcp_size,
                                   /*dcp_rank=*/1,
                                   interleave_size),
            23);
  EXPECT_EQ(compute_dcp_cache_slot(/*logical_slot=*/359,
                                   /*position=*/8,
                                   block_size,
                                   dcp_size,
                                   /*dcp_rank=*/0,
                                   interleave_size),
            359);
}

TEST(ComputeDcpCacheSlot, RejectsSubBlockInterleave) {
  const int32_t block_size = 4;
  const int32_t dcp_size = 2;
  EXPECT_DEATH(compute_dcp_cache_slot(/*logical_slot=*/0,
                                      /*position=*/0,
                                      block_size,
                                      dcp_size,
                                      /*dcp_rank=*/0,
                                      /*interleave_size=*/1),
               "");
}

TEST(ComputeDcpCacheSlot, PreservesNegativeSlots) {
  EXPECT_EQ(compute_dcp_cache_slot(/*logical_slot=*/-1,
                                   /*position=*/0,
                                   /*block_size=*/4,
                                   /*dcp_size=*/2,
                                   /*dcp_rank=*/0,
                                   /*interleave_size=*/4),
            -1);
}

TEST(SelectDcpLocalBlockTable, SelectsOriginalNonContiguousBlockIds) {
  const torch::Tensor global_block_table =
      torch::tensor({{37, 5, 89, 2}, {41, 13, 73, 29}},
                    torch::TensorOptions().dtype(torch::kInt64));

  const torch::Tensor rank_zero_table = select_dcp_local_block_table(
      global_block_table, /*dcp_size=*/2, /*dcp_rank=*/0);
  const torch::Tensor rank_one_table = select_dcp_local_block_table(
      global_block_table, /*dcp_size=*/2, /*dcp_rank=*/1);

  EXPECT_TRUE(
      torch::equal(rank_zero_table,
                   torch::tensor({{37, 89}, {41, 73}},
                                 torch::TensorOptions().dtype(torch::kInt64))));
  EXPECT_TRUE(
      torch::equal(rank_one_table,
                   torch::tensor({{5, 2}, {13, 29}},
                                 torch::TensorOptions().dtype(torch::kInt64))));
}

TEST(SelectDcpLocalBlockTable, AllowsRankWithoutBlockColumns) {
  const torch::Tensor global_block_table =
      torch::tensor({{37}}, torch::TensorOptions().dtype(torch::kInt64));

  const torch::Tensor local_block_table = select_dcp_local_block_table(
      global_block_table, /*dcp_size=*/2, /*dcp_rank=*/1);

  EXPECT_EQ(local_block_table.dim(), 2);
  EXPECT_EQ(local_block_table.size(0), 1);
  EXPECT_EQ(local_block_table.size(1), 0);
}

TEST(DcpCacheLayout, PrefillWritesMatchDecodeLocalBlockTable) {
  const int32_t block_size = 4;
  const int32_t dcp_size = 2;
  const std::vector<int64_t> global_block_ids = {37, 5, 89, 2};
  const torch::Tensor global_block_table = torch::tensor(
      {{37, 5, 89, 2}}, torch::TensorOptions().dtype(torch::kInt64));

  for (int32_t dcp_rank = 0; dcp_rank < dcp_size; ++dcp_rank) {
    const torch::Tensor local_block_table =
        select_dcp_local_block_table(global_block_table, dcp_size, dcp_rank);
    for (int32_t local_block_index = 0;
         local_block_index < local_block_table.size(1);
         ++local_block_index) {
      const int32_t global_block_index =
          dcp_rank + local_block_index * dcp_size;
      const int64_t original_block_id = global_block_ids[global_block_index];
      const int64_t original_slot =
          original_block_id * block_size + (block_size - 1);
      const int64_t position =
          static_cast<int64_t>(global_block_index) * block_size +
          (block_size - 1);
      const int64_t owner_slot =
          compute_dcp_cache_slot(original_slot,
                                 position,
                                 block_size,
                                 dcp_size,
                                 dcp_rank,
                                 /*interleave_size=*/block_size);
      const int64_t decode_block_id =
          local_block_table.index({0, local_block_index}).item<int64_t>();

      EXPECT_EQ(owner_slot, original_slot);
      EXPECT_EQ(owner_slot / block_size, decode_block_id);
    }
  }
}

TEST(DcpPrefixCacheContract,
     SharedBlocksSuffixWritesAndLocalLengthsStayOwnerAligned) {
  const int32_t block_size = 4;
  const int32_t dcp_size = 2;
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(block_size)
      .enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  PrefixTestSequence seed(
      /*index=*/0,
      /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(pool.allocate(seed.get()));
  seed.get()->kv_state().set_kv_cache_tokens_num(seed.get()->num_tokens());
  const Slice<Block> seed_blocks = seed.get()->kv_state().blocks(BlockType::KV);
  ASSERT_EQ(seed_blocks.size(), 3);
  std::vector<int64_t> seed_block_ids;
  seed_block_ids.reserve(seed_blocks.size());
  for (const Block& block : seed_blocks) {
    seed_block_ids.emplace_back(block.id());
  }
  pool.cache(seed.get());
  pool.deallocate(seed.get());

  PrefixTestSequence probe(
      /*index=*/1,
      /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  ASSERT_TRUE(pool.allocate(probe.get(), probe.get()->num_tokens()));
  EXPECT_EQ(probe.get()->kv_state().shared_blocks_num(BlockType::KV), 3);
  EXPECT_EQ(probe.get()->kv_state().kv_cache_tokens_num(), 12);

  const Slice<Block> probe_blocks =
      probe.get()->kv_state().blocks(BlockType::KV);
  ASSERT_EQ(probe_blocks.size(), 4);
  for (int64_t block_index = 0;
       block_index < static_cast<int64_t>(seed_block_ids.size());
       ++block_index) {
    EXPECT_EQ(probe_blocks[block_index].id(), seed_block_ids[block_index]);
  }

  std::vector<int64_t> global_block_ids;
  global_block_ids.reserve(probe_blocks.size());
  for (const Block& block : probe_blocks) {
    global_block_ids.emplace_back(block.id());
  }
  const torch::Tensor global_block_table =
      torch::tensor(global_block_ids,
                    torch::TensorOptions().dtype(torch::kInt64))
          .view({1, static_cast<int64_t>(global_block_ids.size())});

  const int32_t suffix_start =
      static_cast<int32_t>(probe.get()->kv_state().kv_cache_tokens_num());
  const int32_t suffix_end = static_cast<int32_t>(probe.get()->num_tokens());
  const std::vector<int32_t> suffix_slots = probe.get()->kv_state().cache_slots(
      BlockType::KV, suffix_start, suffix_end);
  const torch::Tensor suffix_positions = torch::arange(
      suffix_start, suffix_end, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor suffix_slot_tensor =
      torch::tensor(suffix_slots, torch::TensorOptions().dtype(torch::kInt64));

  const std::vector<int32_t> full_slots = probe.get()->kv_state().cache_slots(
      BlockType::KV, /*pos_start=*/0, suffix_end);
  const torch::Tensor full_positions =
      torch::arange(suffix_end, torch::TensorOptions().dtype(torch::kInt64));
  const torch::Tensor full_slot_tensor =
      torch::tensor(full_slots, torch::TensorOptions().dtype(torch::kInt64));
  const std::vector<int64_t> expected_local_kv_seq_lens = {8, 6};

  for (int32_t dcp_rank = 0; dcp_rank < dcp_size; ++dcp_rank) {
    const torch::Tensor local_block_table =
        select_dcp_local_block_table(global_block_table, dcp_size, dcp_rank);
    int64_t local_block_index = 0;
    for (int64_t global_block_index = dcp_rank;
         global_block_index < static_cast<int64_t>(global_block_ids.size());
         global_block_index += dcp_size) {
      ASSERT_LT(local_block_index, local_block_table.size(1));
      EXPECT_EQ(local_block_table[0][local_block_index].item<int64_t>(),
                global_block_ids[global_block_index]);
      ++local_block_index;
    }
    EXPECT_EQ(local_block_index, local_block_table.size(1));

    const torch::Tensor remapped_suffix =
        remap_dcp_cache_slots(suffix_positions,
                              suffix_slot_tensor,
                              /*interleave_size=*/block_size,
                              dcp_size,
                              dcp_rank);
    for (int64_t suffix_index = 0; suffix_index < remapped_suffix.numel();
         ++suffix_index) {
      const int64_t remapped_slot =
          remapped_suffix[suffix_index].item<int64_t>();
      if (remapped_slot < 0) {
        continue;
      }
      const int64_t position = suffix_positions[suffix_index].item<int64_t>();
      const int64_t global_block_index = position / block_size;
      const int64_t local_block_index = global_block_index / dcp_size;
      ASSERT_LT(local_block_index, local_block_table.size(1));
      EXPECT_EQ(remapped_slot / block_size,
                local_block_table[0][local_block_index].item<int64_t>());
    }

    const torch::Tensor remapped_full =
        remap_dcp_cache_slots(full_positions,
                              full_slot_tensor,
                              /*interleave_size=*/block_size,
                              dcp_size,
                              dcp_rank);
    const std::vector<int64_t> local_kv_seq_lens =
        layer::detail::compute_dcp_local_kv_seq_lens(
            /*global_kv_seq_lens=*/{suffix_end},
            dcp_size,
            dcp_rank,
            block_size);
    ASSERT_EQ(local_kv_seq_lens.size(), 1);
    EXPECT_EQ(local_kv_seq_lens.front(), expected_local_kv_seq_lens[dcp_rank]);
    EXPECT_EQ(local_kv_seq_lens.front(),
              remapped_full.ge(0).sum().item<int64_t>());

    if (dcp_rank == 0) {
      EXPECT_EQ(remapped_full.slice(/*dim=*/0, /*start=*/4, /*end=*/8)
                    .ge(0)
                    .sum()
                    .item<int64_t>(),
                0);
    } else {
      EXPECT_EQ(remapped_full.slice(/*dim=*/0, /*start=*/0, /*end=*/4)
                    .ge(0)
                    .sum()
                    .item<int64_t>(),
                0);
      EXPECT_EQ(remapped_full.slice(/*dim=*/0, /*start=*/8, /*end=*/12)
                    .ge(0)
                    .sum()
                    .item<int64_t>(),
                0);
    }
  }
}

// Regression for the owner float-division bug: a plain `/` on an integer
// position tensor is float true-division, so 0<position<interleave_size yields
// a non-zero fractional owner that never equals dcp_rank, wrongly dropping the
// owner's slot to -1. remap_dcp_cache_slots must use integer floor division.
TEST(RemapDcpCacheSlots, FloorDivisionKeepsOwnerSlotAcrossVirtualCycles) {
  const int32_t block_size = 128;
  const int32_t dcp_size = 2;
  // positions covering: 0<pos<interleave (5), block boundary (133=128+5 ->
  // owner 1), owner-1 interior (134,137), and an L513-class 2nd-virtual-cycle
  // position (523 -> 523/128=4, owner 0).
  const torch::Tensor positions = torch::tensor(
      {5, 133, 134, 137, 523}, torch::TensorOptions().dtype(torch::kInt32));
  const torch::Tensor slots = torch::tensor(
      {5, 133, 134, 137, 523}, torch::TensorOptions().dtype(torch::kInt32));

  // rank0 owns positions whose (pos/128)%2==0: 5(->0), 523(->4%2=0). Others -1.
  const torch::Tensor r0 = remap_dcp_cache_slots(positions,
                                                 slots,
                                                 /*interleave_size=*/block_size,
                                                 dcp_size,
                                                 /*dcp_rank=*/0);
  EXPECT_EQ(r0[0].item<int64_t>(), 5);    // pos 5: float bug would give -1
  EXPECT_EQ(r0[1].item<int64_t>(), -1);   // pos 133: owner 1
  EXPECT_EQ(r0[2].item<int64_t>(), -1);   // pos 134: owner 1
  EXPECT_EQ(r0[3].item<int64_t>(), -1);   // pos 137: owner 1
  EXPECT_EQ(r0[4].item<int64_t>(), 523);  // pos 523: owner 0 (2nd cycle)

  // rank1 owns (pos/128)%2==1: 133,134,137. 5 and 523 -> -1.
  const torch::Tensor r1 = remap_dcp_cache_slots(positions,
                                                 slots,
                                                 /*interleave_size=*/block_size,
                                                 dcp_size,
                                                 /*dcp_rank=*/1);
  EXPECT_EQ(r1[0].item<int64_t>(), -1);
  EXPECT_EQ(r1[1].item<int64_t>(), 133);
  EXPECT_EQ(r1[2].item<int64_t>(), 134);
  EXPECT_EQ(r1[3].item<int64_t>(), 137);
  EXPECT_EQ(r1[4].item<int64_t>(), -1);
}

// Negative slots stay -1 regardless of owner (non-owner or unallocated token).
TEST(RemapDcpCacheSlots, NegativeSlotsStayNegative) {
  const torch::Tensor positions =
      torch::tensor({5, 133}, torch::TensorOptions().dtype(torch::kInt32));
  const torch::Tensor slots =
      torch::tensor({-1, -1}, torch::TensorOptions().dtype(torch::kInt32));
  const torch::Tensor r0 = remap_dcp_cache_slots(positions,
                                                 slots,
                                                 /*interleave_size=*/128,
                                                 /*dcp_size=*/2,
                                                 /*dcp_rank=*/0);
  EXPECT_EQ(r0[0].item<int64_t>(), -1);
  EXPECT_EQ(r0[1].item<int64_t>(), -1);
}

}  // namespace
}  // namespace parallel_state
}  // namespace xllm
