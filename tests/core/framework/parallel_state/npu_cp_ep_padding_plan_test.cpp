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

#include <cstdint>
#include <numeric>
#include <vector>

#include "framework/parallel_state/mapping_npu.h"
#include "framework/parallel_state/npu_cp_ep_padding.h"
#include "framework/parallel_state/npu_cp_prepare.h"

namespace xllm {
namespace {

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

torch::Tensor arange(int64_t start, int64_t end) {
  return torch::arange(start, end, torch::kInt32);
}

// Re-derive the legacy per-rank real token count, used as the oracle for
// "aligned case: local_padded == legacy per-rank real".
int64_t legacy_per_rank_real_tokens(int cp_size,
                                    int cp_rank,
                                    const std::vector<int32_t>& q_seq_lens) {
  const int32_t num_chunks = cp_size * 2;
  int64_t total = 0;
  for (int32_t q_i_signed : q_seq_lens) {
    const int64_t q_i = std::max<int64_t>(0, q_i_signed);
    const int64_t chunk_len = (q_i + num_chunks - 1) / num_chunks;
    auto clamp = [&](int64_t a, int64_t b) {
      a = std::max<int64_t>(0, a);
      b = std::max<int64_t>(0, b);
      a = std::min<int64_t>(a, q_i);
      b = std::min<int64_t>(b, q_i);
      return std::max<int64_t>(0, b - a);
    };
    total += clamp(chunk_len * cp_rank, chunk_len * (cp_rank + 1));
    total += clamp(chunk_len * (num_chunks - 1 - cp_rank),
                   chunk_len * (num_chunks - cp_rank));
  }
  return total;
}

// Expected CpEpPadding indices for the static-EP (non-dynamic) branch, given
// the local-padded length L, attn_tp_size T, attn_tp_rank R, attn_cp_size C.
// Mirrors CpEpPadding::prepare_indices byte-for-byte so the test locks the
// contract that the constructor is purely length-driven.
struct ExpectedIndices {
  torch::Tensor attn_padding_idx;
  torch::Tensor attn_unpadding_idx;
  torch::Tensor ffn_padding_idx;
  torch::Tensor ffn_unpadding_idx;
  torch::Tensor lm_head_skip_padding_token_indices;
  torch::Tensor gather_prenorm_idx;
  torch::Tensor padding_idx;
  torch::Tensor un_padding_idx;
};

ExpectedIndices build_expected(int64_t L, int64_t T, int64_t R, int64_t C) {
  const int64_t padding_length = (T - L % T) % T;
  const int64_t per_group = L + padding_length;
  const int64_t per_rank = per_group / T;

  torch::Tensor attn_padding =
      torch::cat({arange(0, L), torch::zeros({padding_length}, torch::kInt32)});
  torch::Tensor gather_prenorm =
      attn_padding.slice(0, R * per_rank, (R + 1) * per_rank);

  std::vector<torch::Tensor> skip_components;
  skip_components.reserve(C);
  for (int64_t cp_rank = 0; cp_rank < C; ++cp_rank) {
    skip_components.emplace_back(arange(0, L) + cp_rank * per_group);
  }
  torch::Tensor skip = torch::cat(skip_components, 0);

  std::vector<torch::Tensor> ffn_components;
  ffn_components.reserve(C);
  for (int64_t cp_rank = 0; cp_rank < C; ++cp_rank) {
    ffn_components.emplace_back(
        torch::cat({arange(L * cp_rank, L * (cp_rank + 1)),
                    torch::zeros({padding_length}, torch::kInt32)}));
  }
  torch::Tensor ffn_padding = torch::cat(ffn_components, 0);

  ExpectedIndices out;
  out.attn_padding_idx = attn_padding;
  out.attn_unpadding_idx = skip;
  out.ffn_padding_idx = ffn_padding;
  out.ffn_unpadding_idx = arange(0, L);
  out.lm_head_skip_padding_token_indices = skip;
  out.gather_prenorm_idx = gather_prenorm;
  out.padding_idx = attn_padding;
  out.un_padding_idx = torch::zeros({1}, torch::kInt32);
  return out;
}

void expect_padding_equal(const CpEpPaddingData& got,
                          const ExpectedIndices& exp) {
  EXPECT_TRUE(tensor_equal(got.attn_padding_idx(), exp.attn_padding_idx));
  EXPECT_TRUE(tensor_equal(got.attn_unpadding_idx(), exp.attn_unpadding_idx));
  EXPECT_TRUE(tensor_equal(got.ffn_padding_idx(), exp.ffn_padding_idx));
  EXPECT_TRUE(tensor_equal(got.ffn_unpadding_idx(), exp.ffn_unpadding_idx));
  EXPECT_TRUE(tensor_equal(got.lm_head_skip_padding_token_indices(),
                           exp.lm_head_skip_padding_token_indices));
  EXPECT_TRUE(tensor_equal(got.gather_prenorm_idx(), exp.gather_prenorm_idx));
  EXPECT_TRUE(tensor_equal(got.padding_idx(), exp.padding_idx));
  EXPECT_TRUE(tensor_equal(got.un_padding_idx(), exp.un_padding_idx));
}

MappingNPU::Options cp2_tp4_options() {
  // moe_ep_size=1 forces the constraint moe_tp_size == world_size (see
  // MappingNPU::validate); attn_tp_size*dp_size*cp_size must equal world_size.
  MappingNPU::Options options;
  options.dp_size(1)
      .tp_size(4)
      .moe_tp_size(8)
      .moe_ep_size(1)
      .pp_size(1)
      .sp_size(1)
      .cp_size(2)
      .kv_split_size(2);
  return options;
}

TEST(CpEpPaddingPlanTest, AlignedCp2ProducesExpectedIndices) {
  // All seqs divisible by 2*cp_size = 4 -> local_padded == legacy per-rank
  // real.
  const std::vector<int32_t> q_seq_lens = {8, 12, 4};
  const int32_t cp_size = 2;
  const int32_t world_size = 8;
  MappingNPU::Options options = cp2_tp4_options();
  MappingNPU mapping(/*rank_table_file=*/"", world_size, /*rank=*/0, options);
  nlohmann::json data = mapping.to_json();
  const int64_t attn_tp_size = data["attnTpSize"].get<int64_t>();
  const int64_t attn_tp_rank = data["attnTp"]["rank"].get<int64_t>();
  const int64_t attn_cp_size = data["attnCp"]["rankIds"].size();

  torch::Tensor global_positions =
      torch::arange(std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), 0),
                    torch::dtype(torch::kInt32));

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
    ASSERT_EQ(plan.local_padded_token_num,
              legacy_per_rank_real_tokens(cp_size, cp_rank, q_seq_lens))
        << "aligned case: local_padded must equal legacy per-rank real";

    CpEpPadding cp_ep_padding(plan.local_padded_token_num,
                              /*num_experts_per_tok=*/8,
                              data,
                              /*device=*/torch::Device(torch::kCPU),
                              /*dtype=*/torch::kInt32,
                              /*is_prefill=*/true);
    CpEpPaddingData got = cp_ep_padding.build();

    ExpectedIndices exp = build_expected(
        plan.local_padded_token_num, attn_tp_size, attn_tp_rank, attn_cp_size);
    expect_padding_equal(got, exp);
  }
}

TEST(CpEpPaddingPlanTest, LocalPaddedTokenNumIsRankInvariant) {
  // Odd / short seqs: legacy per-rank real differs, but the plan must keep
  // local_padded_token_num identical across ranks so CpEpPadding builds the
  // same indices on every rank (required for the post-decoder all-gather).
  const std::vector<int32_t> q_seq_lens = {5, 7, 1, 3};
  const int32_t cp_size = 2;
  const int32_t world_size = 8;
  MappingNPU::Options options = cp2_tp4_options();
  MappingNPU mapping(/*rank_table_file=*/"", world_size, /*rank=*/0, options);
  nlohmann::json data = mapping.to_json();

  torch::Tensor global_positions =
      torch::arange(std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), 0),
                    torch::dtype(torch::kInt32));

  int64_t local_padded_ref = -1;
  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    NpuCpPrefillPlan plan =
        build_npu_cp_prefill_plan(cp_size,
                                  cp_rank,
                                  q_seq_lens,
                                  global_positions,
                                  false,
                                  std::vector<int32_t>(q_seq_lens.size(), 0),
                                  128,
                                  cp_size);
    if (local_padded_ref < 0) {
      local_padded_ref = plan.local_padded_token_num;
    } else {
      ASSERT_EQ(plan.local_padded_token_num, local_padded_ref)
          << "local_padded_token_num must be identical across CP ranks";
    }
    // CpEpPadding must construct and build cleanly from the plan length.
    CpEpPadding cp_ep_padding(plan.local_padded_token_num,
                              8,
                              data,
                              torch::Device(torch::kCPU),
                              torch::kInt32,
                              true);
    CpEpPaddingData got = cp_ep_padding.build();
    EXPECT_TRUE(got.attn_padding_idx().defined());
    EXPECT_TRUE(got.lm_head_skip_padding_token_indices().defined());
  }
}

TEST(CpEpPaddingPlanTest, EmptyShardClampsToOne) {
  // Empty / fake-input shard: local_padded_token_num == 0 must clamp to 1
  // inside the constructor (mirrors the legacy `max(numel(), 1)` guard). The
  // clamp is observable via ffn_unpadding_idx = arange(input_length_), whose
  // length equals the clamped input_length_.
  MappingNPU::Options options = cp2_tp4_options();
  MappingNPU mapping(
      /*rank_table_file=*/"", /*world_size=*/8, /*rank=*/0, options);
  nlohmann::json data = mapping.to_json();
  CpEpPadding cp_ep_padding(/*local_padded_token_num=*/0,
                            8,
                            data,
                            torch::Device(torch::kCPU),
                            torch::kInt32,
                            true);
  CpEpPaddingData got = cp_ep_padding.build();
  EXPECT_EQ(got.ffn_unpadding_idx().numel(), 1);
  EXPECT_TRUE(got.attn_padding_idx().defined());
}

}  // namespace
}  // namespace xllm
