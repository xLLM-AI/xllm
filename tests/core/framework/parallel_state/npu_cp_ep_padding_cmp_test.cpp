// Copyright 2025-2026 The xLLM Authors.
//
// CpEpPadding comparison test (refactored vs pre-refactoring), CPU-only.
//
// Goal: confirm that the CpEpPadding index tensors produced by the REFACTORED
// path (constructor fed with NpuCpPrefillPlan::local_padded_token_num) are
// byte-identical to the PRE-REFACTORING path (constructor fed with the legacy
// per-rank real token count, i.e. cp_partition_inplace's input_ids.numel())
// for ALIGNED prompts, and intentionally diverge for non-aligned prompts
// (NEW uses local-padded so every CP rank builds the same indices for the
// post-decoder all-gather; OLD used local-real which is rank-dependent).
//
// CpEpPadding is purely length-driven: the constructor only consumes
// input_length_ (= max(local_padded_token_num, 1) for NEW, = max(numel, 1) for
// OLD) plus attn_tp_size / attn_tp_rank / attn_cp_size / MoE-EP config. It does
// NOT depend on prefix cache or kv_split_size, so those flags are irrelevant
// here (slots / gather index comparison is covered by the prefill-plan cmp
// test). We still vary attn_tp_size to exercise the tp>1 padding path.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "framework/parallel_state/npu_cp_ep_padding.h"
#include "framework/parallel_state/npu_cp_prepare.h"

namespace xllm {
namespace {

// Faithful re-derivation of worker cp_partition_inplace zigzag gather per-rank
// REAL token count. This is exactly what OLD CpEpPadding saw as
// input_ids.numel() after cp_partition_inplace sliced the token stream.
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
  if (!a.defined() && !b.defined()) return true;
  if (!a.defined() || !b.defined()) return false;
  if (a.sizes() != b.sizes()) return false;
  return a.to(torch::kCPU).eq(b.to(torch::kCPU)).all().item<bool>();
}

std::string tensor_str(const torch::Tensor& t, int max_elems = 48) {
  if (!t.defined()) return std::string("<undef>");
  std::ostringstream os;
  os << "shape=[" << t.sizes() << "] [";
  torch::Tensor tc =
      t.to(torch::kCPU).to(torch::kInt64).contiguous().reshape({-1});
  int64_t n = tc.numel();
  int64_t show = std::min<int64_t>(n, max_elems);
  for (int64_t i = 0; i < show; ++i) {
    if (i) os << ",";
    os << tc[i].item<int64_t>();
  }
  if (n > show) os << ",...";
  os << "]";
  return os.str();
}

bool is_aligned(const std::vector<int32_t>& q_seq_lens, int cp_size) {
  int num_chunks = cp_size * 2;
  for (int32_t q : q_seq_lens) {
    if (q > 0 && q % num_chunks != 0) return false;
  }
  return true;
}

// Build a minimal mapping_npu JSON that CpEpPadding consumes. moeEpSize=1
// forces is_dynamic_ep_=false so handle_expert_parallel() only emits
// placeholders (the common prefill case we want to compare).
nlohmann::json build_mapping_npu(int cp_size,
                                 int attn_tp_size,
                                 int attn_tp_rank) {
  nlohmann::json j;
  j["attnTpSize"] = static_cast<int64_t>(attn_tp_size);
  j["attnCpSize"] = static_cast<int64_t>(cp_size);
  j["moeEpSize"] = static_cast<int64_t>(1);
  j["attnTp"]["rank"] = static_cast<int64_t>(attn_tp_rank);
  std::vector<int64_t> rank_ids(cp_size, 0);
  j["attnCp"]["rankIds"] = rank_ids;
  return j;
}

struct CmpCase {
  int cp_size;
  int attn_tp_size = 1;
  int attn_tp_rank = 0;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> pos_starts;
};

void expect_padding_equal(const CpEpPaddingData& neu,
                          const CpEpPaddingData& old,
                          int cp_rank) {
  EXPECT_TRUE(tensor_equal(neu.attn_padding_idx(), old.attn_padding_idx()))
      << "cp_rank=" << cp_rank << " attn_padding_idx";
  EXPECT_TRUE(tensor_equal(neu.attn_unpadding_idx(), old.attn_unpadding_idx()))
      << "cp_rank=" << cp_rank << " attn_unpadding_idx";
  EXPECT_TRUE(tensor_equal(neu.ffn_padding_idx(), old.ffn_padding_idx()))
      << "cp_rank=" << cp_rank << " ffn_padding_idx";
  EXPECT_TRUE(tensor_equal(neu.ffn_unpadding_idx(), old.ffn_unpadding_idx()))
      << "cp_rank=" << cp_rank << " ffn_unpadding_idx";
  EXPECT_TRUE(tensor_equal(neu.lm_head_skip_padding_token_indices(),
                           old.lm_head_skip_padding_token_indices()))
      << "cp_rank=" << cp_rank << " lm_head_skip_padding_token_indices";
  EXPECT_TRUE(tensor_equal(neu.gather_prenorm_idx(), old.gather_prenorm_idx()))
      << "cp_rank=" << cp_rank << " gather_prenorm_idx";
  EXPECT_TRUE(tensor_equal(neu.padding_idx(), old.padding_idx()))
      << "cp_rank=" << cp_rank << " padding_idx";
  EXPECT_TRUE(tensor_equal(neu.un_padding_idx(), old.un_padding_idx()))
      << "cp_rank=" << cp_rank << " un_padding_idx";
}

void run_compare(const CmpCase& cs) {
  const int cp_size = cs.cp_size;
  const auto& q_seq_lens = cs.q_seq_lens;
  const auto& pos_starts = cs.pos_starts;
  const bool aligned = is_aligned(q_seq_lens, cp_size);
  const nlohmann::json mapping =
      build_mapping_npu(cp_size, cs.attn_tp_size, cs.attn_tp_rank);
  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);

  int64_t local_padded_ref = -1;
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

    // Sanity: NEW plan's local_real must equal legacy cp_partition_inplace
    // per-rank real count (this is exactly OLD's input_ids.numel()).
    const int64_t legacy_real =
        legacy_per_rank_real_tokens(cp_size, cp_rank, q_seq_lens);
    ASSERT_EQ(plan.local_real_token_num, legacy_real)
        << "cp_rank=" << cp_rank << " plan.local_real must match legacy";

    // NEW path: CpEpPadding fed with local_padded_token_num.
    CpEpPadding neu_pad(plan.local_padded_token_num,
                        /*num_experts_per_tok=*/8,
                        mapping,
                        /*device=*/torch::Device(torch::kCPU),
                        /*dtype=*/torch::kInt32,
                        /*is_prefill=*/true);
    CpEpPaddingData neu = neu_pad.build();

    // OLD path (simulated): CpEpPadding fed with local_real_token_num, which
    // is what input_ids.numel() was after cp_partition_inplace. The NEW
    // constructor is purely length-driven, so passing local_real reproduces
    // the OLD output byte-for-byte.
    CpEpPadding old_pad(legacy_real,
                        8,
                        mapping,
                        torch::Device(torch::kCPU),
                        torch::kInt32,
                        /*is_prefill=*/true);
    CpEpPaddingData old = old_pad.build();

    LOG(INFO) << "=== cp_size=" << cp_size << " cp_rank=" << cp_rank
              << " attn_tp_size=" << cs.attn_tp_size
              << " attn_tp_rank=" << cs.attn_tp_rank
              << " aligned=" << (aligned ? 1 : 0)
              << " local_padded=" << plan.local_padded_token_num
              << " local_real=" << legacy_real << " ===";
    LOG(INFO) << "[NEW ] attn_padding_idx           ="
              << tensor_str(neu.attn_padding_idx());
    LOG(INFO) << "[OLD ] attn_padding_idx           ="
              << tensor_str(old.attn_padding_idx());
    LOG(INFO) << "[NEW ] attn_unpadding_idx          ="
              << tensor_str(neu.attn_unpadding_idx());
    LOG(INFO) << "[OLD ] attn_unpadding_idx          ="
              << tensor_str(old.attn_unpadding_idx());
    LOG(INFO) << "[NEW ] ffn_padding_idx             ="
              << tensor_str(neu.ffn_padding_idx());
    LOG(INFO) << "[OLD ] ffn_padding_idx             ="
              << tensor_str(old.ffn_padding_idx());
    LOG(INFO) << "[NEW ] gather_prenorm_idx          ="
              << tensor_str(neu.gather_prenorm_idx());
    LOG(INFO) << "[OLD ] gather_prenorm_idx          ="
              << tensor_str(old.gather_prenorm_idx());
    LOG(INFO) << "[NEW ] lm_head_skip_padding_idx    ="
              << tensor_str(neu.lm_head_skip_padding_token_indices());
    LOG(INFO) << "[OLD ] lm_head_skip_padding_idx    ="
              << tensor_str(old.lm_head_skip_padding_token_indices());

    // local_padded_token_num must be identical across CP ranks (required for
    // the post-decoder all-gather). local_real is rank-dependent when
    // non-aligned.
    if (local_padded_ref < 0) {
      local_padded_ref = plan.local_padded_token_num;
    } else {
      ASSERT_EQ(plan.local_padded_token_num, local_padded_ref)
          << "cp_rank=" << cp_rank
          << " local_padded_token_num must be rank-invariant";
    }

    if (aligned) {
      // Aligned: local_padded == local_real, so NEW and OLD must be
      // byte-identical.
      ASSERT_EQ(plan.local_padded_token_num, legacy_real)
          << "cp_rank=" << cp_rank
          << " aligned: local_padded must equal local_real";
      expect_padding_equal(neu, old, cp_rank);
      LOG(INFO) << "[OK  ] aligned: NEW == OLD (byte-identical)";
    } else {
      // Non-aligned: NEW uses local_padded (rank-invariant), OLD used
      // local_real (rank-dependent). They intentionally differ. We only
      // assert NEW is well-formed and rank-invariant; the divergence vs OLD
      // is the P0 fix (recorded, not a failure).
      EXPECT_TRUE(neu.attn_padding_idx().defined());
      EXPECT_TRUE(neu.lm_head_skip_padding_token_indices().defined());
      if (plan.local_padded_token_num != legacy_real) {
        LOG(INFO) << "[OK  ] non-aligned: NEW(local_padded="
                  << plan.local_padded_token_num
                  << ") != OLD(local_real=" << legacy_real
                  << ") on cp_rank=" << cp_rank
                  << " -- intended P0 behavior (rank-invariant all-gather)";
      }
    }
  }
}

}  // namespace

TEST(NpuCpEpPaddingCmpTest, AlignedCp2Tp1MatchesOld) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/1,
               /*attn_tp_rank=*/0,
               {8, 12, 4},
               {0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, AlignedCp2Tp4MatchesOld) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/4,
               /*attn_tp_rank=*/0,
               {8, 12, 4},
               {0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, AlignedCp2Tp4Rank1MatchesOld) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/4,
               /*attn_tp_rank=*/1,
               {8, 12, 4},
               {0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, AlignedCp4Tp2MatchesOld) {
  run_compare({/*cp_size=*/4,
               /*attn_tp_size=*/2,
               /*attn_tp_rank=*/0,
               {16, 8, 24},
               {0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, AlignedCp2WithPosOffsetMatchesOld) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/1,
               /*attn_tp_rank=*/0,
               {8, 12},
               {100, 200}});
}

TEST(NpuCpEpPaddingCmpTest, NonAlignedCp2Dump) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/1,
               /*attn_tp_rank=*/0,
               {5, 7, 1, 3},
               {0, 0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, NonAlignedCp2Tp4Dump) {
  run_compare({/*cp_size=*/2,
               /*attn_tp_size=*/4,
               /*attn_tp_rank=*/0,
               {5, 7, 1, 3},
               {0, 0, 0, 0}});
}

TEST(NpuCpEpPaddingCmpTest, NonAlignedCp2ShortSingleDump) {
  for (int32_t q : {1, 2, 3, 13, 20, 21}) {
    run_compare(
        {/*cp_size=*/2, /*attn_tp_size=*/1, /*attn_tp_rank=*/0, {q}, {0}});
  }
}

}  // namespace xllm
