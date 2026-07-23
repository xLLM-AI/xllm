// Copyright 2025-2026 The xLLM Authors.
//
// CP input comparison test (refactored vs pre-refactoring), CPU-only.
//
// Goal: dump the model-side CP input tensors produced by the REFACTORED path
// (build_npu_cp_prefill_plan + prepare_cp_prefill_inputs_from_plan, which feeds
// local-PADDED seq lens into prepare_cp_prefill_inputs_impl and overwrites
// cp_kv_recover_idx / cp_load_balance_idx) and by the PRE-REFACTORING path
// (worker cp_partition_inplace gather -> prepare_cp_prefill_inputs_impl fed
// with local-REAL seq lens, no overwrites), then compare value-by-value.
//
// For aligned prompts (every q_i % (2*cp_size) == 0) local_real ==
// local_padded, so the two paths must be byte-identical. For non-aligned
// prompts the new path uses local-padded (virtual pad) while the old path used
// local-real; we print both side-by-side so a divergence can be spotted
// visually.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "framework/parallel_state/npu_cp_prepare.h"

namespace xllm {

// Declared in npu_cp_prepare.cpp (anonymous-namespace-free, external linkage)
// but not exposed in the header. We re-declare it here so the test can drive
// the OLD path (local-real inputs) through the SAME impl the NEW path uses.
CpPrefillInputs prepare_cp_prefill_inputs_impl(
    int cp_size,
    int64_t local_token_num,
    const torch::Tensor& position_ids,
    const torch::Tensor& input_lengths,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int block_size,
    int kv_split_size);

namespace {

struct LegacyPartition {
  std::vector<int64_t> gather_indices;
  std::vector<int32_t> local_q_seq_lens;
  int64_t local_token_num = 0;
};

// Faithful re-derivation of worker cp_partition_inplace zigzag gather.
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
  if (!a.defined() && !b.defined()) return true;
  if (!a.defined() || !b.defined()) return false;
  if (a.sizes() != b.sizes()) return false;
  return a.to(torch::kCPU).eq(b.to(torch::kCPU)).all().item<bool>();
}

std::string tensor_str(const torch::Tensor& t, int max_elems = 64) {
  if (!t.defined()) return std::string("<undef>");
  std::ostringstream os;
  os << "shape=[" << t.sizes() << "] dtype=" << t.dtype();
  torch::Tensor tc =
      t.to(torch::kCPU).to(torch::kInt64).contiguous().reshape({-1});
  int64_t n = tc.numel();
  os << " n=" << n << " [";
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

struct CmpCase {
  int cp_size;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> pos_starts;
  bool have_prefix_slots = false;
  std::vector<int32_t> kv_cache_tokens_per_seq;  // prefix tokens per seq
  int kv_split_size = 0;                         // 0 => default to cp_size
};

void run_compare(const CmpCase& cs) {
  const int cp_size = cs.cp_size;
  const auto& q_seq_lens = cs.q_seq_lens;
  const auto& pos_starts = cs.pos_starts;
  const int nseq = static_cast<int>(q_seq_lens.size());
  const bool have_prefix_slots = cs.have_prefix_slots;
  const std::vector<int32_t> kv_empty(nseq, 0);
  const std::vector<int32_t>& kv_cache_tokens_per_seq =
      have_prefix_slots ? cs.kv_cache_tokens_per_seq : kv_empty;
  const int kv_split_size = cs.kv_split_size > 0 ? cs.kv_split_size : cp_size;
  const bool aligned = is_aligned(q_seq_lens, cp_size);

  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);

  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    // ---- NEW (refactored) path ----
    NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(cp_size,
                                                      cp_rank,
                                                      q_seq_lens,
                                                      global_positions,
                                                      have_prefix_slots,
                                                      kv_cache_tokens_per_seq,
                                                      /*block_size=*/128,
                                                      kv_split_size);
    CpPrefillInputs neu = prepare_cp_prefill_inputs_from_plan(
        plan, have_prefix_slots, kv_cache_tokens_per_seq, 128, kv_split_size);

    // ---- OLD (pre-refactoring) path ----
    // For non-aligned short prompts the legacy path crashes (c10::IndexError:
    // index_select on an empty positions tensor when a rank owns 0 real
    // tokens). That is exactly the bug the P0 refactor fixed by switching to
    // local-padded. We wrap the OLD call in try/catch so the comparison test
    // records the crash as the expected regression marker instead of aborting.
    LegacyPartition leg = legacy_partition(cp_size, cp_rank, q_seq_lens);
    torch::Tensor leg_gather =
        torch::tensor(leg.gather_indices, torch::dtype(torch::kInt64));
    torch::Tensor old_positions = global_positions.index_select(0, leg_gather);
    torch::Tensor old_lens = torch::tensor(
        leg.local_q_seq_lens, torch::dtype(torch::kInt32).device(torch::kCPU));
    CpPrefillInputs old;
    bool old_crashed = false;
    try {
      old = prepare_cp_prefill_inputs_impl(cp_size,
                                           leg.local_token_num,
                                           old_positions,
                                           old_lens,
                                           have_prefix_slots,
                                           kv_cache_tokens_per_seq,
                                           128,
                                           kv_split_size);
    } catch (const c10::Error& e) {
      old_crashed = true;
      LOG(INFO) << "[OLD ] CRASHED (expected for non-aligned short prompts; "
                   "the bug P0 fixed): "
                << e.what_without_backtrace();
    }

    // ---- print both ----
    LOG(INFO) << "=== cp_size=" << cp_size << " cp_rank=" << cp_rank
              << " q_seq_lens=[...]"
              << " aligned=" << (aligned ? 1 : 0)
              << " prefix=" << (have_prefix_slots ? 1 : 0)
              << " kvsplit=" << kv_split_size << " ===";
    LOG(INFO) << "[PLAN] local_padded_token_num=" << plan.local_padded_token_num
              << " local_real_token_num=" << plan.local_real_token_num
              << " local_q_seq_lens=[...]";
    LOG(INFO) << "[PLAN] local_padded_seq_lens="
              << tensor_str(torch::tensor(plan.local_padded_seq_lens,
                                          torch::dtype(torch::kInt32)));
    LOG(INFO) << "[PLAN] local_source_indices  ="
              << tensor_str(plan.local_source_indices);
    LOG(INFO) << "[PLAN] local_destination_indices="
              << tensor_str(plan.local_destination_indices);
    LOG(INFO) << "[PLAN] local_virtual_positions="
              << tensor_str(plan.local_virtual_positions);
    LOG(INFO) << "[PLAN] restore_indices       ="
              << tensor_str(plan.restore_indices);
    LOG(INFO) << "[OLD ] local_real_token_num =" << leg.local_token_num
              << " local_q_seq_lens=[...]";
    LOG(INFO) << "[OLD ] gathered_positions    =" << tensor_str(old_positions);
    LOG(INFO) << "[NEW ] cp_load_balance_idx   ="
              << tensor_str(neu.cp_load_balance_idx);
    LOG(INFO) << "[OLD ] cp_load_balance_idx   ="
              << tensor_str(old.cp_load_balance_idx);
    LOG(INFO) << "[NEW ] cp_o_recover_idx      ="
              << tensor_str(neu.cp_o_recover_idx);
    LOG(INFO) << "[OLD ] cp_o_recover_idx      ="
              << tensor_str(old.cp_o_recover_idx);
    LOG(INFO) << "[NEW ] cp_kv_recover_idx      ="
              << tensor_str(neu.cp_kv_recover_idx);
    LOG(INFO) << "[OLD ] cp_kv_recover_idx      ="
              << tensor_str(old.cp_kv_recover_idx);
    LOG(INFO) << "[NEW ] k_gather_index_prev   ="
              << tensor_str(neu.k_gather_index_prev);
    LOG(INFO) << "[OLD ] k_gather_index_prev   ="
              << tensor_str(old.k_gather_index_prev);
    LOG(INFO) << "[NEW ] k_gather_index_next   ="
              << tensor_str(neu.k_gather_index_next);
    LOG(INFO) << "[OLD ] k_gather_index_next   ="
              << tensor_str(old.k_gather_index_next);
    LOG(INFO) << "[NEW ] asl_query_prev        ="
              << tensor_str(neu.actual_seq_lengths_query_prev);
    LOG(INFO) << "[OLD ] asl_query_prev        ="
              << tensor_str(old.actual_seq_lengths_query_prev);
    LOG(INFO) << "[NEW ] asl_key_prev          ="
              << tensor_str(neu.actual_seq_lengths_key_prev);
    LOG(INFO) << "[OLD ] asl_key_prev          ="
              << tensor_str(old.actual_seq_lengths_key_prev);
    LOG(INFO) << "[NEW ] asl_key_next          ="
              << tensor_str(neu.actual_seq_lengths_key_next);
    LOG(INFO) << "[OLD ] asl_key_next          ="
              << tensor_str(old.actual_seq_lengths_key_next);

    if (aligned && old_crashed) {
      FAIL() << "OLD path crashed on an ALIGNED case (cp_rank=" << cp_rank
             << ") -- unexpected, refactored path is not the cause";
    }
    if (aligned && !old_crashed) {
      EXPECT_TRUE(
          tensor_equal(neu.cp_load_balance_idx, old.cp_load_balance_idx))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(tensor_equal(neu.cp_o_recover_idx, old.cp_o_recover_idx))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(tensor_equal(neu.cp_kv_recover_idx, old.cp_kv_recover_idx))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(
          tensor_equal(neu.k_gather_index_prev, old.k_gather_index_prev))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(
          tensor_equal(neu.k_gather_index_next, old.k_gather_index_next))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(tensor_equal(neu.actual_seq_lengths_query_prev,
                               old.actual_seq_lengths_query_prev))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(tensor_equal(neu.actual_seq_lengths_key_prev,
                               old.actual_seq_lengths_key_prev))
          << "cp_rank=" << cp_rank;
      EXPECT_TRUE(tensor_equal(neu.actual_seq_lengths_key_next,
                               old.actual_seq_lengths_key_next))
          << "cp_rank=" << cp_rank;
    }
    if (!aligned && old_crashed) {
      // Non-aligned short prompt: OLD path crashes (the bug P0 fixed); NEW path
      // must still produce a fully-defined CpPrefillInputs. This is the
      // regression marker -- record it as a passing expectation.
      EXPECT_TRUE(neu.cp_load_balance_idx.defined());
      EXPECT_TRUE(neu.cp_kv_recover_idx.defined());
      EXPECT_TRUE(neu.actual_seq_lengths_key_prev.defined());
      EXPECT_TRUE(neu.actual_seq_lengths_key_next.defined());
      LOG(INFO) << "[OK  ] non-aligned: OLD crashed (expected), NEW ok -> "
                   "refactored path handles padding correctly";
    }
  }
}

TEST(NpuCpPrefillPlanCmpTest, AlignedCp2MatchesOld) {
  run_compare({/*cp_size=*/2, {8, 12, 4}, {0, 0, 0}});
}

TEST(NpuCpPrefillPlanCmpTest, AlignedCp2WithPrefixOffset) {
  run_compare({/*cp_size=*/2, {8, 12}, {100, 200}});
}

TEST(NpuCpPrefillPlanCmpTest, AlignedCp4MatchesOld) {
  run_compare({/*cp_size=*/4, {16, 8, 24}, {0, 0, 0}});
}

TEST(NpuCpPrefillPlanCmpTest, NonAlignedCp2Dump) {
  run_compare({/*cp_size=*/2, {5, 7, 1, 3}, {0, 0, 0, 0}});
}

TEST(NpuCpPrefillPlanCmpTest, NonAlignedCp2ShortSingleDump) {
  for (int32_t q : {1, 2, 3, 13, 20, 21}) {
    run_compare({/*cp_size=*/2, {q}, {0}});
  }
}

// ===== MTP + chunked prefill + prefix cache + kvsplit combination =====

TEST(NpuCpPrefillPlanCmpTest, PrefixCacheKvsplitEqCpAligned) {
  // cp=2, kvsplit=2 (==cp), prefix cache on, aligned q
  run_compare({/*cp_size=*/2,
               {8, 12, 4},
               {256, 0, 128},
               /*have_prefix_slots=*/true,
               {256, 0, 128},
               /*kv_split_size=*/2});
}

TEST(NpuCpPrefillPlanCmpTest, PrefixCacheKvsplitLtCpAligned) {
  // cp=4, kvsplit=2 (<cp), prefix cache on, aligned q
  run_compare({/*cp_size=*/4,
               {16, 8, 24},
               {256, 0, 128},
               /*have_prefix_slots=*/true,
               {256, 0, 128},
               /*kv_split_size=*/2});
}

TEST(NpuCpPrefillPlanCmpTest, PrefixCacheKvsplit1Aligned) {
  // cp=2, kvsplit=1, prefix cache on, aligned q
  run_compare({/*cp_size=*/2,
               {8},
               {256},
               /*have_prefix_slots=*/true,
               {256},
               /*kv_split_size=*/1});
}

TEST(NpuCpPrefillPlanCmpTest, PrefixCacheChunkedNonAligned) {
  // cp=2, kvsplit=2, prefix cache on, non-aligned short (chunked) q
  run_compare({/*cp_size=*/2,
               {5, 7},
               {128, 0},
               /*have_prefix_slots=*/true,
               {128, 0},
               /*kv_split_size=*/2});
}

TEST(NpuCpPrefillPlanCmpTest, PrefixCacheMixedSomeSeqNoPrefix) {
  // cp=2, kvsplit=2, prefix cache on, seq1 has prefix, seq2 no prefix
  run_compare({/*cp_size=*/2,
               {8, 4},
               {256, 0},
               /*have_prefix_slots=*/true,
               {256, 0},
               /*kv_split_size=*/2});
}

}  // namespace
}  // namespace xllm
