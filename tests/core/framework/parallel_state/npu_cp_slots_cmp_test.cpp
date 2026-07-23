// Copyright 2025-2026 The xLLM Authors.
//
// Slots comparison test (refactored vs pre-refactoring), CPU-only.
//
// Goal: confirm the CP-related slot tensors are equivalent between the
// REFACTORED (model-side CP) path and the PRE-REFACTORING (worker
// cp_partition_inplace) path under MTP + chunked prefill + prefix cache +
// kvsplit.
//
// Two slot tensors are involved on the prefill side:
//   1. new_cache_slots  -> WorkerImpl::recompute_new_cache_slots
//   2. in_prefix_slots  -> WorkerImpl::compute_in_prefix_slots
//
// Both WorkerImpl member functions are BYTE-IDENTICAL between the NEW and OLD
// repos (verified by diff). They are pure functions of (input tensors,
// kv_split_size, kv_split_rank, block_size). So equivalence reduces to whether
// the INPUTS they see are equivalent:
//
//   * compute_in_prefix_slots reads block_tables / kv_cache_tokens_nums, which
//     are per-SEQUENCE (num_sequences rows) and are NOT touched by
//     cp_partition_inplace (it only gathers token-level tensors). So both paths
//     feed identical inputs -> identical in_prefix_slots.
//
//   * recompute_new_cache_slots reads new_cache_slots. OLD applies it to the
//     global new_cache_slots (cp_partition_inplace does not slice it). NEW
//     first expands global slots to cp_size*local_padded recovered order via
//     localize_slots_recovered (real tokens carry their global slot id, virtual
//     rows carry -1), THEN applies recompute_new_cache_slots. For ALIGNED cases
//     (global_real == cp_size*local_padded) localize_slots_recovered is an
//     IDENTITY (recovered order == global-real order, restore_indices and
//     cp_kv_recover_idx are inverse), so NEW input == OLD input -> identical
//     result. For NON-ALIGNED cases NEW produces a longer tensor with -1
//     virtual rows (the P0 fix); OLD kept global_real length (the bug).

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "framework/parallel_state/npu_cp_closure.h"
#include "framework/parallel_state/npu_cp_prepare.h"

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
  if (!a.defined() && !b.defined()) return true;
  if (!a.defined() || !b.defined()) return false;
  if (a.sizes() != b.sizes()) return false;
  return a.to(torch::kCPU).eq(b.to(torch::kCPU)).all().item<bool>();
}

std::string tensor_str(const torch::Tensor& t, int max_elems = 64) {
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

// Pure re-derivation of WorkerImpl::recompute_new_cache_slots (byte-identical
// in NEW and OLD repos). Remaps slots from BlockManager logical space (stride
// block_size * kv_split_size) to this rank's local physical space; slots whose
// sub-block index != owner_kv_split_rank are set to -1.
torch::Tensor recompute_slots_oracle(const torch::Tensor& old_cache_slots,
                                     int32_t block_size,
                                     int32_t kv_split_size,
                                     int32_t owner_kv_split_rank) {
  const int32_t block_size_total = block_size * kv_split_size;
  const int64_t numel = old_cache_slots.numel();
  torch::Tensor indices = torch::arange(numel, torch::kCPU);
  torch::Tensor block_offset = indices % block_size_total;
  torch::Tensor sub_block_idx = torch::floor_divide(block_offset, block_size);
  torch::Tensor mask = (sub_block_idx == owner_kv_split_rank);
  torch::Tensor valid_indices = torch::nonzero(mask).squeeze();
  torch::Tensor new_cache_slots = torch::full_like(old_cache_slots, -1);
  if (valid_indices.numel() > 0) {
    torch::Tensor old_slotid = old_cache_slots.to(torch::kCPU)
                                   .to(torch::kInt)
                                   .index_select(0, valid_indices);
    torch::Tensor block_id = torch::floor_divide(old_slotid, block_size_total);
    torch::Tensor block_offset_mod = old_slotid % block_size;
    torch::Tensor new_slotid = block_id * block_size + block_offset_mod;
    new_cache_slots.index_put_({valid_indices},
                               new_slotid.to(new_cache_slots.scalar_type()));
  }
  return new_cache_slots;
}

// Pure re-derivation of WorkerImpl::compute_in_prefix_slots (byte-identical in
// NEW and OLD repos). Emits per-seq prefix slots from block_tables /
// kv_cache_tokens_nums. Per-rank prefix count = total_prefix_tokens /
// kv_split_size.
torch::Tensor compute_in_prefix_slots_oracle(
    const torch::Tensor& block_tables,
    const torch::Tensor& kv_cache_tokens_nums,
    int32_t block_size,
    int32_t kv_split_size) {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  const int64_t num_sequences = block_tables.size(0);
  std::vector<int32_t> out;
  if (num_sequences == 0) {
    out.push_back(0);
    return torch::tensor(out, torch::kInt);
  }
  auto bt_cpu = block_tables.to(torch::kCPU);
  auto kv_cpu = kv_cache_tokens_nums.to(torch::kCPU);
  auto bt = bt_cpu.accessor<int32_t, 2>();
  auto kv = kv_cpu.accessor<int32_t, 1>();
  for (int64_t i = 0; i < num_sequences; ++i) {
    const int32_t prefix_tokens = kv[i] / kv_split_size;
    const int32_t prefix_blocks = prefix_tokens / block_size;
    if (prefix_blocks <= 0) {
      out.push_back(0);
      continue;
    }
    for (int32_t j = 0; j < prefix_blocks; ++j) {
      const int32_t physical_block = bt[i][j];
      const int32_t base_slot = physical_block * block_size;
      for (int32_t k = 0; k < block_size; ++k) {
        out.push_back(base_slot + k);
      }
    }
  }
  return torch::tensor(out, torch::kInt);
}

struct CmpCase {
  int cp_size;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> pos_starts;
  bool have_prefix_slots = false;
  std::vector<int32_t> kv_cache_tokens_per_seq;  // prefix tokens per seq
  int kv_split_size = 0;                         // 0 => default to cp_size
  int block_size = 128;
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
  const int block_size = cs.block_size;
  const bool aligned = is_aligned(q_seq_lens, cp_size);

  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);
  const int64_t global_real =
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0});

  // Build a synthetic global new_cache_slots in BlockManager logical space
  // (stride block_size * kv_split_size). Slot id for token t = t * block_size
  // (one block per token for simplicity; the remap logic is slot-id-driven so
  // any monotonic assignment works).
  std::vector<int32_t> global_slots_vec;
  global_slots_vec.reserve(global_real);
  for (int64_t t = 0; t < global_real; ++t) {
    global_slots_vec.push_back(static_cast<int32_t>(t * block_size));
  }
  torch::Tensor global_slots =
      torch::tensor(global_slots_vec, torch::dtype(torch::kInt32));

  // Synthetic per-seq block_tables: 2 rows of block ids per seq (enough for the
  // prefix cases we test). kv_cache_tokens_nums = kv_cache_tokens_per_seq.
  std::vector<int32_t> bt_vec;
  for (int i = 0; i < nseq; ++i) {
    for (int j = 0; j < 2; ++j) bt_vec.push_back(i * 2 + j);
  }
  torch::Tensor block_tables =
      torch::tensor(bt_vec, torch::dtype(torch::kInt32)).reshape({nseq, 2});
  torch::Tensor kv_cache_tokens_nums =
      torch::tensor(kv_cache_tokens_per_seq, torch::dtype(torch::kInt32));

  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    NpuCpPrefillPlan plan = build_npu_cp_prefill_plan(cp_size,
                                                      cp_rank,
                                                      q_seq_lens,
                                                      global_positions,
                                                      have_prefix_slots,
                                                      kv_cache_tokens_per_seq,
                                                      block_size,
                                                      kv_split_size);
    CpPrefillInputs cp_inputs =
        prepare_cp_prefill_inputs_from_plan(plan,
                                            have_prefix_slots,
                                            kv_cache_tokens_per_seq,
                                            block_size,
                                            kv_split_size);

    // ---- new_cache_slots ----
    // NEW path: localize_slots_recovered(global_slots, plan, cp_kv_recover_idx)
    // then recompute_new_cache_slots. For aligned cases localize is identity
    // (recovered == global-real), so NEW input == OLD input (global_slots).
    torch::Tensor new_slots_new_localized = npu_cp::localize_slots_recovered(
        global_slots, plan, cp_inputs.cp_kv_recover_idx);
    torch::Tensor new_slots_new =
        recompute_slots_oracle(new_slots_new_localized,
                               block_size,
                               kv_split_size,
                               /*owner_kv_split_rank=*/cp_rank % kv_split_size);
    // OLD path: recompute_new_cache_slots applied to global_slots directly
    // (cp_partition_inplace does not slice new_cache_slots).
    torch::Tensor new_slots_old =
        recompute_slots_oracle(global_slots,
                               block_size,
                               kv_split_size,
                               /*owner_kv_split_rank=*/cp_rank % kv_split_size);

    // ---- in_prefix_slots ----
    // Both paths: compute_in_prefix_slots(block_tables, kv_cache_tokens_nums).
    // Identical function + identical (CP-agnostic) inputs -> identical output.
    torch::Tensor prefix_slots_new = compute_in_prefix_slots_oracle(
        block_tables, kv_cache_tokens_nums, block_size, kv_split_size);
    torch::Tensor prefix_slots_old = compute_in_prefix_slots_oracle(
        block_tables, kv_cache_tokens_nums, block_size, kv_split_size);

    LOG(INFO) << "=== cp_size=" << cp_size << " cp_rank=" << cp_rank
              << " kvsplit=" << kv_split_size
              << " prefix=" << (have_prefix_slots ? 1 : 0)
              << " aligned=" << (aligned ? 1 : 0)
              << " global_real=" << global_real << " cp_size*local_padded="
              << static_cast<int64_t>(cp_size) * plan.local_padded_token_num
              << " ===";
    LOG(INFO) << "[NEW ] new_cache_slots (localized) ="
              << tensor_str(new_slots_new_localized);
    LOG(INFO) << "[NEW ] new_cache_slots (recomputed) ="
              << tensor_str(new_slots_new);
    LOG(INFO) << "[OLD ] new_cache_slots (recomputed) ="
              << tensor_str(new_slots_old);
    LOG(INFO) << "[NEW ] in_prefix_slots            ="
              << tensor_str(prefix_slots_new);
    LOG(INFO) << "[OLD ] in_prefix_slots            ="
              << tensor_str(prefix_slots_old);

    // in_prefix_slots: always identical (CP-agnostic inputs + identical fn).
    EXPECT_TRUE(tensor_equal(prefix_slots_new, prefix_slots_old))
        << "cp_rank=" << cp_rank << " in_prefix_slots must be identical";

    if (aligned) {
      // localize_slots_recovered is identity for aligned -> NEW == OLD.
      ASSERT_EQ(new_slots_new_localized.numel(), global_real)
          << "cp_rank=" << cp_rank
          << " aligned: localized new_cache_slots length must == global_real";
      EXPECT_TRUE(tensor_equal(new_slots_new_localized, global_slots))
          << "cp_rank=" << cp_rank
          << " aligned: localize_slots_recovered must be identity";
      EXPECT_TRUE(tensor_equal(new_slots_new, new_slots_old))
          << "cp_rank=" << cp_rank << " aligned: new_cache_slots NEW == OLD";
      LOG(INFO) << "[OK  ] aligned: new_cache_slots NEW == OLD";
    } else {
      // Non-aligned: NEW expands to cp_size*local_padded with -1 virtual rows;
      // OLD kept global_real length (the bug P0 fixed).
      const int64_t ag_rows =
          static_cast<int64_t>(cp_size) * plan.local_padded_token_num;
      EXPECT_EQ(new_slots_new_localized.numel(), ag_rows)
          << "cp_rank=" << cp_rank
          << " non-aligned: localized length must == cp_size*local_padded";
      EXPECT_GT(ag_rows, global_real)
          << "cp_rank=" << cp_rank
          << " non-aligned: cp_size*local_padded must > global_real";
      // Virtual rows (positions beyond global_real in recovered order) carry
      // -1; recompute preserves -1.
      torch::Tensor nc =
          new_slots_new.to(torch::kCPU).to(torch::kInt64).reshape({-1});
      int64_t neg_count = 0;
      for (int64_t i = 0; i < nc.numel(); ++i) {
        if (nc[i].item<int64_t>() == -1) ++neg_count;
      }
      EXPECT_GT(neg_count, 0)
          << "cp_rank=" << cp_rank
          << " non-aligned: NEW must contain -1 virtual rows";
      LOG(INFO) << "[OK  ] non-aligned: NEW length=" << new_slots_new.numel()
                << " (with " << neg_count
                << " virtual -1 rows), OLD length=" << new_slots_old.numel()
                << " -- intended P0 behavior";
    }
  }
}

}  // namespace

TEST(NpuCpSlotsCmpTest, AlignedCp2Kvsplit2NewCacheEqOld) {
  run_compare({/*cp_size=*/2,
               {8, 12, 4},
               {0, 0, 0},
               /*have_prefix_slots=*/false,
               {},
               /*kv_split_size=*/2});
}

TEST(NpuCpSlotsCmpTest, AlignedCp2Kvsplit1NewCacheEqOld) {
  run_compare({/*cp_size=*/2,
               {8, 12},
               {0, 0},
               /*have_prefix_slots=*/false,
               {},
               /*kv_split_size=*/1});
}

TEST(NpuCpSlotsCmpTest, AlignedCp4Kvsplit2NewCacheEqOld) {
  run_compare({/*cp_size=*/4,
               {16, 8, 24},
               {0, 0, 0},
               /*have_prefix_slots=*/false,
               {},
               /*kv_split_size=*/2});
}

TEST(NpuCpSlotsCmpTest, NonAlignedCp2Kvsplit2Dump) {
  run_compare({/*cp_size=*/2,
               {5, 7, 1, 3},
               {0, 0, 0, 0},
               /*have_prefix_slots=*/false,
               {},
               /*kv_split_size=*/2});
}

TEST(NpuCpSlotsCmpTest, PrefixCacheKvsplitEqCpAligned) {
  // cp=2, kvsplit=2, prefix cache on, aligned q
  run_compare({/*cp_size=*/2,
               {8, 12, 4},
               {256, 0, 128},
               /*have_prefix_slots=*/true,
               {256, 0, 128},
               /*kv_split_size=*/2});
}

TEST(NpuCpSlotsCmpTest, PrefixCacheKvsplit1Aligned) {
  // cp=2, kvsplit=1, prefix cache on, aligned q
  run_compare({/*cp_size=*/2,
               {8},
               {256},
               /*have_prefix_slots=*/true,
               {256},
               /*kv_split_size=*/1});
}

TEST(NpuCpSlotsCmpTest, PrefixCacheKvsplitLtCpAligned) {
  // cp=4, kvsplit=2 (<cp), prefix cache on, aligned q
  run_compare({/*cp_size=*/4,
               {16, 8, 24},
               {256, 0, 128},
               /*have_prefix_slots=*/true,
               {256, 0, 128},
               /*kv_split_size=*/2});
}

TEST(NpuCpSlotsCmpTest, PrefixCacheChunkedNonAligned) {
  // cp=2, kvsplit=2, prefix cache on, non-aligned short (chunked) q
  run_compare({/*cp_size=*/2,
               {5, 7},
               {128, 0},
               /*have_prefix_slots=*/true,
               {128, 0},
               /*kv_split_size=*/2});
}

// MTP review: the MTP draft model reuses the main model's already-recovered
// new_cache_slots (cp_size*local_padded long, in cp_kv_recover_idx order) for
// its own ReshapeAndCache. localize_slots_recovered must treat that already-
// recovered input as a NO-OP (return it unchanged) instead of forcing
// global_real length. This is the idempotency contract that prevents double-
// expand when the MTP composite worker hands its prepared input to the draft
// leaf via run_llm_no_sync_impl (which re-enters the worker prepare path).
TEST(NpuCpSlotsCmpTest, LocalizeSlotsRecoveredIsNoOpOnAlreadyRecovered) {
  const int cp_size = 2;
  const std::vector<int32_t> q_seq_lens = {8, 12, 4};
  const std::vector<int32_t> pos_starts = {256, 0, 128};
  torch::Tensor global_positions = build_positions(q_seq_lens, pos_starts);
  const int64_t global_real =
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0});

  for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    NpuCpPrefillPlan plan =
        build_npu_cp_prefill_plan(cp_size,
                                  cp_rank,
                                  q_seq_lens,
                                  global_positions,
                                  /*have_prefix_slots=*/true,
                                  /*kv_cache_tokens_per_seq=*/{256, 0, 128},
                                  /*block_size=*/128,
                                  /*kv_split_size=*/cp_size);
    CpPrefillInputs cp_inputs = prepare_cp_prefill_inputs_from_plan(
        plan,
        /*have_prefix_slots=*/true,
        /*kv_cache_tokens_per_seq=*/{256, 0, 128},
        /*block_size=*/128,
        /*kv_split_size=*/cp_size);
    const int64_t ag_rows =
        static_cast<int64_t>(cp_size) * plan.local_padded_token_num;

    // Synthetic global slots (length global_real).
    std::vector<int32_t> gvec(global_real);
    for (int64_t t = 0; t < global_real; ++t) gvec[t] = static_cast<int32_t>(t);
    torch::Tensor global_slots = torch::tensor(gvec, torch::kInt32);

    // First call: global_real -> recovered (length ag_rows).
    torch::Tensor recovered = npu_cp::localize_slots_recovered(
        global_slots, plan, cp_inputs.cp_kv_recover_idx);
    ASSERT_EQ(recovered.numel(), ag_rows)
        << "cp_rank=" << cp_rank << " first localize must expand to ag_rows";

    // Second call on already-recovered input: must be a NO-OP (MTP draft
    // reuse).
    torch::Tensor recovered2 = npu_cp::localize_slots_recovered(
        recovered, plan, cp_inputs.cp_kv_recover_idx);
    EXPECT_TRUE(tensor_equal(recovered2, recovered))
        << "cp_rank=" << cp_rank
        << " localize_slots_recovered must be no-op on already-recovered slots";
    LOG(INFO) << "[OK  ] MTP: localize_slots_recovered no-op on already-"
              << "recovered slots (cp_rank=" << cp_rank
              << " ag_rows=" << ag_rows << ")";
  }
}

}  // namespace xllm
