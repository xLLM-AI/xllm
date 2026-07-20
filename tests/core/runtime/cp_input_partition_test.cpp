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

#include "runtime/cp_input_partition.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "framework/batch/batch_forward_type.h"
#include "framework/model/model_input_params.h"
#include "runtime/forward_params.h"

namespace xllm::cp {
namespace {

// Mirror the deserialization layout used by `params_utils.cpp` so the test's
// ForwardInput looks identical to what arrives at WorkerImpl on the wire.
//
// On NPU the per-seq vectors are stored as raw lengths; on CUDA/MLU/ILU they
// are stored as cumulative-sums-with-leading-zero. We mirror this convention
// for q_seq_lens_vec / kv_seq_lens_vec so the round-trip equivalence test
// matches what the worker actually sees.
std::vector<int32_t> to_layout_seq_lens(const std::vector<int32_t>& lens) {
#if defined(USE_NPU)
  return lens;
#else
  std::vector<int32_t> out;
  out.reserve(lens.size() + 1);
  out.emplace_back(0);
  int32_t sum = 0;
  for (int32_t len : lens) {
    sum += len;
    out.emplace_back(sum);
  }
  return out;
#endif
}

torch::TensorOptions int32_cpu() {
  return torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
}

std::vector<int32_t> tensor_to_vec(const torch::Tensor& t) {
  if (!t.defined() || t.numel() == 0) {
    return {};
  }
  auto cpu = t.to(torch::kCPU).to(torch::kInt32).contiguous();
  const int32_t* data = cpu.data_ptr<int32_t>();
  return {data, data + cpu.numel()};
}

// Build a ForwardInput that matches what `params_utils.cpp::pb_to_forward`
// would produce for a corresponding RawForwardInput. Only the fields that
// `cp_partition_inplace` reads or writes need to be populated.
ForwardInput make_forward_input(
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& selected_idxes = {},
    const std::vector<int32_t>& mtp_shifted = {},
    BatchForwardType type = BatchForwardType::PREFILL) {
  ForwardInput fi;
  fi.token_ids = torch::tensor(tokens, int32_cpu());
  fi.positions = torch::tensor(positions, int32_cpu());

  auto& p = fi.input_params;
  p.meta.batch_forward_type = type;
  p.meta.num_sequences = static_cast<int32_t>(q_seq_lens.size());
  p.attention.host.q_seq_lens = to_layout_seq_lens(q_seq_lens);
  p.attention.host.kv_seq_lens = p.attention.host.q_seq_lens;
  p.attention.device.q_seq_lens =
      torch::tensor(p.attention.host.q_seq_lens, int32_cpu());
  p.attention.device.kv_seq_lens =
      torch::tensor(p.attention.host.kv_seq_lens, int32_cpu());

  std::vector<int32_t> q_cu;
  q_cu.reserve(q_seq_lens.size());
  std::partial_sum(
      q_seq_lens.begin(), q_seq_lens.end(), std::back_inserter(q_cu));
  p.attention.host.q_cu_seq_lens = q_cu;
  p.attention.device.q_cu_seq_lens = torch::tensor(q_cu, int32_cpu());
  p.meta.q_max_seq_len =
      q_seq_lens.empty()
          ? 0
          : *std::max_element(q_seq_lens.begin(), q_seq_lens.end());
  p.meta.kv_max_seq_len = p.meta.q_max_seq_len;

  if (!mtp_shifted.empty()) {
    p.embedding.mtp_shifted_token_ids = torch::tensor(mtp_shifted, int32_cpu());
  }
  if (!selected_idxes.empty()) {
    fi.sampling_params.selected_token_idxes =
        torch::tensor(selected_idxes, int32_cpu());
  }
  return fi;
}

TEST(CpInputPartitionTest, NoOpWhenCpSizeOne) {
  auto fi = make_forward_input({1, 2, 3, 4}, {0, 1, 2, 3}, /*q_seq_lens=*/{4});
  auto orig_tokens = tensor_to_vec(fi.token_ids);
  auto orig_q_seq = fi.input_params.attention.host.q_seq_lens;

  cp_partition_inplace(fi, /*cp_rank=*/0, /*cp_size=*/1);

  EXPECT_EQ(tensor_to_vec(fi.token_ids), orig_tokens);
  EXPECT_EQ(fi.input_params.attention.host.q_seq_lens, orig_q_seq);
}

TEST(CpInputPartitionTest, MluMtpDraftKeepsUnevenChunkedLayout) {
  auto input = make_forward_input({10, 11, 20, 21, 22, 23, 24},
                                  {0, 1, 0, 1, 2, 3, 4},
                                  /*q_seq_lens=*/{2, 5},
                                  /*selected_idxes=*/{1, 6},
                                  /*mtp_shifted=*/{11, -1, 21, 22, 23, 24, -1},
                                  BatchForwardType::CHUNKED_PREFILL);
  const std::vector<int32_t> expected_tokens = tensor_to_vec(input.token_ids);
  const std::vector<int32_t> expected_shifted =
      tensor_to_vec(input.input_params.embedding.mtp_shifted_token_ids);
  const std::vector<int32_t> expected_q_lens =
      input.input_params.attention.host.q_seq_lens;

  cp_partition_inplace(input, /*cp_rank=*/0, /*cp_size=*/1);

  EXPECT_EQ(tensor_to_vec(input.token_ids), expected_tokens);
  EXPECT_EQ(tensor_to_vec(input.input_params.embedding.mtp_shifted_token_ids),
            expected_shifted);
  EXPECT_EQ(input.input_params.attention.host.q_seq_lens, expected_q_lens);
  EXPECT_EQ(tensor_to_vec(input.sampling_params.selected_token_idxes),
            std::vector<int32_t>({1, 6}));
}

TEST(CpInputPartitionTest, CpPartitionedFlagPropagatesThroughCopy) {
  // Confirms ForwardInput::cp_partitioned propagates through
  // ForwardInput::to(device, dtype). This is what stops sub-worker calls
  // (MTP target/draft) from re-partitioning an already-partitioned tensor
  // that lives on device.
  ForwardInput fi;
  fi.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  fi.input_params.meta.num_sequences = 1;
  fi.token_ids = torch::tensor(std::vector<int32_t>{1}, int32_cpu());
  fi.cp_partitioned = true;

  ForwardInput moved = fi.to(torch::kCPU, torch::kFloat32);
  EXPECT_TRUE(moved.cp_partitioned);
}

TEST(CpInputPartitionTest, MixedBatchPartitionsLikePrefill) {
  std::vector<int32_t> tokens(8);
  std::iota(tokens.begin(), tokens.end(), 100);
  std::vector<int32_t> positions(8);
  std::iota(positions.begin(), positions.end(), 0);

  auto rank0 = make_forward_input(tokens,
                                  positions,
                                  /*q_seq_lens=*/{8},
                                  /*selected_idxes=*/{},
                                  /*mtp_shifted=*/{},
                                  BatchForwardType::MIXED);
  cp_partition_inplace(rank0, /*cp_rank=*/0, /*cp_size=*/2);
  EXPECT_EQ(tensor_to_vec(rank0.token_ids),
            std::vector<int32_t>({100, 101, 106, 107}));
}

TEST(CpInputPartitionTest, NoOpWhenDecode) {
  auto fi = make_forward_input({1, 2, 3, 4},
                               {0, 1, 2, 3},
                               /*q_seq_lens=*/{4},
                               /*selected_idxes=*/{},
                               /*mtp_shifted=*/{},
                               BatchForwardType::DECODE);
  auto orig_tokens = tensor_to_vec(fi.token_ids);

  cp_partition_inplace(fi, /*cp_rank=*/0, /*cp_size=*/2);

  EXPECT_EQ(tensor_to_vec(fi.token_ids), orig_tokens);
}

TEST(CpInputPartitionTest, NoOpWhenNoSequences) {
  ForwardInput fi;
  fi.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  fi.input_params.meta.num_sequences = 0;
  fi.token_ids = torch::tensor(std::vector<int32_t>{}, int32_cpu());

  cp_partition_inplace(fi, /*cp_rank=*/0, /*cp_size=*/2);
  // No crash, fields stay empty.
  EXPECT_EQ(fi.input_params.meta.num_sequences, 0);
}

TEST(CpInputPartitionTest, SingleSequenceCp2EvenLength) {
  // tokens: [a0..a7], cp_size=2, num_chunks=4, chunk_len=2.
  //  rank 0 takes chunks {0, 3} -> indices {0,1, 6,7}
  //  rank 1 takes chunks {1, 2} -> indices {2,3, 4,5}
  std::vector<int32_t> tokens(8);
  std::iota(tokens.begin(), tokens.end(), 100);
  std::vector<int32_t> positions(8);
  std::iota(positions.begin(), positions.end(), 0);

  auto rank0 = make_forward_input(tokens, positions, {8});
  cp_partition_inplace(rank0, /*cp_rank=*/0, /*cp_size=*/2);
  EXPECT_EQ(tensor_to_vec(rank0.token_ids),
            std::vector<int32_t>({100, 101, 106, 107}));
  EXPECT_EQ(tensor_to_vec(rank0.positions), std::vector<int32_t>({0, 1, 6, 7}));

  auto rank1 = make_forward_input(tokens, positions, {8});
  cp_partition_inplace(rank1, /*cp_rank=*/1, /*cp_size=*/2);
  EXPECT_EQ(tensor_to_vec(rank1.token_ids),
            std::vector<int32_t>({102, 103, 104, 105}));
  EXPECT_EQ(tensor_to_vec(rank1.positions), std::vector<int32_t>({2, 3, 4, 5}));
}

TEST(CpInputPartitionTest, MultiSequenceUnevenLengths) {
  // seq lengths {5, 6}, cp_size=2, num_chunks=4
  //   seq0 (len=5): chunk_len = ceil(5/4) = 2; ranges per rank:
  //     rank 0 -> [0,2)+[6,8) clamped to len5 -> [0,2)+empty
  //     rank 1 -> [2,4)+[4,6) clamped to len5 -> [2,4)+[4,5)
  //   seq1 (len=6): chunk_len = ceil(6/4) = 2; ranges:
  //     rank 0 -> [0,2)+[6,8) clamped to len6 -> [0,2)+empty
  //     rank 1 -> [2,4)+[4,6)
  std::vector<int32_t> tokens(11);
  std::iota(tokens.begin(), tokens.end(), 0);
  std::vector<int32_t> positions = tokens;

  auto rank0 = make_forward_input(tokens, positions, {5, 6});
  cp_partition_inplace(rank0, 0, 2);
  // seq0 contributes [0,1], seq1 contributes [5,6]
  EXPECT_EQ(tensor_to_vec(rank0.token_ids), std::vector<int32_t>({0, 1, 5, 6}));
  EXPECT_EQ(rank0.input_params.attention.host.q_seq_lens,
            to_layout_seq_lens({2, 2}));

  auto rank1 = make_forward_input(tokens, positions, {5, 6});
  cp_partition_inplace(rank1, 1, 2);
  // seq0 contributes [2,3,4], seq1 contributes [7,8,9,10]
  EXPECT_EQ(tensor_to_vec(rank1.token_ids),
            std::vector<int32_t>({2, 3, 4, 7, 8, 9, 10}));
  EXPECT_EQ(rank1.input_params.attention.host.q_seq_lens,
            to_layout_seq_lens({3, 4}));
}

TEST(CpInputPartitionTest, Cp4Partition) {
  // single seq, len = 8, cp_size = 4, num_chunks = 8, chunk_len = 1
  //   rank r takes chunks {r, 7-r} -> tokens {r, 7-r}
  std::vector<int32_t> tokens(8);
  std::iota(tokens.begin(), tokens.end(), 10);
  std::vector<int32_t> positions(8);
  std::iota(positions.begin(), positions.end(), 0);

  for (int r = 0; r < 4; ++r) {
    auto fi = make_forward_input(tokens, positions, {8});
    cp_partition_inplace(fi, r, 4);
    EXPECT_EQ(tensor_to_vec(fi.token_ids),
              std::vector<int32_t>({10 + r, 10 + 7 - r}))
        << "rank=" << r;
  }
}

TEST(CpInputPartitionTest, MtpShiftedTokensFollowGather) {
  std::vector<int32_t> tokens(8);
  std::iota(tokens.begin(), tokens.end(), 100);
  std::vector<int32_t> positions(8);
  std::iota(positions.begin(), positions.end(), 0);
  std::vector<int32_t> shifted(8);
  std::iota(shifted.begin(), shifted.end(), 1000);

  auto rank1 = make_forward_input(tokens, positions, {8}, {}, shifted);
  cp_partition_inplace(rank1, /*cp_rank=*/1, /*cp_size=*/2);
  EXPECT_EQ(tensor_to_vec(rank1.input_params.embedding.mtp_shifted_token_ids),
            std::vector<int32_t>({1002, 1003, 1004, 1005}));
}

// End-to-end pinning across cp_rank/cp_size combinations on a non-trivial
// batch. Expected outputs are derived by tracing the algorithm by hand, and
// also cross-validated by the Python harness in
// tools/tests/test_cp_input_partition_cpu.py (which exercises the same
// algorithm and asserts equivalence with the legacy slicer).
TEST(CpInputPartitionTest, MultiSeqMultiRankPinning) {
  const std::vector<int32_t> q_seq_lens = {6, 8};
  std::vector<int32_t> tokens(14);
  std::iota(tokens.begin(), tokens.end(), 0);
  std::vector<int32_t> positions = tokens;

  struct Expect {
    int32_t cp_size;
    int32_t cp_rank;
    std::vector<int32_t> tokens;
    std::vector<int32_t> q_lens;
  };
  // Tracing the algorithm by hand (cp_size c, num_chunks=2c, chunk_len =
  // ceil(seq_len / num_chunks)):
  // cp_size=2, num_chunks=4:
  //   seq0 (len 6, chunk_len=2): rank0 -> [0,2)+[6,8)∩[0,6) = {0,1};
  //                              rank1 -> [2,4)+[4,6) = {2,3,4,5}
  //   seq1 (len 8, chunk_len=2): rank0 -> [0,2)+[6,8) = {0,1,6,7};
  //                              rank1 -> [2,4)+[4,6) = {2,3,4,5} (offset 6)
  // cp_size=4, num_chunks=8:
  //   seq0 (len 6, chunk_len=1): rank r -> {r}+{7-r}∩[0,6)
  //     r=0 {0}; r=1 {1}; r=2 {2,5}; r=3 {3,4}
  //   seq1 (len 8, chunk_len=1): rank r -> {r}+{7-r}, all in-range
  //     r=0 {0,7}; r=1 {1,6}; r=2 {2,5}; r=3 {3,4}; (offset 6)
  std::vector<Expect> expectations = {
      {2, 0, {0, 1, 6, 7, 12, 13}, {2, 4}},
      {2, 1, {2, 3, 4, 5, 8, 9, 10, 11}, {4, 4}},
      {4, 0, {0, 6, 13}, {1, 2}},
      {4, 1, {1, 7, 12}, {1, 2}},
      {4, 2, {2, 5, 8, 11}, {2, 2}},
      {4, 3, {3, 4, 9, 10}, {2, 2}},
  };

  for (const auto& e : expectations) {
    auto fi = make_forward_input(tokens, positions, q_seq_lens);
    cp_partition_inplace(fi, e.cp_rank, e.cp_size);
    EXPECT_EQ(tensor_to_vec(fi.token_ids), e.tokens)
        << "cp_size=" << e.cp_size << ",cp_rank=" << e.cp_rank;
    EXPECT_EQ(fi.input_params.attention.host.q_seq_lens,
              to_layout_seq_lens(e.q_lens))
        << "cp_size=" << e.cp_size << ",cp_rank=" << e.cp_rank;
  }
}

}  // namespace
}  // namespace xllm::cp
