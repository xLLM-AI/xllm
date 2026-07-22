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

// Pure-CPU unit tests for DSAMetadataBuilder::build_cp_local_metadata (M0 of
// the DeepSeek-V4 DSA-CP plan). These tests do not touch any NPU device; they
// validate the host-side token-partition math against hand-computed golden
// vectors, including the worked example from the vllm-ascend
// `_build_local_token_metadata` docstring.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/dsa_metadata_builder.h"

namespace xllm::layer {
namespace {

torch::Tensor i32(const std::vector<int32_t>& v) {
  return torch::tensor(v, torch::dtype(torch::kInt32).device(torch::kCPU));
}

std::vector<int64_t> to_vec(const torch::Tensor& t) {
  auto c = t.to(torch::kCPU).to(torch::kInt64).contiguous();
  return std::vector<int64_t>(c.data_ptr<int64_t>(),
                              c.data_ptr<int64_t>() + c.numel());
}

// query_start_loc / seq_lens for the docstring example:
//   9 requests, seq_lens = [1..9], pure prefill so q_lens == seq_lens,
//   query_start_loc = [0,1,3,6,10,15,21,28,36,45], num_input_tokens = 45.
torch::Tensor docstring_qsl() {
  return i32({0, 1, 3, 6, 10, 15, 21, 28, 36, 45});
}
torch::Tensor docstring_seq_lens() { return i32({1, 2, 3, 4, 5, 6, 7, 8, 9}); }

}  // namespace

// Rank 1 of the docstring example is the canonical golden vector.
TEST(DsaCpMetadataTest, DocstringRank1) {
  auto cp = DSAMetadataBuilder::build_cp_local_metadata(
      docstring_qsl(), docstring_seq_lens(), /*cp_size=*/3, /*cp_rank=*/1);

  EXPECT_EQ(cp.num_tokens_pad, 45);
  EXPECT_EQ(cp.tokens_per_rank, 15);
  EXPECT_EQ(cp.local_start, 15);
  EXPECT_EQ(cp.local_end, 30);
  EXPECT_EQ(to_vec(cp.local_query_start_loc),
            (std::vector<int64_t>{0, 0, 0, 0, 0, 0, 6, 13, 15, 15}));
  EXPECT_EQ(to_vec(cp.local_seq_lens),
            (std::vector<int64_t>{0, 0, 0, 0, 0, 6, 7, 2, 0}));
  // local_kv_start_loc is the leading-0 cumsum of local_seq_lens. The last
  // rank's queries see the full KV, so this differs from local_query_start_loc.
  EXPECT_EQ(to_vec(cp.local_kv_start_loc),
            (std::vector<int64_t>{0, 0, 0, 0, 0, 0, 6, 13, 15, 15}));
}

TEST(DsaCpMetadataTest, DocstringRank0) {
  auto cp = DSAMetadataBuilder::build_cp_local_metadata(
      docstring_qsl(), docstring_seq_lens(), /*cp_size=*/3, /*cp_rank=*/0);
  EXPECT_EQ(cp.local_start, 0);
  EXPECT_EQ(cp.local_end, 15);
  EXPECT_EQ(to_vec(cp.local_query_start_loc),
            (std::vector<int64_t>{0, 1, 3, 6, 10, 15, 15, 15, 15, 15}));
  EXPECT_EQ(to_vec(cp.local_seq_lens),
            (std::vector<int64_t>{1, 2, 3, 4, 5, 0, 0, 0, 0}));
}

TEST(DsaCpMetadataTest, DocstringRank2) {
  auto cp = DSAMetadataBuilder::build_cp_local_metadata(
      docstring_qsl(), docstring_seq_lens(), /*cp_size=*/3, /*cp_rank=*/2);
  EXPECT_EQ(cp.local_start, 30);
  EXPECT_EQ(cp.local_end, 45);
  EXPECT_EQ(to_vec(cp.local_query_start_loc),
            (std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 6, 15}));
  EXPECT_EQ(to_vec(cp.local_seq_lens),
            (std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 8, 9}));
}

// The local query lengths across all ranks must partition the global tokens
// exactly (no token dropped or double-counted).
TEST(DsaCpMetadataTest, RanksPartitionAllQueryTokens) {
  const int32_t cp_size = 3;
  int64_t total = 0;
  for (int32_t r = 0; r < cp_size; ++r) {
    auto cp = DSAMetadataBuilder::build_cp_local_metadata(
        docstring_qsl(), docstring_seq_lens(), cp_size, r);
    total += to_vec(cp.local_query_start_loc).back();  // local total tokens
  }
  EXPECT_EQ(total, 45);
}

// cp_size == 1 must be the identity view.
TEST(DsaCpMetadataTest, CpSizeOneIsIdentity) {
  auto cp = DSAMetadataBuilder::build_cp_local_metadata(
      docstring_qsl(), docstring_seq_lens(), /*cp_size=*/1, /*cp_rank=*/0);
  EXPECT_EQ(cp.num_tokens_pad, 45);
  EXPECT_EQ(cp.tokens_per_rank, 45);
  EXPECT_EQ(cp.local_start, 0);
  EXPECT_EQ(cp.local_end, 45);
  EXPECT_EQ(to_vec(cp.local_query_start_loc), to_vec(docstring_qsl()));
  EXPECT_EQ(to_vec(cp.local_seq_lens), to_vec(docstring_seq_lens()));
  // cp_size==1: local_kv_start_loc is the cumsum of the full seq_lens, which
  // for pure prefill equals the query cu-seqlens.
  EXPECT_EQ(to_vec(cp.local_kv_start_loc), to_vec(docstring_qsl()));
}

// Non-divisible token count must pad up to a multiple of cp_size, and a
// boundary-crossing request must land its tail on the later rank with the full
// KV length visible there.
TEST(DsaCpMetadataTest, PaddingAndBoundaryCross) {
  auto qsl = i32({0, 3, 5});    // 2 reqs, 5 tokens
  auto seq_lens = i32({3, 2});  // pure prefill

  auto r0 = DSAMetadataBuilder::build_cp_local_metadata(
      qsl, seq_lens, /*cp_size=*/2, /*cp_rank=*/0);
  EXPECT_EQ(r0.num_tokens_pad, 6);
  EXPECT_EQ(r0.tokens_per_rank, 3);
  EXPECT_EQ(r0.local_start, 0);
  EXPECT_EQ(r0.local_end, 3);
  EXPECT_EQ(to_vec(r0.local_query_start_loc), (std::vector<int64_t>{0, 3, 3}));
  EXPECT_EQ(to_vec(r0.local_seq_lens), (std::vector<int64_t>{3, 0}));
  EXPECT_EQ(to_vec(r0.local_kv_start_loc), (std::vector<int64_t>{0, 3, 3}));

  auto r1 = DSAMetadataBuilder::build_cp_local_metadata(
      qsl, seq_lens, /*cp_size=*/2, /*cp_rank=*/1);
  EXPECT_EQ(r1.local_start, 3);
  EXPECT_EQ(r1.local_end, 6);
  EXPECT_EQ(to_vec(r1.local_query_start_loc), (std::vector<int64_t>{0, 0, 2}));
  EXPECT_EQ(to_vec(r1.local_seq_lens), (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(to_vec(r1.local_kv_start_loc), (std::vector<int64_t>{0, 0, 2}));
}

// A single long request split across ranks is the real-world DSA-CP prefill
// case (the gpqa regression): every rank's local queries causally see the
// FULL KV stream, so local_kv_start_loc (ori_kv cu-seqlens) must sum to the
// full length on every rank -- and must DIFFER from local_query_start_loc on
// the later ranks, which is exactly the bug the ori_kv path had.
TEST(DsaCpMetadataTest, SingleLongRequestOriKvSeesFullStream) {
  auto qsl = i32({0, 332});    // 1 request, 332 tokens
  auto seq_lens = i32({332});  // pure prefill

  auto r0 = DSAMetadataBuilder::build_cp_local_metadata(
      qsl, seq_lens, /*cp_size=*/2, /*cp_rank=*/0);
  // rank0 owns queries [0,166): it only sees KV [0,166), so here the ori_kv
  // cu-seqlens coincides with the local query cu-seqlens.
  EXPECT_EQ(to_vec(r0.local_query_start_loc), (std::vector<int64_t>{0, 166}));
  EXPECT_EQ(to_vec(r0.local_seq_lens), (std::vector<int64_t>{166}));
  EXPECT_EQ(to_vec(r0.local_kv_start_loc), (std::vector<int64_t>{0, 166}));

  auto r1 = DSAMetadataBuilder::build_cp_local_metadata(
      qsl, seq_lens, /*cp_size=*/2, /*cp_rank=*/1);
  // rank1 owns queries [166,332): its last query (pos 331) sees the FULL KV
  // [0,332). local_query_start_loc sums to 166 (local q count) but the ori_kv
  // cu-seqlens must sum to 332 (full KV) -- they MUST differ.
  EXPECT_EQ(to_vec(r1.local_query_start_loc), (std::vector<int64_t>{0, 166}));
  EXPECT_EQ(to_vec(r1.local_seq_lens), (std::vector<int64_t>{332}));
  EXPECT_EQ(to_vec(r1.local_kv_start_loc), (std::vector<int64_t>{0, 332}));
  EXPECT_NE(to_vec(r1.local_kv_start_loc), to_vec(r1.local_query_start_loc));
}

// RoPE tables padded to num_tokens_pad must be sliced to [local_start,
// local_end).
TEST(DsaCpMetadataTest, RopeTablesSlicedToLocalInterval) {
  const int64_t rope_dim = 4;
  // arange table so the slice is trivially checkable: row i == [i,i,i,i].
  auto full = torch::arange(45, torch::dtype(torch::kFloat32))
                  .view({45, 1})
                  .expand({45, rope_dim})
                  .contiguous();

  auto cp = DSAMetadataBuilder::build_cp_local_metadata(docstring_qsl(),
                                                        docstring_seq_lens(),
                                                        /*cp_size=*/3,
                                                        /*cp_rank=*/1,
                                                        full,
                                                        full);

  ASSERT_TRUE(cp.local_cos.defined());
  ASSERT_TRUE(cp.local_sin.defined());
  EXPECT_EQ(cp.local_cos.size(0), 15);
  EXPECT_EQ(cp.local_cos.size(1), rope_dim);
  // Rows should be positions 15..29.
  EXPECT_FLOAT_EQ(cp.local_cos[0][0].item<float>(), 15.0f);
  EXPECT_FLOAT_EQ(cp.local_cos[14][0].item<float>(), 29.0f);
}

// Empty batch must not crash and returns a single-element leading-0 cumsum.
TEST(DsaCpMetadataTest, EmptyBatch) {
  auto qsl = i32({0});
  auto seq_lens = torch::zeros({0}, torch::dtype(torch::kInt32));
  auto cp = DSAMetadataBuilder::build_cp_local_metadata(
      qsl, seq_lens, /*cp_size=*/2, /*cp_rank=*/0);
  EXPECT_EQ(cp.num_tokens_pad, 0);
  EXPECT_EQ(to_vec(cp.local_query_start_loc), (std::vector<int64_t>{0}));
  EXPECT_EQ(cp.local_seq_lens.numel(), 0);
}

// Invalid cp_rank must be rejected.
TEST(DsaCpMetadataTest, RejectsInvalidRank) {
  EXPECT_THROW(DSAMetadataBuilder::build_cp_local_metadata(
                   docstring_qsl(), docstring_seq_lens(), 2, 2),
               c10::Error);
}

}  // namespace xllm::layer
