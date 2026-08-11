/* Copyright 2026 The xLLM Authors.

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

// Unit tests for the xllm_ops::build_cp_context torch op (zigzag CP index
// math). This op is the C++ lowering of xllm/python/model_executor/cp_utils.py
// build_cp_context; these tests port that module's CPU pytest properties.
//
// build_cp_context is registered on CompositeExplicitAutograd and is pure host
// index math, so it runs on CPU with no NPU. We drive it through the real torch
// dispatcher (findSchemaOrThrow + callBoxed) so the registered schema and impl
// are exercised end to end, then check the four invariants that make the plan
// correct: the shard/all-gather/merge round-trip is exact, shards are disjoint
// and complete, zigzag balances per-rank attention work, and gathering query
// rows over their causal KV prefixes reproduces full causal attention.

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "core/kernels/xllm_torch_ops.h"

namespace xllm {
namespace {

// Field-named view of the op's 9-tuple return, materialized from the boxed
// dispatcher call so tests read like the Python CpContext.
struct CpContext {
  torch::Tensor shard_index;
  torch::Tensor shard_gather_index;
  torch::Tensor shard_valid_mask;
  torch::Tensor restore_index;
  torch::Tensor query_index;
  torch::Tensor kv_gather_index;
  std::vector<int64_t> q_cu_seqlens;
  std::vector<int64_t> kv_cu_seqlens;
  int64_t total_local;
};

std::vector<int64_t> to_int_vector(const c10::IValue& value) {
  std::vector<int64_t> out;
  for (const auto& element : value.toListRef()) {
    out.push_back(element.toInt());
  }
  return out;
}

CpContext build_cp_context(const std::vector<int64_t>& seq_lens,
                           int64_t cp_size,
                           int64_t cp_rank) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::build_cp_context", "");
  std::vector<c10::IValue> stack{c10::IValue(c10::List<int64_t>(seq_lens)),
                                 cp_size,
                                 cp_rank,
                                 c10::IValue(torch::Device(torch::kCPU))};
  op.callBoxed(&stack);

  CpContext ctx;
  ctx.shard_index = stack[0].toTensor();
  ctx.shard_gather_index = stack[1].toTensor();
  ctx.shard_valid_mask = stack[2].toTensor();
  ctx.restore_index = stack[3].toTensor();
  ctx.query_index = stack[4].toTensor();
  ctx.kv_gather_index = stack[5].toTensor();
  ctx.q_cu_seqlens = to_int_vector(stack[6]);
  ctx.kv_cu_seqlens = to_int_vector(stack[7]);
  ctx.total_local = stack[8].toInt();
  return ctx;
}

// Select this rank's rows from a global packed tensor, zeroing padding rows —
// mirrors cp_utils.cp_shard_rows.
torch::Tensor cp_shard_rows(const torch::Tensor& x, const CpContext& ctx) {
  auto local = x.index_select(0, ctx.shard_gather_index);
  auto mask = ctx.shard_valid_mask.view({-1, 1}).to(local.dtype());
  return local * mask;
}

// Concatenate rank-major shards, mirroring all_gather(dim=0, "cp").
torch::Tensor emulate_all_gather(const std::vector<torch::Tensor>& shards) {
  return torch::cat(shards, 0);
}

int64_t total_tokens(const std::vector<int64_t>& seq_lens) {
  int64_t total = 0;
  for (const int64_t length : seq_lens) {
    total += length;
  }
  return total;
}

class BuildCpContextTest : public ::testing::TestWithParam<
                               std::tuple<std::vector<int64_t>, int64_t>> {
 protected:
  static void SetUpTestSuite() { xllm::ensure_xllm_torch_ops_registered(); }
};

// merge(all_gather(shard(x))) == x for arbitrary lengths and ranks.
TEST_P(BuildCpContextTest, ShardMergeRoundTrip) {
  const auto& seq_lens = std::get<0>(GetParam());
  const int64_t cp_size = std::get<1>(GetParam());
  const int64_t total = total_tokens(seq_lens);
  auto x = torch::arange(total * 3, torch::kFloat32).reshape({total, 3});

  std::vector<torch::Tensor> shards;
  for (int64_t r = 0; r < cp_size; ++r) {
    shards.push_back(cp_shard_rows(x, build_cp_context(seq_lens, cp_size, r)));
  }
  auto gathered = emulate_all_gather(shards);

  // Any rank's context restores the same global order (restore_index is
  // rank-independent).
  auto ctx0 = build_cp_context(seq_lens, cp_size, 0);
  auto restored = gathered.index_select(0, ctx0.restore_index);
  EXPECT_TRUE(torch::equal(restored, x));
}

// Every real global row is owned by exactly one rank.
TEST_P(BuildCpContextTest, ShardsAreDisjointAndComplete) {
  const auto& seq_lens = std::get<0>(GetParam());
  const int64_t cp_size = std::get<1>(GetParam());
  const int64_t total = total_tokens(seq_lens);

  std::vector<int32_t> owner(total, -1);
  for (int64_t r = 0; r < cp_size; ++r) {
    auto ctx = build_cp_context(seq_lens, cp_size, r);
    auto real = ctx.shard_index.masked_select(ctx.shard_valid_mask);
    for (int64_t i = 0; i < real.size(0); ++i) {
      const int64_t g = real[i].item<int64_t>();
      ASSERT_GE(g, 0);
      ASSERT_LT(g, total);
      EXPECT_EQ(owner[g], -1) << "row " << g << " owned by >1 rank";
      owner[g] = static_cast<int32_t>(r);
    }
  }
  for (int64_t g = 0; g < total; ++g) {
    EXPECT_NE(owner[g], -1) << "row " << g << " owned by no rank";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Zigzag,
    BuildCpContextTest,
    ::testing::Combine(::testing::Values(std::vector<int64_t>{8},
                                         std::vector<int64_t>{16, 24},
                                         std::vector<int64_t>{7},
                                         std::vector<int64_t>{5, 13, 2},
                                         std::vector<int64_t>{1}),
                       ::testing::Values<int64_t>(2, 4)));

// Zigzag equalizes per-rank attention work (sum of causal prefixes) for a
// length divisible by 2*cp_size.
TEST(BuildCpContextInvariants, QueryLoadIsBalanced) {
  for (const int64_t cp_size : {int64_t{2}, int64_t{4}}) {
    const std::vector<int64_t> seq_lens{4 * cp_size * 2};
    int64_t expected = -1;
    for (int64_t r = 0; r < cp_size; ++r) {
      auto ctx = build_cp_context(seq_lens, cp_size, r);
      ASSERT_FALSE(ctx.kv_cu_seqlens.empty());
      const int64_t mass = ctx.kv_cu_seqlens.back();
      if (expected < 0) {
        expected = mass;
      } else {
        EXPECT_EQ(mass, expected) << "imbalanced at cp_size=" << cp_size;
      }
    }
  }
}

// Gathering query rows + causal KV prefixes reproduces full causal attention:
// emulate one FIA call per segment on CPU (softmax over the segment's causal
// prefix) and check the reassembled global output equals dense causal
// attention over the whole sequence.
TEST(BuildCpContextInvariants, PackedAttentionMatchesReference) {
  torch::manual_seed(0);
  const std::vector<std::vector<int64_t>> seq_lens_cases{
      {8}, {16, 24}, {7}, {5, 13, 2}};
  const int64_t dim = 4;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));

  for (const auto& seq_lens : seq_lens_cases) {
    for (const int64_t cp_size : {int64_t{2}, int64_t{4}}) {
      const int64_t total = total_tokens(seq_lens);
      auto q = torch::randn({total, dim});
      auto k = torch::randn({total, dim});
      auto v = torch::randn({total, dim});

      // Reference: dense per-sequence causal attention in global order.
      auto ref = torch::zeros({total, dim});
      int64_t base = 0;
      for (const int64_t length : seq_lens) {
        for (int64_t i = 0; i < length; ++i) {
          auto qi = q[base + i];
          auto kk = k.slice(0, base, base + i + 1);
          auto vv = v.slice(0, base, base + i + 1);
          auto w = torch::softmax(torch::matmul(kk, qi) * scale, 0);
          ref[base + i] = torch::matmul(w, vv);
        }
        base += length;
      }

      // CP path: each rank gathers its query rows and their causal KV prefixes,
      // runs segment-local causal attention, scatters back; then merge.
      std::vector<torch::Tensor> out_shards;
      for (int64_t r = 0; r < cp_size; ++r) {
        auto ctx = build_cp_context(seq_lens, cp_size, r);
        auto q_local = cp_shard_rows(q, ctx);
        auto q_real = q_local.index_select(0, ctx.query_index);
        auto out_real = torch::zeros({q_real.size(0), dim});
        int64_t q_prev = 0;
        int64_t kv_prev = 0;
        for (size_t si = 0; si < ctx.q_cu_seqlens.size(); ++si) {
          const int64_t q_end = ctx.q_cu_seqlens[si];
          const int64_t kv_end = ctx.kv_cu_seqlens[si];
          auto seg_kv_idx = ctx.kv_gather_index.slice(0, kv_prev, kv_end);
          auto seg_k = k.index_select(0, seg_kv_idx);
          auto seg_v = v.index_select(0, seg_kv_idx);
          const int64_t qcount = q_end - q_prev;
          const int64_t prefix = kv_end - kv_prev;
          const int64_t start = prefix - qcount;  // right-aligned causal
          for (int64_t j = 0; j < qcount; ++j) {
            const int64_t allowed = start + j + 1;
            auto seg_q_j = q_real[q_prev + j];
            auto w = torch::softmax(
                torch::matmul(seg_k.slice(0, 0, allowed), seg_q_j) * scale, 0);
            out_real[q_prev + j] = torch::matmul(w, seg_v.slice(0, 0, allowed));
          }
          q_prev = q_end;
          kv_prev = kv_end;
        }
        auto out_local = torch::zeros({ctx.total_local, dim});
        out_local.index_copy_(0, ctx.query_index, out_real);
        out_shards.push_back(out_local);
      }

      auto gathered = emulate_all_gather(out_shards);
      auto ctx0 = build_cp_context(seq_lens, cp_size, 0);
      auto merged = gathered.index_select(0, ctx0.restore_index);
      EXPECT_TRUE(torch::allclose(merged, ref, /*rtol=*/1e-4, /*atol=*/1e-5))
          << "max abs diff = " << (merged - ref).abs().max().item<float>()
          << " (cp_size=" << cp_size << ")";
    }
  }
}

TEST(BuildCpContextInvariants, CpSizeOneRejected) {
  xllm::ensure_xllm_torch_ops_registered();
  EXPECT_THROW(build_cp_context({8}, 1, 0), c10::Error);
}

}  // namespace
}  // namespace xllm
