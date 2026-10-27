/* Copyright 2026 The xLLM Authors.

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

#include <cmath>
#include <cstdint>
#include <limits>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using kernel::mlu::dcp_correct_attn_out;
using kernel::mlu::dcp_correct_attn_transpose;

// Frozen FP32 arithmetic from xllm-dcp-attn-out/reference.py. The separate
// concatenated-attention oracle below does not use this correction formula.
torch::Tensor reference(const torch::Tensor& src,
                        const torch::Tensor& lse,
                        const torch::Tensor& slots,
                        int64_t rank,
                        bool base_e) {
  const torch::Tensor raw = lse.to(torch::kFloat32);
  const torch::Tensor clean =
      raw.masked_fill(torch::isnan(raw) | (raw == INFINITY), -INFINITY);
  torch::Tensor maximum = std::get<0>(clean.max(0));
  maximum.masked_fill_(maximum == -INFINITY, 0);
  const torch::Tensor shifted = clean - maximum;
  const torch::Tensor total =
      base_e ? shifted.exp().sum(0).log() : shifted.exp2().sum(0).log2();
  torch::Tensor diff = raw[rank] - (total + maximum);
  diff.masked_fill_(torch::isnan(diff) | (diff == INFINITY), -INFINITY);
  const torch::Tensor factor = base_e ? diff.exp() : diff.exp2();
  const torch::Tensor zero = (slots < 0).unsqueeze(-1) | (factor == 0);
  return (src.to(torch::kFloat32) * factor.unsqueeze(-1))
      .masked_fill(zero.unsqueeze(-1), 0)
      .to(src.scalar_type());
}

void check_close(const torch::Tensor& actual,
                 const torch::Tensor& expected,
                 double atol = 1e-3) {
  const torch::Tensor a = actual.cpu().to(torch::kFloat32);
  const torch::Tensor e = expected.cpu().to(torch::kFloat32);
  ASSERT_EQ(a.sizes(), e.sizes());
  EXPECT_TRUE(torch::allclose(a, e, 1e-2, atol, /*equal_nan=*/true));
  EXPECT_TRUE((a.masked_select(e == 0) == 0).all().item<bool>());
}

class DcpCorrectionTest : public ::testing::Test {
 protected:
  void SetUp() override { torch::manual_seed(2026); }
  const torch::Device device{torch::kPrivateUse1, 0};

  void check_case(const torch::Tensor& src,
                  const torch::Tensor& lse,
                  const torch::Tensor& slots,
                  bool base_e = true) {
    const torch::DeviceGuard guard(device);
    const torch::Tensor cpu_src = src.cpu();
    const torch::Tensor cpu_lse = lse.cpu();
    const torch::Tensor cpu_slots = slots.cpu();
    const torch::Tensor source_bits = cpu_src.contiguous().view(torch::kUInt8);
    const torch::Tensor lse_bits = cpu_lse.contiguous().view(torch::kUInt8);
    const int64_t b = src.size(0);
    const int64_t h = src.size(1);
    const int64_t v = src.size(2);
    for (int64_t rank = 0; rank < lse.size(0); ++rank) {
      SCOPED_TRACE(::testing::Message()
                   << "B=" << b << " rank=" << rank << " base_e=" << base_e);
      const torch::Tensor expected =
          reference(cpu_src, cpu_lse, cpu_slots, rank, base_e);
      torch::Tensor storage = torch::full({src.numel() + 2}, 19, src.options());
      torch::Tensor dst = storage.slice(0, 1, -1).view({h, b, v});
      dst.fill_(NAN);
      dcp_correct_attn_transpose(src, lse, slots, rank, dst, base_e);
      check_close(dst, expected.transpose(0, 1));
      EXPECT_TRUE(dst.is_contiguous());
      EXPECT_EQ(storage[0].item<float>(), 19);
      EXPECT_EQ(storage[-1].item<float>(), 19);
      EXPECT_TRUE(torch::equal(src.cpu().contiguous().view(torch::kUInt8),
                               source_bits));
      EXPECT_TRUE(
          torch::equal(lse.cpu().contiguous().view(torch::kUInt8), lse_bits));
      EXPECT_TRUE(torch::equal(slots.cpu(), cpu_slots));
      torch::Tensor backing =
          torch::full({src.numel() * 2 + 2}, 37, src.options());
      torch::Tensor out = backing.slice(0, 1, -1, 2).view_as(src);
      out.copy_(src);
      dcp_correct_attn_out(out, lse, slots, rank, base_e);
      check_close(out, expected);
      EXPECT_EQ(backing[0].item<float>(), 37);
      EXPECT_TRUE((backing.slice(0, 2, -1, 2) == 37).all().item<bool>());
      out = src.clone(torch::MemoryFormat::Contiguous);
      dcp_correct_attn_out(out, lse, slots, rank, base_e);
      check_close(out, expected);
    }
  }

  void check_rows(int64_t b) {
    const auto opts = torch::TensorOptions().device(device);
    torch::Tensor src =
        torch::randn({b, 64, 512}, opts.dtype(torch::kBFloat16));
    torch::Tensor lse =
        torch::randn({4, b, 64}, opts.dtype(torch::kFloat32)) * 3;
    torch::Tensor slots = torch::arange(b, opts.dtype(torch::kInt32));
    check_case(src, lse, slots);
  }
};

// Use exec-based death tests: never fork an initialized MLU runtime.
TEST(DcpCorrectionDeathTest, RejectsInvalidDestinationsAndRank) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (int32_t kind = 0; kind < 4; ++kind) {
    EXPECT_DEATH(
        {
          const torch::Device device(torch::kPrivateUse1, 0);
          const auto opts = torch::TensorOptions().device(device);
          torch::Tensor src =
              torch::zeros({3, 8, 16}, opts.dtype(torch::kBFloat16));
          const torch::Tensor lse =
              torch::zeros({4, 3, 8}, opts.dtype(torch::kFloat32));
          const torch::Tensor slots =
              torch::arange(3, opts.dtype(torch::kInt32));
          torch::Tensor dst = torch::empty({8, 3, 16}, src.options());
          int64_t rank = 0;
          switch (kind) {
            case 0:
              dst = src.view({8, 3, 16});
              break;
            case 1:
              dst = dst.slice(0, 0, 7);
              break;
            case 2:
              rank = 4;
              break;
            case 3: {
              torch::Tensor storage =
                  torch::empty({src.numel() * 2}, src.options());
              src = storage.slice(0, 0, src.numel()).view_as(src);
              dst = storage.slice(0, src.numel()).view({8, 3, 16});
              break;
            }
          }
          dcp_correct_attn_transpose(src, lse, slots, rank, dst);
        },
        "Check failed");
  }
}

TEST_F(DcpCorrectionTest, DecodeRowsAndDispatchBoundaries) {
  for (int64_t b = 1; b <= 128; ++b) {
    check_rows(b);
  }
  for (int64_t b : {0, 129, 255, 256, 257}) {
    check_rows(b);
  }
}

TEST_F(DcpCorrectionTest, Prefill8192MatchesReference) {
  const auto opts = torch::TensorOptions().device(device);
  const torch::Tensor src =
      torch::randn({8192, 64, 512}, opts.dtype(torch::kBFloat16));
  const torch::Tensor lse =
      torch::randn({4, 8192, 64}, opts.dtype(torch::kFloat32));
  torch::Tensor slots = torch::arange(8192, opts.dtype(torch::kInt32));
  slots.slice(0, 0, 8192, 3).fill_(-1);
  const torch::Tensor cpu_src = src.cpu();
  const torch::Tensor cpu_lse = lse.cpu();
  const torch::Tensor cpu_slots = slots.cpu();
  for (int64_t rank = 0; rank < 4; ++rank) {
    torch::Tensor out = src.clone();
    dcp_correct_attn_out(out, lse, slots, rank);
    check_close(out, reference(cpu_src, cpu_lse, cpu_slots, rank, true));
  }
}

TEST_F(DcpCorrectionTest, ShardedHeadPrefillBoundariesAndPadding) {
  const auto opts = torch::TensorOptions().device(device);
  for (int64_t b : {0, 1, 128, 129, 255, 256, 257, 513}) {
    const torch::Tensor src =
        torch::randn({b, 32, 512}, opts.dtype(torch::kBFloat16));
    const torch::Tensor lse =
        torch::randn({4, b, 32}, opts.dtype(torch::kFloat32)) * 3;
    const torch::Tensor rows = torch::arange(b, opts.dtype(torch::kInt32));
    // Each 64-head tile spans two tokens with independent padding states.
    const torch::Tensor slots = torch::where(
        (rows % 4 == 0) | (rows % 4 == 3), torch::full_like(rows, -1), rows);
    check_case(src, lse, slots);
  }
}

TEST_F(DcpCorrectionTest, ShardedHeadPrefill8192MatchesReference) {
  const auto opts = torch::TensorOptions().device(device);
  const torch::Tensor src =
      torch::randn({8192, 32, 512}, opts.dtype(torch::kBFloat16));
  const torch::Tensor lse =
      torch::randn({4, 8192, 32}, opts.dtype(torch::kFloat32));
  torch::Tensor slots = torch::arange(8192, opts.dtype(torch::kInt32));
  slots.slice(0, 1, 8192, 2).fill_(-1);
  const torch::Tensor cpu_src = src.cpu();
  const torch::Tensor cpu_lse = lse.cpu();
  const torch::Tensor cpu_slots = slots.cpu();
  for (int64_t rank = 0; rank < 4; ++rank) {
    SCOPED_TRACE(::testing::Message() << "rank=" << rank);
    torch::Tensor out = src.clone();
    dcp_correct_attn_out(out, lse, slots, rank);
    check_close(out, reference(cpu_src, cpu_lse, cpu_slots, rank, true));
  }
}

TEST_F(DcpCorrectionTest, ShardedHeadPrefillZeroWeightsAndNonfiniteValues) {
  const auto opts = torch::TensorOptions().device(device);
  torch::Tensor src =
      torch::randn({257, 32, 512}, opts.dtype(torch::kBFloat16));
  torch::Tensor lse = torch::zeros({4, 257, 32}, opts.dtype(torch::kFloat32));
  torch::Tensor slots = torch::arange(257, opts.dtype(torch::kInt32));
  lse.select(2, 0).fill_(NAN);
  lse.select(2, 1).fill_(INFINITY);
  lse.select(2, 2).fill_(-INFINITY);
  lse[0].select(1, 3).fill_(-INFINITY);
  src.slice(1, 0, 4).fill_(NAN);
  lse[0].select(1, 4).fill_(-102);
  src.select(1, 4).fill_(INFINITY);
  slots.slice(0, 1, 257, 2).fill_(-1);
  check_case(src, lse, slots);
}

TEST_F(DcpCorrectionTest, NonfinitePaddingAndSubnormalWeights) {
  const auto opts = torch::TensorOptions().device(device);
  for (int64_t b : {1, 32, 128, 129, 256}) {
    for (bool base_e : {true, false}) {
      torch::Tensor src =
          torch::randn({b, 64, 512}, opts.dtype(torch::kBFloat16));
      torch::Tensor lse = torch::zeros({4, b, 64}, opts.dtype(torch::kFloat32));
      torch::Tensor slots = torch::arange(b, opts.dtype(torch::kInt32));
      lse.select(2, 0).fill_(NAN);
      lse.select(2, 1).fill_(INFINITY);
      lse.select(2, 2).fill_(-INFINITY);
      lse[0].select(1, 3).fill_(-INFINITY);
      src.slice(1, 0, 4).fill_(NAN);
      lse.slice(2, 4, 8).fill_(0);
      lse[0].slice(1, 4, 8).fill_(base_e ? -102 : -148);
      src.select(1, 4).fill_(INFINITY);
      src.select(1, 5).fill_(-INFINITY);
      src.select(1, 6).fill_(NAN);
      lse.select(2, 8).fill_(1e20);
      lse.select(2, 9).fill_(-1e20);
      lse.select(2, 10).fill_(10000);
      lse.select(2, 11).fill_(-10000);
      // Both bases must exercise a nonzero FP32 subnormal weight, not just
      // the zero-weight case. Such a weight must preserve an infinite output.
      const torch::Tensor expected =
          reference(src.cpu(), lse.cpu(), slots.cpu(), /*rank=*/0, base_e);
      EXPECT_EQ(expected[0][4][0].item<float>(), INFINITY);
      check_case(src, lse, slots, base_e);
      slots.slice(0, 0, b, 2).fill_(-1);
      check_case(src, lse, slots, base_e);
    }
  }
}

TEST_F(DcpCorrectionTest, StridesDtypesAndMasks) {
  const auto opts = torch::TensorOptions().device(device);
  for (int64_t n : {1, 2, 3, 5, 8}) {
    for (torch::ScalarType dtype :
         {torch::kBFloat16, torch::kFloat16, torch::kFloat32}) {
      torch::Tensor src = torch::randn({4, 5, 34}, opts.dtype(dtype))
                              .slice(0, 1)
                              .slice(2, 0, 34, 2);
      torch::Tensor lse = torch::randn({n, 5, 4}, opts.dtype(dtype))
                              .slice(2, 1)
                              .transpose(1, 2);
      torch::Tensor slots =
          torch::arange(4, opts.dtype(torch::kInt32)).slice(0, 1);
      for (bool base_e : {true, false}) {
        check_case(src, lse, slots, base_e);
      }
    }
  }
}

TEST_F(DcpCorrectionTest, ConcatenatedAttentionOracle) {
  const torch::Tensor scores = torch::randn({4, 3, 64, 7}, torch::kFloat64);
  const torch::Tensor values =
      torch::randn({4, 3, 64, 7, 512}, torch::kFloat64);
  const torch::Tensor local =
      (scores.softmax(-1).unsqueeze(-1) * values).sum(-2);
  const torch::Tensor lse = scores.logsumexp(-1).to(torch::kFloat32).to(device);
  const torch::Tensor full_scores = scores.permute({1, 2, 0, 3}).flatten(-2);
  const torch::Tensor full_values =
      values.permute({1, 2, 0, 3, 4}).flatten(2, 3);
  const torch::Tensor expected =
      (full_scores.softmax(-1).unsqueeze(-1) * full_values).sum(-2);
  const torch::Tensor slots = torch::arange(3, torch::kInt32).to(device);
  torch::Tensor merged = torch::zeros({64, 3, 512}, torch::kFloat32);
  torch::Tensor dst =
      torch::empty({64, 3, 512}, lse.options().dtype(torch::kBFloat16));
  for (int64_t rank = 0; rank < 4; ++rank) {
    const torch::Tensor src = local[rank].to(torch::kBFloat16).to(device);
    dcp_correct_attn_transpose(src, lse, slots, rank, dst);
    merged += dst.cpu().to(torch::kFloat32);
  }
  check_close(merged.transpose(0, 1), expected, /*atol=*/5e-3);
}

}  // namespace
}  // namespace xllm
