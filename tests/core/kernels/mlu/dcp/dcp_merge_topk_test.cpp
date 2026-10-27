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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/kernels/mlu/dcp/dcp_topk_test_utils.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using test::GuardedTopk;
using test::kSentinel;
using test::local_reference;
using test::TopkResult;

using kernel::mlu::dcp_localize_topk;
using kernel::mlu::dcp_merge_topk;

TopkResult merge_reference(const torch::Tensor& scores,
                           const torch::Tensor& slots,
                           const torch::Tensor& mapping,
                           int64_t k) {
  const int64_t d = scores.size(0);
  const int64_t q = scores.size(1);
  const int64_t c = scores.size(2);
  auto output = torch::full({q, k}, -1, torch::kInt32);
  auto lengths = torch::zeros({q}, torch::kInt32);
  const auto values = scores.accessor<float, 3>();
  const auto src = slots.accessor<int32_t, 3>();
  const auto map = mapping.accessor<int32_t, 1>();
  auto dst = output.accessor<int32_t, 2>();
  auto lens = lengths.accessor<int32_t, 1>();
  for (int64_t row = 0; row < q; ++row) {
    if (map[row] < 0) {
      continue;
    }
    std::vector<int64_t> candidates;
    candidates.reserve(d * c);
    for (int64_t index = 0; index < d * c; ++index) {
      if (src[index / c][row][index % c] < 0 ||
          std::isnan(values[index / c][row][index % c])) {
        continue;
      }
      candidates.emplace_back(index);
    }
    std::stable_sort(
        candidates.begin(), candidates.end(), [&](int64_t left, int64_t right) {
          return values[left / c][row][left % c] >
                 values[right / c][row][right % c];
        });
    lens[row] = static_cast<int32_t>(std::min<int64_t>(k, candidates.size()));
    for (int64_t col = 0; col < lens[row]; ++col) {
      const int64_t index = candidates[col];
      dst[row][col] = src[index / c][row][index % c];
    }
  }
  return {output, lengths};
}

std::pair<torch::Tensor, torch::Tensor> score_templates(int64_t d, int64_t c) {
  auto scores = torch::empty({d, 10, c}, torch::kFloat32);
  auto slots = torch::empty({d, 10, c}, torch::kInt32);
  auto values = scores.accessor<float, 3>();
  auto ids = slots.accessor<int32_t, 3>();
  const std::vector<uint32_t> bits{0,
                                   0x80000000,
                                   1,
                                   2,
                                   0x80000001,
                                   0x80000002,
                                   0x3f800000,
                                   0x3f800001,
                                   0xbf800000,
                                   0xbf800001,
                                   0x7f7fffff,
                                   0xff7fffff,
                                   0x7f800000,
                                   0xff800000,
                                   0x7fc00001,
                                   0xffc00001};
  for (int64_t rank = 0; rank < d; ++rank) {
    for (int64_t col = 0; col < c; ++col) {
      const int64_t index = rank * c + col;
      for (int64_t row = 0; row < 10; ++row) {
        ids[rank][row][col] = static_cast<int32_t>((index * 37) % 131072);
      }
      values[rank][0][col] = std::bit_cast<float>(bits[index % bits.size()]);
      values[rank][1][col] = index % 2 ? -0.0f : 0.0f;
      values[rank][2][col] = INFINITY;
      values[rank][3][col] = NAN;
      values[rank][4][col] = -INFINITY;
      ids[rank][4][col] = index < 5 ? static_cast<int32_t>(index) : -1;
      values[rank][5][col] = index < c - 1 ? 1.0f : -1.0f;
      values[rank][6][col] = index < c ? 1.0f : -1.0f;
      values[rank][7][col] = index < c + 1 ? 1.0f : -1.0f;
      values[rank][8][col] = static_cast<float>((index * 7919) % (d * c));
      uint32_t random_bits = static_cast<uint32_t>(index + 72317);
      random_bits ^= random_bits << 13;
      random_bits ^= random_bits >> 17;
      random_bits ^= random_bits << 5;
      values[rank][9][col] = std::bit_cast<float>(random_bits);
      ids[rank][9][col] = index % 11 == 0 ? -1 : static_cast<int32_t>(index);
    }
  }
  return {scores, slots};
}

// Compare selected scores and candidate membership, allowing arbitrary choices
// within a tied score band. The templates use unique slots within each row.
void check_selection(const TopkResult& actual,
                     const TopkResult& expected,
                     const torch::Tensor& scores,
                     const torch::Tensor& slots) {
  ASSERT_TRUE(torch::equal(actual.second, expected.second));
  const auto src_scores = scores.accessor<float, 3>();
  const auto src_slots = slots.accessor<int32_t, 3>();
  std::vector<std::unordered_map<int32_t, float>> candidates(scores.size(1));
  for (int64_t row = 0; row < scores.size(1); ++row) {
    candidates[row].reserve(scores.size(0) * scores.size(2));
    for (int64_t rank = 0; rank < scores.size(0); ++rank) {
      for (int64_t col = 0; col < scores.size(2); ++col) {
        const int32_t slot = src_slots[rank][row][col];
        const float score = src_scores[rank][row][col];
        if (slot < 0 || std::isnan(score)) {
          continue;
        }
        ASSERT_TRUE(candidates[row].emplace(slot, score).second);
      }
    }
  }
  const auto got = actual.first.accessor<int32_t, 2>();
  const auto want = expected.first.accessor<int32_t, 2>();
  const auto lengths = expected.second.accessor<int32_t, 1>();
  for (int64_t row = 0; row < actual.first.size(0); ++row) {
    SCOPED_TRACE(::testing::Message() << "row=" << row);
    const auto& valid = candidates[row % scores.size(1)];
    std::unordered_set<int32_t> seen;
    seen.reserve(lengths[row]);
    for (int64_t col = 0; col < lengths[row]; ++col) {
      const auto found = valid.find(got[row][col]);
      ASSERT_NE(found, valid.end());
      ASSERT_TRUE(seen.emplace(got[row][col]).second);
      ASSERT_EQ(found->second, valid.at(want[row][col]));
    }
    for (int64_t col = lengths[row]; col < actual.first.size(1); ++col) {
      ASSERT_EQ(got[row][col], -1);
    }
  }
}

class DcpMergeTopkTest : public ::testing::Test {
 protected:
  const torch::Device device{torch::kPrivateUse1, 0};

  void check_case(int64_t q,
                  int64_t d,
                  int64_t c,
                  int64_t k,
                  int32_t layout = 0,
                  bool compose = false) {
    SCOPED_TRACE(::testing::Message() << "Q=" << q << " D=" << d << " C=" << c
                                      << " K=" << k << " layout=" << layout);
    const torch::DeviceGuard guard(device);
    const auto templates = score_templates(d, c);
    const auto selector = torch::arange(q, torch::kInt64).remainder(10);
    const auto cpu_scores = templates.first.index_select(1, selector);
    const auto cpu_slots = templates.second.index_select(1, selector);
    auto cpu_mapping = torch::zeros({q}, torch::kInt32);
    cpu_mapping.slice(0, 8, q, 11).fill_(-1);
    const auto reference = merge_reference(templates.first,
                                           templates.second,
                                           torch::zeros({10}, torch::kInt32),
                                           k);
    TopkResult expected{reference.first.index_select(0, selector),
                        reference.second.index_select(0, selector)};
    expected.first.index_put_({cpu_mapping < 0}, -1);
    expected.second.index_put_({cpu_mapping < 0}, 0);
    auto scores = cpu_scores.to(device);
    auto slots = cpu_slots.to(device);
    auto mapping = cpu_mapping.to(device);
    if (layout == 1) {
      scores = torch::empty({d, q * 2, c * 2}, scores.options())
                   .slice(1, 0, q * 2, 2)
                   .slice(2, 0, c * 2, 2);
      slots = torch::empty({d, q * 3, c * 3}, slots.options())
                  .slice(1, 0, q * 3, 3)
                  .slice(2, 1, c * 3, 3);
    } else if (layout == 2) {
      scores = torch::empty({c, q, d}, scores.options()).permute({2, 1, 0});
      slots = torch::empty({c, q, d}, slots.options()).permute({2, 1, 0});
    }
    scores.copy_(cpu_scores);
    slots.copy_(cpu_slots);
    if (layout != 0) {
      mapping = torch::full({q * 3}, kSentinel, mapping.options())
                    .slice(0, 0, q * 3, 3);
      mapping.copy_(cpu_mapping);
    }
    GuardedTopk out(q,
                    k,
                    device,
                    /*strided=*/layout != 0,
                    /*strided_lens=*/layout != 0);
    dcp_merge_topk(scores, slots, mapping, k, out.output, out.lengths);
    TopkResult actual{out.output.cpu(), out.lengths.cpu()};
    check_selection(actual, expected, templates.first, templates.second);
    out.check(actual);
    EXPECT_TRUE(torch::equal(scores.cpu().contiguous().view(torch::kInt32),
                             cpu_scores.contiguous().view(torch::kInt32)));
    EXPECT_TRUE(torch::equal(slots.cpu(), cpu_slots));
    EXPECT_TRUE(torch::equal(mapping.cpu(), cpu_mapping));
    if (compose) {
      GuardedTopk local(q, k, device);
      for (int32_t rank = 0; rank < d; ++rank) {
        dcp_localize_topk(out.output,
                          out.lengths,
                          rank,
                          static_cast<int32_t>(d),
                          /*physical_block_size=*/16,
                          local.output,
                          local.lengths);
        local.check(local_reference(actual.first,
                                    actual.second,
                                    rank,
                                    static_cast<int32_t>(d),
                                    /*physical_block_size=*/16));
      }
      out.check(actual);
    }
    mapping.fill_(-1);
    dcp_merge_topk(scores, slots, mapping, k, out.output, out.lengths);
    out.check({torch::full({q, k}, -1, torch::kInt32),
               torch::zeros({q}, torch::kInt32)});
    mapping.copy_(cpu_mapping);
    dcp_merge_topk(scores, slots, mapping, k, out.output, out.lengths);
    actual = {out.output.cpu(), out.lengths.cpu()};
    check_selection(actual, expected, templates.first, templates.second);
    out.check(actual);
  }
};

TEST_F(DcpMergeTopkTest, GenericShapesAndEmptyQueries) {
  for (const auto& [d, q, c, k] :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>{
           {1, 0, 3, 2},
           {1, 3, 1, 1},
           {4, 8, 13, 17},
           {8, 8, 5, 40},
           {4, 8, 65, 129},
           {4, 8, 2048, 1},
           {4, 8, 2048, 17},
           {4, 8, 2048, 2047},
           {4, 128, 17, 68}}) {
    check_case(q, d, c, k);
  }
}

TEST_F(DcpMergeTopkTest, QueryBandsAndIEEEOrdering) {
  for (int64_t q :
       {1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 259, 260, 261}) {
    check_case(q, /*d=*/4, /*c=*/2048, /*k=*/2048);
  }
}

TEST_F(DcpMergeTopkTest, StridedAndTransposedBands) {
  for (int32_t layout : {1, 2}) {
    for (int64_t q : {63, 64, 127, 128, 129, 255, 256, 257, 259, 260, 261}) {
      check_case(q, /*d=*/4, /*c=*/2048, /*k=*/2048, layout);
    }
  }
}

TEST_F(DcpMergeTopkTest, LargeQueriesPreserveAllRows) {
  for (int64_t q :
       {511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 8191, 8192, 8193}) {
    check_case(q, /*d=*/4, /*c=*/2048, /*k=*/2048);
  }
}

TEST_F(DcpMergeTopkTest, GlobalStateSupportsRepeatedOwnershipProjection) {
  for (int64_t q : {5, 64, 128, 257}) {
    check_case(q,
               /*d=*/4,
               /*c=*/2048,
               /*k=*/2048,
               /*layout=*/0,
               /*compose=*/true);
  }
}

TEST_F(DcpMergeTopkTest, ValidNegativeInfinityPrecedesInvalidTail) {
  const torch::DeviceGuard guard(device);
  // Duplicate valid slots remain separate candidates, including slot zero.
  auto scores = torch::tensor({NAN,
                               -INFINITY,
                               -INFINITY,
                               -INFINITY,
                               1.0f,
                               -INFINITY,
                               NAN,
                               -INFINITY})
                    .view({2, 1, 4})
                    .to(device);
  auto slots = torch::tensor({7, 0, -1, 0, -1, 16, 32, -1}, torch::kInt32)
                   .view({2, 1, 4})
                   .to(device);
  auto mapping = torch::zeros({1}, slots.options());
  for (int64_t k : {1, 2, 3, 5, 8}) {
    GuardedTopk out(/*q=*/1, k, device);
    dcp_merge_topk(scores, slots, mapping, k, out.output, out.lengths);
    const int64_t count = std::min<int64_t>(k, 3);
    const auto cpu_output = out.output.cpu();
    ASSERT_EQ(out.lengths.cpu().item<int32_t>(), count);
    const auto selected = cpu_output[0].slice(0, 0, count);
    EXPECT_TRUE(((selected == 0) | (selected == 16)).all().item<bool>());
    EXPECT_LE((selected == 0).sum().item<int64_t>(), 2);
    EXPECT_LE((selected == 16).sum().item<int64_t>(), 1);
    EXPECT_TRUE((cpu_output[0].slice(0, count) == -1).all().item<bool>());
    out.check({cpu_output, torch::full({1}, count, torch::kInt32)});
    GuardedTopk local(/*q=*/1, k, device);
    for (int32_t rank : {0, 1}) {
      dcp_localize_topk(out.output,
                        out.lengths,
                        rank,
                        /*size=*/2,
                        /*physical_block_size=*/16,
                        local.output,
                        local.lengths);
      local.check(local_reference(cpu_output,
                                  out.lengths.cpu(),
                                  rank,
                                  /*size=*/2,
                                  /*physical_block_size=*/16));
    }
  }
}

TEST(DcpMergeTopkDeathTest, RejectsInvalidContracts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const torch::Device device{torch::kPrivateUse1, 0};
  const auto opts = torch::TensorOptions().device(device);
  auto scores = torch::zeros({1, 2, 3}, opts.dtype(torch::kFloat32));
  auto slots = torch::zeros({1, 2, 3}, opts.dtype(torch::kInt32));
  auto mapping = torch::zeros({2}, slots.options());
  auto output = torch::empty({2, 3}, slots.options());
  auto lengths = torch::empty({2}, slots.options());
  EXPECT_DEATH(dcp_merge_topk(scores, slots, mapping, 4, output, lengths),
               "topk");
  EXPECT_DEATH(
      dcp_merge_topk(
          scores.to(torch::kFloat64), slots, mapping, 3, output, lengths),
      "scalar_type");
  EXPECT_DEATH(
      dcp_merge_topk(scores, slots, mapping.slice(0, 0, 1), 3, output, lengths),
      "mapping.size");
  auto alias = slots[0];
  EXPECT_DEATH(dcp_merge_topk(scores, slots, mapping, 3, alias, lengths),
               "alias");
  auto overlap = output.as_strided({2, 3}, {1, 1});
  EXPECT_DEATH(dcp_merge_topk(scores, slots, mapping, 3, overlap, lengths),
               "stride");
}

}  // namespace
}  // namespace xllm
