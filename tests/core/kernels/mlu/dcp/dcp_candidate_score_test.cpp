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

#include <ATen/Parallel.h>
#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using Output = std::pair<torch::Tensor, torch::Tensor>;
constexpr int64_t kHeads = 32;
constexpr int64_t kDim = 128;
constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
constexpr double kTolerance = 1e-4;
const torch::Device kDevice{torch::kPrivateUse1, 0};

struct Inputs {
  torch::Tensor q;
  torch::Tensor weights;
  torch::Tensor cache;
  torch::Tensor slots;
  // Valid candidate prefix length, not the KV sequence length.
  torch::Tensor counts;
  int64_t size = 4;
  int64_t rank = 0;
  int64_t page = 16;
};

struct Workload {
  int64_t rows;
  int64_t width;
  int64_t context = 256;
  int64_t batch = 1;
  bool prefill = false;
  int64_t rank = 0;
  int64_t size = 4;
  int64_t page = 16;
  int64_t seed = 20260913;
};

Inputs make_inputs(const Workload& work) {
  torch::manual_seed(work.seed);
  const int64_t tokens = work.rows / work.batch;
  const int64_t logical_page = work.page * work.size;
  const int64_t final_length = work.context + (work.prefill ? 0 : tokens);
  const int64_t pages = (final_length + logical_page - 1) / logical_page;
  const int64_t pool_pages = std::max<int64_t>(1, work.batch * pages);
  std::mt19937 rng(work.seed);
  std::vector<int64_t> page_table(pool_pages);
  std::iota(page_table.begin(), page_table.end(), 0);
  std::shuffle(page_table.begin(), page_table.end(), rng);
  auto slots = torch::full({work.rows, work.width}, -1, torch::kInt32);
  auto counts = torch::zeros({work.rows}, torch::kInt32);
  auto slot_data = slots.accessor<int32_t, 2>();
  auto count_data = counts.accessor<int32_t, 1>();
  for (int64_t seq = 0; seq < work.batch; ++seq) {
    std::vector<int64_t> priorities(pages * work.page);
    std::iota(priorities.begin(), priorities.end(), 0);
    std::shuffle(priorities.begin(), priorities.end(), rng);
    for (int64_t offset = 0; offset < tokens; ++offset) {
      const int64_t row = seq * tokens + offset;
      const int64_t position =
          work.context - (work.prefill ? tokens : 0) + offset + 1;
      const int64_t visible =
          position / logical_page * work.page +
          std::clamp(position % logical_page - work.rank * work.page,
                     int64_t{0},
                     work.page);
      std::vector<int64_t> eligible;
      eligible.reserve(priorities.size());
      std::copy_if(priorities.begin(),
                   priorities.end(),
                   std::back_inserter(eligible),
                   [visible](int64_t slot) { return slot < visible; });
      const int64_t count = std::min<int64_t>(work.width, eligible.size());
      count_data[row] = static_cast<int32_t>(count);
      const int64_t start =
          work.prefill && count > 0 &&
                  static_cast<int64_t>(eligible.size()) > count
              ? rng() % eligible.size()
              : 0;
      for (int64_t col = 0; col < count; ++col) {
        const int64_t candidate = eligible[(start + col) % eligible.size()];
        slot_data[row][col] = static_cast<int32_t>(
            page_table[seq * pages + candidate / work.page] * work.page +
            candidate % work.page);
      }
    }
  }
  auto q = torch::randn({work.batch, tokens, kHeads, kDim}, torch::kBFloat16);
  auto weights =
      (torch::randn({work.batch, tokens, kHeads}) / 32).to(torch::kBFloat16);
  auto cache = torch::randn({pool_pages, work.page, 1, kDim}, torch::kBFloat16);
  if (work.prefill) {
    q = q.view({work.rows, kHeads, kDim});
    weights = weights.view({work.rows, kHeads});
  }
  return {q.to(kDevice),
          weights.to(kDevice),
          cache.to(kDevice),
          slots.to(kDevice),
          counts.to(kDevice),
          work.size,
          work.rank,
          work.page};
}

Output score(const Inputs& in) {
  CHECK_EQ(in.slots.dim(), 2);
  const int64_t rows = in.slots.size(0);
  CHECK_EQ(in.counts.dim(), 1);
  CHECK_EQ(in.counts.size(0), rows);
  CHECK_GT(in.size, 0);
  CHECK_GE(in.rank, 0);
  CHECK_LT(in.rank, in.size);
  CHECK_GT(in.page, 0);
  for (const auto& tensor : {in.q, in.weights, in.cache}) {
    CHECK(tensor.scalar_type() == torch::kBFloat16) << "BF16 required";
  }
  for (const auto& tensor : {in.slots, in.counts}) {
    CHECK(tensor.scalar_type() == torch::kInt32);
  }
  for (const auto& tensor : {in.weights, in.cache, in.slots, in.counts}) {
    CHECK(tensor.device() == in.q.device());
  }
  CHECK_GE(in.q.dim(), 3);
  CHECK_EQ(in.q.size(-2), kHeads);
  CHECK_EQ(in.q.size(-1), kDim);
  CHECK_GE(in.weights.dim(), 2);
  CHECK_EQ(in.weights.size(-1), kHeads);
  CHECK(in.cache.dim() == 2 || in.cache.dim() == 4);
  if (in.cache.dim() == 4) {
    CHECK((in.cache.size(1) == in.page && in.cache.size(2) == 1) ||
          (in.cache.size(1) == 1 && in.cache.size(2) == in.page));
  }
  CHECK_EQ(in.cache.size(-1), kDim);
  const auto query = in.q.view({rows, kHeads, kDim});
  const auto weights = in.weights.view({rows, kHeads});
  const auto cache = in.cache.view({-1, kDim});
  CHECK_GT(cache.size(0), 0);
  auto [scores, slots] = kernel::mlu::dcp_score_candidates(
      query, weights, cache, in.slots, in.counts, in.size, in.rank, in.page);
  return {std::move(scores), std::move(slots)};
}

// Independent CPU FP64 oracle. Only valid candidates are evaluated, with the
// original slot used for global mapping and a clamped slot used for cache
// reads.
Output oracle(const Inputs& in) {
  const int64_t rows = in.slots.size(0);
  const int64_t width = in.slots.size(1);
  const auto query =
      in.q.cpu().to(torch::kFloat64).reshape({rows, kHeads, kDim});
  const auto weights =
      in.weights.cpu().to(torch::kFloat64).reshape({rows, kHeads});
  const auto cache = in.cache.cpu().to(torch::kFloat64).reshape({-1, kDim});
  const auto slots = in.slots.cpu();
  const auto counts = in.counts.cpu();
  auto scores = torch::full({rows, width}, -INFINITY, torch::kFloat64);
  auto global_slots = torch::full({rows, width}, -1, torch::kInt32);
  auto slot_data = slots.accessor<int32_t, 2>();
  auto count_data = counts.accessor<int32_t, 1>();
  auto global_data = global_slots.accessor<int32_t, 2>();
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t count = std::clamp<int64_t>(count_data[row], 0, width);
    const auto columns =
        torch::nonzero(slots[row].slice(0, 0, count) >= 0).view({-1});
    const auto keys = cache.index_select(0,
                                         slots[row]
                                             .index_select(0, columns)
                                             .to(torch::kInt64)
                                             .clamp_max(cache.size(0) - 1));
    const auto dots = torch::mm(keys, query[row].transpose(0, 1)).clamp_min(0);
    scores[row].index_copy_(0, columns, torch::mv(dots, weights[row]));
    const auto col_data = columns.accessor<int64_t, 1>();
    for (int64_t idx = 0; idx < columns.numel(); ++idx) {
      const int64_t col = col_data[idx];
      const int64_t slot = slot_data[row][col];
      global_data[row][col] = static_cast<int32_t>(
          (slot / in.page * in.size + in.rank) * in.page + slot % in.page);
    }
  }
  return {scores, global_slots};
}

Output reference(Inputs in) {
  // Compute FP32 on CPU so the reference cannot use vendor TF32 matmul modes.
  in.q = in.q.cpu();
  in.weights = in.weights.cpu();
  in.cache = in.cache.cpu();
  in.slots = in.slots.cpu();
  in.counts = in.counts.cpu();
  const int64_t rows = in.slots.size(0);
  const int64_t width = in.slots.size(1);
  const auto query = in.q.to(torch::kFloat32).reshape({rows, kHeads, kDim});
  const auto weights = in.weights.to(torch::kFloat32).reshape({rows, kHeads});
  const auto cache = in.cache.view({-1, kDim});
  const auto safe_slots =
      in.slots.clamp(0, cache.size(0) - 1).to(torch::kInt64);
  auto scores = torch::empty({rows, width}, query.options());
  const int64_t chunk = std::min<int64_t>(
      64,
      std::max<int64_t>(1,
                        (128 * 1024 * 1024) / (2 * std::max<int64_t>(1, width) *
                                               (kDim + kHeads) * 4)));
  for (int64_t start = 0; start < rows; start += chunk) {
    const int64_t end = std::min(rows, start + chunk);
    const auto keys =
        cache.index_select(0, safe_slots.slice(0, start, end).reshape({-1}))
            .view({end - start, width, kDim})
            .to(torch::kFloat32);
    const auto dots =
        torch::bmm(query.slice(0, start, end), keys.transpose(1, 2)).relu();
    scores.slice(0, start, end)
        .copy_((dots * weights.slice(0, start, end).unsqueeze(-1)).sum(1));
  }
  const auto columns = torch::arange(width, in.slots.options());
  const auto valid =
      (columns.unsqueeze(0) < in.counts.unsqueeze(1)) & (in.slots >= 0);
  const auto slots = in.slots.to(torch::kInt64).clamp_min(0);
  auto global_slots =
      (torch::floor_divide(slots, in.page) * in.size + in.rank) * in.page +
      slots.remainder(in.page);
  return {scores.masked_fill(~valid, -INFINITY),
          global_slots.masked_fill(~valid, -1).to(torch::kInt32)};
}

void compare(const Output& actual, const Output& expected) {
  ASSERT_EQ(actual.first.scalar_type(), torch::kFloat32);
  ASSERT_EQ(actual.second.scalar_type(), torch::kInt32);
  ASSERT_TRUE(actual.first.is_contiguous());
  ASSERT_TRUE(actual.second.is_contiguous());
  ASSERT_EQ(actual.first.sizes(), expected.first.sizes());
  ASSERT_EQ(actual.second.sizes(), expected.second.sizes());
  const auto scores = actual.first.cpu().to(torch::kFloat64);
  const auto wanted = expected.first.cpu().to(torch::kFloat64);
  EXPECT_TRUE(torch::equal(actual.second.cpu(), expected.second.cpu()));
  EXPECT_TRUE(torch::equal(torch::isneginf(scores), torch::isneginf(wanted)));
  const auto finite = torch::isfinite(wanted);
  EXPECT_TRUE(torch::isfinite(scores.masked_select(finite)).all().item<bool>());
  EXPECT_TRUE(torch::allclose(scores, wanted, kTolerance, kTolerance));
}

void check(const Inputs& in) { compare(score(in), oracle(in)); }

torch::Tensor strided_copy(const torch::Tensor& tensor,
                           int64_t step,
                           int64_t offset) {
  auto shape = tensor.sizes().vec();
  shape.back() = shape.back() * step + offset;
  auto view = torch::empty(shape, tensor.options())
                  .slice(-1, offset, shape.back(), step);
  view.copy_(tensor);
  return view;
}

void check_sample(const Inputs& in,
                  const Output& actual,
                  const std::vector<int64_t>& row_ids,
                  const std::vector<int64_t>& col_ids) {
  const auto rows = torch::tensor(row_ids, torch::kInt64).to(kDevice);
  const auto cols = torch::tensor(col_ids, torch::kInt64).to(kDevice);
  Inputs sample = in;
  sample.q =
      in.q.reshape({in.slots.size(0), kHeads, kDim}).index_select(0, rows);
  sample.weights =
      in.weights.reshape({in.slots.size(0), kHeads}).index_select(0, rows);
  sample.slots = in.slots.index_select(0, rows).index_select(1, cols);
  sample.counts =
      (cols.unsqueeze(0) < in.counts.index_select(0, rows).unsqueeze(1))
          .sum(1)
          .to(torch::kInt32);
  compare({actual.first.index_select(0, rows).index_select(1, cols),
           actual.second.index_select(0, rows).index_select(1, cols)},
          oracle(sample));
}

struct Ranking {
  int64_t mismatches = 0;
  int64_t ties = 0;
  std::vector<double> gaps;
};

Ranking rank_outputs(const Output& actual,
                     const Output& expected,
                     int64_t topk) {
  const auto scores = expected.first.cpu().to(torch::kFloat64);
  const auto slots = expected.second.cpu();
  const auto got_scores = actual.first.cpu();
  const auto got_slots = actual.second.cpu();
  Ranking result;
  result.gaps.reserve(scores.size(0));
  for (int64_t row = 0; row < scores.size(0); ++row) {
    const int64_t valid = torch::isfinite(scores[row]).sum().item<int64_t>();
    const int64_t count = std::min(topk, valid);
    if (count == 0) {
      continue;
    }
    const auto order = scores[row].argsort(/*dim=*/-1, /*descending=*/true);
    const auto got_order =
        got_scores[row].argsort(/*dim=*/-1, /*descending=*/true);
    const double boundary = scores[row][order[count - 1]].item<double>();
    result.gaps.emplace_back(
        count < valid ? boundary - scores[row][order[count]].item<double>()
                      : INFINITY);
    std::unordered_set<int32_t> ref_set;
    std::unordered_set<int32_t> got_set;
    std::unordered_map<int32_t, double> score_by_slot;
    for (int64_t col = 0; col < scores.size(1); ++col) {
      score_by_slot[slots[row][col].item<int32_t>()] =
          scores[row][col].item<double>();
    }
    for (int64_t col = 0; col < count; ++col) {
      ref_set.insert(slots[row][order[col]].item<int32_t>());
      got_set.insert(got_slots[row][got_order[col]].item<int32_t>());
    }
    if (ref_set == got_set) {
      continue;
    }
    const auto on_boundary = [&](const auto& left, const auto& right) {
      return std::all_of(left.begin(), left.end(), [&](int32_t slot) {
        const auto found = score_by_slot.find(slot);
        return right.count(slot) != 0 ||
               (found != score_by_slot.end() && found->second == boundary);
      });
    };
    const bool tied = static_cast<int64_t>(got_set.size()) == count &&
                      on_boundary(ref_set, got_set) &&
                      on_boundary(got_set, ref_set);
    result.ties += tied;
    result.mismatches += !tied;
  }
  return result;
}

class DcpCandidateScoreTest : public ::testing::Test {
 protected:
  const torch::DeviceGuard guard_{kDevice};

  void SetUp() override { torch::set_num_threads(8); }
};

TEST_F(DcpCandidateScoreTest, CausalScores) {
  for (int64_t rank = 0; rank < 4; ++rank) {
    for (int64_t width : {1, 33, 65}) {
      SCOPED_TRACE(::testing::Message()
                   << "rank=" << rank << " width=" << width);
      check(make_inputs({17, width, 64, 1, true, rank}));
    }
  }
}

TEST_F(DcpCandidateScoreTest, EmptyAndRowBoundaries) {
  for (int64_t rows : {0, 1, 51, 52, 64, 65, 161}) {
    SCOPED_TRACE(rows);
    check(make_inputs({rows, 3, 64}));
  }
  check(make_inputs({4, 0, 64}));
}

TEST_F(DcpCandidateScoreTest, MasksAndMapping) {
  for (const auto& [size, rank, page] :
       std::vector<std::tuple<int64_t, int64_t, int64_t>>{
           {1, 0, 16}, {2, 1, 8}, {4, 3, 16}, {3, 2, 7}, {5, 4, 17}}) {
    SCOPED_TRACE(::testing::Message() << size << "/" << rank << "/" << page);
    auto in = make_inputs({4, 7, 64, 1, false, rank, size, page});
    const int32_t cache_size = static_cast<int32_t>(in.cache.numel() / kDim);
    const int32_t block = static_cast<int32_t>(page);
    in.slots = torch::tensor({{0, block - 1, block, -1, cache_size + 3, 2, 1},
                              {0, -17, 123456, kMax, kMin, 5, 6},
                              {0, 1, 2, 3, 4, 5, 6},
                              {cache_size + 2, -1, 0, 1, 2, 3, 4}},
                             torch::kInt32)
                   .to(kDevice);
    in.counts = torch::tensor({7, 1, 0, 1}, torch::kInt32).to(kDevice);
    const auto actual = score(in);
    compare(actual, oracle(in));
    EXPECT_GE(actual.second[3][0].item<int32_t>(), 0);
    EXPECT_TRUE(torch::isfinite(actual.first[3][0]).item<bool>());
    EXPECT_TRUE(torch::isneginf(actual.first[2]).all().item<bool>());
  }
}

TEST_F(DcpCandidateScoreTest, StridesAndStorageOffsets) {
  auto in = make_inputs({8, 33, 256, 2});
  for (auto* tensor : {&in.q, &in.weights, &in.cache, &in.slots, &in.counts}) {
    *tensor = strided_copy(*tensor, /*step=*/2, /*offset=*/0);
    ASSERT_FALSE(tensor->is_contiguous());
  }
  check(in);
  for (int64_t idx = 0; idx < 5; ++idx) {
    SCOPED_TRACE(idx);
    in = make_inputs({4, 129, 512, 1, false, 0, 4, 16, 319});
    const std::vector<torch::Tensor*> tensors{
        &in.q, &in.weights, &in.cache, &in.slots, &in.counts};
    *tensors[idx] = strided_copy(*tensors[idx], /*step=*/3, /*offset=*/1);
    ASSERT_EQ(tensors[idx]->storage_offset(), 1);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, SignedWeights) {
  auto in = make_inputs({4, 17, 256});
  in.weights.fill_(-1);
  const auto actual = score(in);
  compare(actual, oracle(in));
  EXPECT_TRUE((actual.first <= 0).all().item<bool>());
}

TEST_F(DcpCandidateScoreTest, FourShardMerge) {
  std::vector<torch::Tensor> scores, slots, ref_scores, ref_slots;
  scores.reserve(4);
  slots.reserve(4);
  ref_scores.reserve(4);
  ref_slots.reserve(4);
  for (int64_t rank = 0; rank < 4; ++rank) {
    const auto in = make_inputs({4, 17, 128, 1, false, rank});
    const auto got = score(in);
    const auto ref = oracle(in);
    scores.emplace_back(got.first);
    slots.emplace_back(got.second);
    ref_scores.emplace_back(ref.first);
    ref_slots.emplace_back(ref.second);
  }
  const Output actual{torch::cat(scores, 1), torch::cat(slots, 1)};
  const Output expected{torch::cat(ref_scores, 1), torch::cat(ref_slots, 1)};
  compare(actual, expected);
  EXPECT_EQ(rank_outputs(actual, expected, /*topk=*/20).mismatches, 0);
}

TEST(DcpCandidateRankingTest, ExactTiesOnly) {
  const auto slots = torch::tensor({{10, 11, 12, 13}}, torch::kInt32);
  auto ref_scores = torch::tensor({{3.0f, 2.0f, 2.0f, 1.0f}});
  const Output got{torch::tensor({{3.0f, 2.0f, 2.000001f, 1.0f}}), slots};
  const auto tied = rank_outputs(got, {ref_scores, slots}, /*topk=*/2);
  EXPECT_EQ(tied.mismatches, 0);
  EXPECT_EQ(tied.ties, 1);
  ref_scores[0][2] = 1.999999;
  const auto distinct = rank_outputs(got, {ref_scores, slots}, /*topk=*/2);
  EXPECT_EQ(distinct.mismatches, 1);
  ASSERT_EQ(distinct.gaps.size(), 1);
  EXPECT_GT(distinct.gaps[0], 0);
  EXPECT_LT(distinct.gaps[0], 1e-5);
}

TEST_F(DcpCandidateScoreTest, TileAndRowTails) {
  for (const auto& [rows, width] :
       std::vector<std::pair<int64_t, int64_t>>{{3, 127},
                                                {4, 128},
                                                {5, 129},
                                                {31, 7},
                                                {32, 7},
                                                {33, 7},
                                                {127, 7},
                                                {128, 7},
                                                {129, 7},
                                                {257, 3}}) {
    SCOPED_TRACE(::testing::Message() << rows << "/" << width);
    auto in = make_inputs({rows, width, 1024, 1, false, 0, 4, 16, 9157});
    for (int64_t row = 0; row < rows; ++row) {
      in.slots[row].copy_(in.slots[row].roll({row % width}));
    }
    in.counts =
        (torch::arange(rows, in.counts.options()) * 17).remainder(width + 1);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, DuplicateCandidatesAndLayouts) {
  for (bool prefill : {true, false}) {
    auto in = make_inputs({8, 129, 1024, 2, prefill, 0, 4, 16, 831});
    in.slots.slice(1, 1, 129, 3).copy_(in.slots.slice(1, 0, 1));
    in.counts.fill_(129);
    const auto expected = oracle(in);
    compare(score(in), expected);
    in.q = in.q.view({8, kHeads, kDim});
    in.weights = in.weights.view({8, kHeads});
    in.cache = in.cache.view({-1, kDim});
    compare(score(in), expected);
  }
  auto in = make_inputs({4, 33, 128});
  const auto expected = oracle(in);
  in.cache = in.cache.transpose(1, 2);
  ASSERT_EQ(in.cache.size(1), 1);
  ASSERT_EQ(in.cache.size(2), 16);
  compare(score(in), expected);
}

TEST_F(DcpCandidateScoreTest, ZeroStrideInputs) {
  auto in = make_inputs({5, 65, 256, 1, false, 0, 4, 16, 409});
  in.q = in.q.view({5, kHeads, kDim}).slice(0, 0, 1).expand({5, -1, -1});
  in.weights = in.weights.view({5, kHeads}).slice(0, 0, 1).expand({5, -1});
  in.cache = in.cache.view({-1, kDim}).slice(0, 0, 1).expand({64, -1});
  check(in);
}

TEST_F(DcpCandidateScoreTest, WideGlobalizeAndInvalidTile) {
  for (int64_t rank = 0; rank < 4; ++rank) {
    SCOPED_TRACE(rank);
    auto in = make_inputs({4, 257, 128, 1, false, rank});
    in.slots.fill_(kMax);
    in.slots.select(1, 0).fill_(536870911);
    in.slots.select(1, 1).fill_(kMin);
    in.slots[0].slice(0, 2).fill_(-1);
    in.counts = torch::tensor({257, 1, 0, -1}, torch::kInt32).to(kDevice);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, ExtremeCandidateCounts) {
  auto in = make_inputs({4, 129, 256});
  in.slots.fill_(1);
  in.counts = torch::tensor({130, kMax, kMin, -1}, torch::kInt32).to(kDevice);
  check(in);
}

TEST_F(DcpCandidateScoreTest, SharedRowsDistinctCounts) {
  for (bool shared : {true, false}) {
    auto in = make_inputs({128, 33, 1024, 1, false, 0, 4, 16, 2401});
    in.slots = in.slots.slice(0, 0, 1).expand({128, -1}).clone();
    in.counts =
        torch::tensor({0, 33, 1, kMax}, torch::kInt32).repeat({32}).to(kDevice);
    if (!shared) {
      in.slots.slice(0, 3, 128, 4).select(1, 32).fill_(-1);
    }
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, CandidateIdPreloadBoundary) {
  for (int64_t width : {4095, 4096, 4097}) {
    SCOPED_TRACE(width);
    check(make_inputs({33, width, 128, 1, false, 0, 4, 16, 733}));
  }
}

TEST_F(DcpCandidateScoreTest, ValidPathRechecksMutations) {
  for (int64_t width : {4095, 4096, 4097}) {
    SCOPED_TRACE(width);
    auto in = make_inputs({33, width, 128, 1, false, 0, 4, 16, 739});
    in.slots.copy_(torch::arange(width, in.slots.options()).unsqueeze(0));
    in.slots.select(1, 0).fill_(536870911);
    in.counts.fill_(kMax);
    const auto verify = [&]() {
      const auto actual = score(in);
      compare(actual, reference(in));
      check_sample(in, actual, {0, 1, 2, 32}, {0, width / 2, width - 1});
    };
    verify();
    in.slots[0][width - 1] = -1;
    in.counts[1] = width - 1;
    in.counts[2] = kMin;
    verify();
  }
}

TEST_F(DcpCandidateScoreTest, ProofChecksLaterRows) {
  for (int64_t rows : {64, 65, 128, 256}) {
    SCOPED_TRACE(rows);
    constexpr int64_t kWidth = 257;
    auto in = make_inputs({rows, kWidth, 128, 1, false, 0, 4, 16, 741});
    in.slots.copy_(torch::arange(kWidth, in.slots.options()).unsqueeze(0));
    in.slots.select(1, kWidth - 1).fill_(536870911);
    in.counts.fill_(kMax);
    const auto verify = [&]() {
      const auto expected = oracle(in);
      compare(reference(in), expected);
      compare(score(in), expected);
    };
    verify();
    in.slots[rows - 1][kWidth / 2] = -1;
    verify();
    in.slots[rows - 1][kWidth / 2] = kWidth / 2;
    in.counts[rows - 1] = kWidth - 1;
    verify();
  }
}

TEST_F(DcpCandidateScoreTest, Fp32CancellationAndReluOrder) {
  for (int64_t rows : {1, 130, 160}) {
    SCOPED_TRACE(rows);
    auto in = make_inputs({rows, 3, 64});
    in.q = torch::zeros({rows, kHeads, kDim}, in.q.options());
    in.q.select(2, 0).fill_(256);
    in.q.select(2, 1).fill_(1);
    in.q.select(2, 2).fill_(-256);
    in.q.slice(1, 1, kHeads, 2).neg_();
    in.cache = torch::ones({16, kDim}, in.cache.options());
    in.weights = torch::ones({rows, kHeads}, in.weights.options());
    in.weights.slice(1, 1, kHeads, 2).fill_(-2);
    in.slots.fill_(0);
    in.counts.fill_(3);
    const auto actual = score(in);
    compare(actual, oracle(in));
    EXPECT_TRUE(torch::equal(actual.first, torch::full_like(actual.first, 16)));
  }
}

TEST_F(DcpCandidateScoreTest, LargeRowScheduling) {
  for (int64_t rows : {511, 512, 513, 8192, 16383, 16384, 16385}) {
    SCOPED_TRACE(rows);
    const auto in = make_inputs(
        {rows, 3, std::max<int64_t>(8192, rows), 1, true, 0, 4, 16, 974});
    const auto actual = score(in);
    compare(actual, reference(in));
    check_sample(in, actual, {0, 31, rows - 2, rows - 1}, {0, 1, 2});
  }
}

class CompileLogSink final : public google::LogSink {
 public:
  CompileLogSink() { google::AddLogSink(this); }
  ~CompileLogSink() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity /*severity*/,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*logmsgtime*/,
            const char* message,
            size_t length) override {
    if (std::string_view(message, length)
            .find("triton_jit: compiling tmo_dcp_score_candidates_kernel ") !=
        std::string_view::npos) {
      misses_.fetch_add(1);
    }
  }

  int64_t misses() const { return misses_.load(); }

 private:
  std::atomic<int64_t> misses_{0};
};

TEST_F(DcpCandidateScoreTest, ReusesRowBuckets) {
  // Slice one allocation so cache size, strides, and pointer alignment stay
  // constant. In particular, row count divisibility must not split the key.
  auto base = make_inputs({30000, 3, 32768, 1, true});
  CompileLogSink sink;
  for (const auto& group :
       std::vector<std::vector<int64_t>>{{1059, 1072, 1572, 2048},
                                         {3597, 4050, 4096},
                                         {4097, 8191, 8192},
                                         {8193, 16383, 16384},
                                         {16385, 20000, 30000}}) {
    bool first = true;
    int64_t warmed_misses = 0;
    for (int64_t rows : group) {
      SCOPED_TRACE(rows);
      Inputs in = base;
      in.q = base.q.narrow(0, 0, rows);
      in.weights = base.weights.narrow(0, 0, rows);
      in.slots = base.slots.narrow(0, 0, rows);
      in.counts = base.counts.narrow(0, 0, rows);
      const auto actual = score(in);
      compare(actual, reference(in));
      check_sample(in, actual, {0, 31, rows - 2, rows - 1}, {0, 1, 2});
      if (first) {
        warmed_misses = sink.misses();
        first = false;
      }
      EXPECT_EQ(sink.misses(), warmed_misses);
      compare(score(in), actual);
      EXPECT_EQ(sink.misses(), warmed_misses);
    }
  }
}

TEST_F(DcpCandidateScoreTest, RowPolicyBoundaries) {
  const auto* properties = torch_mlu::getDeviceProperties(kDevice.index());
  const int64_t cores =
      properties->cluster_count * properties->core_num_per_cluster;
  for (int64_t rows : std::vector<int64_t>{cores - 1,
                                           cores,
                                           cores + 1,
                                           2 * cores - 1,
                                           2 * cores,
                                           2 * cores + 1,
                                           127,
                                           128,
                                           129,
                                           255,
                                           256,
                                           257,
                                           4095,
                                           4096,
                                           4097}) {
    SCOPED_TRACE(rows);
    check(make_inputs({rows, 3, 8192, 1, true}));
  }
}

TEST_F(DcpCandidateScoreTest, ResidentCountLayouts) {
  constexpr int64_t kRows = 257;
  for (int64_t stride : {0, 3}) {
    SCOPED_TRACE(stride);
    auto in = make_inputs({kRows, 33, 1024, 1, true, 0, 4, 16, 976});
    if (stride == 0) {
      in.counts = torch::full({1}, 17, in.counts.options()).expand({kRows});
    } else {
      in.counts = torch::full({kRows * stride + 1}, -999, in.counts.options())
                      .slice(0, 1, kRows * stride + 1, stride);
      const auto values =
          torch::tensor({kMin, -1, 0, 1, 17, 33, kMax}, torch::kInt32)
              .to(kDevice);
      in.counts.copy_(values.index_select(
          0,
          torch::arange(kRows, in.counts.options().dtype(torch::kInt64))
              .remainder(values.numel())));
    }
    ASSERT_EQ(in.counts.stride(0), stride);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, FullCountsRecheckInvalidSlots) {
  constexpr int64_t kRows = 257;
  constexpr int64_t kWidth = 33;
  for (int64_t stride : {0, 3}) {
    SCOPED_TRACE(stride);
    auto in = make_inputs({kRows, kWidth, 32768, 1, true, 0, 4, 16, 977});
    in.counts =
        stride == 0
            ? torch::full({1}, kMax, in.counts.options()).expand({kRows})
            : torch::full({kRows * stride + 1}, kMax, in.counts.options())
                  .slice(0, 1, kRows * stride + 1, stride);
    in.slots.fill_(1);
    in.slots[0][kWidth - 1] = kMin;
    in.slots[kRows - 1][0] = 536870911;
    for (int32_t count : {kMax, kMin, 32, 33}) {
      SCOPED_TRACE(count);
      in.counts[kRows - 1] = count;
      check(in);
    }
  }
}

TEST_F(DcpCandidateScoreTest, DenseCacheCountBoundaries) {
  constexpr int64_t kWidth = 2048;
  for (int64_t rows : {130, 256}) {
    SCOPED_TRACE(rows);
    auto in = make_inputs({rows, kWidth, 8192, 1, true, 0, 4, 16, 478});
    const auto columns = torch::arange(kWidth, in.slots.options());
    in.slots.copy_((columns.unsqueeze(0) +
                    torch::arange(rows, in.slots.options()).unsqueeze(1) * 17)
                       .remainder(kWidth));
    in.slots.slice(0, 0, rows, 3).select(1, 17).fill_(kMin);
    in.slots.select(1, kWidth - 2).fill_(kWidth + 123);
    in.slots.select(1, kWidth - 1).fill_(536870911);
    const auto thresholds =
        torch::tensor({1535, 1536, 1537, 2047, 2048, kMax}, torch::kInt32)
            .to(kDevice);
    in.counts.copy_(thresholds.index_select(
        0,
        torch::arange(rows, in.counts.options().dtype(torch::kInt64))
            .remainder(thresholds.numel())));
    const auto verify = [&]() {
      const auto actual = score(in);
      compare(actual, reference(in));
      check_sample(in,
                   actual,
                   {0, 1, 2, 3, 4, 5, rows - 1},
                   {0, 17, 1535, 1536, 1537, 2046, 2047});
    };
    verify();
    in.slots.select(1, 1536).fill_(-1);
    in.counts[rows - 1] = 1536;
    verify();
  }
}

TEST_F(DcpCandidateScoreTest, RepeatedCallReadsMutations) {
  auto in = make_inputs({4, 33, 512, 1, false, 0, 4, 16, 193});
  const auto first = score(in);
  const Output snapshot{first.first.clone(), first.second.clone()};
  in.q.neg_();
  in.cache.mul_(2);
  in.weights.neg_();
  in.slots.slice(1, 0, 3).fill_(0);
  in.counts.fill_(17);
  check(in);
  EXPECT_TRUE(torch::equal(first.first, snapshot.first));
  EXPECT_TRUE(torch::equal(first.second, snapshot.second));
}

bool wide_tests_enabled() {
  const char* value = std::getenv("XLLM_DCP_WIDE_TESTS");
  return value != nullptr && std::string(value) == "1";
}

TEST_F(DcpCandidateScoreTest, WideRowAddresses) {
  if (!wide_tests_enabled()) {
    GTEST_SKIP() << "Set XLLM_DCP_WIDE_TESTS=1 for multi-GiB address tests";
  }
  for (bool query : {true, false}) {
    SCOPED_TRACE(query);
    auto in = make_inputs({2, 3, 64});
    in.q = in.q.view({2, kHeads, kDim});
    in.cache = in.cache.view({-1, kDim}).slice(0, 0, 2);
    in.slots.fill_(1);
    in.counts.fill_(3);
    auto& tensor = query ? in.q : in.cache;
    auto strides = tensor.strides().vec();
    strides[0] = (int64_t{1} << 30) + 256;
    auto wide = torch::empty_strided(tensor.sizes(), strides, tensor.options());
    wide.copy_(tensor);
    ASSERT_TRUE(torch::equal(wide.cpu(), tensor.cpu()));
    tensor = wide;
    ASSERT_GT(wide.stride(0) * wide.element_size(), int64_t{1} << 31);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, PageArithmeticBoundaries) {
  for (int64_t page : std::vector<int64_t>{
           1, 7, int64_t{1} << 30, int64_t{1} << 31, (int64_t{1} << 31) + 1}) {
    SCOPED_TRACE(page);
    auto in = make_inputs({4, 7, 64});
    in.cache = in.cache.view({-1, kDim});
    in.page = page;
    in.size = 1;
    in.rank = 0;
    in.slots.copy_(torch::tensor({0, 1, 63, kMax, -1, kMin, 7}, torch::kInt32)
                       .to(kDevice));
    in.counts.fill_(7);
    check(in);
  }
}

TEST_F(DcpCandidateScoreTest, WeightedHeadCancellation) {
  for (int64_t rows : {32, 130, 160}) {
    SCOPED_TRACE(rows);
    const auto base = torch::cat({torch::full({8}, 256.0f),
                                  torch::full({8}, -256.0f),
                                  torch::full({16}, std::ldexp(1.0f, -16))});
    torch::manual_seed(0);
    auto terms = base.index_select(0, torch::randperm(kHeads));
    auto row_terms = terms.unsqueeze(0).expand({rows, -1});
    auto q = torch::zeros({rows, kHeads, kDim}, torch::kBFloat16);
    q.select(2, 0).copy_((row_terms.abs() / 64).to(torch::kBFloat16));
    auto cache = torch::zeros({1, kDim}, torch::kBFloat16);
    cache.select(1, 0).fill_(64);
    Inputs in{q.to(kDevice),
              row_terms.sign().to(torch::kBFloat16).to(kDevice),
              cache.to(kDevice),
              torch::zeros({rows, 3}, torch::kInt32).to(kDevice),
              torch::full({rows}, 3, torch::kInt32).to(kDevice)};
    const auto expected = oracle(in);
    compare(reference(in), expected);
    compare(score(in), expected);
  }
}

TEST_F(DcpCandidateScoreTest, WideInnerAddresses) {
  if (!wide_tests_enabled()) {
    GTEST_SKIP() << "Set XLLM_DCP_WIDE_TESTS=1 for multi-GiB address tests";
  }
  for (bool weights : {true, false}) {
    SCOPED_TRACE(weights);
    auto in = make_inputs({2, 3, 64});
    in.weights = in.weights.view({2, kHeads});
    in.cache = in.cache.view({-1, kDim}).slice(0, 0, 2);
    in.slots.fill_(1);
    in.counts.fill_(3);
    const auto expected = oracle(in);
    auto& tensor = weights ? in.weights : in.cache;
    const int64_t stride = weights ? int64_t{1} << 27 : 17 * (int64_t{1} << 20);
    auto wide =
        torch::empty_strided(tensor.sizes(), {1, stride}, tensor.options());
    // Tiny contiguous columns avoid testing another kernel's large-stride copy.
    for (int64_t col = 0; col < tensor.size(1); ++col) {
      wide.select(1, col).copy_(tensor.select(1, col));
      ASSERT_TRUE(
          torch::equal(wide.select(1, col).cpu(), tensor.select(1, col).cpu()));
    }
    ASSERT_GT((tensor.size(1) - 1) * stride, kMax);
    tensor = wide;
    compare(score(in), expected);
  }
}

}  // namespace
}  // namespace xllm
