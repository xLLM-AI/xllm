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
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

// Independent torch reference: append bonus, then retain the first rejected
// target and replace every subsequent ID with -1. The bonus never rejects.
std::tuple<torch::Tensor, torch::Tensor> torch_reference(
    const torch::Tensor& draft,
    const torch::Tensor& target,
    const torch::Tensor& bonus,
    bool mask) {
  const torch::Tensor full = torch::cat({target, bonus}, -1).to(torch::kInt32);
  if (!mask) {
    return {full, torch::Tensor()};
  }
  const torch::Tensor rejected = torch::cat(
      {draft.ne(target), torch::ones({draft.size(0), 1}, torch::kBool)}, -1);
  const torch::Tensor first_reject =
      rejected.to(torch::kInt64).argmax(/*dim=*/1, /*keepdim=*/true);
  const torch::Tensor positions =
      torch::arange(draft.size(1) + 1, torch::kInt64).unsqueeze(0);
  return {full, torch::where(positions <= first_reject, full, -1)};
}

void expect_matches(const torch::Tensor& draft,
                    const torch::Tensor& target,
                    const torch::Tensor& bonus,
                    bool mask) {
  SCOPED_TRACE(::testing::Message()
               << "shape=" << draft.sizes() << " mask=" << mask);
  const torch::Tensor draft_before = draft.cpu().clone();
  const torch::Tensor target_before = target.cpu().clone();
  const torch::Tensor bonus_before = bonus.cpu().clone();
  const auto [expected_full, expected_masked] =
      torch_reference(draft_before, target_before, bonus_before, mask);
  const auto [full, masked] = greedy_prefix_verify(draft, target, bonus, mask);

  ASSERT_TRUE(full.defined());
  EXPECT_EQ(full.scalar_type(), torch::kInt32);
  EXPECT_EQ(full.device(), target.device());
  EXPECT_EQ(full.sizes(),
            torch::IntArrayRef({draft.size(0), draft.size(1) + 1}));
  EXPECT_TRUE(full.is_contiguous());
  EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
  ASSERT_EQ(masked.defined(), mask);
  if (mask) {
    EXPECT_EQ(masked.scalar_type(), torch::kInt32);
    EXPECT_EQ(masked.device(), full.device());
    EXPECT_EQ(masked.sizes(), full.sizes());
    EXPECT_TRUE(masked.is_contiguous());
    EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
    if (full.numel() != 0) {
      EXPECT_NE(full.data_ptr(), masked.data_ptr());
    }
  }
  EXPECT_TRUE(torch::equal(draft.cpu(), draft_before));
  EXPECT_TRUE(torch::equal(target.cpu(), target_before));
  EXPECT_TRUE(torch::equal(bonus.cpu(), bonus_before));
}

// cc_test supplies the NPU/Python runtime.
class GreedyPrefixVerifyTest : public ::testing::Test {
 protected:
  const torch::Device device_{"npu:0"};
  const torch::TensorOptions i32_ =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
};

TEST_F(GreedyPrefixVerifyTest, PreservesFullOutputAcrossFirstRejection) {
  const torch::Tensor draft = torch::tensor(
      {{9, 22, 33}, {11, 22, 33}, {11, 25, 99}, {11, 25, 33}}, i32_);
  const torch::Tensor target =
      torch::tensor({{11, 25, 33}}, i32_).repeat({4, 1});
  const torch::Tensor bonus = torch::full({4, 1}, 44, i32_);

  for (const bool mask : {false, true}) {
    expect_matches(draft, target, bonus, mask);
  }
  const auto [full, masked] =
      greedy_prefix_verify(draft, target, bonus, /*mask=*/true);
  EXPECT_TRUE(torch::equal(
      full.cpu(),
      torch::tensor({{11, 25, 33, 44}}, torch::kInt32).repeat({4, 1})));
  EXPECT_TRUE(torch::equal(masked.cpu(),
                           torch::tensor({{11, -1, -1, -1},
                                          {11, 25, -1, -1},
                                          {11, 25, 33, -1},
                                          {11, 25, 33, 44}},
                                         torch::kInt32)));
}

TEST_F(GreedyPrefixVerifyTest, VerifiesFiveDraftStepsAcrossBatchMatrix) {
  constexpr int64_t kDraftWidth = 5;
  constexpr int64_t kOutputWidth = kDraftWidth + 1;
  constexpr int64_t kStorageOffset = 3;
  const std::array<int64_t, 8> values = {16777216,
                                         16777217,
                                         std::numeric_limits<int32_t>::max(),
                                         std::numeric_limits<int32_t>::min(),
                                         -16777216,
                                         -16777217,
                                         0,
                                         1};
  const std::array<const char*, 6> patterns = {
      "first", "middle", "last", "all", "rematch", "mixed"};
  const std::array<int64_t, 5> reject_positions = {0, 2, 4, 5, 2};
  for (const int64_t batch : {1, 2, 4, 8, 16, 24, 32}) {
    for (int64_t pattern = 0; pattern < 6; ++pattern) {
      torch::Tensor storage_cpu = torch::full(
          {kStorageOffset + batch * kOutputWidth + 4}, -777, torch::kInt64);
      torch::Tensor ids_cpu =
          storage_cpu.narrow(0, kStorageOffset, batch * kOutputWidth)
              .view({batch, kOutputWidth});
      torch::Tensor draft_cpu =
          torch::empty({batch, kDraftWidth}, torch::kInt64);
      auto ids = ids_cpu.accessor<int64_t, 2>();
      auto draft_ids = draft_cpu.accessor<int64_t, 2>();
      for (int64_t row = 0; row < batch; ++row) {
        for (int64_t column = 0; column < kOutputWidth; ++column) {
          ids[row][column] =
              values[static_cast<size_t>(row + column) % values.size()];
        }
        const int64_t first_reject =
            pattern == 5 ? row % kOutputWidth
                         : reject_positions[static_cast<size_t>(pattern)];
        for (int64_t column = 0; column < kDraftWidth; ++column) {
          const bool mismatch =
              column == first_reject || (pattern < 3 && column > first_reject);
          draft_ids[row][column] =
              mismatch ? ids[row][column] ^ int64_t { 1 } : ids[row][column];
        }
      }
      const torch::Tensor storage = storage_cpu.to(device_);
      const torch::Tensor ids_view =
          storage.narrow(0, kStorageOffset, batch * kOutputWidth)
              .view({batch, kOutputWidth});
      const torch::Tensor target = ids_view.slice(1, 0, kDraftWidth);
      const torch::Tensor bonus = ids_view.slice(1, kDraftWidth, kOutputWidth);
      const torch::Tensor draft = draft_cpu.to(device_);
      ASSERT_EQ(target.stride(0), kOutputWidth);
      ASSERT_EQ(target.stride(1), 1);
      ASSERT_EQ(target.storage_offset(), kStorageOffset);
      ASSERT_EQ(bonus.stride(0), kOutputWidth);
      ASSERT_EQ(bonus.storage_offset(), kStorageOffset + kDraftWidth);
      for (const bool mask : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "batch=" << batch
                     << " pattern=" << patterns[static_cast<size_t>(pattern)]
                     << " mask=" << mask);
        expect_matches(draft, target, bonus, mask);
        EXPECT_TRUE(torch::equal(storage.cpu(), storage_cpu));
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyTest, AcceptsIntegerDtypesInStridedViews) {
  constexpr int32_t kAdjacent = (int32_t{1} << 24) + 1;
  constexpr int32_t kSmallest = std::numeric_limits<int32_t>::min();
  constexpr int32_t kLargest = std::numeric_limits<int32_t>::max();
  for (const torch::ScalarType draft_dtype : {torch::kInt32, torch::kInt64}) {
    for (const torch::ScalarType target_dtype :
         {torch::kInt32, torch::kInt64}) {
      for (const torch::ScalarType bonus_dtype :
           {torch::kInt32, torch::kInt64}) {
        SCOPED_TRACE(::testing::Message()
                     << "dtypes=" << draft_dtype << "/" << target_dtype << "/"
                     << bonus_dtype);
        const torch::Tensor storage =
            torch::tensor({{kAdjacent, kAdjacent + 1, kLargest, kSmallest},
                           {kSmallest, -1, kLargest, kAdjacent}},
                          i32_.dtype(target_dtype));
        const torch::Tensor target = storage.slice(1, 0, 3);
        const torch::Tensor bonus_storage = storage.to(bonus_dtype);
        const torch::Tensor bonus = bonus_storage.flatten()
                                        .slice(0, 3, storage.numel(), 4)
                                        .view({2, 1});
        const torch::Tensor draft_storage =
            torch::tensor({{kAdjacent, 0, kAdjacent, 0, kLargest, 0},
                           {kSmallest, 0, -1, 0, kLargest, 0}},
                          i32_.dtype(draft_dtype));
        const torch::Tensor draft = draft_storage.slice(1, 0, 6, 2);
        ASSERT_EQ(draft.stride(1), 2);
        ASSERT_EQ(target.stride(0), 4);
        ASSERT_EQ(bonus.stride(0), 4);
        ASSERT_EQ(bonus.storage_offset(), 3);
        for (const bool mask : {false, true}) {
          expect_matches(draft, target, bonus, mask);
        }
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyTest, HandlesEmptyAndIrregularShapes) {
  for (const int64_t batch :
       {0, 1, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49, 97}) {
    for (const int64_t width :
         {0, 1, 2, 3, 4, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
      const torch::Tensor full_cpu =
          (torch::arange(batch * (width + 1), torch::kInt32) + 10)
              .reshape({batch, width + 1});
      torch::Tensor draft_cpu = full_cpu.slice(1, 0, width).clone();
      auto values = draft_cpu.accessor<int32_t, 2>();
      for (int64_t row = 0; row < batch && width > 0; row += 2) {
        values[row][(row / 2) % width] += int32_t{1} << 24;
      }
      const torch::Tensor full = full_cpu.to(device_);
      const torch::Tensor draft = draft_cpu.to(device_);
      for (const bool mask : {false, true}) {
        expect_matches(draft,
                       full.slice(1, 0, width),
                       full.slice(1, width, width + 1),
                       mask);
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyTest, ReadsNoncontiguousAndAliasedInputs) {
  constexpr int64_t kBatch = 5;
  constexpr int64_t kWidth = 7;
  const torch::Tensor full_cpu =
      (torch::arange(kBatch * (kWidth + 1), torch::kInt32) + 10)
          .reshape({kBatch, kWidth + 1});
  torch::Tensor target_storage =
      torch::full({kBatch + 2, 2 * kWidth + 5}, -777, i32_);
  torch::Tensor target =
      target_storage.slice(0, 1, kBatch + 1).slice(1, 3, 3 + 2 * kWidth, 2);
  torch::Tensor bonus = target_storage.slice(0, 1, kBatch + 1)
                            .slice(1, 2 * kWidth + 3, 2 * kWidth + 4);
  torch::Tensor draft_storage =
      torch::full({2 * kWidth + 3, kBatch + 2}, -888, i32_);
  torch::Tensor draft = draft_storage.slice(0, 1, 1 + 2 * kWidth, 2)
                            .slice(1, 1, kBatch + 1)
                            .transpose(0, 1);
  target.copy_(full_cpu.slice(1, 0, kWidth).to(device_));
  bonus.copy_(full_cpu.slice(1, kWidth, kWidth + 1).to(device_));
  draft.copy_(target);
  draft.index_put_({1, 3}, 999);
  const torch::Tensor target_storage_before = target_storage.cpu();
  const torch::Tensor draft_storage_before = draft_storage.cpu();

  ASSERT_EQ(target.stride(1), 2);
  ASSERT_GT(draft.stride(1), 1);
  ASSERT_GT(target.storage_offset(), 0);
  ASSERT_GT(draft.storage_offset(), 0);
  for (const bool mask : {false, true}) {
    expect_matches(draft, target, bonus, mask);
  }
  EXPECT_TRUE(torch::equal(target_storage.cpu(), target_storage_before));
  EXPECT_TRUE(torch::equal(draft_storage.cpu(), draft_storage_before));

  const torch::Tensor scalar = torch::tensor({{int32_t{1} << 30}}, i32_);
  const torch::Tensor aliased_draft = scalar.expand({9, 3});
  const torch::Tensor aliased_target = scalar.expand({9, 3});
  const torch::Tensor aliased_bonus = scalar.expand({9, 1});
  ASSERT_EQ(aliased_draft.stride(0), 0);
  ASSERT_EQ(aliased_draft.stride(1), 0);
  for (const bool mask : {false, true}) {
    expect_matches(aliased_draft, aliased_target, aliased_bonus, mask);
  }

  const torch::Tensor target_row = torch::tensor({{11, 25, 33}}, i32_);
  const torch::Tensor draft_row = torch::tensor({{11, 22, 33}}, i32_);
  for (const bool mask : {false, true}) {
    expect_matches(draft_row.expand({9, 3}),
                   target_row.expand({9, 3}),
                   aliased_bonus,
                   mask);
  }
}

TEST_F(GreedyPrefixVerifyTest, ReadsStridesLargerThanTheUbTile) {
  constexpr int64_t kBatch = 3;
  constexpr int64_t kWidth = 3;
  constexpr int64_t kColumnStride = 65537;
  constexpr int64_t kRowStride = kWidth * kColumnStride + 5;
  constexpr int64_t kOffset = 3;
  constexpr int64_t kStorageSize = kOffset + kBatch * kRowStride;
  torch::Tensor target_storage = torch::full({kStorageSize}, -777, i32_);
  torch::Tensor draft_storage = torch::full({kStorageSize}, -888, i32_);
  torch::Tensor target = target_storage.as_strided(
      {kBatch, kWidth}, {kRowStride, kColumnStride}, kOffset);
  torch::Tensor draft = draft_storage.as_strided(
      {kBatch, kWidth}, {kRowStride, kColumnStride}, kOffset);
  const torch::Tensor values =
      torch::tensor({{11, 25, 33}}, i32_).expand({kBatch, kWidth});
  target.copy_(values);
  draft.copy_(target);
  draft.index_put_({1, 1}, (int32_t{1} << 24) + 25);
  const torch::Tensor bonus = torch::full({kBatch, 1}, 44, i32_);
  const torch::Tensor target_storage_before = target_storage.cpu();
  const torch::Tensor draft_storage_before = draft_storage.cpu();

  for (const bool mask : {false, true}) {
    expect_matches(draft, target, bonus, mask);
  }
  EXPECT_TRUE(torch::equal(target_storage.cpu(), target_storage_before));
  EXPECT_TRUE(torch::equal(draft_storage.cpu(), draft_storage_before));
}

TEST_F(GreedyPrefixVerifyTest, PreservesExtremeBitsAcrossDtypesAndTiles) {
  const torch::Tensor values =
      torch::tensor({std::numeric_limits<int32_t>::min(),
                     -16777217,
                     -16777216,
                     -1,
                     0,
                     16777216,
                     16777217,
                     std::numeric_limits<int32_t>::max()},
                    torch::kInt32);
  for (const torch::ScalarType dtype : {torch::kInt32, torch::kInt64}) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << dtype);
    for (const int64_t batch : {0, 7}) {
      for (const int64_t width : {0, 1, 3, 65, 129}) {
        const int64_t count = batch * (width + 1);
        const torch::Tensor full_cpu = values.repeat({(count + 7) / 8})
                                           .narrow(0, 0, count)
                                           .reshape({batch, width + 1});
        torch::Tensor draft_cpu = full_cpu.slice(1, 0, width).clone();
        auto ids = draft_cpu.accessor<int32_t, 2>();
        // Single-bit differences after 2^24 and at int32 limits must reject;
        // later matching IDs must not restore the accepted prefix.
        for (int64_t row = 0; row + 1 < batch && width > 0; ++row) {
          ids[row][row * (width - 1) / (batch - 2)] ^= int32_t{1};
        }
        const torch::Tensor full = full_cpu.to(i32_.dtype(dtype));
        const torch::Tensor draft = draft_cpu.to(i32_.dtype(dtype));
        for (const bool mask : {false, true}) {
          expect_matches(draft,
                         full.slice(1, 0, width),
                         full.slice(1, width, width + 1),
                         mask);
        }
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyTest, RepeatedCallsDoNotReuseMaskedTail) {
  const torch::Tensor target = torch::arange(33 * 3, i32_).view({33, 3}) + 100;
  const torch::Tensor bonus = torch::arange(33, i32_).view({33, 1}) + 1000;
  for (int64_t iteration = 0; iteration < 8; ++iteration) {
    torch::Tensor draft = target.clone();
    draft.index_put_({iteration, iteration % 3}, -123);
    expect_matches(draft, target, bonus, iteration % 2 == 0);
  }
}

TEST_F(GreedyPrefixVerifyTest, UsesTheInputDevicesCurrentStream) {
  const auto stream =
      c10_npu::getStreamFromPool(/*isHighPriority=*/false, device_.index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  const torch::Tensor target = torch::tensor({{11, 25, 33}}, i32_);
  const torch::Tensor draft = torch::tensor({{11, 22, 33}}, i32_);
  const torch::Tensor bonus = torch::tensor({{44}}, i32_);
  expect_matches(draft, target, bonus, /*mask=*/true);
  EXPECT_EQ(aclrtSynchronizeStream(stream.stream()), ACL_SUCCESS);
}

TEST_F(GreedyPrefixVerifyTest, RejectsMetadataOutsideInt32BeforeAllocation) {
  constexpr int64_t kTooLarge =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  const torch::Tensor scalar = torch::ones({1}, i32_);
  const torch::Tensor ids = scalar.view({1, 1});
  const torch::Tensor huge_stride = scalar.as_strided({1, 1}, {kTooLarge, 1});
  const torch::Tensor huge_batch = scalar.expand({kTooLarge, 1});
  const torch::Tensor huge_output = scalar.expand({kTooLarge / 2, 1});
  const torch::Tensor huge_width = scalar.expand({1, kTooLarge - 1});
  EXPECT_DEATH_IF_SUPPORTED(
      static_cast<void>(greedy_prefix_verify(huge_stride, ids, ids, true)),
      "stride exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_batch, huge_batch, huge_batch, true)),
                            "shape exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_output, huge_output, huge_output, true)),
                            "output element count exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_width, huge_width, ids, true)),
                            "output width exceeds INT32");
}

TEST_F(GreedyPrefixVerifyTest, RejectsInvalidMetadataBeforeLaunch) {
  const torch::Tensor target = torch::ones({2, 3}, i32_);
  const torch::Tensor bonus = torch::ones({2, 1}, i32_);
  const torch::Tensor invalid_bonus = torch::ones({2, 2}, i32_);
  const torch::Tensor wrong_dtype = target.to(torch::kFloat32);
  const torch::Tensor wrong_bonus_dtype = bonus.to(torch::kFloat32);
  const torch::Tensor cpu_target = target.cpu();
  const torch::Tensor wrong_rank = target.flatten();
  EXPECT_DEATH_IF_SUPPORTED(
      static_cast<void>(greedy_prefix_verify(target, cpu_target, bonus, true)),
      "must be on NPU");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                wrong_rank, wrong_rank, bonus, true)),
                            "must be rank two");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                target, target, invalid_bonus, true)),
                            "bonus.size\\(1\\) == 1");
  for (const bool mask : {false, true}) {
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  wrong_dtype, target, bonus, mask)),
                              "must contain int32 or int64 token IDs");
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  target, wrong_dtype, bonus, mask)),
                              "must contain int32 or int64 token IDs");
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  target, target, wrong_bonus_dtype, mask)),
                              "must contain int32 or int64 token IDs");
    EXPECT_DEATH_IF_SUPPORTED(
        static_cast<void>(greedy_prefix_verify(
            wrong_dtype, wrong_dtype, wrong_bonus_dtype, mask)),
        "must contain int32 or int64 token IDs");
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
