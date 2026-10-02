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

#include <limits>
#include <vector>

#include "core/layers/npu_torch/qwen3_5_gated_delta_net.h"

namespace xllm::layer {
namespace {

using qwen3_5_gdn_internal::get_or_build_prefill_indices;

class Qwen3_5GatedDeltaNetIndicesDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

constexpr int64_t kNumSlots = 16;
constexpr int64_t kCheckpointStride = 4;
const torch::Device kDevice(torch::kCPU);

void expect_int32_vector(const torch::Tensor& tensor,
                         const std::vector<int32_t>& expected) {
  EXPECT_EQ(tensor.scalar_type(), torch::kInt32);
  EXPECT_EQ(tensor.device(), kDevice);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.sizes(),
            torch::IntArrayRef({static_cast<int64_t>(expected.size())}));
  EXPECT_TRUE(torch::equal(tensor, torch::tensor(expected, torch::kInt32)));
}

TEST(Qwen3_5GatedDeltaNetIndicesTest,
     BuildsColdWarmOrderedPaddingAndStridedIndices) {
  AttentionMetadata metadata;
  const std::vector<int32_t> slots = {3, 0, 7, 0};
  const std::vector<int64_t> validity = {0, 0, 1, 0};
  const MegaGdnPrefillIndicesCache& indices = get_or_build_prefill_indices(
      metadata, slots, validity, kCheckpointStride, kNumSlots, kDevice);

  expect_int32_vector(indices.conv_read, {-1, -1, 7, -1});
  expect_int32_vector(indices.conv_write, {3, 0, 7, 0});
  expect_int32_vector(indices.ssm_read, {-1, -1, 28, -1});
  expect_int32_vector(indices.ssm_write, {12, 0, 28, 0});
}

TEST(Qwen3_5GatedDeltaNetIndicesTest,
     ReusesAllTensorStorageForMatchingForwardKey) {
  AttentionMetadata metadata;
  const std::vector<int32_t> slots = {2, 5};
  const std::vector<int64_t> validity = {0, 1};
  const MegaGdnPrefillIndicesCache& first = get_or_build_prefill_indices(
      metadata, slots, validity, kCheckpointStride, kNumSlots, kDevice);
  const void* conv_read = first.conv_read.data_ptr();
  const void* conv_write = first.conv_write.data_ptr();
  const void* ssm_read = first.ssm_read.data_ptr();
  const void* ssm_write = first.ssm_write.data_ptr();

  for (int64_t layer = 1; layer < 64; ++layer) {
    const MegaGdnPrefillIndicesCache& reused = get_or_build_prefill_indices(
        metadata, slots, validity, kCheckpointStride, kNumSlots, kDevice);
    EXPECT_EQ(reused.conv_read.data_ptr(), conv_read);
    EXPECT_EQ(reused.conv_write.data_ptr(), conv_write);
    EXPECT_EQ(reused.ssm_read.data_ptr(), ssm_read);
    EXPECT_EQ(reused.ssm_write.data_ptr(), ssm_write);
  }
}

TEST(Qwen3_5GatedDeltaNetIndicesTest, NewMetadataMaterializesNewTensorSet) {
  AttentionMetadata first_metadata;
  AttentionMetadata next_metadata;
  const std::vector<int32_t> slots = {2, 5};
  const std::vector<int64_t> validity = {0, 1};
  const MegaGdnPrefillIndicesCache& first = get_or_build_prefill_indices(
      first_metadata, slots, validity, kCheckpointStride, kNumSlots, kDevice);
  const MegaGdnPrefillIndicesCache& next = get_or_build_prefill_indices(
      next_metadata, slots, validity, kCheckpointStride, kNumSlots, kDevice);

  EXPECT_NE(next.conv_read.data_ptr(), first.conv_read.data_ptr());
  EXPECT_NE(next.conv_write.data_ptr(), first.conv_write.data_ptr());
  EXPECT_NE(next.ssm_read.data_ptr(), first.ssm_read.data_ptr());
  EXPECT_NE(next.ssm_write.data_ptr(), first.ssm_write.data_ptr());
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsInvalidValidity) {
  AttentionMetadata metadata;
  EXPECT_DEATH(
      get_or_build_prefill_indices(metadata, {1}, {2}, 1, kNumSlots, kDevice),
      "validity must be 0 or 1");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsValiditySizeMismatch) {
  AttentionMetadata metadata;
  EXPECT_DEATH(get_or_build_prefill_indices(
                   metadata, {1, 2}, {1}, 1, kNumSlots, kDevice),
               "validity_mask must be sequence-scoped");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsNegativeSlot) {
  AttentionMetadata metadata;
  EXPECT_DEATH(
      get_or_build_prefill_indices(metadata, {-1}, {1}, 1, kNumSlots, kDevice),
      "live slot must be non-negative");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsSlotOutOfBounds) {
  AttentionMetadata metadata;
  EXPECT_DEATH(get_or_build_prefill_indices(
                   metadata, {1, 16}, {1, 1}, 1, kNumSlots, kDevice),
               "live slot exceeds cache capacity");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsDuplicateNonzeroSlot) {
  AttentionMetadata metadata;
  EXPECT_DEATH(get_or_build_prefill_indices(
                   metadata, {3, 3}, {1, 0}, 1, kNumSlots, kDevice),
               "write slots must be unique");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsInvalidStride) {
  AttentionMetadata metadata;
  EXPECT_DEATH(
      get_or_build_prefill_indices(metadata, {1}, {1}, 0, kNumSlots, kDevice),
      "checkpoint stride must be positive");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsStrideOverflow) {
  AttentionMetadata metadata;
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata,
          {1},
          {1},
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
          kNumSlots,
          kDevice),
      "checkpoint stride does not fit");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsIndexOverflow) {
  AttentionMetadata metadata;
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata, {1}, {1}, std::numeric_limits<int32_t>::max(), 3, kDevice),
      "SSM state index does not fit");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedBatchSize) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1}, {1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata, {1, 2}, {1, 1}, kCheckpointStride, kNumSlots, kDevice),
      "batch size changed");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedSlotGeometry) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1}, {1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata, {1}, {1}, kCheckpointStride, kNumSlots + 1, kDevice),
      "slot geometry changed");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedCheckpointStride) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1}, {1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata, {1}, {1}, kCheckpointStride + 1, kNumSlots, kDevice),
      "checkpoint stride changed");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedSlotOrder) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1, 2}, {1, 1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(
      get_or_build_prefill_indices(
          metadata, {2, 1}, {1, 1}, kCheckpointStride, kNumSlots, kDevice),
      "linear state ids changed");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedValidity) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1}, {1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(get_or_build_prefill_indices(
                   metadata, {1}, {0}, kCheckpointStride, kNumSlots, kDevice),
               "validity changed");
}

TEST_F(Qwen3_5GatedDeltaNetIndicesDeathTest, RejectsChangedDevice) {
  AttentionMetadata metadata;
  get_or_build_prefill_indices(
      metadata, {1}, {1}, kCheckpointStride, kNumSlots, kDevice);
  EXPECT_DEATH(get_or_build_prefill_indices(metadata,
                                            {1},
                                            {1},
                                            kCheckpointStride,
                                            kNumSlots,
                                            torch::Device(torch::kMeta)),
               "cache device changed");
}

}  // namespace
}  // namespace xllm::layer
