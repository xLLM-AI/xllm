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

#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

StaticKVCacheLayoutSpec make_spec(size_t num_units = 12,
                                  size_t checkpoint_stride = 1) {
  StaticKVCacheLayoutSpec spec;
  spec.num_units = num_units;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.checkpoint_stride = checkpoint_stride;
  spec.dtype = torch::kFloat32;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {0, 3};
  spec.linear_layer_ids = {1, 2};
  return spec;
}

torch::Tensor allocate_float_tensor(StaticKVCacheTensorAllocator& allocator,
                                    KVCacheTensorRole role,
                                    const std::vector<int64_t>& shape) {
  return allocator.allocate(role, shape, torch::kFloat32, torch::kCPU);
}

TEST(StaticKVCacheTensorAllocatorTest, PlansGeometryWithoutAllocatingCapacity) {
  const StaticKVCacheLayout layout(make_spec(/*num_units=*/0));
  EXPECT_EQ(layout.unit_bytes(), 128);
  EXPECT_EQ(layout.full_block_units(), 1);
  EXPECT_EQ(layout.linear_state_units(), 2);
  EXPECT_EQ(layout.role_row_stride(KVCacheTensorRole::KEY), 8);
  EXPECT_EQ(layout.role_row_stride(KVCacheTensorRole::VALUE), 8);
  EXPECT_EQ(layout.role_row_stride(KVCacheTensorRole::CONV), 16);
  EXPECT_EQ(layout.role_row_stride(KVCacheTensorRole::SSM), 4);
  EXPECT_EQ(layout.bytes_used(), 0);
}

TEST(StaticKVCacheTensorAllocatorTest, GroupsFourFullLayersAcrossPaddedPages) {
  auto spec = make_spec(/*num_units=*/6);
  spec.conv_row_bytes = 24;
  spec.full_layer_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  spec.linear_layer_ids = {9};
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(spec),
                                         torch::kCPU);
  const auto& layout = allocator.layout();
  EXPECT_EQ(layout.unit_bytes(), 192);
  EXPECT_EQ(layout.full_page_payload_bytes(), 128);
  EXPECT_EQ(layout.linear_page_payload_bytes(), 56);
  EXPECT_EQ(layout.full_block_units(), 3);
  EXPECT_EQ(layout.linear_state_units(), 1);
  const auto& first = layout.tensor_layout(0, KVCacheTensorRole::KEY);
  const auto& fourth = layout.tensor_layout(3, KVCacheTensorRole::KEY);
  const auto& fifth = layout.tensor_layout(4, KVCacheTensorRole::KEY);
  const auto& ninth = layout.tensor_layout(8, KVCacheTensorRole::KEY);
  const auto& ninth_value = layout.tensor_layout(8, KVCacheTensorRole::VALUE);
  EXPECT_EQ(first.page_group_index, 0);
  EXPECT_EQ(fourth.page_group_index, 0);
  EXPECT_EQ(fifth.page_group_index, 1);
  EXPECT_EQ(ninth.page_group_index, 2);
  EXPECT_EQ(fourth.offset_bytes, 96);
  EXPECT_EQ(fifth.offset_bytes, 192);
  EXPECT_EQ(ninth.offset_bytes, 384);
  EXPECT_EQ(ninth_value.offset_bytes, 400);
  // The last group has one live K/V pair and padding for three absent layers.
  EXPECT_EQ(3 * layout.unit_bytes() - (ninth_value.offset_bytes + 16), 160);

  allocator.begin_layer(0);
  auto key0 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {6, 2, 2});
  allocator.begin_layer(4);
  auto key4 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {6, 2, 2});
  allocator.begin_layer(8);
  auto key8 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {6, 2, 2});
  ASSERT_EQ(key8.size(0), 37);
  key0[36].fill_(1);
  key4[36].fill_(2);
  key8[36].fill_(3);
  EXPECT_TRUE(key0[36].eq(1).all().item<bool>());
  EXPECT_TRUE(key4[36].eq(2).all().item<bool>());
  EXPECT_TRUE(key8[36].eq(3).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest, PadsDefaultQwenGeometryToCommonPages) {
  auto spec = make_spec(/*num_units=*/0);
  spec.full_row_bytes = 128 * 2 * 256 * 2;
  spec.conv_row_bytes = 3 * (128 * 16 * 2 + 128 * 32) * 2;
  spec.ssm_row_bytes = 32 * 128 * 128 * 2;
  spec.dtype = torch::kBFloat16;
  spec.ssm_dtype = torch::kBFloat16;
  spec.full_layer_ids.clear();
  spec.linear_layer_ids.clear();
  for (int64_t layer = 0; layer < 48; ++layer) {
    auto& group =
        (layer + 1) % 4 == 0 ? spec.full_layer_ids : spec.linear_layer_ids;
    group.emplace_back(layer);
  }
  const StaticKVCacheLayout next_layout(spec);
  EXPECT_EQ(next_layout.unit_bytes(), 3 * 1024 * 1024);
  EXPECT_EQ(next_layout.full_page_payload_bytes(), 1024 * 1024);
  EXPECT_EQ(next_layout.linear_page_payload_bytes(), 1024 * 1024 + 48 * 1024);
  EXPECT_EQ(next_layout.full_block_units(), 3);
  EXPECT_EQ(next_layout.linear_state_units(), 36);
  spec.ssm_row_bytes *= 2;
  spec.ssm_dtype = torch::kFloat32;
  const StaticKVCacheLayout qwen35_layout(spec);
  EXPECT_EQ(qwen35_layout.unit_bytes(), 6 * 1024 * 1024);
  EXPECT_EQ(qwen35_layout.full_page_payload_bytes(), 1024 * 1024);
  EXPECT_EQ(qwen35_layout.linear_page_payload_bytes(),
            2 * 1024 * 1024 + 48 * 1024);
}

TEST(StaticKVCacheTensorAllocatorTest,
     KeepsSimultaneousResourceExtentsIsolated) {
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                         torch::kCPU);
  allocator.begin_layer(0);
  auto key0 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  auto value0 =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  allocator.begin_layer(3);
  auto key3 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  auto value3 =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  allocator.begin_layer(1);
  auto conv1 =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  auto ssm1 =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {12, 2, 4});
  allocator.begin_layer(2);
  auto conv2 =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  auto ssm2 =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {12, 2, 4});

  ASSERT_EQ(allocator.bytes_used(), 1536);
  ASSERT_EQ(key0.size(0), 89);
  ASSERT_EQ(value0.size(0), key0.size(0));
  ASSERT_EQ(key3.size(0), key0.size(0));
  ASSERT_EQ(value3.size(0), key0.size(0));
  ASSERT_EQ(conv1.size(0), 161);
  ASSERT_EQ(ssm1.size(0), 41);
  for (const auto& tensor :
       {key0, value0, key3, value3, conv1, ssm1, conv2, ssm2}) {
    ASSERT_TRUE(tensor.is_contiguous());
  }

  // Full extent [1, 2) and recurrent extent [6, 8) are live together.
  key0[8].fill_(11);
  value0[8].fill_(12);
  key3[8].fill_(13);
  value3[8].fill_(14);
  conv1[96].fill_(21);
  ssm1[24].fill_(22);
  conv2[96].fill_(23);
  ssm2[24].fill_(24);
  EXPECT_TRUE(key0[8].eq(11).all().item<bool>());
  EXPECT_TRUE(value0[8].eq(12).all().item<bool>());
  EXPECT_TRUE(key3[8].eq(13).all().item<bool>());
  EXPECT_TRUE(value3[8].eq(14).all().item<bool>());
  EXPECT_TRUE(conv1[96].eq(21).all().item<bool>());
  EXPECT_TRUE(ssm1[24].eq(22).all().item<bool>());
  EXPECT_TRUE(conv2[96].eq(23).all().item<bool>());
  EXPECT_TRUE(ssm2[24].eq(24).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest, ReusesTheSameBytesAcrossAttentionTypes) {
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                         torch::kCPU);
  allocator.begin_layer(0);
  auto key =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  key[16].fill_(17);
  allocator.begin_layer(1);
  auto conv =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});

  // Creating a role view cannot zero bytes already used by another view.
  EXPECT_TRUE(key[16].eq(17).all().item<bool>());
  EXPECT_EQ(key[16].data_ptr<float>(), conv[32].data_ptr<float>());
  EXPECT_TRUE(conv[32].eq(17).all().item<bool>());
  conv[32].fill_(29);
  EXPECT_TRUE(key[16].reshape({4}).slice(0, 0, 2).eq(29).all().item<bool>());
  EXPECT_TRUE(key[16].reshape({4}).slice(0, 2, 4).eq(17).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest, FitsLastExtentAndAllCheckpointRows) {
  StaticKVCacheTensorAllocator allocator(
      StaticKVCacheLayout(make_spec(/*num_units=*/12, /*checkpoint_stride=*/2)),
      torch::kCPU);
  ASSERT_EQ(allocator.layout().linear_state_units(), 2);
  allocator.begin_layer(1);
  auto conv =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  auto ssm =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});
  allocator.begin_layer(2);
  auto last_ssm =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});
  ASSERT_EQ(conv.size(0), 161);
  ASSERT_EQ(ssm.size(0), 42);
  ASSERT_EQ(last_ssm.size(0), 42);
  const int64_t last_start = 10;
  conv[last_start * 16].fill_(3);
  ssm[last_start * 4].fill_(4);
  ssm[last_start * 4 + 1].fill_(5);
  last_ssm[last_start * 4].fill_(6);
  last_ssm[last_start * 4 + 1].fill_(7);
  EXPECT_TRUE(conv[160].eq(3).all().item<bool>());
  EXPECT_TRUE(ssm[40].eq(4).all().item<bool>());
  EXPECT_TRUE(ssm[41].eq(5).all().item<bool>());
  EXPECT_TRUE(last_ssm[40].eq(6).all().item<bool>());
  EXPECT_TRUE(last_ssm[41].eq(7).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest, TensorViewRetainsArenaOwnership) {
  torch::Tensor retained;
  {
    StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                           torch::kCPU);
    allocator.begin_layer(0);
    retained =
        allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
    retained[88].fill_(43);
  }
  EXPECT_TRUE(retained[88].eq(43).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest, RejectsInvalidLayerAndRowGeometry) {
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                         torch::kCPU);
  EXPECT_DEATH(
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2}),
      "explicit layer");
  EXPECT_DEATH(allocator.begin_layer(7), "Unknown static cache layer");
  allocator.begin_layer(0);
  EXPECT_DEATH(
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {11, 2, 2}),
      "logical capacity");
  EXPECT_DEATH(
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 3}),
      "inner row geometry");
  EXPECT_DEATH(
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2}),
      "has no conv");
  allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  EXPECT_DEATH(
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2}),
      "role cannot be allocated twice");
  EXPECT_DEATH(allocator.begin_layer(0), "layer cannot be allocated twice");
}

}  // namespace
}  // namespace xllm
