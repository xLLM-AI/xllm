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
#include <limits>
#include <vector>

namespace xllm {
namespace {

StaticKVCacheLayoutSpec make_spec(size_t num_slots = 12,
                                  size_t checkpoint_stride = 1,
                                  size_t num_slabs = 12) {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = num_slots;
  spec.num_slabs = num_slabs;
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
  const StaticKVCacheLayout layout(make_spec(
      /*num_slots=*/0, /*checkpoint_stride=*/1, /*num_slabs=*/0));
  EXPECT_EQ(layout.slab_bytes(), 32);
  EXPECT_EQ(layout.num_slabs(), 0);
  EXPECT_EQ(layout.num_reserved_slabs(), 2);
  EXPECT_EQ(layout.full_read_padding_row(), 2);
  EXPECT_EQ(layout.full_pair_bytes(), 32);
  EXPECT_EQ(layout.ssm_bundle_bytes(), 32);
  EXPECT_EQ(layout.full_layer_count(), 2);
  EXPECT_EQ(layout.linear_layer_count(), 2);
  EXPECT_EQ(layout.num_linear_groups(), 4);
  EXPECT_EQ(layout.full_layer_index(3), 1);
  EXPECT_EQ(layout.linear_group(1, KVCacheTensorRole::CONV), 0);
  EXPECT_EQ(layout.linear_group(1, KVCacheTensorRole::SSM), 1);
  EXPECT_EQ(layout.linear_group(2, KVCacheTensorRole::CONV), 2);
  EXPECT_EQ(layout.linear_group(2, KVCacheTensorRole::SSM), 3);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::KEY), 2);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::VALUE), 2);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::CONV), 4);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::SSM), 1);
  EXPECT_EQ(layout.bytes_used(), 0);
}

TEST(StaticKVCacheTensorAllocatorTest,
     SlabsFitWholeObjectsWithoutLayerPadding) {
  auto spec = make_spec(/*num_slots=*/100,
                        /*checkpoint_stride=*/2,
                        /*num_slabs=*/3);
  spec.conv_row_bytes = 24;
  const StaticKVCacheLayout first(spec);
  EXPECT_EQ(first.slab_bytes(), 192);
  EXPECT_EQ(first.bytes_used(), 576);
  EXPECT_EQ(first.rows_per_slab(KVCacheTensorRole::KEY), 12);
  EXPECT_EQ(first.rows_per_slab(KVCacheTensorRole::CONV), 8);
  EXPECT_EQ(first.rows_per_slab(KVCacheTensorRole::SSM), 6);
  spec.full_layer_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  spec.linear_layer_ids = {9};
  const StaticKVCacheLayout many_layers(spec);
  EXPECT_EQ(many_layers.slab_bytes(), first.slab_bytes());
  EXPECT_EQ(many_layers.bytes_used(), first.bytes_used());
  EXPECT_EQ(many_layers.tensor_layout(8, KVCacheTensorRole::KEY).offset_bytes,
            0);
  EXPECT_EQ(many_layers.tensor_layout(8, KVCacheTensorRole::VALUE).offset_bytes,
            16);
  EXPECT_EQ(many_layers.tensor_layout(9, KVCacheTensorRole::CONV).offset_bytes,
            0);
  EXPECT_EQ(many_layers.tensor_layout(9, KVCacheTensorRole::SSM).offset_bytes,
            0);
}

TEST(StaticKVCacheTensorAllocatorTest, DerivesQwen38SlabsFromRealObjectSizes) {
  auto spec = make_spec(/*num_slots=*/0,
                        /*checkpoint_stride=*/5,
                        /*num_slabs=*/0);
  spec.full_row_bytes = 128 * 2 * 256 * 2;
  spec.conv_row_bytes = 7 * (2 * 8 * 128 + 24 * 128) * 2;
  spec.ssm_row_bytes = 24 * 128 * 128 * 4;
  spec.dtype = torch::kBFloat16;
  spec.ssm_dtype = torch::kFloat32;
  const StaticKVCacheLayout layout(spec);
  EXPECT_EQ(layout.full_pair_bytes(), 262144);
  EXPECT_EQ(layout.conv_row_bytes(), 71680);
  EXPECT_EQ(layout.ssm_bundle_bytes(), 7864320);
  EXPECT_EQ(layout.slab_bytes(), 55050240);
  EXPECT_EQ(layout.num_reserved_slabs(), 1);
  EXPECT_EQ(layout.full_read_padding_row(), 60);
  EXPECT_GE(layout.full_read_padding_row() * layout.full_row_bytes(),
            layout.ssm_bundle_bytes());
  EXPECT_LE(layout.full_read_padding_row() * layout.full_row_bytes() +
                layout.full_pair_bytes(),
            layout.num_reserved_slabs() * layout.slab_bytes());
  EXPECT_EQ(layout.slab_bytes() % layout.full_pair_bytes(), 0);
  EXPECT_EQ(layout.slab_bytes() % layout.conv_row_bytes(), 0);
  EXPECT_EQ(layout.slab_bytes() % layout.ssm_bundle_bytes(), 0);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::KEY), 420);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::CONV), 768);
  EXPECT_EQ(layout.rows_per_slab(KVCacheTensorRole::SSM), 35);
}

TEST(StaticKVCacheTensorAllocatorTest,
     KeepsReadPaddingFiniteAfterEveryWritableScratchRoleChanges) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2));
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  allocator.begin_layer(0);
  const torch::Tensor key0 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  const torch::Tensor value0 =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  allocator.begin_layer(3);
  const torch::Tensor key3 =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  const torch::Tensor value3 =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  allocator.begin_layer(1);
  const torch::Tensor conv =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  const torch::Tensor ssm =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});

  key0[0].fill_(std::numeric_limits<float>::quiet_NaN());
  value0[0].fill_(std::numeric_limits<float>::quiet_NaN());
  conv[0].fill_(std::numeric_limits<float>::quiet_NaN());
  ssm.narrow(0, 0, layout.checkpoint_stride())
      .fill_(std::numeric_limits<float>::quiet_NaN());

  const int64_t padding_row =
      static_cast<int64_t>(layout.full_read_padding_row());
  EXPECT_TRUE(key0[padding_row].eq(0).all().item<bool>());
  EXPECT_TRUE(value0[padding_row].eq(0).all().item<bool>());
  EXPECT_EQ(key0[padding_row].data_ptr(), key3[padding_row].data_ptr());
  EXPECT_EQ(value0[padding_row].data_ptr(), value3[padding_row].data_ptr());
}

TEST(StaticKVCacheTensorAllocatorTest,
     InitializesOnlyFreshFullLeasesAndPreservesPrefixes) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2));
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  allocator.begin_layer(0);
  const torch::Tensor key =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  const torch::Tensor value =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  key[0].fill_(47);
  value[0].fill_(48);
  key[12].fill_(53);
  value[12].fill_(54);
  for (const int64_t row : {8, 10}) {
    key[row].fill_(std::numeric_limits<float>::quiet_NaN());
    value[row].fill_(std::numeric_limits<float>::quiet_NaN());
  }
  const std::vector<int32_t> rows = {0, 0, 0, 1, 8, 10};
  const std::vector<uint64_t> epochs = {0, 1};
  EXPECT_TRUE(allocator.needs_full_block_initialization(rows, epochs));
  allocator.initialize_full_blocks(rows, epochs);
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
    key[row].fill_(11);
    value[row].fill_(12);
  }
  EXPECT_FALSE(allocator.needs_full_block_initialization(rows, epochs));
  allocator.initialize_full_blocks(rows, epochs);
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(11).all().item<bool>());
    EXPECT_TRUE(value[row].eq(12).all().item<bool>());
  }
  EXPECT_TRUE(key[12].eq(53).all().item<bool>());
  EXPECT_TRUE(value[12].eq(54).all().item<bool>());
  EXPECT_TRUE(key[0].eq(47).all().item<bool>());
  EXPECT_TRUE(value[0].eq(48).all().item<bool>());
  EXPECT_FALSE(allocator.needs_full_block_initialization(
      /*rows=*/{0, 0, 0}, /*lease_epochs=*/{0}));
  const int64_t padding_row =
      static_cast<int64_t>(layout.full_read_padding_row());
  EXPECT_TRUE(key[padding_row].eq(0).all().item<bool>());
  EXPECT_TRUE(value[padding_row].eq(0).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest,
     ClearsReusedFullLeasesAtTheSameAndDifferentPhysicalRows) {
  StaticKVCacheTensorAllocator allocator(
      StaticKVCacheLayout(make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2)),
      torch::kCPU);
  allocator.begin_layer(0);
  const torch::Tensor key =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  const torch::Tensor value =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  const std::vector<int32_t> first_rows = {1, 8, 10};
  EXPECT_TRUE(allocator.needs_full_block_initialization(first_rows,
                                                        /*lease_epochs=*/{1}));
  allocator.initialize_full_blocks(first_rows, /*lease_epochs=*/{1});
  for (const int64_t row : {8, 10}) {
    key[row].fill_(std::numeric_limits<float>::quiet_NaN());
    value[row].fill_(std::numeric_limits<float>::quiet_NaN());
  }
  EXPECT_TRUE(allocator.needs_full_block_initialization(first_rows,
                                                        /*lease_epochs=*/{2}));
  allocator.initialize_full_blocks(first_rows, /*lease_epochs=*/{2});
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
    key[row].fill_(31);
    value[row].fill_(32);
  }
  // Retyping can return the same logical ID with another layer-row mapping.
  const std::vector<int32_t> moved_rows = {1, 12, 14};
  for (const int64_t row : {12, 14}) {
    key[row].fill_(std::numeric_limits<float>::quiet_NaN());
    value[row].fill_(std::numeric_limits<float>::quiet_NaN());
  }
  EXPECT_TRUE(allocator.needs_full_block_initialization(moved_rows,
                                                        /*lease_epochs=*/{3}));
  allocator.initialize_full_blocks(moved_rows, /*lease_epochs=*/{3});
  EXPECT_FALSE(allocator.needs_full_block_initialization(moved_rows,
                                                         /*lease_epochs=*/{3}));
  for (const int64_t row : {12, 14}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
  }
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(31).all().item<bool>());
    EXPECT_TRUE(value[row].eq(32).all().item<bool>());
  }
}

TEST(StaticKVCacheTensorAllocatorTest,
     ClearsDisjointReusedPairsWithoutChangingLiveNeighborRoles) {
  auto spec = make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2);
  spec.dtype = torch::kBFloat16;
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(spec),
                                         torch::kCPU);
  allocator.begin_layer(0);
  const torch::Tensor key = allocator.allocate(
      KVCacheTensorRole::KEY, {12, 2, 4}, torch::kBFloat16, torch::kCPU);
  const torch::Tensor value = allocator.allocate(
      KVCacheTensorRole::VALUE, {12, 2, 4}, torch::kBFloat16, torch::kCPU);
  allocator.begin_layer(1);
  const torch::Tensor conv = allocator.allocate(
      KVCacheTensorRole::CONV, {12, 4}, torch::kBFloat16, torch::kCPU);
  const torch::Tensor ssm = allocator.allocate(
      KVCacheTensorRole::SSM, {24, 2, 4}, torch::kFloat32, torch::kCPU);

  allocator.initialize_full_blocks(/*rows=*/{3, 22, 24}, /*lease_epochs=*/{1});
  for (const int64_t row : {22, 24}) {
    key[row].fill_(13);
    value[row].fill_(17);
  }
  // Former recurrent objects cover two adjacent pairs in each range. Live
  // SSM and Conv objects occupy the gap; an existing Full lease follows it.
  ssm.narrow(/*dim=*/0, /*start=*/4, /*length=*/2)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  conv.narrow(/*dim=*/0, /*start=*/36, /*length=*/8)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  ssm.narrow(/*dim=*/0, /*start=*/6, /*length=*/2).fill_(29);
  conv[32].fill_(31);
  const std::vector<int32_t> rows = {0, 0, 0, 1, 20, 8, 3, 22, 24, 2, 18, 10};
  const std::vector<uint64_t> epochs = {0, 1, 1, 1};
  EXPECT_TRUE(allocator.needs_full_block_initialization(rows, epochs));
  allocator.initialize_full_blocks(rows, epochs);
  EXPECT_FALSE(allocator.needs_full_block_initialization(rows, epochs));
  EXPECT_TRUE(ssm.narrow(/*dim=*/0, /*start=*/4, /*length=*/2)
                  .eq(0)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(conv.narrow(/*dim=*/0, /*start=*/36, /*length=*/8)
                  .eq(0)
                  .all()
                  .item<bool>());
  for (const int64_t row : {8, 10, 18, 20}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
    key[row].fill_(7);
    value[row].fill_(9);
  }
  allocator.initialize_full_blocks(rows, epochs);
  for (const int64_t row : {8, 10, 18, 20}) {
    EXPECT_TRUE(key[row].eq(7).all().item<bool>());
    EXPECT_TRUE(value[row].eq(9).all().item<bool>());
  }

  // Renew only one logical block; the neighboring new block remains live.
  allocator.initialize_full_blocks(/*rows=*/{1, 8, 20}, /*lease_epochs=*/{2});
  for (const int64_t row : {8, 20}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
  }
  for (const int64_t row : {10, 18}) {
    EXPECT_TRUE(key[row].eq(7).all().item<bool>());
    EXPECT_TRUE(value[row].eq(9).all().item<bool>());
  }
  for (const int64_t row : {22, 24}) {
    EXPECT_TRUE(key[row].eq(13).all().item<bool>());
    EXPECT_TRUE(value[row].eq(17).all().item<bool>());
  }
  EXPECT_TRUE(ssm.narrow(/*dim=*/0, /*start=*/6, /*length=*/2)
                  .eq(29)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(conv[32].eq(31).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest,
     KeepsSimultaneousResourceExtentsAndScratchIsolated) {
  StaticKVCacheTensorAllocator allocator(
      StaticKVCacheLayout(make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2)),
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
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});
  allocator.begin_layer(2);
  auto conv2 =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  auto ssm2 =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});

  ASSERT_EQ(allocator.bytes_used(), 768);
  ASSERT_EQ(key0.size(0), 48);
  ASSERT_EQ(value0.size(0), 47);
  ASSERT_EQ(key3.size(0), key0.size(0));
  ASSERT_EQ(value3.size(0), value0.size(0));
  ASSERT_EQ(conv1.size(0), 96);
  ASSERT_EQ(ssm1.size(0), 24);
  EXPECT_EQ(key0.data_ptr(), key3.data_ptr());
  EXPECT_EQ(value0.data_ptr(), key0[1].data_ptr());
  for (const auto& tensor :
       {key0, value0, key3, value3, conv1, ssm1, conv2, ssm2}) {
    ASSERT_TRUE(tensor.is_contiguous());
  }

  // Live objects begin after the writable scratch and immutable read padding.
  key0[0].fill_(7);
  key0[8].fill_(11);
  value0[8].fill_(12);
  key3[10].fill_(13);
  value3[10].fill_(14);
  conv1[24].fill_(21);
  ssm1.narrow(0, 8, 2).fill_(22);
  conv2[40].fill_(23);
  ssm2.narrow(0, 12, 2).fill_(24);
  EXPECT_TRUE(key0[0].eq(7).all().item<bool>());
  EXPECT_TRUE(key0[8].eq(11).all().item<bool>());
  EXPECT_TRUE(value0[8].eq(12).all().item<bool>());
  EXPECT_TRUE(key3[10].eq(13).all().item<bool>());
  EXPECT_TRUE(value3[10].eq(14).all().item<bool>());
  EXPECT_TRUE(conv1[24].eq(21).all().item<bool>());
  EXPECT_TRUE(ssm1.narrow(0, 8, 2).eq(22).all().item<bool>());
  EXPECT_TRUE(conv2[40].eq(23).all().item<bool>());
  EXPECT_TRUE(ssm2.narrow(0, 12, 2).eq(24).all().item<bool>());
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
      StaticKVCacheLayout(make_spec(/*num_slots=*/12, /*checkpoint_stride=*/2)),
      torch::kCPU);
  allocator.begin_layer(0);
  auto key =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
  auto value =
      allocate_float_tensor(allocator, KVCacheTensorRole::VALUE, {12, 2, 2});
  allocator.begin_layer(1);
  auto conv =
      allocate_float_tensor(allocator, KVCacheTensorRole::CONV, {12, 2});
  auto ssm =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {24, 2, 4});
  key[46].fill_(3);
  value[46].fill_(4);
  EXPECT_TRUE(key[46].eq(3).all().item<bool>());
  EXPECT_TRUE(value[46].eq(4).all().item<bool>());
  conv[95].fill_(5);
  EXPECT_TRUE(conv[95].eq(5).all().item<bool>());
  ssm[22].fill_(6);
  ssm[23].fill_(7);
  EXPECT_TRUE(ssm[22].eq(6).all().item<bool>());
  EXPECT_TRUE(ssm[23].eq(7).all().item<bool>());
  EXPECT_EQ(value[46].data_ptr(), key[47].data_ptr());
}

TEST(StaticKVCacheTensorAllocatorTest,
     SeparatesLogicalCapacityFromPhysicalArenaCapacity) {
  StaticKVCacheTensorAllocator allocator(
      StaticKVCacheLayout(make_spec(/*num_slots=*/100,
                                    /*checkpoint_stride=*/2,
                                    /*num_slabs=*/3)),
      torch::kCPU);
  allocator.begin_layer(0);
  auto key =
      allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {100, 2, 2});
  allocator.begin_layer(1);
  auto ssm =
      allocate_float_tensor(allocator, KVCacheTensorRole::SSM, {200, 2, 4});
  EXPECT_EQ(allocator.layout().num_slots(), 100);
  EXPECT_EQ(allocator.layout().num_slabs(), 3);
  EXPECT_EQ(allocator.bytes_used(), 192);
  EXPECT_EQ(key.size(0), 12);
  EXPECT_EQ(ssm.size(0), 6);
}

TEST(StaticKVCacheTensorAllocatorTest, TensorViewRetainsArenaOwnership) {
  torch::Tensor retained;
  {
    StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                           torch::kCPU);
    allocator.begin_layer(0);
    retained =
        allocate_float_tensor(allocator, KVCacheTensorRole::KEY, {12, 2, 2});
    retained[23].fill_(43);
  }
  EXPECT_TRUE(retained[23].eq(43).all().item<bool>());
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
