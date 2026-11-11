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

#include <acl/acl.h>
#include <gtest/gtest.h>

#include <limits>

#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

namespace xllm {
namespace {

TEST(StaticKVCacheTensorAllocatorNpuTest, KeepsDenseNdViewsInOnePhysicalArena) {
  uint32_t device_count = 0;
  ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
  if (device_count == 0) {
    GTEST_SKIP() << "An NPU is required to verify storage descriptors.";
  }
  const torch::Device device("npu:0");
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 8;
  spec.num_slabs = 20;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.dtype = torch::kBFloat16;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {0};
  spec.linear_layer_ids = {1};
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(spec), device);
  allocator.begin_layer(0);
  auto key = allocator.allocate(
      KVCacheTensorRole::KEY, {8, 2, 4}, torch::kBFloat16, device);
  auto value = allocator.allocate(
      KVCacheTensorRole::VALUE, {8, 2, 4}, torch::kBFloat16, device);
  allocator.begin_layer(1);
  auto conv = allocator.allocate(
      KVCacheTensorRole::CONV, {8, 4}, torch::kBFloat16, device);
  auto ssm = allocator.allocate(
      KVCacheTensorRole::SSM, {8, 2, 4}, torch::kFloat32, device);

  for (const auto& tensor : {key, value, conv, ssm}) {
    EXPECT_TRUE(tensor.is_contiguous());
#ifdef TORCH_HIGHER_THAN_PTA6
    EXPECT_EQ(at_npu::native::get_npu_format(tensor), ACL_FORMAT_ND);
#else
    EXPECT_EQ(at_npu::native::NPUNativeFunctions::get_npu_format(tensor),
              ACL_FORMAT_ND);
#endif
  }
  EXPECT_EQ(key[16].data_ptr(), conv[32].data_ptr());
  EXPECT_EQ(key[16].data_ptr(), ssm[8].data_ptr());
  EXPECT_EQ(value[16].data_ptr(), key[17].data_ptr());
  key[16].fill_(17);
  EXPECT_TRUE(conv[32].eq(17).all().item<bool>());
  conv[32].fill_(29);
  EXPECT_TRUE(key[16].reshape({8}).slice(0, 0, 4).eq(29).all().item<bool>());
  EXPECT_TRUE(key[16].reshape({8}).slice(0, 4, 8).eq(17).all().item<bool>());

  // A Full extent and a recurrent extent keep independent live state.
  key[8].fill_(3);
  value[8].fill_(4);
  conv[64].fill_(5);
  ssm[10].fill_(6);
  EXPECT_TRUE(key[8].eq(3).all().item<bool>());
  EXPECT_TRUE(value[8].eq(4).all().item<bool>());
  EXPECT_TRUE(conv[64].eq(5).all().item<bool>());
  EXPECT_TRUE(ssm[10].eq(6).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorNpuTest,
     InitializesFreshFullPairsWithoutChangingScratchOrPrefixes) {
  uint32_t device_count = 0;
  ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
  if (device_count == 0) {
    GTEST_SKIP() << "An NPU is required to verify shared arena initialization.";
  }
  const torch::Device device("npu:0");
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 8;
  spec.num_slabs = 20;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.checkpoint_stride = 2;
  spec.dtype = torch::kBFloat16;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {0, 2};
  spec.linear_layer_ids = {1};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, device);
  allocator.begin_layer(0);
  const torch::Tensor key = allocator.allocate(
      KVCacheTensorRole::KEY, {8, 2, 4}, torch::kBFloat16, device);
  const torch::Tensor value = allocator.allocate(
      KVCacheTensorRole::VALUE, {8, 2, 4}, torch::kBFloat16, device);
  allocator.begin_layer(1);
  const torch::Tensor conv = allocator.allocate(
      KVCacheTensorRole::CONV, {8, 4}, torch::kBFloat16, device);
  const torch::Tensor ssm = allocator.allocate(
      KVCacheTensorRole::SSM, {16, 2, 4}, torch::kFloat32, device);
  ssm.narrow(0, 0, 2).fill_(std::numeric_limits<float>::quiet_NaN());
  key[12].fill_(17);
  value[12].fill_(19);
  for (const int64_t row : {8, 10}) {
    key[row].fill_(std::numeric_limits<float>::quiet_NaN());
    value[row].fill_(std::numeric_limits<float>::quiet_NaN());
  }
  const std::vector<int32_t> rows = {0, 0, 0, 1, 8, 10};
  EXPECT_TRUE(
      allocator.needs_full_block_initialization(rows, /*lease_epochs=*/{0, 1}));
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 1});
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
    key[row].fill_(7);
    value[row].fill_(9);
  }
  EXPECT_FALSE(
      allocator.needs_full_block_initialization(rows, /*lease_epochs=*/{0, 1}));
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 1});
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(7).all().item<bool>());
    EXPECT_TRUE(value[row].eq(9).all().item<bool>());
  }
  EXPECT_TRUE(
      allocator.needs_full_block_initialization(rows, /*lease_epochs=*/{0, 2}));
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 2});
  for (const int64_t row : {8, 10}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
  }
  EXPECT_TRUE(torch::isnan(ssm.narrow(0, 0, 2)).all().item<bool>());
  EXPECT_TRUE(key[12].eq(17).all().item<bool>());
  EXPECT_TRUE(value[12].eq(19).all().item<bool>());
  const int64_t padding_row =
      static_cast<int64_t>(layout.full_read_padding_row());
  EXPECT_TRUE(key[padding_row].eq(0).all().item<bool>());
  EXPECT_TRUE(value[padding_row].eq(0).all().item<bool>());

  // Two disjoint new ranges surround live recurrent objects. Existing Full
  // leases and their adjacent physical pairs must keep their contents.
  for (const int64_t row : {8, 10, 38}) {
    key[row].fill_(7);
    value[row].fill_(9);
  }
  ssm.narrow(/*dim=*/0, /*start=*/12, /*length=*/2)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  conv.narrow(/*dim=*/0, /*start=*/68, /*length=*/8)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  ssm.narrow(/*dim=*/0, /*start=*/14, /*length=*/2).fill_(29);
  conv[64].fill_(31);
  const std::vector<int32_t> mixed_rows = {2, 36, 24, 1, 8, 10, 3, 34, 26};
  const std::vector<uint64_t> mixed_epochs = {1, 2, 1};
  EXPECT_TRUE(
      allocator.needs_full_block_initialization(mixed_rows, mixed_epochs));
  allocator.initialize_full_blocks(mixed_rows, mixed_epochs);
  EXPECT_FALSE(
      allocator.needs_full_block_initialization(mixed_rows, mixed_epochs));
  EXPECT_TRUE(ssm.narrow(/*dim=*/0, /*start=*/12, /*length=*/2)
                  .eq(0)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(conv.narrow(/*dim=*/0, /*start=*/68, /*length=*/8)
                  .eq(0)
                  .all()
                  .item<bool>());
  for (const int64_t row : {24, 26, 34, 36}) {
    EXPECT_TRUE(key[row].eq(0).all().item<bool>());
    EXPECT_TRUE(value[row].eq(0).all().item<bool>());
    key[row].fill_(11);
    value[row].fill_(13);
  }
  allocator.initialize_full_blocks(mixed_rows, mixed_epochs);
  for (const int64_t row : {24, 26, 34, 36}) {
    EXPECT_TRUE(key[row].eq(11).all().item<bool>());
    EXPECT_TRUE(value[row].eq(13).all().item<bool>());
  }
  for (const int64_t row : {8, 10, 38}) {
    EXPECT_TRUE(key[row].eq(7).all().item<bool>());
    EXPECT_TRUE(value[row].eq(9).all().item<bool>());
  }
  EXPECT_TRUE(ssm.narrow(/*dim=*/0, /*start=*/14, /*length=*/2)
                  .eq(29)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(conv[64].eq(31).all().item<bool>());
}

}  // namespace
}  // namespace xllm
