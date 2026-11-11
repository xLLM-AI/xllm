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

#include <cstdint>
#include <limits>
#include <vector>

#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

namespace xllm {
namespace {

struct NpuCacheViews {
  std::vector<torch::Tensor> keys;
  std::vector<torch::Tensor> values;
  std::vector<torch::Tensor> conv;
  std::vector<torch::Tensor> ssm;
};

NpuCacheViews allocate_views(StaticKVCacheTensorAllocator& allocator,
                             const StaticKVCacheLayoutSpec& spec,
                             const torch::Device& device) {
  NpuCacheViews views;
  views.keys.reserve(spec.full_layer_ids.size());
  views.values.reserve(spec.full_layer_ids.size());
  views.conv.reserve(spec.linear_layer_ids.size());
  views.ssm.reserve(spec.linear_layer_ids.size());
  for (const int64_t layer : spec.full_layer_ids) {
    allocator.begin_layer(layer);
    views.keys.emplace_back(allocator.allocate(
        KVCacheTensorRole::KEY, {8, 2, 4}, torch::kBFloat16, device));
    views.values.emplace_back(allocator.allocate(
        KVCacheTensorRole::VALUE, {8, 2, 4}, torch::kBFloat16, device));
  }
  for (const int64_t layer : spec.linear_layer_ids) {
    allocator.begin_layer(layer);
    views.conv.emplace_back(allocator.allocate(
        KVCacheTensorRole::CONV, {8, 4}, torch::kBFloat16, device));
    views.ssm.emplace_back(allocator.allocate(
        KVCacheTensorRole::SSM,
        {8 * static_cast<int64_t>(spec.checkpoint_stride), 2, 4},
        torch::kFloat32,
        device));
  }
  return views;
}

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
  spec.full_layer_ids = {0, 2};
  spec.linear_layer_ids = {1, 3};
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(spec), device);
  const NpuCacheViews views = allocate_views(allocator, spec, device);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(8);
  tensors.insert(tensors.end(), views.keys.begin(), views.keys.end());
  tensors.insert(tensors.end(), views.values.begin(), views.values.end());
  tensors.insert(tensors.end(), views.conv.begin(), views.conv.end());
  tensors.insert(tensors.end(), views.ssm.begin(), views.ssm.end());
  for (const torch::Tensor& tensor : tensors) {
    EXPECT_TRUE(tensor.is_contiguous());
#ifdef TORCH_HIGHER_THAN_PTA6
    EXPECT_EQ(at_npu::native::get_npu_format(tensor), ACL_FORMAT_ND);
#else
    EXPECT_EQ(at_npu::native::NPUNativeFunctions::get_npu_format(tensor),
              ACL_FORMAT_ND);
#endif
  }
  EXPECT_NE(views.keys[0][16].data_ptr(), views.keys[1][16].data_ptr());
  EXPECT_EQ(views.keys[0][16].data_ptr(), views.conv[0][32].data_ptr());
  EXPECT_EQ(views.keys[0][16].data_ptr(), views.ssm[0][8].data_ptr());
  EXPECT_EQ(views.keys[1][16].data_ptr(), views.conv[1][112].data_ptr());
  EXPECT_EQ(views.keys[1][16].data_ptr(), views.ssm[1][28].data_ptr());
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][16].fill_(static_cast<int64_t>(17 + layer));
    views.values[layer][16].fill_(static_cast<int64_t>(23 + layer));
  }
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][16].eq(17 + layer).all().item<bool>());
    EXPECT_TRUE(views.values[layer][16].eq(23 + layer).all().item<bool>());
  }
  views.conv[0][32].fill_(29);
  EXPECT_TRUE(
      views.keys[0][16].reshape({8}).slice(0, 0, 4).eq(29).all().item<bool>());
  EXPECT_TRUE(
      views.keys[0][16].reshape({8}).slice(0, 4, 8).eq(17).all().item<bool>());
  EXPECT_TRUE(views.keys[1][16].eq(18).all().item<bool>());
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
  spec.full_layer_ids = {0, 2, 4, 6};
  spec.linear_layer_ids = {1, 3};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, device);
  const NpuCacheViews views = allocate_views(allocator, spec, device);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][0].fill_(std::numeric_limits<float>::quiet_NaN());
    views.values[layer][0].fill_(std::numeric_limits<float>::quiet_NaN());
    views.keys[layer][8].fill_(std::numeric_limits<float>::quiet_NaN());
    views.values[layer][8].fill_(std::numeric_limits<float>::quiet_NaN());
    views.keys[layer][12].fill_(17);
    views.values[layer][12].fill_(19);
  }
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    const int64_t bank_conv_offset =
        static_cast<int64_t>(layer * layout.bank_bytes() / spec.conv_row_bytes);
    const int64_t bank_ssm_offset =
        static_cast<int64_t>(layer * layout.bank_bytes() / spec.ssm_row_bytes);
    views.conv[layer][bank_conv_offset + 40].fill_(31);
    views.ssm[layer].narrow(0, bank_ssm_offset + 12, 2).fill_(43);
  }
  const std::vector<int32_t> rows = {0, 0, 0, 0, 0, 1, 8, 8, 8, 8};
  EXPECT_TRUE(
      allocator.needs_full_block_initialization(rows, /*lease_epochs=*/{0, 1}));
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 1});
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][8].eq(0).all().item<bool>());
    EXPECT_TRUE(views.values[layer][8].eq(0).all().item<bool>());
    views.keys[layer][8].fill_(7);
    views.values[layer][8].fill_(9);
  }
  EXPECT_FALSE(
      allocator.needs_full_block_initialization(rows, /*lease_epochs=*/{0, 1}));
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 1});
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][8].eq(7).all().item<bool>());
    EXPECT_TRUE(views.values[layer][8].eq(9).all().item<bool>());
  }
  allocator.initialize_full_blocks(rows, /*lease_epochs=*/{0, 2});
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][8].eq(0).all().item<bool>());
    views.keys[layer][8].fill_(7);
    views.values[layer][8].fill_(9);
    for (const int64_t page : {16, 28}) {
      views.keys[layer][page].fill_(std::numeric_limits<float>::quiet_NaN());
      views.values[layer][page].fill_(std::numeric_limits<float>::quiet_NaN());
    }
  }
  // Fresh bundles surround live Conv and SSM groups in both banks.
  const std::vector<int32_t> mixed_rows = {
      3, 16, 16, 16, 16, 1, 8, 8, 8, 8, 4, 28, 28, 28, 28};
  const std::vector<uint64_t> epochs = {1, 2, 1};
  EXPECT_TRUE(allocator.needs_full_block_initialization(mixed_rows, epochs));
  allocator.initialize_full_blocks(mixed_rows, epochs);
  EXPECT_FALSE(allocator.needs_full_block_initialization(mixed_rows, epochs));
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    for (const int64_t page : {16, 28}) {
      EXPECT_TRUE(views.keys[layer][page].eq(0).all().item<bool>());
      EXPECT_TRUE(views.values[layer][page].eq(0).all().item<bool>());
      views.keys[layer][page].fill_(11);
      views.values[layer][page].fill_(13);
    }
    EXPECT_TRUE(torch::isnan(views.keys[layer][0]).all().item<bool>());
    EXPECT_TRUE(torch::isnan(views.values[layer][0]).all().item<bool>());
    EXPECT_TRUE(views.keys[layer][8].eq(7).all().item<bool>());
    EXPECT_TRUE(views.values[layer][8].eq(9).all().item<bool>());
    EXPECT_TRUE(views.keys[layer][12].eq(17).all().item<bool>());
    EXPECT_TRUE(views.values[layer][12].eq(19).all().item<bool>());
    EXPECT_TRUE(views.keys[layer][layout.full_read_padding_page()]
                    .eq(0)
                    .all()
                    .item<bool>());
    EXPECT_TRUE(views.values[layer][layout.full_read_padding_page()]
                    .eq(0)
                    .all()
                    .item<bool>());
  }
  allocator.initialize_full_blocks(mixed_rows, epochs);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    for (const int64_t page : {16, 28}) {
      EXPECT_TRUE(views.keys[layer][page].eq(11).all().item<bool>());
      EXPECT_TRUE(views.values[layer][page].eq(13).all().item<bool>());
    }
  }
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    const int64_t conv_row = static_cast<int64_t>(layer * layout.bank_bytes() /
                                                  spec.conv_row_bytes) +
                             40;
    const int64_t ssm_row =
        static_cast<int64_t>(layer * layout.bank_bytes() / spec.ssm_row_bytes) +
        12;
    EXPECT_TRUE(views.conv[layer][conv_row].eq(31).all().item<bool>());
    EXPECT_TRUE(
        views.ssm[layer].narrow(0, ssm_row, 2).eq(43).all().item<bool>());
  }
}

}  // namespace
}  // namespace xllm
