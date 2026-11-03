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
  spec.num_units = 8;
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
  EXPECT_EQ(key[16].data_ptr(), ssm[7].data_ptr());
  key[16].fill_(17);
  EXPECT_TRUE(conv[32].eq(17).all().item<bool>());
  conv[32].fill_(29);
  EXPECT_TRUE(key[16].reshape({8}).slice(0, 0, 4).eq(29).all().item<bool>());
  EXPECT_TRUE(key[16].reshape({8}).slice(0, 4, 8).eq(17).all().item<bool>());

  // A Full extent and a recurrent extent keep independent live state.
  key[8].fill_(3);
  value[8].fill_(4);
  conv[64].fill_(5);
  ssm[16].fill_(6);
  EXPECT_TRUE(key[8].eq(3).all().item<bool>());
  EXPECT_TRUE(value[8].eq(4).all().item<bool>());
  EXPECT_TRUE(conv[64].eq(5).all().item<bool>());
  EXPECT_TRUE(ssm[16].eq(6).all().item<bool>());
}

}  // namespace
}  // namespace xllm
