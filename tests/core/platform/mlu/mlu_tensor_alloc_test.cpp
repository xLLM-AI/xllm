/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/mlu/mlu_tensor_alloc.h"

#include <cnrt.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/model/model_args.h"
#include "core/platform/device.h"
#include "core/platform/mlu/mlu_rdma_memory_plan.h"
#include "core/platform/platform.h"

namespace xllm::mlu {
namespace {

constexpr size_t kScaleBlockBytes = 96 * sizeof(float);
constexpr size_t kLogicalBytes = 2 * kScaleBlockBytes;
constexpr size_t kExpectedRegisteredBytes = 2097408;
const std::vector<int64_t> kScaleShape = {2, 96, 1};

class MluTensorAllocTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for tensor allocator tests.";
    }
    device_ = torch::Device(Platform::type_torch(), /*device_index=*/0);
    Device device(device_);
    device.set_device();
  }

  torch::Device device_{torch::kCPU};
};

TEST_F(MluTensorAllocTest, RegularAllocationKeepsLogicalStorageSize) {
  const torch::Tensor tensor =
      alloc_zero_tensor(kScaleShape, torch::kFloat32, device_);

  EXPECT_EQ(static_cast<size_t>(tensor.nbytes()), kLogicalBytes);
  EXPECT_EQ(tensor.storage().nbytes(), kLogicalBytes);
}

TEST_F(MluTensorAllocTest,
       RdmaRegisterableAllocationPreservesShapeAndExpandsStorage) {
  const torch::Tensor tensor = alloc_rdma_registerable_zero_tensor(
      kScaleShape, torch::kFloat32, device_);

  EXPECT_EQ(tensor.sizes().vec(), kScaleShape);
  EXPECT_EQ(static_cast<size_t>(tensor.nbytes()), kLogicalBytes);
  EXPECT_EQ(tensor.storage().nbytes(), kExpectedRegisteredBytes);
  EXPECT_EQ(get_rdma_registerable_nbytes(tensor), kExpectedRegisteredBytes);
}

TEST_F(MluTensorAllocTest,
       EstimatedCapacityIsLargestThatFitsActualMultiLayerScaleStorage) {
  constexpr int64_t kNumLayers = 4;
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kCacheBudgetBytes = 10 * 1024 * 1024;
  ModelArgs model_args;
  model_args.n_layers(kNumLayers)
      .head_dim(16)
      .model_type("deepseek_v32")
      .index_n_heads(1)
      .index_head_dim(16);
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat16;
  options.kv_cache_dtype = "auto";
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = kCacheBudgetBytes;
  options.block_size = kBlockSize;
  options.n_local_kv_heads = 2;
  options.max_seqs_per_batch = 8;
  options.max_concurrent_requests = 8;
  options.enable_rdma_scale_padding = true;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  ASSERT_EQ(capacity.n_blocks(), 227);

  const std::vector<int64_t> scale_shape = {capacity.n_blocks(), kBlockSize, 1};
  std::vector<torch::Tensor> tensors;
  tensors.reserve(kNumLayers);

  size_t scale_storage_bytes = 0;
  for (int64_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
    tensors.emplace_back(alloc_rdma_registerable_zero_tensor(
        scale_shape, torch::kFloat32, device_));
    scale_storage_bytes += tensors.back().storage().nbytes();
  }

  const int64_t non_scale_slot_bytes =
      capacity.slot_size() + capacity.index_slot_size() - sizeof(float);
  const size_t non_scale_storage_bytes =
      static_cast<size_t>(kNumLayers) *
      static_cast<size_t>(capacity.n_blocks()) *
      static_cast<size_t>(kBlockSize) *
      static_cast<size_t>(non_scale_slot_bytes);
  EXPECT_LE(scale_storage_bytes + non_scale_storage_bytes,
            static_cast<size_t>(kCacheBudgetBytes));

  const int64_t next_block_count = capacity.n_blocks() + 1;
  const size_t next_scale_logical_bytes =
      static_cast<size_t>(next_block_count) * static_cast<size_t>(kBlockSize) *
      sizeof(float);
  const RdmaMemoryPlan next_scale_plan = make_rdma_memory_plan(
      next_scale_logical_bytes, static_cast<size_t>(next_block_count));
  const size_t next_total_storage_bytes =
      static_cast<size_t>(kNumLayers) * next_scale_plan.registered_bytes +
      static_cast<size_t>(kNumLayers) * static_cast<size_t>(next_block_count) *
          static_cast<size_t>(kBlockSize) *
          static_cast<size_t>(non_scale_slot_bytes);
  EXPECT_GT(next_total_storage_bytes, static_cast<size_t>(kCacheBudgetBytes));
}

TEST_F(MluTensorAllocTest,
       RdmaRegisterableAllocationZerosTheEntireBackingStorage) {
  const torch::Tensor tensor = alloc_rdma_registerable_zero_tensor(
      kScaleShape, torch::kFloat32, device_);
  std::vector<uint8_t> host_bytes(kExpectedRegisteredBytes, 1);

  const cnrtRet_t ret = cnrtMemcpy(host_bytes.data(),
                                   tensor.data_ptr(),
                                   host_bytes.size(),
                                   cnrtMemcpyDevToHost);

  ASSERT_EQ(ret, cnrtSuccess);
  EXPECT_TRUE(std::all_of(host_bytes.begin(),
                          host_bytes.end(),
                          [](uint8_t value) { return value == 0; }));
}

TEST_F(MluTensorAllocTest, ResolverRejectsInsufficientBackingStorage) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  EXPECT_DEATH(
      {
        torch::Device device(Platform::type_torch(), /*device_index=*/0);
        Device xllm_device(device);
        xllm_device.set_device();
        const torch::Tensor tensor =
            alloc_zero_tensor(kScaleShape, torch::kFloat32, device);
        (void)get_rdma_registerable_nbytes(tensor);
      },
      "required_bytes=.*available_bytes");
}

TEST_F(MluTensorAllocTest, ResolverRejectsNonzeroStorageOffset) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  EXPECT_DEATH(
      {
        torch::Device device(Platform::type_torch(), /*device_index=*/0);
        Device xllm_device(device);
        xllm_device.set_device();
        const torch::Tensor tensor = alloc_rdma_registerable_zero_tensor(
            {3, 96, 1}, torch::kFloat32, device);
        const torch::Tensor offset_tensor = tensor.narrow(
            /*dim=*/0, /*start=*/1, /*length=*/2);
        (void)get_rdma_registerable_nbytes(offset_tensor);
      },
      "storage offset must be zero");
}

}  // namespace
}  // namespace xllm::mlu
