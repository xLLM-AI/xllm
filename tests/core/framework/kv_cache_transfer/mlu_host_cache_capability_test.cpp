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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm {
namespace {

class MLUHostCacheCapabilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for host cache capability tests.";
    }
    Device device(/*device_index=*/0);
    device.set_device();
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }

  torch::Tensor mlu_tensor(torch::ScalarType dtype = torch::kBFloat16) const {
    return torch::zeros(
        {2, 1, 1, 8},
        torch::TensorOptions().dtype(dtype).device(
            torch::Device(Platform::type_torch(), /*device_index=*/0)));
  }

  HierarchyKVCacheTransfer::Options transfer_options(
      bool enable_graph = false) const {
    HierarchyKVCacheTransfer::Options options;
    options.tp_rank(0)
        .tp_size(1)
        .layers(1)
        .host_blocks_factor(2.0)
        .layers_wise_copy_batchs(1)
        .enable_graph(enable_graph);
    return options;
  }

  KVCacheCreateOptions create_options() const {
    KVCacheCreateOptions options;
    options.device(torch::Device(Platform::type_torch(), /*device_index=*/0))
        .dtype(torch::kBFloat16)
        .num_layers(1)
        .host_blocks_factor(2.0);
    return options;
  }

  void construct_transfer(const HierarchyKVCacheTransfer::Options& options,
                          std::vector<KVCache>* caches,
                          const KVCacheCreateOptions& create_options) const {
    const torch::Device device(Platform::type_torch(), /*device_index=*/0);
    const KVCacheShape shape;
    HierarchyKVCacheTransfer transfer(
        options, device, caches, shape, create_options);
  }
};

TEST_F(MLUHostCacheCapabilityTest, RejectsInt8KVCache) {
  EXPECT_DEATH(
      {
        KVCacheCreateOptions options = create_options();
        options.enable_kv_cache_quant(true);
        std::vector<KVCache> caches;
        caches.emplace_back(
            KVCacheTensors{mlu_tensor(torch::kInt8), mlu_tensor(torch::kInt8)});
        construct_transfer(transfer_options(), &caches, options);
      },
      "kv_cache_dtype=auto");
}

TEST_F(MLUHostCacheCapabilityTest, RejectsLightingIndexerCache) {
  EXPECT_DEATH(
      {
        IndexedKVCacheTensors tensors;
        tensors.kv_cache_tensors =
            (KVCacheTensors{mlu_tensor(), torch::Tensor()});
        tensors.index_cache = mlu_tensor();
        std::vector<KVCache> caches;
        caches.emplace_back(tensors);
        construct_transfer(transfer_options(), &caches, create_options());
      },
      "KEY.*VALUE");
}

TEST_F(MLUHostCacheCapabilityTest, RejectsLinearAttentionCache) {
  EXPECT_DEATH(
      {
        std::vector<KVCache> caches;
        caches.emplace_back(
            LinearAttentionKVCacheTensors{mlu_tensor(), mlu_tensor()});
        construct_transfer(transfer_options(), &caches, create_options());
      },
      "BlockType::KV");
}

TEST_F(MLUHostCacheCapabilityTest, RejectsDeepSeekV4Cache) {
  EXPECT_DEATH(
      {
        DeepSeekV4KVCacheTensors tensors;
        tensors.swa_cache = mlu_tensor();
        tensors.compressed_block_type = BlockType::SWA;
        std::vector<KVCache> caches;
        caches.emplace_back(tensors);
        construct_transfer(transfer_options(), &caches, create_options());
      },
      "BlockType::KV");
}

TEST_F(MLUHostCacheCapabilityTest, RejectsMLUGraph) {
  EXPECT_DEATH(
      {
        std::vector<KVCache> caches;
        caches.emplace_back(KVCacheTensors{mlu_tensor(), mlu_tensor()});
        construct_transfer(
            transfer_options(/*enable_graph=*/true), &caches, create_options());
      },
      "disable.*graph.*host cache");
}

}  // namespace
}  // namespace xllm
