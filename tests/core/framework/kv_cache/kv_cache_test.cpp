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

#include "kv_cache.h"

#include <gtest/gtest.h>

#include <vector>

#include "kv_cache_shape.h"

namespace xllm {

namespace {

std::vector<int64_t> shape_vec(const torch::Tensor& tensor) {
  return tensor.sizes().vec();
}

std::vector<int64_t> dsv4_block_shape(int64_t block_count,
                                      int64_t block_size,
                                      int64_t n_heads,
                                      int64_t head_dim) {
#if defined(USE_MLU)
  return {block_count, n_heads, block_size, head_dim};
#else
  return {block_count, block_size, n_heads, head_dim};
#endif
}

}  // namespace

TEST(KVCacheTest, DeepSeekV4FourDimCachesUseDeviceLayout) {
  constexpr int64_t kSwaCount = 10;
  constexpr int64_t kC4Count = 32;
  constexpr int64_t kC128Count = 1;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 16;
  constexpr int64_t kIndexHeadDim = 8;

  KVCacheCapacity capacity;
  capacity.block_size(kBlockSize)
      .swa_count(kSwaCount)
      .c4_count(kC4Count)
      .c128_count(kC128Count);

  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(3)
      .model_type("deepseek_v4")
      .block_size(kBlockSize)
      .head_dim(kHeadDim)
      .index_head_dim(kIndexHeadDim)
      .window_size(/*window_size=*/512)
      .compress_ratios({1, 4, 128});

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  ASSERT_EQ(caches.size(), 3u);

  EXPECT_EQ(shape_vec(caches[0].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  EXPECT_FALSE(caches[0].get_compress_kv_state().defined());

  EXPECT_EQ(shape_vec(caches[1].get_k_cache()),
            dsv4_block_shape(kC4Count, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[1].get_index_cache()),
            dsv4_block_shape(kC4Count, kBlockSize, 1, kIndexHeadDim));
  EXPECT_EQ(shape_vec(caches[1].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  if (caches[1].get_indexer_cache_scale().defined()) {
    EXPECT_EQ(shape_vec(caches[1].get_indexer_cache_scale()),
              (std::vector<int64_t>{kC4Count, kBlockSize, 1}));
  }
  EXPECT_EQ(shape_vec(caches[1].get_compress_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_index_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kIndexHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_index_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kIndexHeadDim}));

  EXPECT_EQ(shape_vec(caches[2].get_k_cache()),
            dsv4_block_shape(kC128Count, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_compress_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
  EXPECT_EQ(shape_vec(caches[2].get_compress_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
}

}  // namespace xllm
