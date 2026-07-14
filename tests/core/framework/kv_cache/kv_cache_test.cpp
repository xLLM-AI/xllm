/* Copyright 2025-2026 The xLLM Authors.

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

#include <string>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "kv_cache_estimation.h"
#include "kv_cache_shape.h"
#include "worker.pb.h"

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

class IndexerCacheDtypeConfigGuard final {
 public:
  explicit IndexerCacheDtypeConfigGuard(const std::string& indexer_cache_dtype)
      : old_indexer_cache_dtype_(
            KVCacheConfig::get_instance().indexer_cache_dtype()) {
    KVCacheConfig::get_instance().indexer_cache_dtype(indexer_cache_dtype);
  }

  ~IndexerCacheDtypeConfigGuard() {
    KVCacheConfig::get_instance().indexer_cache_dtype(old_indexer_cache_dtype_);
  }

 private:
  std::string old_indexer_cache_dtype_;
};

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
  const std::optional<torch::Tensor> indexer_cache_scale =
      caches[1].get_indexer_cache_scale();
  if (indexer_cache_scale.has_value()) {
    EXPECT_EQ(shape_vec(indexer_cache_scale.value()),
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

TEST(KVCacheTest, DeepSeekV4KVCacheExposesIndexerScaleThroughSharedContract) {
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({2, 4, 1, 64});
  tensors.index_cache =
      torch::zeros({2, 4, 1, 32}, torch::TensorOptions().dtype(torch::kInt8));
  tensors.indexer_cache_scale =
      torch::zeros({2, 4, 1}, torch::TensorOptions().dtype(torch::kFloat16));
  tensors.swa_cache = torch::zeros({2, 4, 1, 64});
  KVCache cache(tensors);

  const std::optional<torch::Tensor> cached_scale =
      cache.get_indexer_cache_scale();
  ASSERT_TRUE(cached_scale.has_value());
  EXPECT_EQ(shape_vec(cached_scale.value()), (std::vector<int64_t>{2, 4, 1}));

  const std::vector<KVCacheTensor> cache_tensors = cache.get_cache_tensors();
  ASSERT_EQ(cache_tensors.size(), 3U);
  EXPECT_EQ(cache_tensors[0].role, KVCacheTensorRole::KEY);
  EXPECT_EQ(cache_tensors[1].role, KVCacheTensorRole::INDEX);
  EXPECT_EQ(cache_tensors[2].role, KVCacheTensorRole::INDEX_SCALE);
  EXPECT_EQ(shape_vec(cache_tensors[2].tensor),
            (std::vector<int64_t>{2, 4, 1}));
}

TEST(KVCacheTest, CacheVariantsNormalizeInvalidIndexerScaleToNullopt) {
  DeepSeekV4KVCacheTensors deepseek_v4_tensors;
  deepseek_v4_tensors.swa_cache = torch::zeros({1, 1, 1, 1});
  KVCache deepseek_v4_cache(deepseek_v4_tensors);
  EXPECT_FALSE(deepseek_v4_cache.get_indexer_cache_scale().has_value());

  const torch::Tensor key_cache = torch::zeros({1, 1, 1, 1});
  const torch::Tensor value_cache = torch::zeros({1, 1, 1, 1});
  const torch::Tensor index_cache = torch::zeros({1, 1, 1, 1});
  const torch::Tensor empty_scale = torch::empty({0});
  KVCache indexed_cache(IndexedKVCacheTensors{
      KVCacheTensors{key_cache, value_cache}, index_cache, empty_scale});
  EXPECT_FALSE(indexed_cache.get_indexer_cache_scale().has_value());
}

TEST(KVCacheTest, KVCacheShapeRoundTripsIndexCacheScaleShape) {
  proto::KVCacheShape proto_shape;
  proto_shape.add_key_cache_shape(8);
  proto_shape.add_key_cache_shape(16);
  proto_shape.add_key_cache_shape(1);
  proto_shape.add_key_cache_shape(64);
  proto_shape.add_index_cache_shape(8);
  proto_shape.add_index_cache_shape(16);
  proto_shape.add_index_cache_shape(1);
  proto_shape.add_index_cache_shape(32);
  proto_shape.add_index_cache_scale_shape(8);
  proto_shape.add_index_cache_scale_shape(1);
  proto_shape.add_index_cache_scale_shape(16);

  KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  EXPECT_TRUE(shape.has_index_cache_scale_shape());
  EXPECT_EQ(shape.index_cache_scale_shape(), (std::vector<int64_t>{8, 1, 16}));

  proto::KVCacheShape roundtrip_shape;
  shape.to_proto(&roundtrip_shape);
  EXPECT_EQ(roundtrip_shape.index_cache_scale_shape_size(), 3);
  EXPECT_EQ(roundtrip_shape.index_cache_scale_shape(0), 8);
  EXPECT_EQ(roundtrip_shape.index_cache_scale_shape(1), 1);
  EXPECT_EQ(roundtrip_shape.index_cache_scale_shape(2), 16);
}

TEST(KVCacheTest, IndexerInt8WithoutIndexerDoesNotCreateScaleShape) {
  KVCacheCapacity capacity;
  capacity.n_blocks(8).block_size(16).enable_indexer_cache_quant(true);

  ModelArgs model_args;
  model_args.n_heads(8).n_kv_heads(2).head_dim(64).index_n_heads(0);

  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  EXPECT_FALSE(shape.has_index_cache_shape());
  EXPECT_FALSE(shape.has_index_cache_scale_shape());
}

TEST(KVCacheTest, IndexedKVCacheExposesIndexerCacheScaleTensor) {
  torch::Tensor key_cache = torch::zeros({2, 4, 1, 64});
  torch::Tensor value_cache = torch::zeros({2, 4, 1, 64});
  torch::Tensor index_cache =
      torch::zeros({2, 1, 4, 32}, torch::TensorOptions().dtype(torch::kInt8));
  torch::Tensor index_cache_scale =
      torch::zeros({2, 1, 4}, torch::TensorOptions().dtype(torch::kFloat32));

  KVCache cache(IndexedKVCacheTensors{
      KVCacheTensors{key_cache, value_cache}, index_cache, index_cache_scale});

  const std::optional<torch::Tensor> cached_scale =
      cache.get_indexer_cache_scale();
  ASSERT_TRUE(cached_scale.has_value());
  EXPECT_EQ(shape_vec(cached_scale.value()), (std::vector<int64_t>{2, 1, 4}));

  std::vector<std::vector<int64_t>> shapes = cache.get_shapes();
  ASSERT_EQ(shapes.size(), 4u);
  EXPECT_EQ(shapes[3], (std::vector<int64_t>{2, 1, 4}));

  std::vector<KVCacheTensor> tensors = cache.get_cache_tensors();
  ASSERT_EQ(tensors.size(), 4u);
  EXPECT_EQ(tensors[0].role, KVCacheTensorRole::KEY);
  EXPECT_EQ(tensors[1].role, KVCacheTensorRole::VALUE);
  EXPECT_EQ(tensors[2].role, KVCacheTensorRole::INDEX);
  EXPECT_EQ(tensors[3].role, KVCacheTensorRole::INDEX_SCALE);
  EXPECT_EQ(shape_vec(tensors[3].tensor), (std::vector<int64_t>{2, 1, 4}));
}

TEST(KVCacheTest, IndexedKVCacheExposesQuantizedKvScaleTensors) {
  torch::Tensor key_cache =
      torch::zeros({2, 4, 1, 64}, torch::TensorOptions().dtype(torch::kInt8));
  torch::Tensor value_cache =
      torch::zeros({2, 4, 1, 64}, torch::TensorOptions().dtype(torch::kInt8));
  torch::Tensor index_cache = torch::zeros({2, 1, 4, 32});
  torch::Tensor key_cache_scale =
      torch::zeros({2, 4, 1}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor value_cache_scale =
      torch::zeros({2, 4, 1}, torch::TensorOptions().dtype(torch::kFloat32));

  KVCache cache(IndexedKVCacheTensors{KVCacheTensors{key_cache, value_cache},
                                      index_cache,
                                      std::nullopt,
                                      key_cache_scale,
                                      value_cache_scale});

  std::optional<torch::Tensor> cached_key_scale = cache.get_k_cache_scale();
  ASSERT_TRUE(cached_key_scale.has_value());
  EXPECT_EQ(shape_vec(cached_key_scale.value()),
            (std::vector<int64_t>{2, 4, 1}));

  std::optional<torch::Tensor> cached_value_scale = cache.get_v_cache_scale();
  ASSERT_TRUE(cached_value_scale.has_value());
  EXPECT_EQ(shape_vec(cached_value_scale.value()),
            (std::vector<int64_t>{2, 4, 1}));
}

#if defined(USE_MLU)
TEST(KVCacheTest,
     MluQuantizedIndexedKVCacheAllocatesInt8KvAndScalesOnCpuDevice) {
  constexpr int64_t kBlockCount = 8;
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeadDim = 64;
  constexpr int64_t kIndexHeadDim = 32;
  constexpr int64_t kKvHeadCount = 2;

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);

  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .n_heads(8)
      .n_kv_heads(kKvHeadCount)
      .head_dim(kHeadDim)
      .index_n_heads(1)
      .index_head_dim(kIndexHeadDim);

  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true)
      .enable_kv_cache_quant(true);

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  ASSERT_EQ(caches.size(), 1u);
  EXPECT_EQ(caches[0].get_k_cache().scalar_type(), torch::kChar);
  EXPECT_EQ(caches[0].get_v_cache().scalar_type(), torch::kChar);

  std::optional<torch::Tensor> key_cache_scale = caches[0].get_k_cache_scale();
  ASSERT_TRUE(key_cache_scale.has_value());
  EXPECT_EQ(shape_vec(key_cache_scale.value()),
            (std::vector<int64_t>{kBlockCount, kKvHeadCount, kBlockSize}));

  std::optional<torch::Tensor> value_cache_scale =
      caches[0].get_v_cache_scale();
  ASSERT_TRUE(value_cache_scale.has_value());
  EXPECT_EQ(shape_vec(value_cache_scale.value()),
            (std::vector<int64_t>{kBlockCount, kKvHeadCount, kBlockSize}));
}

TEST(KVCacheTest, MluIndexerInt8ScaleShapeMatchesQuantPagedCacheContract) {
  IndexerCacheDtypeConfigGuard config_guard(
      /*indexer_cache_dtype=*/"int8");
  constexpr int64_t kBlockCount = 8;
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeadDim = 64;
  constexpr int64_t kIndexHeadDim = 32;
  constexpr int64_t kKvHeadCount = 2;

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .enable_indexer_cache_quant(true);

  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .n_heads(8)
      .n_kv_heads(kKvHeadCount)
      .head_dim(kHeadDim)
      .index_n_heads(1)
      .index_head_dim(kIndexHeadDim);

  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  EXPECT_EQ(shape.index_cache_shape(),
            (std::vector<int64_t>{kBlockCount, 1, kBlockSize, kIndexHeadDim}));
  ASSERT_TRUE(shape.has_index_cache_scale_shape());
  EXPECT_EQ(shape.index_cache_scale_shape(),
            (std::vector<int64_t>{kBlockCount, 1, kBlockSize}));
}

TEST(KVCacheTest, IndexerInt8ShapeUsesCapacityDecisionWhenGlobalIsAuto) {
  IndexerCacheDtypeConfigGuard config_guard(
      /*indexer_cache_dtype=*/"auto");
  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .n_layers(1)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(64)
      .index_n_heads(1)
      .index_head_dim(32);

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 16;
  options.n_local_kv_heads = 2;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  EXPECT_TRUE(capacity.enable_indexer_cache_quant());
  EXPECT_TRUE(shape.has_index_cache_scale_shape());
}

TEST(KVCacheTest, IndexerAutoShapeUsesCapacityDecisionWhenGlobalIsInt8) {
  IndexerCacheDtypeConfigGuard config_guard(
      /*indexer_cache_dtype=*/"int8");
  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .n_layers(1)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(64)
      .index_n_heads(1)
      .index_head_dim(32);

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.indexer_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 16;
  options.n_local_kv_heads = 2;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  EXPECT_FALSE(capacity.enable_indexer_cache_quant());
  EXPECT_FALSE(shape.has_index_cache_scale_shape());
}

TEST(KVCacheTest, MluIndexerAutoUsesDefaultCacheShapeWithoutScale) {
  IndexerCacheDtypeConfigGuard config_guard(
      /*indexer_cache_dtype=*/"auto");
  constexpr int64_t kBlockCount = 8;
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeadDim = 64;
  constexpr int64_t kIndexHeadDim = 32;
  constexpr int64_t kKvHeadCount = 2;

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);

  ModelArgs model_args;
  model_args.model_type("deepseek_v32")
      .n_heads(8)
      .n_kv_heads(kKvHeadCount)
      .head_dim(kHeadDim)
      .index_n_heads(1)
      .index_head_dim(kIndexHeadDim);

  KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  EXPECT_EQ(shape.index_cache_shape(),
            (std::vector<int64_t>{kBlockCount, 1, kBlockSize, kIndexHeadDim}));
  EXPECT_FALSE(shape.has_index_cache_scale_shape());

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true);

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  ASSERT_EQ(caches.size(), 1U);
  EXPECT_EQ(caches[0].get_index_cache().scalar_type(), torch::kBFloat16);
  EXPECT_FALSE(caches[0].get_indexer_cache_scale().has_value());
}
#endif

}  // namespace xllm
