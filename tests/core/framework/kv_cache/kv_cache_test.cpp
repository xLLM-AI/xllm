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

#include <algorithm>
#include <string>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "framework/block/block.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/deepseek_v4_kv_cache_impl.h"
#include "framework/kv_cache/kv_cache_tensor_allocator.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "kv_cache_estimation.h"
#include "kv_cache_shape.h"
#include "platform/device.h"
#if defined(USE_MLU)
#include "platform/mlu/mlu_tensor_alloc.h"
#endif
#include "platform/platform.h"
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

#if defined(USE_MLU)
class RecordingKVCacheTensorAllocator final : public KVCacheTensorAllocator {
 public:
  struct Request {
    KVCacheTensorRole role;
    std::vector<int64_t> shape;
    torch::ScalarType dtype;
    torch::Device device;
  };

  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    requests.emplace_back(Request{role, shape, dtype, device});
    return torch::zeros(shape, torch::dtype(dtype).device(device));
  }

  std::vector<Request> requests;
};
#endif

}  // namespace

TEST(Dsv4StateCacheTest, SplitStateReturnsInputsAndSwapsBoth) {
  torch::Tensor kv = torch::tensor({{{1.0F}}, {{2.0F}}, {{3.0F}}});
  torch::Tensor score = torch::tensor({{{4.0F}}, {{5.0F}}, {{6.0F}}});
  Dsv4StateCache state = Dsv4StateCache::from_split(kv, score);

  EXPECT_EQ(state.kv().data_ptr(), kv.data_ptr());
  EXPECT_EQ(state.score().data_ptr(), score.data_ptr());
  EXPECT_FALSE(state.packed().defined());

  torch::Tensor src = torch::tensor({0}, torch::kLong);
  torch::Tensor dst = torch::tensor({2}, torch::kLong);
  state.swap_blocks(src, dst);

  EXPECT_TRUE(
      torch::equal(state.kv(), torch::tensor({{{1.0F}}, {{2.0F}}, {{1.0F}}})));
  EXPECT_TRUE(torch::equal(state.score(),
                           torch::tensor({{{4.0F}}, {{5.0F}}, {{4.0F}}})));
}

TEST(Dsv4StateCacheTest, PackedStateReturnsWritableViewsAndSwapsOwner) {
  torch::Tensor packed = torch::arange(24, torch::kFloat32).reshape({3, 2, 4});
  Dsv4StateCache state =
      Dsv4StateCache::from_packed(packed, torch::Tensor(), torch::Tensor());

  EXPECT_EQ(shape_vec(state.kv()), (std::vector<int64_t>{3, 2, 2}));
  EXPECT_EQ(shape_vec(state.score()), (std::vector<int64_t>{3, 2, 2}));
  EXPECT_EQ(state.kv().storage().unsafeGetStorageImpl(),
            state.packed().storage().unsafeGetStorageImpl());
  EXPECT_EQ(state.score().storage().unsafeGetStorageImpl(),
            state.packed().storage().unsafeGetStorageImpl());
  state.kv().fill_(42.0F);
  EXPECT_TRUE(torch::equal(
      state.packed().narrow(/*dim=*/2, /*start=*/0, /*length=*/2), state.kv()));
  state.packed().narrow(/*dim=*/2, /*start=*/2, /*length=*/2).fill_(17.0F);
  EXPECT_TRUE(torch::equal(state.score(), torch::full({3, 2, 2}, 17.0F)));

  torch::Tensor src = torch::tensor({0}, torch::kLong);
  torch::Tensor dst = torch::tensor({2}, torch::kLong);
  torch::Tensor expected = state.packed().clone();
  expected.index_copy_(0, dst, torch::index_select(expected, 0, src));
  state.swap_blocks(src, dst);

  EXPECT_TRUE(torch::equal(state.packed(), expected));
  EXPECT_TRUE(torch::equal(
      state.kv(), expected.narrow(/*dim=*/2, /*start=*/0, /*length=*/2)));
}

TEST(Dsv4StateCacheTest, MissingPackedFallsBackWithoutSwappingSplit) {
  torch::Tensor kv = torch::tensor({{{1.0F}}, {{2.0F}}});
  torch::Tensor score = torch::tensor({{{3.0F}}, {{4.0F}}});
  Dsv4StateCache state =
      Dsv4StateCache::from_packed(torch::Tensor(), kv, score);

  EXPECT_EQ(state.kv().data_ptr(), kv.data_ptr());
  EXPECT_EQ(state.score().data_ptr(), score.data_ptr());
  EXPECT_FALSE(state.packed().defined());

  torch::Tensor src = torch::tensor({0}, torch::kLong);
  torch::Tensor dst = torch::tensor({1}, torch::kLong);
  state.swap_blocks(src, dst);

  EXPECT_TRUE(torch::equal(state.kv(), kv));
  EXPECT_TRUE(torch::equal(state.score(), score));
}

// Host prefix-cache allocation registers page-aligned host memory with the NPU
// via aclrtHostRegister, which requires a live device context. Set one up once.
class HostKVCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
  }
};

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
#if defined(USE_MLU)
  // MLU merges the split states into one owning tensor of shape
  // [swa_count, block_size, 2 * coff_dim]; the split getters are narrow views
  // into it (sizes unchanged).
  EXPECT_EQ(shape_vec(caches[1].get_compress_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 4 * kHeadDim}));
  EXPECT_EQ(shape_vec(caches[1].get_compress_index_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 4 * kIndexHeadDim}));
  EXPECT_TRUE(caches[1].get_compress_state().is_contiguous());
  EXPECT_TRUE(caches[1].get_compress_index_state().is_contiguous());
#endif

  EXPECT_EQ(shape_vec(caches[2].get_k_cache()),
            dsv4_block_shape(kC128Count, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_swa_cache()),
            dsv4_block_shape(kSwaCount, kBlockSize, 1, kHeadDim));
  EXPECT_EQ(shape_vec(caches[2].get_compress_kv_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
  EXPECT_EQ(shape_vec(caches[2].get_compress_score_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, kHeadDim}));
#if defined(USE_MLU)
  EXPECT_EQ(shape_vec(caches[2].get_compress_state()),
            (std::vector<int64_t>{kSwaCount, kBlockSize, 2 * kHeadDim}));
  EXPECT_FALSE(caches[2].get_compress_index_state().defined());
#endif

  auto expect_tensor_group = [](const KVCache& cache,
                                KVCacheTensorRole role,
                                BlockType expected_block_type) {
    const auto tensors = cache.get_cache_tensors();
    const auto it = std::find_if(
        tensors.begin(), tensors.end(), [role](const KVCacheTensor& tensor) {
          return tensor.role == role;
        });
    ASSERT_NE(it, tensors.end()) << "missing role=" << role.to_string();
    EXPECT_EQ(it->group_id, cache_group_id(expected_block_type))
        << "role=" << role.to_string();
    EXPECT_FALSE(it->sequence_scoped);
  };
  expect_tensor_group(caches[0], KVCacheTensorRole::WINDOW, BlockType::SWA);
  expect_tensor_group(caches[1], KVCacheTensorRole::KEY, BlockType::C4);
  expect_tensor_group(caches[1], KVCacheTensorRole::INDEX, BlockType::C4);
  expect_tensor_group(caches[1], KVCacheTensorRole::KV_STATE, BlockType::SWA);
  expect_tensor_group(caches[2], KVCacheTensorRole::KEY, BlockType::C128);
  expect_tensor_group(
      caches[2], KVCacheTensorRole::SCORE_STATE, BlockType::SWA);
}

TEST(KVCacheTest, GroupedCacheCapabilitySurvivesProtoRoundTrip) {
  KVCacheCapacity capacity;
  capacity.block_size(128).swa_count(10).c4_count(32).c128_count(1);

  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  const KVCacheShape original(capacity, model_args, /*world_size=*/1);
  ASSERT_TRUE(original.has_grouped_cache_layout());

  proto::KVCacheShape serialized;
  original.to_proto(&serialized);
  const KVCacheShape restored = KVCacheShape::from_proto(serialized);

  EXPECT_TRUE(restored.has_grouped_cache_layout());
  EXPECT_EQ(restored.key_cache_shape(), original.key_cache_shape());
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
  ASSERT_EQ(cache_tensors.size(), 4U);
  EXPECT_EQ(cache_tensors[0].role, KVCacheTensorRole::WINDOW);
  EXPECT_EQ(cache_tensors[1].role, KVCacheTensorRole::KEY);
  EXPECT_EQ(cache_tensors[2].role, KVCacheTensorRole::INDEX);
  EXPECT_EQ(cache_tensors[3].role, KVCacheTensorRole::INDEX_SCALE);
  EXPECT_EQ(shape_vec(cache_tensors[3].tensor),
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

TEST(KVCacheTest, MluSharedDsaLayersAllocateOnlyMlaCache) {
  constexpr int64_t kBlockCount = 8;
  constexpr int64_t kBlockSize = 16;

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);

  ModelArgs model_args;
  model_args.model_type("glm_moe_dsa")
      .enable_mla(true)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(64)
      .kv_lora_rank(64)
      .qk_rope_head_dim(16)
      .index_n_heads(1)
      .index_head_dim(32);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(4)
      .model_type("glm_moe_dsa")
      .enable_lighting_indexer(true)
      .indexer_cache_enabled_layers(
          std::vector<bool>{true, false, false, true});

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  ASSERT_EQ(caches.size(), 4U);
  EXPECT_TRUE(caches[0].get_index_cache().defined());
  EXPECT_FALSE(caches[1].get_index_cache().defined());
  EXPECT_FALSE(caches[2].get_index_cache().defined());
  EXPECT_TRUE(caches[3].get_index_cache().defined());
  for (const KVCache& cache : caches) {
    EXPECT_TRUE(cache.get_k_cache().defined());
    EXPECT_FALSE(cache.get_v_cache().defined());
    EXPECT_FALSE(cache.empty());
  }

  caches[1].get_k_cache()[0].fill_(1);
  torch::Tensor source_block = torch::tensor({0}, torch::kLong);
  torch::Tensor destination_block = torch::tensor({1}, torch::kLong);
  caches[1].swap_blocks(source_block, destination_block);
  EXPECT_TRUE(
      torch::equal(caches[1].get_k_cache()[0], caches[1].get_k_cache()[1]));
}

TEST(KVCacheTest, MluMixedDsaLayersUseInjectedAllocatorForActualRoles) {
  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .enable_indexer_cache_quant(true);

  ModelArgs model_args;
  model_args.model_type("glm_moe_dsa")
      .enable_mla(true)
      .n_heads(8)
      .n_kv_heads(2)
      .head_dim(8)
      .kv_lora_rank(8)
      .qk_rope_head_dim(4)
      .index_n_heads(1)
      .index_head_dim(4);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/1);

  auto allocator = std::make_shared<RecordingKVCacheTensorAllocator>();
  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kBFloat16)
      .num_layers(4)
      .model_type("glm_moe_dsa")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true)
      .indexer_cache_enabled_layers({true, false, true, false})
      .tensor_allocator(allocator);

  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);

  const std::vector<KVCacheTensorRole> expected_roles = {
      KVCacheTensorRole::KEY,
      KVCacheTensorRole::INDEX,
      KVCacheTensorRole::INDEX_SCALE,
      KVCacheTensorRole::KEY,
      KVCacheTensorRole::KEY,
      KVCacheTensorRole::INDEX,
      KVCacheTensorRole::INDEX_SCALE,
      KVCacheTensorRole::KEY};
  ASSERT_EQ(allocator->requests.size(), expected_roles.size());
  for (size_t i = 0; i < expected_roles.size(); ++i) {
    EXPECT_EQ(allocator->requests[i].role, expected_roles[i]);
    EXPECT_EQ(allocator->requests[i].device, torch::Device(torch::kCPU));
  }
  EXPECT_EQ(allocator->requests[1].dtype, torch::kChar);
  EXPECT_EQ(allocator->requests[2].dtype, torch::kFloat32);
  EXPECT_EQ(allocator->requests[2].shape, shape.index_cache_scale_shape());

  ASSERT_EQ(caches.size(), 4U);
  EXPECT_EQ(caches[0].get_cache_tensors().size(), 3U);
  EXPECT_EQ(caches[1].get_cache_tensors().size(), 1U);
  EXPECT_EQ(caches[2].get_cache_tensors().size(), 3U);
  EXPECT_EQ(caches[3].get_cache_tensors().size(), 1U);
  EXPECT_FALSE(caches[1].get_index_cache().defined());
  EXPECT_FALSE(caches[3].get_indexer_cache_scale().has_value());
}

TEST(KVCacheTensorAllocatorTest,
     MluMooncakePadsOnlyIndexScaleAndDoesNotOwnTensorLifetime) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for allocator storage checks.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  std::shared_ptr<KVCacheTensorAllocator> allocator =
      mlu_mooncake_tensor_allocator();
  std::weak_ptr<KVCacheTensorAllocator> allocator_lifetime = allocator;

  torch::Tensor key = allocator->allocate(
      KVCacheTensorRole::KEY, {2, 4}, torch::kBFloat16, torch_device);
  torch::Tensor index_scale =
      allocator->allocate(KVCacheTensorRole::INDEX_SCALE,
                          {2, 96, 1},
                          torch::kFloat32,
                          torch_device);

  EXPECT_EQ(key.storage().nbytes(), key.nbytes());
  EXPECT_EQ(index_scale.sizes().vec(), (std::vector<int64_t>{2, 96, 1}));
  EXPECT_EQ(index_scale.scalar_type(), torch::kFloat32);
  EXPECT_EQ(index_scale.nbytes(), 2 * 96 * sizeof(float));
  EXPECT_GE(index_scale.storage().nbytes(), index_scale.nbytes());
  EXPECT_EQ(mlu::get_rdma_registerable_nbytes(index_scale),
            index_scale.storage().nbytes());

  allocator.reset();
  EXPECT_TRUE(allocator_lifetime.expired());
  key.fill_(1);
  index_scale.fill_(2);
  device.synchronize_default_stream();
  EXPECT_TRUE(key.defined());
  EXPECT_TRUE(index_scale.defined());
}
#endif

TEST_F(HostKVCacheTest, HostKVCacheNormalLayoutAddsLayerDim) {
  constexpr int64_t kNumBlocks = 16;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 64;
  constexpr int64_t kNumHeads = 4;
  constexpr int64_t kLayerCount = 5;
  constexpr double kHostFactor = 2.0;

  KVCacheCapacity capacity;
  capacity.n_blocks(kNumBlocks).block_size(kBlockSize);

  ModelArgs model_args;
  model_args.model_type("qwen");
  model_args.n_kv_heads(kNumHeads);
  model_args.head_dim(kHeadDim);
  KVCacheShape shape(capacity, model_args, /*world_size=*/1);
  ASSERT_TRUE(shape.has_key_cache_shape());

  KVCacheCreateOptions options;
  options.device(torch::Device(torch::kCPU))
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("qwen")
      .host_blocks_factor(kHostFactor);

  KVCache host_cache(shape, options, BlockType::KV, kLayerCount);

  const BlockTypeTensorMap tensors =
      host_cache.get_block_type_tensors(BlockType::KV);
  ASSERT_TRUE(tensors.count(KVCacheTensorRole::KEY) > 0);

  const std::vector<int64_t> base_key_shape = shape.key_cache_shape();
  const torch::Tensor& host_key = tensors.at(KVCacheTensorRole::KEY);
  EXPECT_TRUE(host_key.is_contiguous());
  EXPECT_EQ(host_key.device().type(), torch::kCPU);
  // host shape == [scaled_blocks, layer_count, ...per_block_dims]
  ASSERT_EQ(host_key.dim(), static_cast<int64_t>(base_key_shape.size()) + 1);
  EXPECT_EQ(host_key.size(0),
            scale_host_block_count(base_key_shape[0], kHostFactor));
  EXPECT_EQ(host_key.size(1), kLayerCount);
  for (size_t i = 1; i < base_key_shape.size(); ++i) {
    EXPECT_EQ(host_key.size(static_cast<int64_t>(i) + 1), base_key_shape[i]);
  }
}

TEST_F(HostKVCacheTest, HostKVCacheDeepSeekV4PerBlockType) {
  constexpr int64_t kSwaCount = 10;
  constexpr int64_t kC4Count = 32;
  constexpr int64_t kC128Count = 4;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kHeadDim = 16;
  constexpr int64_t kIndexHeadDim = 8;
  constexpr double kHostFactor = 3.0;

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
      .compress_ratios({1, 4, 128})
      .host_blocks_factor(kHostFactor);

  // SWA host cache: 1 layer in this 3-layer config (compress_ratio == 1).
  KVCache swa_host(shape, options, BlockType::SWA, /*layer_count=*/1);
  const BlockTypeTensorMap swa_tensors =
      swa_host.get_block_type_tensors(BlockType::SWA);
  ASSERT_TRUE(swa_tensors.count(KVCacheTensorRole::SWA) > 0);
  const torch::Tensor& swa = swa_tensors.at(KVCacheTensorRole::SWA);
  EXPECT_TRUE(swa.is_contiguous());
  EXPECT_EQ(swa.size(0), scale_host_block_count(kSwaCount, kHostFactor));
  EXPECT_EQ(swa.size(1), 1);

  // C4 host cache: key + index, index uses the DSV4 index dtype.
  KVCache c4_host(shape, options, BlockType::C4, /*layer_count=*/1);
  const BlockTypeTensorMap c4_tensors =
      c4_host.get_block_type_tensors(BlockType::C4);
  ASSERT_TRUE(c4_tensors.count(KVCacheTensorRole::KEY) > 0);
  ASSERT_TRUE(c4_tensors.count(KVCacheTensorRole::INDEX) > 0);
  EXPECT_EQ(c4_tensors.at(KVCacheTensorRole::KEY).size(0),
            scale_host_block_count(kC4Count, kHostFactor));
  EXPECT_EQ(c4_tensors.at(KVCacheTensorRole::INDEX).scalar_type(),
            get_dsv4_cache_policy(options.dtype()).index_dtype);

  // C128 host cache: key only (no index).
  KVCache c128_host(shape, options, BlockType::C128, /*layer_count=*/1);
  const BlockTypeTensorMap c128_tensors =
      c128_host.get_block_type_tensors(BlockType::C128);
  ASSERT_TRUE(c128_tensors.count(KVCacheTensorRole::KEY) > 0);
  EXPECT_TRUE(c128_tensors.count(KVCacheTensorRole::INDEX) == 0);
  EXPECT_EQ(c128_tensors.at(KVCacheTensorRole::KEY).size(0),
            scale_host_block_count(kC128Count, kHostFactor));
}

}  // namespace xllm
