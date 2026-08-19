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

#include <framework/core/MLUStream.h>
#include <framework/core/stream_guard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

using torch::indexing::Slice;

constexpr int64_t kGuard = 17;
constexpr float kGuardValue = -31.0f;

struct CacheWithBacking {
  torch::Tensor cache;
  torch::Tensor backing;
};

torch::Tensor make_pattern(int64_t batch,
                           int64_t beam,
                           int64_t heads,
                           int64_t steps,
                           int64_t dim,
                           int64_t layer,
                           int64_t kind) {
  auto values = torch::arange(batch * beam * heads * steps * dim,
                              torch::TensorOptions().dtype(torch::kFloat32));
  values = values.view({batch, beam, heads, steps, dim});
  return (values.remainder(29) + batch * 5 + beam * 3 + layer * 11 + kind * 7)
      .to(torch::kBFloat16);
}

CacheWithBacking make_cache(const torch::Tensor& cpu_values,
                            const torch::Device& device,
                            bool with_guards) {
  const int64_t guard = with_guards ? kGuard : 0;
  auto backing_cpu = torch::full(
      {cpu_values.numel() + 2 * guard}, kGuardValue, cpu_values.options());
  backing_cpu.narrow(0, guard, cpu_values.numel())
      .view(cpu_values.sizes())
      .copy_(cpu_values);
  auto backing = backing_cpu.to(device);
  auto cache =
      backing.narrow(0, guard, cpu_values.numel()).view(cpu_values.sizes());
  return {cache, backing};
}

torch::Tensor make_indices(const std::vector<int32_t>& values,
                           const torch::Device& device) {
  auto cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32));
  return cpu.view({static_cast<int64_t>(values.size()), 1}).to(device);
}

torch::Tensor reference(const torch::Tensor& original,
                        const std::vector<int32_t>& indices,
                        int64_t decode_step,
                        int64_t beam_width) {
  auto expected = original.clone();
  const int64_t batch = original.size(0);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t new_beam = 0; new_beam < beam_width; ++new_beam) {
      const int64_t row = b * beam_width + new_beam;
      const int64_t parent = indices[row] / beam_width;
      expected.index({b, new_beam, Slice(), Slice(0, decode_step + 1), Slice()})
          .copy_(original.index(
              {b, parent, Slice(), Slice(0, decode_step + 1), Slice()}));
    }
  }
  return expected;
}

void expect_metadata_unchanged(const torch::Tensor& before,
                               const torch::Tensor& after) {
  EXPECT_EQ(after.sizes(), before.sizes());
  EXPECT_EQ(after.strides(), before.strides());
  EXPECT_EQ(after.storage_offset(), before.storage_offset());
  EXPECT_EQ(after.data_ptr(), before.data_ptr());
  EXPECT_EQ(after.storage().unsafeGetStorageImpl(),
            before.storage().unsafeGetStorageImpl());
}

class CacheSelectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (torch_mlu::device_count() == 0) {
      GTEST_SKIP() << "MLU is not available";
    }
  }

  const torch::Device device_{"mlu:0"};
};

TEST_F(CacheSelectTest, ReparentsUnsortedRepeatedAndCyclicParents) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kBeam = 4;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kSteps = 3;
  constexpr int64_t kDim = 3;
  const std::vector<int32_t> indices = {
      9, 3, 12, 13, 5, 4, 10, 0};  // parents: 2, 0, 3, 3, 1, 1, 2, 0
  auto out_token_index = make_indices(indices, device_);
  auto index_before = out_token_index.cpu().clone();
  auto index_storage = out_token_index.storage().unsafeGetStorageImpl();
  auto index_ptr = out_token_index.data_ptr();
  std::vector<torch::Tensor> k_caches;
  std::vector<torch::Tensor> v_caches;
  std::vector<torch::Tensor> expected_k;
  std::vector<torch::Tensor> expected_v;
  std::vector<torch::Tensor> k_before;
  std::vector<torch::Tensor> v_before;

  for (int64_t layer = 0; layer < 2; ++layer) {
    auto k_cpu = make_pattern(kBatch, kBeam, kHeads, kSteps, kDim, layer, 0);
    auto v_cpu = make_pattern(kBatch, kBeam, kHeads, kSteps, kDim, layer, 1);
    auto k = make_cache(k_cpu, device_, false).cache;
    auto v = make_cache(v_cpu, device_, false).cache;
    expected_k.push_back(reference(k_cpu, indices, /*decode_step=*/1, kBeam));
    expected_v.push_back(reference(v_cpu, indices, /*decode_step=*/1, kBeam));
    k_before.push_back(k);
    v_before.push_back(v);
    k_caches.push_back(k);
    v_caches.push_back(v);
  }

  cache_select(out_token_index, k_caches, v_caches, /*decode_step=*/1, kBeam);

  EXPECT_TRUE(torch::equal(out_token_index.cpu(), index_before));
  EXPECT_EQ(out_token_index.data_ptr(), index_ptr);
  EXPECT_EQ(out_token_index.storage().unsafeGetStorageImpl(), index_storage);
  for (size_t layer = 0; layer < k_caches.size(); ++layer) {
    EXPECT_TRUE(torch::equal(k_caches[layer].cpu(), expected_k[layer]));
    EXPECT_TRUE(torch::equal(v_caches[layer].cpu(), expected_v[layer]));
    expect_metadata_unchanged(k_before[layer], k_caches[layer]);
    expect_metadata_unchanged(v_before[layer], v_caches[layer]);
  }
}

TEST_F(CacheSelectTest, PreservesCacheForIdentityParents) {
  constexpr int64_t kBeam = 4;
  const std::vector<int32_t> indices = {0, 5, 10, 15};
  auto k_cpu = make_pattern(1, kBeam, 2, 3, 2, 0, 0);
  auto v_cpu = make_pattern(1, kBeam, 2, 3, 2, 0, 1);
  auto k_caches =
      std::vector<torch::Tensor>{make_cache(k_cpu, device_, false).cache};
  auto v_caches =
      std::vector<torch::Tensor>{make_cache(v_cpu, device_, false).cache};

  cache_select(make_indices(indices, device_),
               k_caches,
               v_caches,
               /*decode_step=*/2,
               kBeam);

  EXPECT_TRUE(torch::equal(k_caches[0].cpu(), k_cpu));
  EXPECT_TRUE(torch::equal(v_caches[0].cpu(), v_cpu));
}

TEST_F(CacheSelectTest, ReparentsFirstAndFinalHistoryStep) {
  constexpr int64_t kBeam = 3;
  const std::vector<int32_t> indices = {5, 0, 4};
  for (int64_t decode_step : {0, 2}) {
    auto cpu = make_pattern(1, kBeam, 1, 3, 2, 0, 0);
    auto cache = make_cache(cpu, device_, false).cache;
    auto caches = std::vector<torch::Tensor>{cache};
    auto values =
        std::vector<torch::Tensor>{make_cache(cpu + 5, device_, false).cache};
    cache_select(
        make_indices(indices, device_), caches, values, decode_step, kBeam);
    EXPECT_TRUE(torch::equal(caches[0].cpu(),
                             reference(cpu, indices, decode_step, kBeam)));
    EXPECT_TRUE(torch::equal(values[0].cpu(),
                             reference(cpu + 5, indices, decode_step, kBeam)));
  }
}

TEST_F(CacheSelectTest, UsesCurrentStreamAndPreservesGuardedStorage) {
  constexpr int64_t kBeam = 4;
  const std::vector<int32_t> indices = {11, 1, 12, 6};
  auto cpu = make_pattern(1, kBeam, 2, 3, 2, 0, 0);
  auto k = make_cache(cpu, device_, true);
  auto v = make_cache(cpu + 4, device_, true);
  auto expected_k = reference(cpu, indices, /*decode_step=*/1, kBeam);
  auto expected_v = reference(cpu + 4, indices, /*decode_step=*/1, kBeam);
  auto k_before = k.cache;
  auto v_before = v.cache;
  auto stream = torch_mlu::getStreamFromPool(/*isHighPriority=*/false, 0);
  {
    torch_mlu::mlu::MLUStreamGuard guard(stream);
    auto k_caches = std::vector<torch::Tensor>{k.cache};
    auto v_caches = std::vector<torch::Tensor>{v.cache};
    cache_select(make_indices(indices, device_),
                 k_caches,
                 v_caches,
                 /*decode_step=*/1,
                 kBeam);
  }
  stream.synchronize();

  EXPECT_TRUE(torch::equal(k.cache.cpu(), expected_k));
  EXPECT_TRUE(torch::equal(v.cache.cpu(), expected_v));
  expect_metadata_unchanged(k_before, k.cache);
  expect_metadata_unchanged(v_before, v.cache);
  auto k_backing = k.backing.cpu();
  auto v_backing = v.backing.cpu();
  EXPECT_TRUE(
      torch::all(k_backing.slice(0, 0, kGuard) == kGuardValue).item<bool>());
  EXPECT_TRUE(
      torch::all(k_backing.slice(0, kGuard + cpu.numel()) == kGuardValue)
          .item<bool>());
  EXPECT_TRUE(
      torch::all(v_backing.slice(0, 0, kGuard) == kGuardValue).item<bool>());
  EXPECT_TRUE(
      torch::all(v_backing.slice(0, kGuard + cpu.numel()) == kGuardValue)
          .item<bool>());
}

TEST_F(CacheSelectTest, RejectsInvalidStructureBeforeWritingCaches) {
  constexpr int64_t kBeam = 2;
  auto cpu = make_pattern(1, kBeam, 1, 2, 2, 0, 0);
  auto k = make_cache(cpu, device_, false).cache;
  auto v = make_cache(cpu + 3, device_, false).cache;
  auto unchanged = k.cpu().clone();
  auto indices = make_indices({0, 3}, device_);
  auto empty = std::vector<torch::Tensor>{};
  auto k_caches = std::vector<torch::Tensor>{k};
  auto v_caches = std::vector<torch::Tensor>{v};
  auto overlapping = std::vector<torch::Tensor>{k};

  EXPECT_THROW(cache_select(indices, empty, v_caches, 0, kBeam), c10::Error);
  EXPECT_THROW(cache_select(indices, k_caches, v_caches, 2, kBeam), c10::Error);
  EXPECT_THROW(cache_select(indices, k_caches, overlapping, 0, kBeam),
               c10::Error);
  EXPECT_TRUE(torch::equal(k.cpu(), unchanged));
}

TEST_F(CacheSelectTest, ReparentsBusinessShape) {
  constexpr int64_t kBatch = 4;
  constexpr int64_t kBeam = 256;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kSteps = 2;
  constexpr int64_t kDim = 128;
  constexpr int64_t kLayers = 28;
  std::vector<int32_t> indices;
  indices.reserve(kBatch * kBeam);
  for (int64_t batch = 0; batch < kBatch; ++batch) {
    for (int64_t beam = 0; beam < kBeam; ++beam) {
      const int32_t parent = static_cast<int32_t>((beam * 37 + 19) % kBeam);
      indices.push_back(parent * kBeam +
                        static_cast<int32_t>((beam * 11) % kBeam));
    }
  }
  auto out_token_index = make_indices(indices, device_);
  std::vector<torch::Tensor> k_caches;
  std::vector<torch::Tensor> v_caches;
  std::vector<torch::Tensor> expected_k;
  std::vector<torch::Tensor> expected_v;
  for (int64_t layer = 0; layer < kLayers; ++layer) {
    auto k_cpu = make_pattern(kBatch, kBeam, kHeads, kSteps, kDim, layer, 0);
    auto v_cpu = make_pattern(kBatch, kBeam, kHeads, kSteps, kDim, layer, 1);
    expected_k.push_back(reference(k_cpu, indices, /*decode_step=*/0, kBeam));
    expected_v.push_back(reference(v_cpu, indices, /*decode_step=*/0, kBeam));
    k_caches.push_back(make_cache(k_cpu, device_, false).cache);
    v_caches.push_back(make_cache(v_cpu, device_, false).cache);
  }

  cache_select(out_token_index, k_caches, v_caches, /*decode_step=*/0, kBeam);

  for (int64_t layer = 0; layer < kLayers; ++layer) {
    EXPECT_TRUE(torch::equal(k_caches[layer].cpu(), expected_k[layer]));
    EXPECT_TRUE(torch::equal(v_caches[layer].cpu(), expected_v[layer]));
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu
