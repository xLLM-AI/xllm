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
#include <torch/torch.h>

#include <string>
#include <vector>

#include "layers/npu_torch/mega_moe_runtime.h"

namespace xllm::layer {
namespace {

TEST(MegaMoeRuntimeTest, DefersCacheUntilW13ThenW2ShardsAreComplete) {
  int build_count = 0;
  const auto observe = [&](bool ready, bool w13_loaded, bool w2_loaded) {
    const auto action = plan_mega_moe_weight_cache(
        true, ready, w13_loaded, w2_loaded, false);
    if (action == MegaMoeWeightCacheAction::BUILD) {
      ++build_count;
    }
    return action;
  };

  EXPECT_EQ(observe(false, true, false),
            MegaMoeWeightCacheAction::WAIT_FOR_WEIGHTS);
  EXPECT_EQ(build_count, 0);
  EXPECT_EQ(observe(false, true, true),
            MegaMoeWeightCacheAction::BUILD);
  EXPECT_EQ(build_count, 1);
  EXPECT_EQ(observe(true, true, true),
            MegaMoeWeightCacheAction::SKIP);
  EXPECT_EQ(build_count, 1);
}

TEST(MegaMoeRuntimeTest, DefersCacheUntilW2ThenW13ShardsAreComplete) {
  int build_count = 0;
  const auto observe = [&](bool ready, bool w13_loaded, bool w2_loaded) {
    const auto action = plan_mega_moe_weight_cache(
        true, ready, w13_loaded, w2_loaded, false);
    if (action == MegaMoeWeightCacheAction::BUILD) {
      ++build_count;
    }
    return action;
  };

  EXPECT_EQ(observe(false, false, true),
            MegaMoeWeightCacheAction::WAIT_FOR_WEIGHTS);
  EXPECT_EQ(build_count, 0);
  EXPECT_EQ(observe(false, true, true),
            MegaMoeWeightCacheAction::BUILD);
  EXPECT_EQ(build_count, 1);
  EXPECT_EQ(observe(true, true, true),
            MegaMoeWeightCacheAction::SKIP);
  EXPECT_EQ(build_count, 1);
}

TEST(MegaMoeRuntimeTest, ForcedForwardFailsClosedWhenShardIsMissing) {
  EXPECT_EQ(plan_mega_moe_weight_cache(true, false, true, false, true),
            MegaMoeWeightCacheAction::FAIL_MISSING_WEIGHTS);
  EXPECT_EQ(plan_mega_moe_weight_cache(true, false, false, true, true),
            MegaMoeWeightCacheAction::FAIL_MISSING_WEIGHTS);
}

TEST(MegaMoeRuntimeTest, DisabledModesNeverBuildOrRequireWeightCache) {
  EXPECT_EQ(plan_mega_moe_weight_cache(false, false, true, true, false),
            MegaMoeWeightCacheAction::SKIP);
  EXPECT_EQ(plan_mega_moe_weight_cache(false, false, false, false, true),
            MegaMoeWeightCacheAction::SKIP);
}

TEST(MegaMoeRuntimeTest, BuildsIndependentContiguousWeightCaches) {
  constexpr int64_t kExperts = 3;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  torch::Tensor canonical_w13 =
      torch::arange(kExperts * 2 * kIntermediate * kHidden, options)
          .reshape({kExperts, 2 * kIntermediate, kHidden});
  torch::Tensor canonical_w2 =
      torch::arange(kExperts * kHidden * kIntermediate, options)
          .reshape({kExperts, kHidden, kIntermediate});
  const torch::Tensor original_w13 = canonical_w13.clone();
  const torch::Tensor original_w2 = canonical_w2.clone();

  MegaMoeWeightCache cache = build_mega_moe_weight_cache(
      canonical_w13, canonical_w2, kHidden, kIntermediate);

  ASSERT_TRUE(cache.ready());
  EXPECT_EQ(cache.weight1_storage.sizes(),
            torch::IntArrayRef({kExperts, kHidden, 2 * kIntermediate}));
  EXPECT_EQ(cache.weight2_storage.sizes(),
            torch::IntArrayRef({kExperts, kIntermediate, kHidden}));
  EXPECT_TRUE(cache.weight1_storage.is_contiguous());
  EXPECT_TRUE(cache.weight2_storage.is_contiguous());
  EXPECT_NE(cache.weight1_storage.data_ptr(), canonical_w13.data_ptr());
  EXPECT_NE(cache.weight2_storage.data_ptr(), canonical_w2.data_ptr());
  EXPECT_EQ(cache.weight1.size(), kExperts);
  EXPECT_EQ(cache.weight2.size(), kExperts);
  EXPECT_TRUE(torch::equal(cache.weight1[1], canonical_w13[1].transpose(0, 1)));
  EXPECT_TRUE(torch::equal(cache.weight2[2], canonical_w2[2].transpose(0, 1)));

  cache.weight1_storage.zero_();
  cache.weight2_storage.zero_();
  EXPECT_TRUE(torch::equal(canonical_w13, original_w13));
  EXPECT_TRUE(torch::equal(canonical_w2, original_w2));

  const int64_t expected_bytes =
      (canonical_w13.numel() + canonical_w2.numel()) *
      static_cast<int64_t>(canonical_w13.element_size());
  EXPECT_EQ(cache.memory_bytes, expected_bytes);
}

TEST(MegaMoeRuntimeTest, AccountsSuccessfulWeightCacheAllocation) {
  constexpr int64_t kExperts = 2;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto canonical_w13 =
      torch::empty({kExperts, 2 * kIntermediate, kHidden}, options);
  const auto canonical_w2 =
      torch::empty({kExperts, kHidden, kIntermediate}, options);
  const int64_t expected_bytes =
      (canonical_w13.numel() + canonical_w2.numel()) *
      static_cast<int64_t>(canonical_w13.element_size());
  const int64_t before = current_mega_moe_weight_cache_bytes();
  MegaMoeWeightBudgetStatus status;

  auto cache = try_build_mega_moe_weight_cache(canonical_w13,
                                                canonical_w2,
                                                kHidden,
                                                kIntermediate,
                                                expected_bytes,
                                                &status);

  ASSERT_TRUE(cache.has_value());
  EXPECT_TRUE(status.allowed);
  EXPECT_EQ(status.estimated_bytes, expected_bytes);
  EXPECT_EQ(status.current_bytes, before);
  EXPECT_EQ(status.limit_bytes, expected_bytes);
  EXPECT_EQ(current_mega_moe_weight_cache_bytes(), before + expected_bytes);
}

TEST(MegaMoeRuntimeTest, RejectsWeightCacheAllocationOverBudget) {
  constexpr int64_t kExperts = 2;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto canonical_w13 =
      torch::empty({kExperts, 2 * kIntermediate, kHidden}, options);
  const auto canonical_w2 =
      torch::empty({kExperts, kHidden, kIntermediate}, options);
  const int64_t expected_bytes =
      (canonical_w13.numel() + canonical_w2.numel()) *
      static_cast<int64_t>(canonical_w13.element_size());
  const int64_t before = current_mega_moe_weight_cache_bytes();
  MegaMoeWeightBudgetStatus status;

  auto cache = try_build_mega_moe_weight_cache(canonical_w13,
                                                canonical_w2,
                                                kHidden,
                                                kIntermediate,
                                                expected_bytes - 1,
                                                &status);

  EXPECT_FALSE(cache.has_value());
  EXPECT_FALSE(status.allowed);
  EXPECT_EQ(status.estimated_bytes, expected_bytes);
  EXPECT_EQ(status.current_bytes, before);
  EXPECT_EQ(status.limit_bytes, expected_bytes - 1);
  EXPECT_EQ(current_mega_moe_weight_cache_bytes(), before);
}

TEST(MegaMoeRuntimeTest, ReturnsWeightCacheAccountingOnDestruction) {
  constexpr int64_t kExperts = 2;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto canonical_w13 =
      torch::empty({kExperts, 2 * kIntermediate, kHidden}, options);
  const auto canonical_w2 =
      torch::empty({kExperts, kHidden, kIntermediate}, options);
  const int64_t expected_bytes =
      (canonical_w13.numel() + canonical_w2.numel()) *
      static_cast<int64_t>(canonical_w13.element_size());
  const int64_t before = current_mega_moe_weight_cache_bytes();

  {
    MegaMoeWeightBudgetStatus status;
    auto cache = try_build_mega_moe_weight_cache(canonical_w13,
                                                  canonical_w2,
                                                  kHidden,
                                                  kIntermediate,
                                                  expected_bytes,
                                                  &status);
    ASSERT_TRUE(cache.has_value());
    EXPECT_EQ(current_mega_moe_weight_cache_bytes(),
              before + expected_bytes);
  }

  EXPECT_EQ(current_mega_moe_weight_cache_bytes(), before);
}

TEST(MegaMoeRuntimeTest, ReleasesTensorFieldsBeforeBudgetReservation) {
  constexpr int64_t kExperts = 2;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto canonical_w13 =
      torch::empty({kExperts, 2 * kIntermediate, kHidden}, options);
  const auto canonical_w2 =
      torch::empty({kExperts, kHidden, kIntermediate}, options);
  const int64_t expected_bytes =
      (canonical_w13.numel() + canonical_w2.numel()) *
      static_cast<int64_t>(canonical_w13.element_size());
  std::vector<std::string> events;
  MegaMoeWeightCacheDestructionObserver observer;
  observer.on_tensor_fields_released =
      [&]() { events.push_back("tensor_fields_released"); };
  observer.on_reservation_released =
      [&]() { events.push_back("reservation_released"); };

  {
    MegaMoeWeightBudgetStatus status;
    auto cache = try_build_mega_moe_weight_cache(canonical_w13,
                                                  canonical_w2,
                                                  kHidden,
                                                  kIntermediate,
                                                  expected_bytes,
                                                  &status,
                                                  &observer);
    ASSERT_TRUE(cache.has_value());
    EXPECT_TRUE(events.empty());
  }

  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0], "tensor_fields_released");
  EXPECT_EQ(events[1], "reservation_released");
}

TEST(MegaMoeRuntimeTest, RejectsCanonicalShapeMismatch) {
  constexpr int64_t kHidden = 4;
  constexpr int64_t kIntermediate = 2;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor canonical_w13 =
      torch::empty({3, 2 * kIntermediate, kHidden + 1}, options);
  const torch::Tensor canonical_w2 =
      torch::empty({3, kHidden, kIntermediate}, options);

  EXPECT_THROW(build_mega_moe_weight_cache(
                   canonical_w13, canonical_w2, kHidden, kIntermediate),
               c10::Error);
}

TEST(MegaMoeRuntimeTest, PreparesBfloat16TopkWeightsForA3MegaMoe) {
  const torch::Tensor topk_weights =
      torch::tensor({{0.125F, 0.25F, 0.625F}},
                    torch::TensorOptions().dtype(torch::kBFloat16));

  const torch::Tensor prepared =
      prepare_mega_moe_topk_weights(topk_weights);

  EXPECT_EQ(prepared.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(prepared.is_contiguous());
  EXPECT_EQ(prepared.sizes(), topk_weights.sizes());
  EXPECT_TRUE(torch::equal(prepared,
                           topk_weights.to(torch::kFloat32)));
  EXPECT_NE(prepared.data_ptr(), topk_weights.data_ptr());
}

}  // namespace
}  // namespace xllm::layer
