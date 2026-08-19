/* Copyright 2025-2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

namespace xllm::kernel::cuda {
namespace test {
namespace {

void apply_reference_prefill_reshape_and_cache(const torch::Tensor& proj_k,
                                               const torch::Tensor& proj_v,
                                               torch::Tensor& shared_k_cache,
                                               torch::Tensor& shared_v_cache) {
  const int64_t shared_len = proj_k.size(0);
  shared_k_cache.slice(0, 0, shared_len).copy_(proj_k);
  shared_v_cache.slice(0, 0, shared_len).copy_(proj_v);
}

void run_and_check_prefill_reshape_and_cache(const torch::Tensor& proj_k,
                                             const torch::Tensor& proj_v,
                                             torch::Tensor& shared_k_cache,
                                             torch::Tensor& shared_v_cache) {
  const int64_t shared_len = proj_k.size(0);
  const int64_t cache_len = shared_k_cache.size(0);

  auto tail_k_before = shared_k_cache.slice(0, shared_len, cache_len).clone();
  auto tail_v_before = shared_v_cache.slice(0, shared_len, cache_len).clone();

  auto ref_k_cache = shared_k_cache.clone();
  auto ref_v_cache = shared_v_cache.clone();
  apply_reference_prefill_reshape_and_cache(
      proj_k, proj_v, ref_k_cache, ref_v_cache);

  prefill_reshape_and_cache(proj_k, proj_v, shared_k_cache, shared_v_cache);
  torch::cuda::synchronize();

  EXPECT_TRUE(torch::equal(shared_k_cache, ref_k_cache));
  EXPECT_TRUE(torch::equal(shared_v_cache, ref_v_cache));

  EXPECT_TRUE(torch::equal(shared_k_cache.slice(0, shared_len, cache_len),
                           tail_k_before));
  EXPECT_TRUE(torch::equal(shared_v_cache.slice(0, shared_len, cache_len),
                           tail_v_before));
}

class PrefillReshapeAndCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(PrefillReshapeAndCacheTest, MatchesReferenceVectorizedPath) {
  const int64_t shared_len = 7;
  const int64_t cache_len = 11;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;

  auto float_opts = torch::TensorOptions().device(device_).dtype(torch::kHalf);
  auto proj_k = torch::randn({shared_len, kv_heads, head_dim}, float_opts);
  auto proj_v = torch::randn({shared_len, kv_heads, head_dim}, float_opts);

  auto shared_k_cache =
      torch::full({cache_len, kv_heads, head_dim}, -7.0f, float_opts);
  auto shared_v_cache =
      torch::full({cache_len, kv_heads, head_dim}, 5.0f, float_opts);

  run_and_check_prefill_reshape_and_cache(
      proj_k, proj_v, shared_k_cache, shared_v_cache);
}

TEST_F(PrefillReshapeAndCacheTest, MatchesReferenceQkvSliceFp16) {
  const int64_t shared_len = 9;
  const int64_t cache_len = 13;
  const int64_t num_q_heads = 16;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t q_size = num_q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;

  auto float_opts = torch::TensorOptions().device(device_).dtype(torch::kHalf);
  auto qkv = torch::randn({shared_len, q_size + 2 * kv_size}, float_opts) * 0.2;

  auto proj_k = qkv.slice(1, q_size, q_size + kv_size)
                    .view({shared_len, kv_heads, head_dim});
  auto proj_v = qkv.slice(1, q_size + kv_size, q_size + 2 * kv_size)
                    .view({shared_len, kv_heads, head_dim});

  ASSERT_FALSE(proj_k.is_contiguous());
  ASSERT_FALSE(proj_v.is_contiguous());
  ASSERT_EQ(proj_k.stride(0), q_size + 2 * kv_size);
  ASSERT_EQ(proj_v.stride(0), q_size + 2 * kv_size);
  ASSERT_EQ(proj_k.stride(1), head_dim);
  ASSERT_EQ(proj_v.stride(1), head_dim);
  ASSERT_EQ(proj_k.stride(2), 1);
  ASSERT_EQ(proj_v.stride(2), 1);

  auto shared_k_cache =
      torch::full({cache_len, kv_heads, head_dim}, -3.0f, float_opts);
  auto shared_v_cache =
      torch::full({cache_len, kv_heads, head_dim}, 9.0f, float_opts);

  run_and_check_prefill_reshape_and_cache(
      proj_k, proj_v, shared_k_cache, shared_v_cache);
}

TEST_F(PrefillReshapeAndCacheTest, MatchesReferenceQkvSliceBf16) {
  const int64_t shared_len = 6;
  const int64_t cache_len = 10;
  const int64_t num_q_heads = 8;
  const int64_t kv_heads = 4;
  const int64_t head_dim = 128;
  const int64_t q_size = num_q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;

  auto bf16_opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto qkv = torch::randn({shared_len, q_size + 2 * kv_size}, bf16_opts) * 0.2;

  auto proj_k = qkv.slice(1, q_size, q_size + kv_size)
                    .view({shared_len, kv_heads, head_dim});
  auto proj_v = qkv.slice(1, q_size + kv_size, q_size + 2 * kv_size)
                    .view({shared_len, kv_heads, head_dim});

  ASSERT_FALSE(proj_k.is_contiguous());
  ASSERT_FALSE(proj_v.is_contiguous());
  ASSERT_EQ(proj_k.stride(0), q_size + 2 * kv_size);
  ASSERT_EQ(proj_v.stride(0), q_size + 2 * kv_size);
  ASSERT_EQ(proj_k.stride(1), head_dim);
  ASSERT_EQ(proj_v.stride(1), head_dim);
  ASSERT_EQ(proj_k.stride(2), 1);
  ASSERT_EQ(proj_v.stride(2), 1);

  auto shared_k_cache =
      torch::full({cache_len, kv_heads, head_dim}, -3.0f, bf16_opts);
  auto shared_v_cache =
      torch::full({cache_len, kv_heads, head_dim}, 9.0f, bf16_opts);

  run_and_check_prefill_reshape_and_cache(
      proj_k, proj_v, shared_k_cache, shared_v_cache);
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::cuda
