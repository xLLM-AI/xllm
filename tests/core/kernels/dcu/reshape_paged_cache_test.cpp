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
#include <hip/hip_runtime_api.h>
#include <torch/torch.h>

#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
constexpr int32_t kDeviceId = 0;

// CPU reference: scatter keys/values into paged cache using slot_ids
void reference_reshape_paged_cache(const torch::Tensor& slot_ids_cpu,
                                   const torch::Tensor& keys_cpu,
                                   const torch::Tensor& values_cpu,
                                   torch::Tensor& key_cache_cpu,
                                   torch::Tensor& value_cache_cpu,
                                   int64_t block_size) {
  int64_t n_tokens = keys_cpu.size(0);
  int64_t n_kv_heads = keys_cpu.size(1);
  int64_t head_dim = keys_cpu.size(2);

  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t slot_id = slot_ids_cpu[t].item<int32_t>();
    int64_t block_idx = slot_id / block_size;
    int64_t block_offset = slot_id % block_size;
    key_cache_cpu[block_idx][block_offset] = keys_cpu[t];
    value_cache_cpu[block_idx][block_offset] = values_cpu[t];
  }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class ReshapePagedCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
      GTEST_SKIP() << "No DCU/HIP device available";
    }
    ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
    torch::manual_seed(2026);
    device_ = torch::Device(c10::DeviceType::CUDA, kDeviceId);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

// ---------------------------------------------------------------------------
// Test: kernel output matches CPU reference for LLM-like shapes
// ---------------------------------------------------------------------------
struct TestCase {
  int64_t n_tokens;
  int64_t n_blocks;
  int64_t block_size;
  int64_t n_kv_heads;
  int64_t head_dim;
  torch::ScalarType dtype;
};

void run_and_check(const TestCase& c, const torch::Device& device) {
  auto opts = torch::TensorOptions().device(device).dtype(c.dtype);
  auto idx_opts = torch::TensorOptions().device(device).dtype(torch::kInt32);

  // Generate random keys/values (simulating attention layer output)
  auto keys = torch::randn({c.n_tokens, c.n_kv_heads, c.head_dim}, opts);
  auto values = torch::randn({c.n_tokens, c.n_kv_heads, c.head_dim}, opts);

  // Generate slot_ids: each token maps to a unique slot in [0,
  // n_blocks*block_size) Shuffle to simulate non-sequential assignment
  // (realistic prefill scenario)
  int64_t total_slots = c.n_blocks * c.block_size;
  auto slot_ids_cpu =
      torch::randperm(total_slots, torch::kInt32).slice(0, 0, c.n_tokens);
  auto slot_ids = slot_ids_cpu.to(device);

  // Init empty caches
  auto key_cache =
      torch::zeros({c.n_blocks, c.block_size, c.n_kv_heads, c.head_dim}, opts);
  auto value_cache = torch::zeros_like(key_cache);

  // Run kernel
  xllm::kernel::cuda::reshape_paged_cache(
      slot_ids, keys, values, key_cache, value_cache);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  // CPU reference
  auto ref_key_cache =
      torch::zeros({c.n_blocks, c.block_size, c.n_kv_heads, c.head_dim},
                   torch::TensorOptions().dtype(torch::kFloat32));
  auto ref_value_cache = torch::zeros_like(ref_key_cache);
  reference_reshape_paged_cache(slot_ids_cpu,
                                keys.cpu().to(torch::kFloat32),
                                values.cpu().to(torch::kFloat32),
                                ref_key_cache,
                                ref_value_cache,
                                c.block_size);

  // Compare
  auto k_diff = (key_cache.cpu().to(torch::kFloat32) - ref_key_cache)
                    .abs()
                    .max()
                    .item<float>();
  auto v_diff = (value_cache.cpu().to(torch::kFloat32) - ref_value_cache)
                    .abs()
                    .max()
                    .item<float>();
  float tol = (c.dtype == torch::kFloat32) ? 1e-5f : 1e-3f;
  EXPECT_LT(k_diff, tol) << "key cache mismatch, max_diff=" << k_diff;
  EXPECT_LT(v_diff, tol) << "value cache mismatch, max_diff=" << v_diff;
}

TEST_F(ReshapePagedCacheTest, MatchesReference) {
  // LLM shapes: block_size=16/64, GQA heads, typical head_dims
  const std::vector<TestCase> cases = {
      // Small: single block, minimal heads
      {4, 1, 16, 1, 64, torch::kFloat32},
      // Qwen2.5-7B style: 4 KV heads, 128 head_dim, block_size=16
      {32, 8, 16, 4, 128, torch::kFloat16},
      // LLaMA-3-8B style: 8 KV heads, 128 head_dim, block_size=64
      {64, 4, 64, 8, 128, torch::kBFloat16},
      // DeepSeek-V3 style: large block_size, many tokens
      {256, 16, 64, 8, 128, torch::kFloat16},
      // Single token decode: batch=1
      {1, 4, 16, 4, 128, torch::kFloat32},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE("tokens=" + std::to_string(c.n_tokens) +
                 " blocks=" + std::to_string(c.n_blocks) +
                 " bs=" + std::to_string(c.block_size) +
                 " heads=" + std::to_string(c.n_kv_heads) +
                 " hd=" + std::to_string(c.head_dim));
    run_and_check(c, device_);
  }
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu
