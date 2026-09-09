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
#include <torch/torch.h>

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

namespace xllm::kernel::cuda {
namespace test {

class DecoderReshapeAndCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
    dtype_ = torch::kFloat16;
  }

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat16;
};

// This test validates `xllm::kernel::cuda::decoder_reshape_and_cache` against
// a PyTorch reference implementation that follows the current CUDA kernel
// contract.
//
// Kernel input contract:
// - proj_k/proj_v: [batch_size, beam_size, kv_heads, head_dim], where
//
// Cache layout:
// - unshared_k/v_cache: [max_num_request, beam_size, max_decode_step, kv_heads,
//   head_dim]
//
//
// The test checks:
// 1) full-tensor equivalence (CUDA output vs reference output)
// 2) per-sequence slice correctness at the target decode step
void torch_reference(const torch::Tensor& proj_k,
                     const torch::Tensor& proj_v,
                     torch::Tensor& unshared_k_cache,
                     torch::Tensor& unshared_v_cache,
                     const torch::Tensor& step) {
  const int64_t batch_size = proj_k.size(0);
  const int64_t beam_size = proj_k.size(1);
  const int64_t kv_heads = proj_k.size(2);
  const int64_t head_dim = proj_k.size(3);

  const int64_t max_num_request = unshared_k_cache.size(0);
  const int64_t max_decode_step = unshared_k_cache.size(2);

  CHECK_EQ(proj_k.dim(), 4) << "proj_k must be 4-dimensional";
  CHECK_EQ(proj_v.dim(), 4) << "proj_v must be 4-dimensional";
  CHECK_EQ(proj_v.sizes(), proj_k.sizes())
      << "proj_v and proj_k must have same shape";

  CHECK_EQ(unshared_k_cache.dim(), 5)
      << "unshared_k_cache must be 5-dimensional";
  CHECK_EQ(unshared_v_cache.sizes(), unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";
  CHECK_LE(batch_size, max_num_request)
      << "batch_size must be <= max_num_request";

  CHECK_EQ(step.dim(), 1) << "step must be 1-dimensional";
  CHECK_EQ(step.size(0), 1) << "step must have shape [1]";
  const int64_t step_value = step[0].item<int64_t>();
  CHECK_GE(step_value, 0) << "step must be >= 0";
  CHECK_LT(step_value, max_decode_step)
      << "step must be less than max_decode_step";

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int64_t beam_idx = 0; beam_idx < beam_size; ++beam_idx) {
      unshared_k_cache[batch_idx][beam_idx]
          .select(0, step_value)
          .copy_(proj_k[batch_idx][beam_idx]);
      unshared_v_cache[batch_idx][beam_idx]
          .select(0, step_value)
          .copy_(proj_v[batch_idx][beam_idx]);
    }
  }
}

TEST_F(DecoderReshapeAndCacheTest, CorrectnessTest) {
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t max_num_request = 2;
  const int64_t max_decode_step = 3;

  torch::Tensor step = torch::tensor({1}, torch::kInt32).to(device_);

  const auto float_opts = torch::TensorOptions().device(device_).dtype(dtype_);
  const auto int_opts =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);

  torch::Tensor proj_k =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, float_opts);
  torch::Tensor proj_v =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, float_opts);

  torch::Tensor unshared_k_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);
  torch::Tensor unshared_v_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);

  torch::Tensor ref_k_cache = unshared_k_cache.clone();
  torch::Tensor ref_v_cache = unshared_v_cache.clone();

  decoder_reshape_and_cache(
      proj_k, proj_v, unshared_k_cache, unshared_v_cache, step);

  torch_reference(proj_k, proj_v, ref_k_cache, ref_v_cache, step);

  EXPECT_TRUE(torch::allclose(unshared_k_cache, ref_k_cache, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(unshared_v_cache, ref_v_cache, 1e-5, 1e-5));

  const int64_t step_value = step[0].item<int64_t>();
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int64_t beam_idx = 0; beam_idx < beam_size; ++beam_idx) {
      torch::Tensor copied_k =
          unshared_k_cache[batch_idx][beam_idx].select(0, step_value);
      torch::Tensor source_k = proj_k[batch_idx][beam_idx];
      EXPECT_TRUE(torch::allclose(copied_k, source_k, 1e-5, 1e-5));
    }
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
