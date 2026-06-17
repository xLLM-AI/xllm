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
#include <torch/cuda.h>
#include <torch/torch.h>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {

namespace {

torch::Tensor apply_reference_rope(const torch::Tensor& input,
                                   const torch::Tensor& cos,
                                   const torch::Tensor& sin,
                                   int64_t rotary_dim,
                                   bool interleaved) {
  auto input_float = input.to(torch::kFloat32);
  auto input_rot = input_float.slice(-1, 0, rotary_dim);
  auto input_pass = input_float.slice(-1, rotary_dim, input_float.size(-1));

  const int64_t num_tokens = input_float.size(0);
  const int64_t num_heads = input_float.size(1);
  const int64_t half_dim = rotary_dim / 2;
  auto cos_view = cos.view({num_tokens, 1, half_dim});
  auto sin_view = sin.view({num_tokens, 1, half_dim});

  torch::Tensor rotated;
  if (interleaved) {
    auto reshaped = input_rot.view({num_tokens, num_heads, half_dim, 2});
    auto even = reshaped.select(-1, 0);
    auto odd = reshaped.select(-1, 1);
    auto out_even = even * cos_view - odd * sin_view;
    auto out_odd = even * sin_view + odd * cos_view;
    rotated = torch::stack({out_even, out_odd}, -1)
                  .view({num_tokens, num_heads, rotary_dim});
  } else {
    auto first = input_rot.slice(-1, 0, half_dim);
    auto second = input_rot.slice(-1, half_dim, rotary_dim);
    rotated = torch::cat({first * cos_view - second * sin_view,
                          second * cos_view + first * sin_view},
                         -1);
  }

  if (rotary_dim < input_float.size(-1)) {
    return torch::cat({rotated, input_pass}, -1).to(input.scalar_type());
  }
  return rotated.to(input.scalar_type());
}

void fused_qk_norm_rope_reference(torch::Tensor& qkv,
                                  int64_t num_heads_q,
                                  int64_t num_heads_k,
                                  int64_t num_heads_v,
                                  int64_t head_dim,
                                  double eps,
                                  const torch::Tensor& q_weight,
                                  const torch::Tensor& k_weight,
                                  const torch::Tensor& cos_sin_cache,
                                  bool interleaved,
                                  const torch::Tensor& position_ids) {
  (void)num_heads_v;
  const int64_t num_tokens = qkv.size(0);
  const int64_t q_size = num_heads_q * head_dim;
  const int64_t k_size = num_heads_k * head_dim;
  const int64_t rotary_dim = cos_sin_cache.size(1);
  const int64_t half_dim = rotary_dim / 2;

  auto q_slice =
      qkv.slice(-1, 0, q_size).view({num_tokens, num_heads_q, head_dim});
  auto k_slice = qkv.slice(-1, q_size, q_size + k_size)
                     .view({num_tokens, num_heads_k, head_dim});

  auto q_float = q_slice.to(torch::kFloat32);
  auto k_float = k_slice.to(torch::kFloat32);
  auto q_weight_float = q_weight.to(torch::kFloat32).view({1, 1, head_dim});
  auto k_weight_float = k_weight.to(torch::kFloat32).view({1, 1, head_dim});

  auto q_rms = torch::rsqrt((q_float * q_float).mean(-1, true) + eps);
  auto k_rms = torch::rsqrt((k_float * k_float).mean(-1, true) + eps);
  auto q = (q_float * q_rms * q_weight_float).to(q_slice.scalar_type());
  auto k = (k_float * k_rms * k_weight_float).to(k_slice.scalar_type());

  auto selected_cos_sin = cos_sin_cache.index_select(0, position_ids);
  auto cos = selected_cos_sin.slice(-1, 0, half_dim);
  auto sin = selected_cos_sin.slice(-1, half_dim, rotary_dim);

  q = apply_reference_rope(q, cos, sin, rotary_dim, interleaved);
  k = apply_reference_rope(k, cos, sin, rotary_dim, interleaved);
  q_slice.copy_(q);
  k_slice.copy_(k);
}

class FusedQKNormRopeTest : public ::testing::Test {
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

TEST_F(FusedQKNormRopeTest, MatchesReferenceNeoX) {
  const int64_t num_tokens = 17;
  const int64_t num_heads_q = 8;
  const int64_t num_heads_k = 4;
  const int64_t num_heads_v = 4;
  const int64_t head_dim = 128;
  const int64_t rotary_dim = 128;
  const int64_t max_position = 512;
  const double eps = 1e-6;

  auto half_opts = torch::TensorOptions().device(device_).dtype(torch::kHalf);
  auto float_opts =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto int_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);

  auto qkv =
      torch::randn(
          {num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim},
          half_opts) *
      0.2;
  auto q_weight = torch::randn({head_dim}, half_opts);
  auto k_weight = torch::randn({head_dim}, half_opts);
  auto cos_sin_cache = torch::randn({max_position, rotary_dim}, float_opts);
  auto position_ids = torch::randint(0, max_position, {num_tokens}, int_opts);

  auto qkv_ref = qkv.clone();
  fused_qk_norm_rope_reference(qkv_ref,
                               num_heads_q,
                               num_heads_k,
                               num_heads_v,
                               head_dim,
                               eps,
                               q_weight,
                               k_weight,
                               cos_sin_cache,
                               false,
                               position_ids);

  auto qkv_out = qkv.clone();
  xllm::kernel::cuda::fused_qk_norm_rope(qkv_out,
                                         num_heads_q,
                                         num_heads_k,
                                         num_heads_v,
                                         head_dim,
                                         eps,
                                         q_weight,
                                         k_weight,
                                         cos_sin_cache,
                                         false,
                                         position_ids);

  EXPECT_TRUE(torch::allclose(qkv_out, qkv_ref, 2e-3, 2e-3));
}

TEST_F(FusedQKNormRopeTest, MatchesReferenceInterleaved) {
  const int64_t num_tokens = 11;
  const int64_t num_heads_q = 6;
  const int64_t num_heads_k = 2;
  const int64_t num_heads_v = 2;
  const int64_t head_dim = 64;
  const int64_t rotary_dim = 64;
  const int64_t max_position = 256;
  const double eps = 1e-6;

  auto bf16_opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto float_opts =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto int_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);

  auto qkv =
      torch::randn(
          {num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim},
          bf16_opts) *
      0.15;
  auto q_weight = torch::randn({head_dim}, bf16_opts);
  auto k_weight = torch::randn({head_dim}, bf16_opts);
  auto cos_sin_cache = torch::randn({max_position, rotary_dim}, float_opts);
  auto position_ids = torch::randint(0, max_position, {num_tokens}, int_opts);

  auto qkv_ref = qkv.clone();
  fused_qk_norm_rope_reference(qkv_ref,
                               num_heads_q,
                               num_heads_k,
                               num_heads_v,
                               head_dim,
                               eps,
                               q_weight,
                               k_weight,
                               cos_sin_cache,
                               true,
                               position_ids);

  auto qkv_out = qkv.clone();
  xllm::kernel::cuda::fused_qk_norm_rope(qkv_out,
                                         num_heads_q,
                                         num_heads_k,
                                         num_heads_v,
                                         head_dim,
                                         eps,
                                         q_weight,
                                         k_weight,
                                         cos_sin_cache,
                                         true,
                                         position_ids);

  EXPECT_TRUE(torch::allclose(qkv_out, qkv_ref, 2e-2, 2e-2));
}

}  // namespace

}  // namespace test
}  // namespace xllm::kernel::dcu
