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

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_recurrent_gated_delta_rule_packed_decode;

torch::Tensor bf16_randn(torch::IntArrayRef shape, const torch::Device& dev) {
  return torch::randn(
      shape, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
}

torch::Tensor fp32_randn(torch::IntArrayRef shape, const torch::Device& dev) {
  return torch::randn(
      shape, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
}

bool tensors_allclose(const torch::Tensor& a,
                      const torch::Tensor& b,
                      double rtol,
                      double atol) {
  torch::Tensor ac = a.to(torch::kCPU).to(torch::kFloat32).contiguous();
  torch::Tensor bc = b.to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (ac.sizes() != bc.sizes()) {
    return false;
  }
  torch::Tensor diff = (ac - bc).abs();
  torch::Tensor tol = atol + rtol * bc.abs();
  return (diff <= tol).all().item<bool>();
}

class FusedRecurrentGdrPackedDecodeJitTest : public ::testing::Test {
 protected:
  torch::Device device() { return torch::Device(torch::kPrivateUse1, 0); }
};

TEST_F(FusedRecurrentGdrPackedDecodeJitTest, SecondCallHitsCache) {
  torch::DeviceGuard guard(device());
  int32_t H = 8;
  int32_t HV = 8;
  int32_t K = 128;
  int32_t V = 128;
  int32_t batch = 4;
  int32_t qkv_dim = 2 * H * K + HV * V;
  double scale = 1.0 / std::sqrt(static_cast<double>(K));

  torch::Tensor mixed_qkv = bf16_randn({batch, qkv_dim}, device());
  torch::Tensor a = bf16_randn({batch, HV}, device());
  torch::Tensor b = bf16_randn({batch, HV}, device());
  torch::Tensor a_log = bf16_randn({HV}, device());
  torch::Tensor dt_bias = torch::ones(
      {HV}, torch::TensorOptions().dtype(torch::kBFloat16).device(device()));
  torch::Tensor ssm_orig = fp32_randn({batch, HV, V, K}, device());
  torch::Tensor ssm_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  torch::Tensor sc1 = ssm_orig.clone();
  torch::Tensor sc2 = ssm_orig.clone();

  auto t0 = std::chrono::steady_clock::now();
  auto [out1, state1] =
      fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv,
                                                     a,
                                                     b,
                                                     a_log,
                                                     dt_bias,
                                                     scale,
                                                     sc1,
                                                     ssm_state_indices,
                                                     /*l2norm=*/true);
  torch_mlu::synchronize();
  auto t1 = std::chrono::steady_clock::now();
  auto [out2, state2] = fused_recurrent_gated_delta_rule_packed_decode(
      mixed_qkv, a, b, a_log, dt_bias, scale, sc2, ssm_state_indices, true);
  torch_mlu::synchronize();
  auto t2 = std::chrono::steady_clock::now();

  int64_t first_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  int64_t second_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "first call (compile+launch): " << first_ms
            << "ms; second call (cached): " << second_ms << "ms";
  EXPECT_TRUE(tensors_allclose(out1, out2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "output mismatch";
  EXPECT_TRUE(tensors_allclose(state1, state2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "ssm_cache mismatch";
}

TEST_F(FusedRecurrentGdrPackedDecodeJitTest, SupportsNoL2Norm) {
  torch::DeviceGuard guard(device());
  int32_t H = 8;
  int32_t HV = 8;
  int32_t K = 128;
  int32_t V = 128;
  int32_t batch = 2;
  int32_t qkv_dim = 2 * H * K + HV * V;
  double scale = 1.0 / std::sqrt(static_cast<double>(K));

  torch::Tensor mixed_qkv = bf16_randn({batch, qkv_dim}, device());
  torch::Tensor a = bf16_randn({batch, HV}, device());
  torch::Tensor b = bf16_randn({batch, HV}, device());
  torch::Tensor a_log = bf16_randn({HV}, device());
  torch::Tensor dt_bias = torch::ones(
      {HV}, torch::TensorOptions().dtype(torch::kBFloat16).device(device()));
  torch::Tensor ssm_cache = fp32_randn({batch, HV, V, K}, device());
  torch::Tensor ssm_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  auto [out, state] =
      fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv,
                                                     a,
                                                     b,
                                                     a_log,
                                                     dt_bias,
                                                     scale,
                                                     ssm_cache,
                                                     ssm_state_indices,
                                                     /*l2norm=*/false);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::isfinite(out.to(torch::kFloat32)).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(state.to(torch::kFloat32)).all().item<bool>());
}

}  // namespace
}  // namespace xllm
