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
#include <cstdint>
#include <utility>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_gdn_gating;

torch::Tensor bf16_randn(torch::IntArrayRef shape, const torch::Device& dev) {
  return torch::randn(
      shape, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
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

class FusedGdnGatingJitTest : public ::testing::Test {
 protected:
  torch::Device device() { return torch::Device(torch::kPrivateUse1, 0); }
};

TEST_F(FusedGdnGatingJitTest, SecondCallHitsCache) {
  torch::DeviceGuard guard(device());
  int32_t nh = 16;
  int32_t batch = 1;
  torch::Tensor A_log = bf16_randn({nh}, device());
  torch::Tensor a = bf16_randn({batch, nh}, device());
  torch::Tensor b = bf16_randn({batch, nh}, device());
  torch::Tensor dt_bias = bf16_randn({nh}, device());

  auto t0 = std::chrono::steady_clock::now();
  auto [g1, beta1] = fused_gdn_gating(A_log, a, b, dt_bias, 1.0f, 20.0f);
  torch_mlu::synchronize();
  auto t1 = std::chrono::steady_clock::now();
  auto [g2, beta2] = fused_gdn_gating(A_log, a, b, dt_bias, 1.0f, 20.0f);
  torch_mlu::synchronize();
  auto t2 = std::chrono::steady_clock::now();

  int64_t first_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  int64_t second_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "first call (compile+launch): " << first_ms
            << "ms; second call (cached): " << second_ms << "ms";
  EXPECT_TRUE(tensors_allclose(g1, g2, /*rtol=*/1e-3, /*atol=*/1e-3));
}

// The JIT path accepts an arbitrary head count (NUM_HEADS=20 here) that was
// never pre-compiled. This is the flexibility win over AOT.
TEST_F(FusedGdnGatingJitTest, AcceptsUncompiledHeadCount) {
  torch::DeviceGuard guard(device());
  int32_t nh = 20;
  int32_t batch = 1;
  torch::Tensor A_log = bf16_randn({nh}, device());
  torch::Tensor a = bf16_randn({batch, nh}, device());
  torch::Tensor b = bf16_randn({batch, nh}, device());
  torch::Tensor dt_bias = bf16_randn({nh}, device());

  auto [g, beta] = fused_gdn_gating(A_log, a, b, dt_bias, 1.0f, 20.0f);
  torch_mlu::synchronize();

  EXPECT_EQ(g.size(2), static_cast<int64_t>(nh));
  EXPECT_TRUE(torch::isfinite(g).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(beta).all().item<bool>());
}

}  // namespace
}  // namespace xllm
