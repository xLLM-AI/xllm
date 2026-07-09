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
#include <optional>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_recurrent_gated_delta_rule;

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

class FusedRecurrentGdrJitTest : public ::testing::Test {
 protected:
  torch::Device device() { return torch::Device(torch::kPrivateUse1, 0); }
};

TEST_F(FusedRecurrentGdrJitTest, SecondCallHitsCache) {
  torch::DeviceGuard guard(device());
  int32_t B = 1;
  int32_t T = 4;
  int32_t H = 8;
  int32_t K = 128;
  int32_t HV = 8;
  int32_t V = 128;

  torch::Tensor q = bf16_randn({B, T, H, K}, device());
  torch::Tensor k = bf16_randn({B, T, H, K}, device());
  torch::Tensor v = bf16_randn({B, T, HV, V}, device());
  torch::Tensor g = bf16_randn({B, T, HV}, device());
  torch::Tensor beta = torch::ones(
      {B, T, HV},
      torch::TensorOptions().dtype(torch::kBFloat16).device(device()));
  torch::Tensor init_orig = torch::zeros(
      {T, HV, V, K},
      torch::TensorOptions().dtype(torch::kFloat32).device(device()));
  // cu_seqlens = [0, T] (one sequence of T tokens); ssm_state_indices[seq, tok]
  // = tok.
  torch::Tensor cu_seqlens = torch::tensor(
      {0, T}, torch::TensorOptions().dtype(torch::kInt32).device(device()));
  torch::Tensor ssm_state_indices =
      torch::arange(
          0, T, torch::TensorOptions().dtype(torch::kInt32).device(device()))
          .reshape({1, T});

  torch::Tensor init1 = init_orig.clone();
  torch::Tensor init2 = init_orig.clone();

  auto t0 = std::chrono::steady_clock::now();
  auto [o1, state1] =
      fused_recurrent_gated_delta_rule(q,
                                       k,
                                       v,
                                       g,
                                       beta,
                                       init1,
                                       /*inplace_final_state=*/true,
                                       cu_seqlens,
                                       ssm_state_indices,
                                       std::nullopt,
                                       /*use_qk_l2norm_in_kernel=*/true);
  torch_mlu::synchronize();
  auto t1 = std::chrono::steady_clock::now();
  auto [o2, state2] = fused_recurrent_gated_delta_rule(q,
                                                       k,
                                                       v,
                                                       g,
                                                       beta,
                                                       init2,
                                                       true,
                                                       cu_seqlens,
                                                       ssm_state_indices,
                                                       std::nullopt,
                                                       true);
  torch_mlu::synchronize();
  auto t2 = std::chrono::steady_clock::now();

  int64_t first_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  int64_t second_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "first call (compile+launch): " << first_ms
            << "ms; second call (cached): " << second_ms << "ms";
  EXPECT_TRUE(tensors_allclose(o1, o2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "output mismatch";
  EXPECT_TRUE(tensors_allclose(state1, state2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "final state mismatch";
}

}  // namespace
}  // namespace xllm
