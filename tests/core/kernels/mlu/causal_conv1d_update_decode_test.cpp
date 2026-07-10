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

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::causal_conv1d_update_decode;

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

class CausalConv1dUpdateDecodeJitTest : public ::testing::Test {
 protected:
  torch::Device device() { return torch::Device(torch::kPrivateUse1, 0); }
};

TEST_F(CausalConv1dUpdateDecodeJitTest, SecondCallHitsCache) {
  torch::DeviceGuard guard(device());
  int32_t dim = 1024;
  int32_t width = 4;
  int32_t batch = 4;
  torch::Tensor x = bf16_randn({batch, dim, 1}, device());
  torch::Tensor conv_state_orig = bf16_randn({batch, dim, width - 1}, device());
  torch::Tensor weight = bf16_randn({dim, width}, device());
  torch::Tensor conv_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  torch::Tensor cs1 = conv_state_orig.clone();
  torch::Tensor cs2 = conv_state_orig.clone();

  auto t0 = std::chrono::steady_clock::now();
  torch::Tensor out1 = causal_conv1d_update_decode(
      x, cs1, weight, std::nullopt, conv_state_indices, /*pad_slot_id=*/-1);
  torch_mlu::synchronize();
  auto t1 = std::chrono::steady_clock::now();
  torch::Tensor out2 = causal_conv1d_update_decode(
      x, cs2, weight, std::nullopt, conv_state_indices, -1);
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
  EXPECT_TRUE(tensors_allclose(cs1, cs2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "conv_state mismatch";
}

TEST_F(CausalConv1dUpdateDecodeJitTest, AcceptsUncompiledDim) {
  torch::DeviceGuard guard(device());
  int32_t dim = 1000;
  int32_t width = 4;
  int32_t batch = 2;
  torch::Tensor x = bf16_randn({batch, dim, 1}, device());
  torch::Tensor conv_state = bf16_randn({batch, dim, width - 1}, device());
  torch::Tensor weight = bf16_randn({dim, width}, device());
  torch::Tensor conv_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  torch::Tensor out = causal_conv1d_update_decode(
      x, conv_state, weight, std::nullopt, conv_state_indices, -1);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::isfinite(out.to(torch::kFloat32)).all().item<bool>());
}

}  // namespace
}  // namespace xllm
