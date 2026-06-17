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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {

namespace {

torch::Tensor torch_reference_act_and_mul(const torch::Tensor& input,
                                          const std::string& act_mode) {
  CHECK_EQ(input.size(-1) % 2, 0) << "Last dim must be even.";
  const int64_t d = input.size(-1) / 2;
  auto x = input.slice(-1, 0, d);
  auto y = input.slice(-1, d, 2 * d);

  if (act_mode == "silu") {
    return (x * torch::sigmoid(x)) * y;
  }
  if (act_mode == "gelu") {
    return torch::gelu(x, "none") * y;
  }
  if (act_mode == "gelu_tanh") {
    return torch::gelu(x, "tanh") * y;
  }
  LOG(FATAL) << "Unsupported act mode in test: " << act_mode;
  return torch::Tensor();
}

std::string dtype_name(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return "float32";
    case torch::kFloat16:
      return "float16";
    case torch::kBFloat16:
      return "bfloat16";
    default:
      return "unknown";
  }
}

class ActAndMulKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  void run_and_check(torch::ScalarType dtype,
                     const std::string& act_mode,
                     int64_t d) const {
    const auto opts = torch::TensorOptions().device(device_).dtype(dtype);
    torch::Tensor input = torch::randn({4, 7, 2 * d}, opts) * 0.5;
    torch::Tensor output = torch::empty({4, 7, d}, opts);

    torch::Tensor reference = torch_reference_act_and_mul(input, act_mode);
    xllm::kernel::cuda::act_and_mul(output, input, act_mode);

    const double atol = dtype == torch::kFloat32 ? 1e-6 : 5e-3;
    const double rtol = dtype == torch::kFloat32 ? 1e-5 : 5e-3;
    EXPECT_TRUE(torch::allclose(output, reference, rtol, atol))
        << "Mismatch for act_mode=" << act_mode
        << ", dtype=" << dtype_name(dtype) << ", d=" << d;
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(ActAndMulKernelTest, MatchesTorchReference) {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};
  const std::vector<std::string> act_modes = {"silu", "gelu", "gelu_tanh"};
  // Cover scalar fallback (d < kVecSize), vectorized path, and tail cleanup.
  const std::vector<int64_t> dims = {3, 64, 129};

  for (auto dtype : dtypes) {
    for (const auto& act_mode : act_modes) {
      for (int64_t d : dims) {
        run_and_check(dtype, act_mode, d);
      }
    }
  }
}

}  // namespace

}  // namespace test
}  // namespace xllm::kernel::dcu
