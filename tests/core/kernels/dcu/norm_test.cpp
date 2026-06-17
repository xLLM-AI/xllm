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

#include <string>
#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
constexpr int32_t kDeviceId = 0;
constexpr float kDefaultEpsilon = 1e-5f;

torch::TensorOptions Opts(torch::ScalarType dtype,
                          const torch::Device& device) {
  return torch::TensorOptions().device(device).dtype(dtype);
}

std::string DtypeName(torch::ScalarType dtype) {
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

// Returns {rtol, atol} for the given dtype.
std::pair<double, double> Tol(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return {1e-5, 1e-5};
    case torch::kFloat16:
      return {1e-3, 1e-3};
    case torch::kBFloat16:
      return {1e-2, 1e-2};
    default:
      return {1e-2, 1e-2};
  }
}

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------
struct RmsNormCase {
  std::string name;
  std::vector<int64_t> shape;
  bool zero_input = false;
  bool strided = false;
};

class RmsNormDcuTest : public ::testing::Test {
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

  void run_and_check(torch::ScalarType dtype, const RmsNormCase& c) const {
    auto opts = Opts(dtype, device_);
    int64_t hidden = c.shape.back();

    torch::Tensor input;
    if (c.zero_input) {
      input = torch::zeros(c.shape, opts);
    } else if (c.strided) {
      // Slice from a wider tensor so stride(-2) != hidden_size
      auto full = torch::randn({c.shape[0], hidden * 2}, opts);
      input = full.slice(1, 0, hidden);
    } else {
      input = torch::randn(c.shape, opts);
    }

    auto weight =
        (torch::randn({hidden}, Opts(torch::kFloat32, device_)) * 0.5 + 1.0)
            .to(dtype);
    auto output = torch::zeros_like(input);

    xllm::kernel::cuda::rms_norm(output, input, weight, kDefaultEpsilon);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Use PyTorch native rms_norm as reference on CPU
    auto expected =
        at::rms_norm(input.cpu(), {hidden}, weight.cpu(), kDefaultEpsilon);
    auto [rtol, atol] = Tol(dtype);

    if (c.zero_input) {
      auto out_cpu = output.cpu();
      EXPECT_FALSE(out_cpu.isnan().any().item<bool>())
          << "NaN: " << c.name << " " << DtypeName(dtype);
      EXPECT_FALSE(out_cpu.isinf().any().item<bool>())
          << "Inf: " << c.name << " " << DtypeName(dtype);
    }

    EXPECT_TRUE(torch::allclose(output.cpu(), expected, rtol, atol))
        << "Mismatch: " << c.name << " " << DtypeName(dtype);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(RmsNormDcuTest, MatchesReference) {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};
  const std::vector<RmsNormCase> cases = {
      {"SmallHidden", {64, 64}},
      {"LargeHidden", {64, 4096}},
      {"SingleToken", {1, 256}},
      {"ZeroInput", {64, 256}, true},
      {"StridedInput", {64, 256}, false, true},
      {"3D_12x8192x4096", {12, 8192, 4096}},
  };
  for (auto dtype : dtypes) {
    for (const auto& c : cases) {
      run_and_check(dtype, c);
    }
  }
}

// ---------------------------------------------------------------------------
// fused_add_rms_norm
// ---------------------------------------------------------------------------
class FusedAddRmsNormDcuTest : public ::testing::Test {
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

  void run_and_check(torch::ScalarType dtype,
                     const std::string& name,
                     const std::vector<int64_t>& shape) const {
    auto opts = Opts(dtype, device_);
    int64_t hidden = shape.back();

    auto input = torch::randn(shape, opts) * 0.3f;
    auto residual = torch::randn(shape, opts) * 0.3f;
    auto weight =
        torch::randn({hidden}, Opts(torch::kFloat32, device_)).to(dtype);

    auto input_copy = input.clone();
    auto residual_copy = residual.clone();

    xllm::kernel::cuda::fused_add_rms_norm(
        input, residual, weight, kDefaultEpsilon);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Reference: residual += input, then rms_norm on updated residual
    auto updated_residual = (input_copy.cpu().to(torch::kFloat32) +
                             residual_copy.cpu().to(torch::kFloat32))
                                .to(dtype);
    auto exp_input =
        at::rms_norm(updated_residual, {hidden}, weight.cpu(), kDefaultEpsilon);

    auto [rtol, atol] = Tol(dtype);
    EXPECT_TRUE(torch::allclose(input.cpu(), exp_input, rtol, atol))
        << "Input mismatch: " << name << " " << DtypeName(dtype);
    EXPECT_TRUE(torch::allclose(residual.cpu(), updated_residual, rtol, atol))
        << "Residual mismatch: " << name << " " << DtypeName(dtype);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(FusedAddRmsNormDcuTest, MatchesReference) {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};
  // Aligned: hidden%8==0 → vectorized.  Unaligned: fallback.
  // LargeHidden: vectorized + loop stride.  ManyTokens: block=256 path.
  struct C {
    std::string name;
    std::vector<int64_t> shape;
  };
  const std::vector<C> cases = {
      {"Aligned_256", {64, 256}},
      {"Unaligned_100", {64, 100}},
      {"LargeHidden_4096", {64, 4096}},
      {"ManyTokens_8192", {8192, 128}},
      {"3D_12x8192x4096", {12, 8192, 4096}},
  };
  for (auto dtype : dtypes) {
    for (const auto& c : cases) {
      run_and_check(dtype, c.name, c.shape);
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu
