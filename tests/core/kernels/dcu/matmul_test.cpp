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

#include <sstream>
#include <string>
#include <vector>

#include "kernels/dcu/dcu_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
constexpr int32_t kDeviceId = 0;

torch::TensorOptions Opts(torch::ScalarType dtype,
                          const torch::Device& device) {
  return torch::TensorOptions().device(device).dtype(dtype);
}

std::string DtypeName(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return "f32";
    case torch::kFloat16:
      return "f16";
    case torch::kBFloat16:
      return "bf16";
    default:
      return "unknown";
  }
}

// Matmul uses different accumulation order on GPU vs CPU, so tolerances are
// looser than element-wise kernel tests (norm/activation).
// For 3D inputs with batch > 1, scale tolerance by sqrt(total_elements) to
// account for accumulation divergence in large batched matmuls.
std::pair<double, double> Tol(torch::ScalarType dtype,
                              int64_t total_elements,
                              bool needs_scale) {
  double scale = 1.0;
  if (needs_scale) {
    scale = std::sqrt(static_cast<double>(total_elements)) /
            2048;  // [2,8k,4k] -> 64M -> scale=4
    if (scale < 1.0) {
      scale = 1.0;
    }
  }
  switch (dtype) {
    case torch::kFloat32:
      return {3e-5 * scale, 5e-5 * scale};
    case torch::kFloat16:
      return {1e-3 * scale, 1e-3 * scale};
    case torch::kBFloat16:
      return {2e-2, 2e-2};
    default:
      return {1e-2 * scale, 1e-2 * scale};
  }
}

// Reference: torch::nn::functional::linear on CPU
torch::Tensor ReferenceMatmul(const torch::Tensor& a,
                              const torch::Tensor& b,
                              const std::optional<torch::Tensor>& bias) {
  namespace F = torch::nn::functional;
  return F::linear(a, b, bias.value_or(torch::Tensor()));
}

// ---------------------------------------------------------------------------
// Parameterized test case descriptor
// ---------------------------------------------------------------------------
struct MatmulCase {
  std::string name;
  std::vector<int64_t> shape_a;
  std::vector<int64_t> shape_b;
  bool with_bias = false;
  torch::ScalarType dtype = torch::kFloat32;
};

// GTest uses PrintToStringParamName — provide a custom printer so the test
// name shows "SingleToken_4096x4096_f32" instead of a memory address.
std::string PrintMatmulCase(const testing::TestParamInfo<MatmulCase>& info) {
  return info.param.name + "_" + DtypeName(info.param.dtype);
}

// ---------------------------------------------------------------------------
// MatmulDcuTest fixture (parameterized)
// ---------------------------------------------------------------------------
class MatmulDcuTest : public ::testing::TestWithParam<MatmulCase> {
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

TEST_P(MatmulDcuTest, MatchesReference) {
  const auto& c = GetParam();
  auto opts = Opts(c.dtype, device_);
  auto a = torch::randn(c.shape_a, opts) * 0.3f;
  auto b = torch::randn(c.shape_b, opts) * 0.3f;

  std::optional<torch::Tensor> bias;
  if (c.with_bias) {
    bias = torch::randn({c.shape_b[0]}, opts) * 0.1f;
  }

  auto output = matmul(a, b, bias);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  auto expected = ReferenceMatmul(
      a.cpu(),
      b.cpu(),
      bias.has_value() ? std::optional<torch::Tensor>(bias->cpu())
                       : std::nullopt);

  // Only scale tolerance for 3D inputs with batch > 1
  bool needs_scale = a.dim() >= 3 && a.size(0) > 1;
  auto [rtol, atol] = Tol(c.dtype, a.numel(), needs_scale);
  EXPECT_TRUE(torch::allclose(output.cpu(), expected, rtol, atol))
      << "Mismatch: " << c.name << " " << DtypeName(c.dtype)
      << " shape_a=" << expected.sizes();
}

// ---------------------------------------------------------------------------
// Generate all (shape × dtype) combinations
// ---------------------------------------------------------------------------
std::vector<MatmulCase> Generate2DCases() {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};
  // shape_a: input [..., in_features], shape_b: weight [out_features,
  // in_features]
  const std::vector<MatmulCase> shapes = {
      {"SingleToken_256x256", {1, 256}, {256, 256}},
      {"SingleToken_4096x4096", {1, 4096}, {4096, 4096}},
      {"Batch8_4096x11008", {8, 4096}, {11008, 4096}},
      {"WithBias_Batch4_11008x4096", {4, 4096}, {11008, 4096}},
  };
  std::vector<MatmulCase> cases;
  cases.reserve(shapes.size() * dtypes.size());
  for (const auto& s : shapes) {
    for (auto dtype : dtypes) {
      MatmulCase c = s;
      c.dtype = dtype;
      cases.push_back(c);
    }
  }
  return cases;
}

std::vector<MatmulCase> Generate3DCases() {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};
  const std::vector<MatmulCase> shapes = {
      {"Prefill_1x128x4096", {1, 128, 4096}, {4096, 4096}},
      {"Prefill_4x11008x4096", {2, 4096, 4096}, {11008, 4096}},
      {"Prefill_WithBias_2x11008x4096", {2, 8192, 4096}, {11008, 4096}, true},
  };
  std::vector<MatmulCase> cases;
  cases.reserve(shapes.size() * dtypes.size());
  for (const auto& s : shapes) {
    for (auto dtype : dtypes) {
      MatmulCase c = s;
      c.dtype = dtype;
      cases.push_back(c);
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(Matmul2D,
                         MatmulDcuTest,
                         testing::ValuesIn(Generate2DCases()),
                         PrintMatmulCase);

INSTANTIATE_TEST_SUITE_P(Matmul3D,
                         MatmulDcuTest,
                         testing::ValuesIn(Generate3DCases()),
                         PrintMatmulCase);

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu
