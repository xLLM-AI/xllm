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

#include "layers/common/activation.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "platform/device.h"

namespace xllm {
namespace layer {
namespace test {

namespace {

// Reference gated activation: split input [..., 2C] -> act(x[:C]) * x[C:]
torch::Tensor ref_gated_act(const torch::Tensor& input,
                            const std::string& act_mode) {
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
  LOG(FATAL) << "Unsupported act_mode in test: " << act_mode;
  return {};
}

// Reference non-gated activation: output = act(input)
torch::Tensor ref_act(const torch::Tensor& input, const std::string& act_mode) {
  if (act_mode == "silu") {
    return input * torch::sigmoid(input);
  }
  if (act_mode == "gelu") {
    return torch::gelu(input, "none");
  }
  if (act_mode == "gelu_tanh") {
    return torch::gelu(input, "tanh");
  }
  LOG(FATAL) << "Unsupported act_mode in test: " << act_mode;
  return {};
}

std::string to_string(torch::ScalarType dtype) {
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

class ActivationImplTest : public ::testing::Test {
 protected:
  void SetUp() override { device_ = torch::Device(Device::type_torch(), 0); }

  void test_gated(torch::ScalarType dtype,
                  const std::string& act_mode,
                  int64_t hidden_dim) const {
    const auto opts = torch::TensorOptions().device(device_).dtype(dtype);
    auto input = torch::randn({4, hidden_dim * 2}, opts) * 0.5;
    auto output = torch::empty({4, hidden_dim}, opts);

    ActivationImpl act(act_mode, /*is_gated=*/true);
    act.forward(input, output);

    auto expected = ref_gated_act(input, act_mode);
    double atol = (dtype == torch::kFloat32)    ? 1e-6
                  : (dtype == torch::kBFloat16) ? 1e-2
                                                : 5e-3;
    double rtol = (dtype == torch::kFloat32)    ? 1e-5
                  : (dtype == torch::kBFloat16) ? 1e-2
                                                : 5e-3;
    EXPECT_TRUE(torch::allclose(output, expected, rtol, atol))
        << "Mismatch: act_mode=" << act_mode << ", dtype=" << to_string(dtype)
        << ", hidden_dim=" << hidden_dim;
  }

  void test_non_gated(torch::ScalarType dtype,
                      const std::string& act_mode,
                      int64_t hidden_dim) const {
    const auto opts = torch::TensorOptions().device(device_).dtype(dtype);
    auto input = torch::randn({4, hidden_dim}, opts) * 0.5;
    auto output = torch::empty_like(input);

    ActivationImpl act(act_mode, /*is_gated=*/false);
    act.forward(input, output);

    auto expected = ref_act(input, act_mode);
    double atol = (dtype == torch::kFloat32)    ? 1e-6
                  : (dtype == torch::kBFloat16) ? 1e-2
                                                : 5e-3;
    double rtol = (dtype == torch::kFloat32)    ? 1e-5
                  : (dtype == torch::kBFloat16) ? 1e-2
                                                : 5e-3;
    EXPECT_TRUE(torch::allclose(output, expected, rtol, atol))
        << "Mismatch: act_mode=" << act_mode << ", dtype=" << to_string(dtype)
        << ", hidden_dim=" << hidden_dim;
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

// ---- Gated activation tests ----

TEST_F(ActivationImplTest, GatedSiluBfloat16) {
  test_gated(torch::kBFloat16, "silu", 128);
}

TEST_F(ActivationImplTest, GatedSiluFloat16) {
  test_gated(torch::kFloat16, "silu", 128);
}

TEST_F(ActivationImplTest, GatedSiluFloat32) {
  test_gated(torch::kFloat32, "silu", 128);
}

TEST_F(ActivationImplTest, GatedGeluBfloat16) {
  test_gated(torch::kBFloat16, "gelu", 128);
}

TEST_F(ActivationImplTest, GatedGeluFloat16) {
  test_gated(torch::kFloat16, "gelu", 128);
}

TEST_F(ActivationImplTest, GatedGeluTanhBfloat16) {
  test_gated(torch::kBFloat16, "gelu_tanh", 128);
}

// ---- Gated with various hidden dims (cover vectorized / scalar / tail paths)
// ----

TEST_F(ActivationImplTest, GatedSiluSmallDim) {
  test_gated(torch::kBFloat16, "silu", 3);
}

TEST_F(ActivationImplTest, GatedSiluMediumDim) {
  test_gated(torch::kBFloat16, "silu", 129);
}

TEST_F(ActivationImplTest, GatedSiluLargeDim) {
  test_gated(torch::kBFloat16, "silu", 4096);
}

// ---- Non-gated activation tests ----

TEST_F(ActivationImplTest, NonGatedSiluBfloat16) {
  GTEST_SKIP() << "DCU ActivationImpl currently dispatches to gated "
                  "act_and_mul, so non-gated activation is not supported.";
  test_non_gated(torch::kBFloat16, "silu", 128);
}

TEST_F(ActivationImplTest, NonGatedGeluBfloat16) {
  GTEST_SKIP() << "DCU ActivationImpl currently dispatches to gated "
                  "act_and_mul, so non-gated activation is not supported.";
  test_non_gated(torch::kBFloat16, "gelu", 128);
}

// ---- Edge cases ----

TEST_F(ActivationImplTest, GatedZerosInput) {
  auto opts = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto input = torch::zeros({2, 256}, opts);
  auto output = torch::empty({2, 128}, opts);

  ActivationImpl act("silu", /*is_gated=*/true);
  act.forward(input, output);

  // silu(0)*0 = 0
  EXPECT_TRUE(torch::allclose(output, torch::zeros_like(output)));
}

TEST_F(ActivationImplTest, GatedLargeBatch) {
  auto opts = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto input = torch::randn({256, 2048}, opts) * 0.5;
  auto output = torch::empty({256, 1024}, opts);

  ActivationImpl act("silu", /*is_gated=*/true);
  act.forward(input, output);

  auto expected = ref_gated_act(input, "silu");
  EXPECT_TRUE(torch::allclose(output, expected, 1e-2, 2e-2));
}

}  // namespace

}  // namespace test
}  // namespace layer
}  // namespace xllm
