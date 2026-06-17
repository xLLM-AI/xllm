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
#include <utility>
#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
constexpr int32_t kDeviceId = 0;

std::string dtype_name(torch::ScalarType dtype) {
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

std::pair<double, double> tolerance(torch::ScalarType dtype) {
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

// CPU reference: log_softmax(input / temperature, dim=-1)
// When temp=0, treat as 1.0 (matching kernel behavior).
torch::Tensor reference_log_softmax(const torch::Tensor& input,
                                    const torch::Tensor& temps) {
  auto x = input.to(torch::kFloat32);
  if (temps.defined()) {
    auto t = temps.clone();
    t.masked_fill_(t == 0.0f, 1.0f);
    x = x / t.unsqueeze(-1);
  }
  return torch::log_softmax(x, -1);
}

// ---------------------------------------------------------------------------
// Parameterized test case
// ---------------------------------------------------------------------------
struct SoftmaxCase {
  std::string name;
  int64_t batch;
  int64_t vocab;
  bool has_temps;
  bool has_zero_temp;
  torch::ScalarType dtype;
};

std::string print_case(const testing::TestParamInfo<SoftmaxCase>& info) {
  const auto& c = info.param;
  std::string tag =
      c.has_temps ? (c.has_zero_temp ? "TempZero" : "WithTemp") : "NoTemp";
  return std::to_string(c.batch) + "x" + std::to_string(c.vocab) + "_" + tag +
         "_" + dtype_name(c.dtype);
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class AirLogSoftmaxTest : public ::testing::TestWithParam<SoftmaxCase> {
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

TEST_P(AirLogSoftmaxTest, MatchesReference) {
  const auto& c = GetParam();
  auto opts = torch::TensorOptions().device(device_).dtype(c.dtype);

  // LLM logits are typically small; use narrow range to avoid fp16 overflow
  auto input = torch::randn({c.batch, c.vocab}, opts) * 2.0f;

  torch::Tensor temps;
  if (c.has_temps) {
    auto t_opts = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    temps = torch::rand({c.batch}, t_opts) * 2.0f + 0.1f;
    if (c.has_zero_temp) {
      // Set one element to 0.0 to trigger the temp==0 to temp=1.0 path.
      temps[0].zero_();
    }
  }

  auto output = xllm::kernel::cuda::air_log_softmax_last_dim(input, temps);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  auto expected =
      reference_log_softmax(input, temps.defined() ? temps : torch::Tensor());

  auto [rtol, atol] = tolerance(c.dtype);
  EXPECT_TRUE(torch::allclose(output, expected, rtol, atol))
      << "Mismatch: " << print_case(testing::TestParamInfo<SoftmaxCase>(c, 0));
}

// ---------------------------------------------------------------------------
// Generate cases
// ---------------------------------------------------------------------------
std::vector<SoftmaxCase> generate_cases() {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat32, torch::kFloat16, torch::kBFloat16};

  // The shared-memory kernel loads a full row into shared memory. The large-k
  // DCU fallback avoids that limit, so keep its batch small to avoid OOM.
  // These shapes cover typical LLM scenarios and the large-k fallback:
  //   - Small vocab / hidden dim:   [B, 256]
  //   - Medium vocab (GPT-2 style): [B, 5024]
  //   - Shared-memory boundary:     [B, 8192]
  //   - Large-vocab fallback:       [B, 128000]
  struct Shape {
    int64_t batch;
    int64_t vocab;
  };
  const std::vector<Shape> shapes = {
      {1, 256},   // minimal / decode single token
      {4, 5024},  // GPT-2 style medium vocab
      {8, 8192},  // near shared memory limit
      {1, 128000},
  };

  std::vector<SoftmaxCase> cases;
  for (auto dtype : dtypes) {
    for (const auto& s : shapes) {
      cases.push_back({"", s.batch, s.vocab, false, false, dtype});
      cases.push_back({"", s.batch, s.vocab, true, false, dtype});
      // temp=0 edge case only needs one shape per dtype
      if (s.batch == 4) {
        cases.push_back({"", s.batch, s.vocab, true, true, dtype});
      }
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(AirLogSoftmax,
                         AirLogSoftmaxTest,
                         testing::ValuesIn(generate_cases()),
                         print_case);

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu
