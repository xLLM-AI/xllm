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

#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/hip/HIPStream.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "kernels/dcu/dcu_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {

namespace {

constexpr int32_t kDeviceId = 0;

at::Generator get_default_generator(c10::DeviceIndex device_index) {
  static std::unordered_map<c10::DeviceIndex, at::Generator> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device_index);
  if (it != cache.end()) {
    return it->second;
  }
  at::globalContext().lazyInitCUDA();
  at::Generator gen = at::cuda::detail::getDefaultCUDAGenerator(device_index);
  cache.emplace(device_index, gen);
  return gen;
}

// ---------------------------------------------------------------------------
// Test case descriptor
// ---------------------------------------------------------------------------
struct SampleCase {
  std::string name;
  int64_t batch_size;
  int64_t vocab_size;
};

std::string PrintSampleCase(const testing::TestParamInfo<SampleCase>& info) {
  return info.param.name;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class RandomSampleDcuTest : public ::testing::TestWithParam<SampleCase> {
 protected:
  void SetUp() override {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
      GTEST_SKIP() << "No DCU/HIP device available";
    }
    ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
    device_ = torch::Device(c10::DeviceType::CUDA, kDeviceId);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

// ---------------------------------------------------------------------------
// TEST 1: Sampling range validity — all sampled indices must be in
// [0, vocab_size). Matches FlashInfer's test_sampling.
// ---------------------------------------------------------------------------
TEST_P(RandomSampleDcuTest, SamplingRangeValidity) {
  const auto& c = GetParam();
  torch::manual_seed(42);
  constexpr int64_t kNumTrials = 5000;

  auto logits = torch::randn({c.batch_size, c.vocab_size}, torch::kFloat32);
  auto probs_cpu = torch::softmax(logits, /*dim=*/-1);
  auto probs_gpu = probs_cpu.to(device_);

  for (int64_t t = 0; t < kNumTrials; ++t) {
    auto output_gpu = dcu::random_sample(probs_gpu);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto output_cpu = output_gpu.cpu().to(torch::kInt64);
    auto acc = output_cpu.accessor<int64_t, 1>();
    for (int64_t b = 0; b < c.batch_size; ++b) {
      EXPECT_GE(acc[b], 0) << "batch=" << b << " trial=" << t;
      EXPECT_LT(acc[b], c.vocab_size) << "batch=" << b << " trial=" << t;
    }
  }
}

// ---------------------------------------------------------------------------
// TEST 2: Sampling frequency — cosine similarity between empirical frequency
// and true probability must exceed 0.99.
// Matches FlashInfer's test_sampling_freq (batch=1 only).
// Uses chunked expansion to batch multiple samples per kernel call.
// ---------------------------------------------------------------------------
TEST_P(RandomSampleDcuTest, SamplingFrequency) {
  const auto& c = GetParam();
  if (c.batch_size != 1) {
    GTEST_SKIP() << "Frequency test only runs for batch=1";
  }

  torch::manual_seed(42);
  auto logits = torch::randn({1, c.vocab_size}, torch::kFloat32);
  auto probs_cpu = torch::softmax(logits, /*dim=*/-1).contiguous();
  auto probs_gpu = probs_cpu.to(device_);

  constexpr int64_t kNumTrials = 5000000;
  const int64_t kChunkSize = std::min<int64_t>(
      5000LL,
      200000000LL / (c.vocab_size * static_cast<int64_t>(sizeof(float))));

  std::vector<int64_t> hist(c.vocab_size, 0);

  for (int64_t start = 0; start < kNumTrials; start += kChunkSize) {
    int64_t n = std::min(kChunkSize, kNumTrials - start);
    auto batch_probs = probs_gpu.repeat({n, 1});
    auto output = dcu::random_sample(batch_probs);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto output_cpu = output.cpu().to(torch::kInt64);
    auto acc = output_cpu.accessor<int64_t, 1>();
    for (int64_t i = 0; i < n; ++i) {
      hist[acc[i]] += 1;
    }
  }

  // Compute cosine similarity between empirical frequency and true probs
  double dot = 0, norm_freq = 0, norm_probs = 0;
  auto probs_acc = probs_cpu.accessor<float, 2>();
  for (int64_t i = 0; i < c.vocab_size; ++i) {
    double f = static_cast<double>(hist[i]) / kNumTrials;
    double p = static_cast<double>(probs_acc[0][i]);
    dot += f * p;
    norm_freq += f * f;
    norm_probs += p * p;
  }
  double similarity =
      dot / (std::sqrt(norm_freq) * std::sqrt(norm_probs) + 1e-12);

  EXPECT_GT(similarity, 0.99)
      << "Cosine similarity too low for " << c.name << ": " << similarity;
}

// ---------------------------------------------------------------------------
// TEST 3: Determinism — same probs + same seed => same output (exact match).
// Matches FlashInfer's test_sampling_from_probs_seed_offset_reproducibility.
// ---------------------------------------------------------------------------
TEST_P(RandomSampleDcuTest, Determinism) {
  const auto& c = GetParam();
  torch::manual_seed(42);

  auto logits = torch::randn({c.batch_size, c.vocab_size}, torch::kFloat32);
  auto probs_cpu = torch::softmax(logits, /*dim=*/-1);

  auto gen = get_default_generator(device_.index());
  {
    std::lock_guard<std::mutex> lock(gen.mutex());
    auto* cuda_gen = at::check_generator<at::CUDAGeneratorImpl>(gen);
    cuda_gen->set_current_seed(12345);
  }

  auto probs_gpu = probs_cpu.to(device_);
  auto out1 = dcu::random_sample(probs_gpu);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  {
    std::lock_guard<std::mutex> lock(gen.mutex());
    auto* cuda_gen = at::check_generator<at::CUDAGeneratorImpl>(gen);
    cuda_gen->set_current_seed(12345);
  }

  auto out2 = dcu::random_sample(probs_gpu);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  EXPECT_TRUE(torch::equal(out1, out2))
      << "Non-deterministic output for " << c.name;
}

// ---------------------------------------------------------------------------
// TEST 4: Edge case — all probability on one token (degenerate distribution)
// ---------------------------------------------------------------------------
TEST_P(RandomSampleDcuTest, DegenerateOneHot) {
  const auto& c = GetParam();

  auto probs_cpu = torch::zeros({c.batch_size, c.vocab_size}, torch::kFloat32);
  auto probs_acc = probs_cpu.accessor<float, 2>();
  for (int64_t b = 0; b < c.batch_size; ++b) {
    probs_acc[b][(b * 7 + 3) % c.vocab_size] = 1.0f;
  }

  auto probs_gpu = probs_cpu.to(device_);
  auto output = dcu::random_sample(probs_gpu);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  auto output_cpu = output.cpu().to(torch::kInt32);

  auto out_acc = output_cpu.accessor<int32_t, 1>();
  for (int64_t b = 0; b < c.batch_size; ++b) {
    int32_t expected = static_cast<int32_t>((b * 7 + 3) % c.vocab_size);
    EXPECT_EQ(out_acc[b], expected)
        << "One-hot mismatch batch=" << b << " expected=" << expected
        << " got=" << out_acc[b];
  }
}

// ---------------------------------------------------------------------------
// Generate test cases matching FlashInfer's parameter space:
//   batch_size in {1, 99, 989}
//   vocab_size in {111, 32000, 128256}
// ---------------------------------------------------------------------------
std::vector<SampleCase> GenerateCases() {
  return {
      {"B1_V111", 1, 111},
      {"B99_V111", 99, 111},
      {"B989_V111", 989, 111},
      {"B1_V32000", 1, 32000},
      {"B99_V32000", 99, 32000},
      {"B989_V32000", 989, 32000},
      {"B1_V128256", 1, 128256},
      {"B99_V128256", 99, 128256},
      {"B989_V128256", 989, 128256},
  };
}

INSTANTIATE_TEST_SUITE_P(RandomSampleDcu,
                         RandomSampleDcuTest,
                         testing::ValuesIn(GenerateCases()),
                         PrintSampleCase);

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu
