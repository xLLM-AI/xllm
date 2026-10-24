/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#if defined(USE_CUDA) && !defined(USE_MACA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "core/kernels/cuda/cuda_ops_api.h"
#endif

#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

#if defined(USE_CUDA) && !defined(USE_MACA)
class TvmFfiStreamGuard final {
 public:
  explicit TvmFfiStreamGuard(const c10::cuda::CUDAStream& stream)
      : device_index_(stream.device_index()) {
    CHECK_EQ(TVMFFIEnvSetStream(kDLCUDA,
                                device_index_,
                                reinterpret_cast<void*>(stream.stream()),
                                &previous_stream_),
             0);
  }

  ~TvmFfiStreamGuard() {
    CHECK_EQ(
        TVMFFIEnvSetStream(kDLCUDA, device_index_, previous_stream_, nullptr),
        0);
  }

  TvmFfiStreamGuard(const TvmFfiStreamGuard&) = delete;
  TvmFfiStreamGuard& operator=(const TvmFfiStreamGuard&) = delete;

 private:
  int32_t device_index_;
  TVMFFIStreamHandle previous_stream_ = nullptr;
};
#endif

SamplingParameters make_mixed_params(int64_t rows,
                                     const torch::Device& device,
                                     torch::ScalarType dtype,
                                     int32_t random_percent = 25) {
  RequestSamplingParam greedy;
  greedy.temperature = 0;
  RequestSamplingParam random;
  random.temperature = 0.8F;
  random.top_k = 40;
  random.top_p = 0.9F;
  random.min_p = 0.01F;
  std::vector<const RequestSamplingParam*> requests;
  std::vector<int32_t> selected;
  requests.reserve(rows);
  selected.reserve(rows);
  for (int64_t row = 0; row < rows; ++row) {
    requests.emplace_back((row + 3) % 4 < random_percent / 25 ? &random
                                                              : &greedy);
    selected.emplace_back(static_cast<int32_t>(row));
  }
  SamplingParameters params;
  params.init(requests, selected, {}, {}, {});
  return params.to(device, dtype);
}

void expect_same_sampling(const torch::Tensor& input,
                          const SamplingParameters& params,
                          const torch::Tensor& mask = {}) {
  SamplingParameters reference = params;
  reference.num_random_sample_rows = -1;
  auto generator = torch::globalContext().defaultGenerator(input.device());
  const auto state = generator.get_state().clone();
  auto original_logits = input.clone();
  const auto expected = Sampler().forward(original_logits, reference, mask);
  const auto expected_state = generator.get_state().clone();
  generator.set_state(state);
  auto actual_logits = input.clone();
  const auto actual = Sampler().forward(actual_logits, params, mask);
  EXPECT_TRUE(torch::equal(actual.next_tokens, expected.next_tokens));
  EXPECT_TRUE(torch::equal(generator.get_state(), expected_state));
  for (const auto member : {&SampleOutput::logprobs,
                            &SampleOutput::top_tokens,
                            &SampleOutput::top_logprobs,
                            &SampleOutput::draft_logits}) {
    ASSERT_EQ((actual.*member).defined(), (expected.*member).defined());
    if ((expected.*member).defined()) {
      EXPECT_TRUE(torch::equal(actual.*member, expected.*member));
    }
  }
  const auto random_rows = params.require_do_sample().nonzero().flatten();
  EXPECT_TRUE(torch::equal(actual_logits.index_select(0, random_rows),
                           original_logits.index_select(0, random_rows)));
}

void check_mixed_sampling(const torch::Device& device) {
#if defined(USE_CUDA) && !defined(USE_MACA)
  if (device.is_cuda()) {
    kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);
  }
#endif
  torch::manual_seed(19);
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    for (const int64_t rows : {8, 33, 63, 64}) {
      SCOPED_TRACE(rows);
      const int64_t large_vocab = rows == 64 ? 131073 : 4097;
      for (const int64_t vocab : {int64_t{257}, large_vocab}) {
        auto params = make_mixed_params(rows, device, dtype);
        const auto options = torch::TensorOptions().device(device).dtype(dtype);
        const auto input = torch::randn({rows, vocab}, options);
        expect_same_sampling(input, params);
        params.seeds = torch::full({rows}, 42, options.dtype(torch::kInt64));
        params.seed_offsets = torch::arange(rows, options.dtype(torch::kInt64));
        expect_same_sampling(input, params);
        params.seeds.slice(0, 1, rows, 8).fill_(-1);
        expect_same_sampling(input, params);
        expect_same_sampling(torch::zeros_like(input), params);
        params.logprobs = true;
        params.max_top_logprobs = 5;
        expect_same_sampling(input, params);
        params.logprobs = false;
        params.return_logits = true;
        expect_same_sampling(input, params);
        params.return_logits = false;
        params.use_beam_search = true;
        expect_same_sampling(input, params);
      }
    }
  }
}

TEST(MixedSamplerTest,
     CpuMatchesFullRowsAndPreservesRngAndDistributionOutputs) {
  check_mixed_sampling(torch::Device(torch::kCPU));
}

TEST(MixedSamplerTest, ConstraintsAndUnsupportedLayoutsPreserveSampling) {
  auto params =
      make_mixed_params(64, torch::Device(torch::kCPU), torch::kFloat32);
  params.logits_bias = torch::zeros({64, 131073});
  params.logits_bias.select(1, 0).fill_(-INFINITY);
  params.min_token_ids = torch::tensor({{0, -1},
                                        {1, 1},
                                        {2, -1},
                                        {3, -1},
                                        {4, -1},
                                        {5, -1},
                                        {6, -1},
                                        {7, -1}},
                                       torch::kInt64)
                             .repeat({8, 1});
  params.min_tokens_remaining = torch::ones({64}, torch::kInt64);
  params.unique_token_ids = torch::zeros({64, 1}, torch::kInt64);
  params.unique_token_counts = torch::ones({64, 1}, torch::kInt32);
  params.unique_token_ids_lens = torch::ones({64}, torch::kInt32);
  params.repetition_penalties = torch::full({64}, 1.2F);
  params.frequency_penalties = torch::full({64}, 0.1F);
  params.presence_penalties = torch::full({64}, 0.2F);
  const auto input = torch::randn({64, 131073});
  const auto mask = torch::zeros_like(input);
  mask.select(1, 64).fill_(-INFINITY);
  expect_same_sampling(input, params, mask);
  params.top_k = torch::Tensor();
  expect_same_sampling(input, params, mask);
  params.top_k = torch::full({64}, 40, torch::kInt64);
  params.top_p = torch::Tensor();
  expect_same_sampling(input, params, mask);
  params.min_p = torch::tensor(0.1F);
  expect_same_sampling(input, params, mask);
  params.min_p = torch::Tensor();
  params.top_k = torch::Tensor();
  params.top_p = torch::Tensor();
  expect_same_sampling(input, params, mask);
}

#if defined(USE_CUDA) && !defined(USE_MACA)
TEST(MixedSamplerTest,
     CudaMatchesFullRowsAndPreservesRngAndDistributionOutputs) {
  ASSERT_TRUE(torch::cuda::is_available());
  check_mixed_sampling(torch::Device(torch::kCUDA, 0));
}

TEST(MixedSamplerTest, CudaUsesCurrentStreamAcrossFusedBatchThresholds) {
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  const TvmFfiStreamGuard ffi_guard(stream);
  const torch::Device device(torch::kCUDA, 0);
  for (const int64_t rows : {64, 128, 256}) {
    SCOPED_TRACE(rows);
    const auto input = torch::randn(
        {rows, 151936},
        torch::TensorOptions().device(device).dtype(torch::kBFloat16));
    for (const int32_t random_percent : {25, 50, 75}) {
      SCOPED_TRACE(random_percent);
      auto params =
          make_mixed_params(rows, device, torch::kBFloat16, random_percent);
      expect_same_sampling(input, params);
      expect_same_sampling(torch::zeros_like(input), params);
      params.do_sample = params.do_sample.roll(1, 0);
      params.temperatures = params.temperatures.roll(1, 0);
      params.min_p = params.min_p.roll(1, 0);
      params.top_k = params.top_k.roll(1, 0);
      params.top_p = params.top_p.roll(1, 0);
      expect_same_sampling(input, params);
    }
  }
}
#endif

}  // namespace
}  // namespace xllm
