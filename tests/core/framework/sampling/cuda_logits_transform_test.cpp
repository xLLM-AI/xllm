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

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/kernels/cuda/fused_logits_transform.h"
#include "core/kernels/ops_api.h"

namespace xllm {
namespace {

SamplingParameters make_transform_params(int64_t rows) {
  SamplingParameters params;
  const auto options = torch::TensorOptions().device(torch::kCUDA);
  params.selected_token_idxes = torch::arange(rows, options.dtype(torch::kInt));
  params.sampling_mode = SamplingMode::RANDOM;
  return params;
}

torch::Tensor effective_top_p(const SamplingParameters& params,
                              torch::ScalarType dtype) {
  if (!params.top_p.defined()) {
    return {};
  }
  return params.top_k.defined() ? params.top_p.to(dtype) : params.top_p;
}

// Independent reference implementations of the now-private transform
// primitives: the reference must recompute, not reuse, the code under test.
void apply_min_p_reference(torch::Tensor& logits, const torch::Tensor& min_p) {
  if (!min_p.defined()) {
    return;
  }
  const torch::Tensor max_logits =
      std::get<0>(logits.max(/*dim=*/-1, /*keepdim=*/true));
  const torch::Tensor threshold =
      max_logits + min_p.to(torch::kFloat32).log().unsqueeze(-1);
  logits.masked_fill_(logits < threshold,
                      -std::numeric_limits<float>::infinity());
}

void apply_top_k_top_p_reference(torch::Tensor& logits,
                                 const torch::Tensor& top_k,
                                 const torch::Tensor& top_p) {
  if (!top_k.defined() && !top_p.defined()) {
    return;
  }
  const int64_t vocab_size = logits.size(-1);
  const float filter_value = -std::numeric_limits<float>::infinity();
  auto [sorted_logits, indices] = logits.sort(/*dim=*/-1, /*descending=*/false);
  if (top_k.defined()) {
    const torch::Tensor k = torch::where(top_k <= 0, vocab_size, top_k)
                                .clamp(1, vocab_size)
                                .to(torch::kLong);
    // Per-row threshold on every leading dim (k is [rows, 1] flat or
    // [batch, 1] grouped): expand the gather index so each row selects its
    // own kth-largest position.
    torch::Tensor threshold_index = (vocab_size - k).unsqueeze(-1);
    if (logits.dim() > 2) {
      threshold_index = threshold_index.expand(
          {logits.size(0), logits.size(1), threshold_index.size(-1)});
    }
    const torch::Tensor threshold =
        sorted_logits.gather(/*dim=*/-1, threshold_index);
    sorted_logits.masked_fill_(sorted_logits < threshold, filter_value);
  }
  if (top_p.defined()) {
    const torch::Tensor probabilities =
        sorted_logits.softmax(-1, torch::kFloat32);
    torch::Tensor mask = probabilities.cumsum(-1) <= (1 - top_p.unsqueeze(-1));
    mask.select(/*dim=*/-1, /*index=*/vocab_size - 1).fill_(false);
    sorted_logits.masked_fill_(mask, filter_value);
  }
  logits.scatter_(/*dim=*/-1, indices, sorted_logits);
}

torch::Tensor reference_transform(torch::Tensor logits,
                                  const SamplingParameters& params) {
  if (params.temperatures.defined()) {
    apply_temperatures(logits, params.temperatures);
  }
  apply_min_p_reference(logits, params.min_p);
  const auto top_p = params.top_k.defined() && params.top_p.defined()
                         ? params.top_p.to(logits.scalar_type())
                         : params.top_p;
  if (params.top_k.defined() || top_p.defined()) {
    apply_top_k_top_p_reference(logits, params.top_k, top_p);
  }
  return logits;
}

void expect_transform(const torch::Tensor& input,
                      const SamplingParameters& params) {
  auto expected = reference_transform(input.clone(), params);
  auto actual = input.clone();
  kernel::FusedLogitsTransformParams transform_params{
      actual,
      params.temperatures,
      params.min_p,
      params.top_k,
      effective_top_p(params, input.scalar_type())};
  ASSERT_TRUE(kernel::supports_fused_logits_transform(transform_params));
  kernel::fused_logits_transform(transform_params);
  ASSERT_TRUE(torch::equal(torch::isfinite(actual), torch::isfinite(expected)));
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-6));
  EXPECT_TRUE(torch::allclose(torch::softmax(actual, -1, torch::kFloat32),
                              torch::softmax(expected, -1, torch::kFloat32),
                              3e-5,
                              1e-6));
  auto logits_only = input.clone();
  process_logits(logits_only, params);
  EXPECT_TRUE(
      torch::equal(torch::isfinite(logits_only), torch::isfinite(expected)));
  EXPECT_TRUE(torch::allclose(logits_only, expected, 1e-5, 1e-6));
}

TEST(CudaLogitsTransformTest, CapabilityCheckRejectsUnsupportedInputs) {
  kernel::FusedLogitsTransformParams params;
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.logits = torch::ones({2, 8});
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.logits = params.logits.to(torch::kCUDA);
  const auto original = params.logits.clone();
  EXPECT_TRUE(kernel::supports_fused_logits_transform(params));

  params.temperatures = torch::ones({2});
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.temperatures = torch::ones({1}, params.logits.options());
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.temperatures =
      torch::ones({2}, params.logits.options().dtype(torch::kFloat64));
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.temperatures =
      torch::ones({4}, params.logits.options()).slice(0, 0, 4, 2);
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.temperatures = torch::Tensor();
  params.top_k = torch::ones({2}, params.logits.options().dtype(torch::kInt32));
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
  params.top_k = params.top_k.to(torch::kInt64);
  EXPECT_TRUE(kernel::supports_fused_logits_transform(params));
  EXPECT_TRUE(torch::equal(params.logits, original));

  params.logits = params.logits.slice(1, 0, 8, 2);
  EXPECT_FALSE(kernel::supports_fused_logits_transform(params));
}

TEST(CudaLogitsTransformTest, MatchesReferenceAcrossShapesAndMixedParameters) {
  torch::manual_seed(19);
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    for (const int64_t width : {1, 17, 255, 257, 2047, 2049, 32000, 131073}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(dtype);
      auto params = make_transform_params(6);
      const auto fp =
          params.selected_token_idxes.options().dtype(torch::kFloat32);
      params.temperatures =
          torch::tensor({0.0F, 0.7F, 1.0F, 1.5F, 0.01F, 0.8F}, fp);
      params.min_p = torch::tensor({0.0F, 0.1F, 0.5F, 0.0F, 0.0F, 1.0F}, fp);
      params.top_k =
          torch::tensor({0, 1, 4, -1, 999999, 32}, fp.dtype(torch::kLong));
      params.top_p = torch::tensor({1.0F, 0.9F, 0.1F, 0.0F, 0.99F, 0.8F}, fp);
      const auto input = torch::randn({6, width}, fp).to(dtype);
      expect_transform(input, params);
    }
  }
}

TEST(CudaLogitsTransformTest, PreservesRandomLargeBatchTopPBoundaries) {
  torch::manual_seed(19);
  auto params = make_transform_params(256);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    SCOPED_TRACE(dtype);
    const auto input = torch::randn({256, 131072}, fp).to(dtype);
    params.temperatures = torch::full({256}, 0.8F, fp);
    params.top_p = torch::full({256}, 0.9F, fp);
    expect_transform(input, params);
    // Place the cutoff on a framework cumulative-probability boundary. A
    // different reduction order must not change the surviving token set.
    params.temperatures = torch::Tensor();
    const auto sorted = std::get<0>(input.sort(-1));
    params.top_p = 1 - sorted.softmax(-1, torch::kFloat32)
                           .cumsum(-1)
                           .select(1, 65536)
                           .contiguous();
    expect_transform(input, params);
  }
}

TEST(CudaLogitsTransformTest, PreservesLargeVocabularyBoundaryTies) {
  torch::manual_seed(71);
  for (const int64_t rows : {1, 8, 32}) {
    auto params = make_transform_params(rows);
    const auto fp =
        params.selected_token_idxes.options().dtype(torch::kFloat32);
    for (const torch::ScalarType dtype :
         {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(dtype);
      params.top_p = torch::full({rows}, 0.9F, fp);
      expect_transform(torch::zeros({rows, 151936}, fp).to(dtype), params);
      // Many equal values straddle both the cumulative cutoff and block
      // boundaries. The survivors must match the native token ordering.
      const auto input = torch::randint(-3, 4, {rows, 151936}, fp).to(dtype);
      params.top_k = torch::arange(rows, fp.dtype(torch::kLong)) * 1024 + 1;
      expect_transform(input, params);
      params.top_k = torch::Tensor();
      const auto sorted = std::get<0>(input.sort(-1));
      const auto cumulative = sorted.softmax(-1, torch::kFloat32).cumsum(-1);
      params.top_p = (1 - cumulative.select(1, 65536)).contiguous();
      expect_transform(input, params);
    }
  }
}

TEST(CudaLogitsTransformTest, SelectsExactTopKWithMixedRanksAndTies) {
  torch::manual_seed(83);
  auto params = make_transform_params(48);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  for (const int64_t width : {4097, 32000, 151936}) {
    params.top_k =
        torch::tensor(std::vector<int64_t>{1, 40, 1024, 4096, width, 0},
                      fp.dtype(torch::kLong))
            .repeat({8});
    for (const torch::ScalarType dtype :
         {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(dtype);
      expect_transform(torch::randn({48, width}, fp).to(dtype), params);
      expect_transform(torch::randint(-8, 9, {48, width}, fp).to(dtype),
                       params);
    }
  }
}

TEST(CudaLogitsTransformTest, PreservesMinPBoundaryAndAdjacentValues) {
  auto params = make_transform_params(3);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.min_p = torch::tensor({0.1F, 0.4F, 0.9F}, fp);
  const auto boundary = params.min_p.log().unsqueeze(1);
  const auto input = torch::cat(
      {torch::zeros({3, 1}, fp),
       boundary,
       torch::nextafter(boundary, torch::full_like(boundary, -INFINITY)),
       torch::nextafter(boundary, torch::full_like(boundary, INFINITY))},
      1);
  expect_transform(input, params);
}

TEST(CudaLogitsTransformTest, PreservesSubnormalLogitsAndTemperatures) {
  auto params = make_transform_params(2);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.temperatures = torch::tensor({1.0F, 0x1p-130F}, fp);
  params.min_p = torch::tensor({1.0F, 0.0F}, fp);
  const auto input =
      torch::tensor({{0x1p-130F, 0x1p-129F, 0.0F}, {1e-20F, 2e-20F, 0.0F}}, fp);
  expect_transform(input, params);
  // Radix keys must distinguish adjacent subnormals and treat both signed
  // zeros as ties, including in mixed-rank rows.
  auto large_params = make_transform_params(64);
  large_params.top_k = torch::arange(64, fp.dtype(torch::kLong)) + 1;
  const auto tiny = torch::tensor(
      {-0.0F, 0.0F, 0x1p-149F, 0x1p-148F, -0x1p-149F, -0x1p-148F}, fp);
  const auto large = tiny.repeat({64, 1000}).slice(1, 0, 4097).contiguous();
  expect_transform(large, large_params);
  auto small_params = make_transform_params(8);
  small_params.top_k = large_params.top_k.slice(0, 0, 8);
  expect_transform(large.slice(0, 0, 8), small_params);
}

TEST(CudaLogitsTransformTest, ParameterDtypesAreIndependentOfLogits) {
  auto params = make_transform_params(3);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  for (const torch::ScalarType parameter_dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    params.temperatures =
        torch::tensor({0.0F, 0.7F, 1.2F}, fp).to(parameter_dtype);
    params.min_p = torch::tensor({0.0F, 0.1F, 0.4F}, fp).to(parameter_dtype);
    params.top_p = torch::tensor({1.0F, 0.9F, 0.1F}, fp).to(parameter_dtype);
    for (const torch::ScalarType logits_dtype :
         {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
      SCOPED_TRACE(parameter_dtype);
      SCOPED_TRACE(logits_dtype);
      params.top_k = torch::Tensor();
      const auto input = torch::randn({3, 4097}, fp).to(logits_dtype);
      expect_transform(input, params);
      params.top_k = torch::tensor({0, 40, 8}, fp.dtype(torch::kLong));
      expect_transform(input, params);
    }
  }
}

TEST(CudaLogitsTransformTest, SupportsEachOptionalTransformAndDisabledRows) {
  auto params = make_transform_params(3);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  const auto input = torch::arange(1031, fp).unsqueeze(0).repeat({3, 1}) / 100;
  expect_transform(input, params);
  params.temperatures = torch::tensor({0.0F, 1.0F, 0.7F}, fp);
  expect_transform(input, params);
  params.min_p = torch::tensor({0.0F, 0.1F, 1.0F}, fp);
  expect_transform(input, params);
  params.temperatures = torch::Tensor();
  params.min_p = torch::Tensor();
  params.top_k = torch::tensor({-1, 40, 2000}, fp.dtype(torch::kLong));
  expect_transform(input, params);
  params.top_k = torch::Tensor();
  params.top_p = torch::tensor({1.0F, 0.9F, 0.0F}, fp);
  expect_transform(input, params);
}

TEST(CudaLogitsTransformTest, PreservesTopKTiesAndTopPEqualityBoundary) {
  auto params = make_transform_params(2);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.top_k = torch::tensor({1, 0}, fp.dtype(torch::kLong));
  auto tied = torch::tensor(
      {{3.0F, 3.0F, 1.0F, -INFINITY}, {1.0F, 2.0F, 3.0F, 4.0F}}, fp);
  expect_transform(tied, params);
  process_logits(tied, params);
  EXPECT_EQ(tied[0].isfinite().sum().item<int64_t>(), 2);
  params.top_k = torch::Tensor();
  params.top_p = torch::full({2}, 0.5F, fp);
  for (const int64_t width : {4, 256, 2048, 4096, 131072}) {
    const auto input = torch::zeros({2, width}, fp);
    expect_transform(input, params);
    auto actual = input.clone();
    process_logits(actual, params);
    EXPECT_EQ(actual[0].isfinite().sum().item<int64_t>(), width / 2);
  }
}

TEST(CudaLogitsTransformTest, SparseRowsAndExtremeLogitsRemainNormalized) {
  auto params = make_transform_params(3);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  auto input = torch::full({3, 4097}, -INFINITY, fp);
  input[0][0] = 10000.0F;
  input[0][4096] = 9999.0F;
  input[1][256] = -10000.0F;
  input[1][257] = -10001.0F;
  input[2][241] = 0.0F;
  params.top_p = torch::tensor({0.9F, 1.0F, 0.5F}, fp);
  expect_transform(input, params);
}

TEST(CudaLogitsTransformTest, NoncontiguousRowsAndEmptyBatches) {
  auto params = make_transform_params(3);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  auto storage = torch::randn({6, 1031}, fp);
  auto input = storage.slice(0, 0, 6, 2);
  const auto untouched = storage.slice(0, 1, 6, 2).clone();
  params.top_k = torch::full({3}, 40, fp.dtype(torch::kLong));
  params.top_p = torch::full({3}, 0.9F, fp);
  const auto expected = reference_transform(input.clone(), params);
  kernel::cuda::fused_logits_transform(
      input, {}, {}, params.top_k, params.top_p);
  EXPECT_TRUE(torch::allclose(input, expected, 1e-5, 1e-6));
  EXPECT_TRUE(torch::equal(storage.slice(0, 1, 6, 2), untouched));
  EXPECT_TRUE(
      torch::allclose(input.softmax(-1), expected.softmax(-1), 3e-5, 1e-6));
  auto empty = torch::empty({0, 17}, fp);
  kernel::cuda::fused_logits_transform(empty, {}, {}, {}, {});
}

TEST(CudaLogitsTransformTest,
     CurrentStreamAndGraphReplayReadUpdatedParameters) {
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  for (const int64_t width : {1031, 151936}) {
    for (const int64_t rows : {4, 64}) {
      auto params = make_transform_params(rows);
      const auto fp =
          params.selected_token_idxes.options().dtype(torch::kFloat32);
      params.temperatures = torch::full({rows}, 0.8F, fp);
      params.min_p = torch::full({rows}, 0.9F, fp);
      params.top_k = torch::full({rows}, 40, fp.dtype(torch::kLong));
      params.top_p = torch::full({rows}, 0.9F, fp);
      const auto source =
          torch::arange(width, fp).unsqueeze(0).repeat({rows, 1}) / 100;
      auto input = source.clone();
      kernel::cuda::fused_logits_transform(
          input, params.temperatures, params.min_p, params.top_k, params.top_p);
      stream.synchronize();
      at::cuda::CUDAGraph graph;
      graph.capture_begin();
      input.copy_(source);
      kernel::cuda::fused_logits_transform(
          input, params.temperatures, params.min_p, params.top_k, params.top_p);
      graph.capture_end();
      for (const float p : {0.9F, 0.5F, 1.0F}) {
        params.top_k.fill_(p == 1.0F ? 0 : p == 0.5F ? 8192 : 40);
        params.top_p.fill_(p);
        params.min_p.fill_(p);
        params.temperatures.fill_(2.0F - p);
        graph.replay();
        const auto expected = reference_transform(source.clone(), params);
        EXPECT_TRUE(torch::allclose(input, expected, 1e-5, 1e-6));
        EXPECT_TRUE(torch::allclose(
            input.softmax(-1), expected.softmax(-1), 3e-5, 1e-6));
      }
    }
  }
}

TEST(CudaLogitsTransformTest, LargeStridedRowsPreservePaddingAndParameters) {
  torch::manual_seed(97);
  auto params = make_transform_params(64);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.top_k = torch::arange(64, fp.dtype(torch::kLong)) * 500;
  params.top_p = torch::full({64}, 0.9F, fp);
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    auto storage = torch::randn({128, 151936}, fp).to(dtype);
    auto input = storage.slice(0, 0, 128, 2);
    const auto padding = storage.slice(0, 1, 128, 2).clone();
    const auto expected = reference_transform(input.clone(), params);
    kernel::cuda::fused_logits_transform(
        input, {}, {}, params.top_k, effective_top_p(params, dtype));
    EXPECT_TRUE(
        torch::equal(torch::isfinite(input), torch::isfinite(expected)));
    EXPECT_TRUE(torch::allclose(input, expected, 1e-5, 1e-6));
    EXPECT_TRUE(torch::allclose(
        input.softmax(-1), expected.softmax(-1, torch::kFloat32), 3e-5, 1e-6));
    EXPECT_TRUE(torch::equal(storage.slice(0, 1, 128, 2), padding));
  }
}

TEST(CudaLogitsTransformTest, MatchesNativeSamplingAndGlobalRngConsumption) {
  torch::manual_seed(109);
  for (const int64_t rows : {8, 64}) {
    auto params = make_transform_params(rows);
    const auto fp =
        params.selected_token_idxes.options().dtype(torch::kFloat32);
    params.temperatures = torch::full({rows}, 0.8F, fp);
    params.top_k = torch::full({rows}, 40, fp.dtype(torch::kLong));
    params.top_p = torch::full({rows}, 0.9F, fp);
    params.sampling_mode = SamplingMode::MIXED;
    params.do_sample =
        torch::arange(rows, fp.dtype(torch::kLong)).remainder(2).eq(0);
    params.logprobs = true;
    params.max_top_logprobs = 5;
    for (const torch::ScalarType dtype :
         {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
      const auto input = torch::randn({rows, 151936}, fp).to(dtype);
      const auto expected = reference_transform(input.clone(), params);
      auto generator = torch::globalContext().defaultGenerator(input.device());
      const auto initial = generator.get_state();
      // forward()'s MIXED leg runs gumbel_noise + gumbel_argmax, which
      // consumes the same global-generator stream as gumbel_sample and
      // produces identical tokens.
      const auto random = gumbel_sample(expected);
      const auto tokens =
          torch::where(params.do_sample, random, expected.argmax(-1));
      const auto final = generator.get_state();
      generator.set_state(initial);
      auto actual = input.clone();
      const auto output = sample(actual, params);
      EXPECT_TRUE(torch::equal(tokens, output.next_tokens));
      EXPECT_TRUE(torch::equal(final, generator.get_state()));
      const auto raw = input.log_softmax(-1, torch::kFloat32);
      EXPECT_TRUE(torch::equal(output.logprobs,
                               raw.gather(1, tokens.unsqueeze(1)).squeeze(1)));
      EXPECT_TRUE(torch::equal(output.top_tokens, std::get<1>(raw.topk(5))));
      EXPECT_TRUE(torch::equal(output.top_logprobs, std::get<0>(raw.topk(5))));
    }
  }
}

TEST(CudaLogitsTransformTest, SamplerPreservesRawLogprobsPenaltiesAndMasks) {
  auto params = make_transform_params(2);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.temperatures = torch::full({2}, 0.8F, fp);
  params.top_k = torch::full({2}, 3, fp.dtype(torch::kLong));
  params.top_p = torch::full({2}, 0.9F, fp);
  params.seeds = torch::tensor({42, 43}, fp.dtype(torch::kLong));
  params.seed_offsets = torch::zeros({2}, params.seeds.options());
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.return_logits = true;
  params.sampling_mode = SamplingMode::MIXED;
  params.do_sample = torch::tensor({true, false}, fp.dtype(torch::kBool));
  params.logits_bias =
      torch::tensor({{-INFINITY, 0.0F, 1.0F, 0.0F}}, fp).repeat({2, 1});
  const auto input =
      torch::tensor({{4.0F, 3.0F, 2.0F, 1.0F}}, fp).repeat({2, 1});
  auto penalized = input.clone();
  apply_sampling_penalties(penalized, params);
  const auto expected = reference_transform(penalized, params);
  const auto raw = input.log_softmax(-1);
  const auto& generator =
      torch::globalContext().defaultGenerator(input.device());
  const auto before = generator.get_state().clone();
  auto logits = input.clone();
  const auto result = sample(logits, params);
  EXPECT_TRUE(torch::allclose(result.draft_logits, expected, 1e-5, 1e-6));
  EXPECT_TRUE(torch::allclose(
      result.logprobs,
      raw.gather(1, result.next_tokens.unsqueeze(1)).squeeze(1)));
  EXPECT_TRUE(torch::equal(result.top_tokens, std::get<1>(raw.topk(2))));
  EXPECT_EQ(result.next_tokens[1].item<int64_t>(),
            expected[1].argmax().item<int64_t>());
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
  logits = input.clone();
  const auto repeat = sample(logits, params);
  EXPECT_TRUE(torch::equal(result.next_tokens, repeat.next_tokens));
  params.filter_bitmask = torch::tensor({{8}, {8}}, fp.dtype(torch::kInt));
  logits = input.clone();
  EXPECT_TRUE(torch::equal(sample(logits, params).next_tokens,
                           torch::full({2}, 3, params.seeds.options())));
}

TEST(CudaLogitsTransformTest, UnseededDistributionExcludesFilteredTokens) {
  constexpr int64_t kRows = 16384;
  auto params = make_transform_params(kRows);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.temperatures = torch::full({kRows}, 0.8F, fp);
  params.top_k = torch::full({kRows}, 3, fp.dtype(torch::kLong));
  params.top_p = torch::full({kRows}, 0.9F, fp);
  auto input =
      torch::tensor({{1.0F, 2.0F, 3.0F, 4.0F, 5.0F}}, fp).repeat({kRows, 1});
  const auto expected = reference_transform(input.clone(), params).softmax(-1);
  const auto& generator =
      torch::globalContext().defaultGenerator(input.device());
  const auto before = generator.get_state().clone();
  const auto sampled = sample(input, params).next_tokens;
  EXPECT_FALSE(torch::equal(before, generator.get_state()));
  const auto counts = torch::bincount(sampled, {}, 5).cpu();
  const auto probabilities = expected[0].cpu();
  for (int32_t token = 0; token < 5; ++token) {
    if (probabilities[token].item<float>() == 0.0F) {
      EXPECT_EQ(counts[token].item<int64_t>(), 0);
    }
    EXPECT_NEAR(counts[token].item<int64_t>() / static_cast<double>(kRows),
                probabilities[token].item<float>(),
                0.025);
  }
}

TEST(CudaLogitsTransformTest,
     GreedySkipsTransformsAndNoncontiguousColumnsUseTorch) {
  auto params = make_transform_params(2);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  params.temperatures = torch::full({2}, 0.7F, fp);
  params.top_k = torch::full({2}, 2, fp.dtype(torch::kLong));
  params.top_p = torch::full({2}, 0.9F, fp);
  const auto source = torch::arange(16, fp).view({2, 8});
  params.sampling_mode = SamplingMode::GREEDY;
  auto input = source.clone();
  const auto output = sample(input, params);
  EXPECT_TRUE(torch::equal(output.next_tokens, source.argmax(-1)));
  EXPECT_TRUE(torch::equal(input, source));
  params.sampling_mode = SamplingMode::RANDOM;
  input = source.clone().slice(1, 0, 8, 2);
  const auto expected = reference_transform(input.clone(), params);
  process_logits(input, params);
  EXPECT_TRUE(torch::allclose(input, expected, 1e-5, 1e-6));
}

TEST(CudaLogitsTransformTest, UnsupportedParameterLayoutsAndDtypesUseTorch) {
  auto params = make_transform_params(2);
  const auto fp = params.selected_token_idxes.options().dtype(torch::kFloat32);
  const auto source = torch::randn({2, 257}, fp);
  params.temperatures =
      torch::tensor({0.7F, 1.0F, 1.2F, 1.0F}, fp).slice(0, 0, 4, 2);
  params.min_p = torch::tensor({0.1F, 0.0F, 0.2F, 0.0F}, fp).slice(0, 0, 4, 2);
  params.top_k =
      torch::tensor({40, 0, 8, 0}, fp.dtype(torch::kLong)).slice(0, 0, 4, 2);
  params.top_p = torch::tensor({0.9F, 1.0F, 0.8F, 1.0F}, fp).slice(0, 0, 4, 2);
  auto actual = source.clone();
  process_logits(actual, params);
  EXPECT_TRUE(
      torch::equal(actual, reference_transform(source.clone(), params)));

  params.temperatures = params.temperatures.to(torch::kFloat64);
  params.min_p = torch::Tensor();
  params.top_k = torch::Tensor();
  params.top_p = params.top_p.to(torch::kFloat64);
  actual = source.clone();
  process_logits(actual, params);
  EXPECT_TRUE(
      torch::equal(actual, reference_transform(source.clone(), params)));

  params.temperatures = torch::Tensor();
  params.top_p = torch::Tensor();
  params.min_p = torch::full({}, 0.1F, fp);
  actual = source.clone();
  process_logits(actual, params);
  EXPECT_TRUE(
      torch::equal(actual, reference_transform(source.clone(), params)));
}

TEST(CudaLogitsTransformDeathTest, RejectsInvalidParameterDeviceDtypeAndShape) {
  EXPECT_DEATH(
      {
        auto params = make_transform_params(2);
        auto logits = torch::ones(
            {2, 4}, params.selected_token_idxes.options().dtype(torch::kFloat));
        kernel::cuda::fused_logits_transform(
            logits, torch::ones({2}), {}, {}, {});
      },
      "device");
  EXPECT_DEATH(
      {
        auto params = make_transform_params(2);
        auto logits = torch::ones(
            {2, 4}, params.selected_token_idxes.options().dtype(torch::kFloat));
        kernel::cuda::fused_logits_transform(
            logits, {}, {}, params.selected_token_idxes, {});
      },
      "scalar_type");
  EXPECT_DEATH(
      {
        auto params = make_transform_params(2);
        auto logits = torch::ones(
            {2, 4}, params.selected_token_idxes.options().dtype(torch::kFloat));
        const auto temperatures = torch::ones({1}, logits.options());
        kernel::cuda::fused_logits_transform(logits, temperatures, {}, {}, {});
      },
      "size");
  EXPECT_DEATH(
      {
        auto params = make_transform_params(2);
        auto logits =
            torch::ones(
                {2, 8},
                params.selected_token_idxes.options().dtype(torch::kFloat))
                .slice(1, 0, 8, 2);
        kernel::cuda::fused_logits_transform(logits, {}, {}, {}, {});
      },
      "stride");
}

}  // namespace
}  // namespace xllm
