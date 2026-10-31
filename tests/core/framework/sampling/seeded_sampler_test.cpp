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
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include <array>
#include <cmath>
#include <limits>

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm {
namespace {

TEST(SeededSamplerTest, NpuVerificationMatchesOrdinaryDrawsAtTokenOffsets) {
  constexpr int64_t kRows = 8;
  constexpr int64_t kSteps = 3;
  constexpr int64_t kVocab = 33;
  const auto options = torch::TensorOptions().device(torch::kPrivateUse1, 0);
  const torch::Tensor original =
      torch::arange(kRows * (kSteps + 1) * kVocab, options)
          .sin()
          .view({kRows, kSteps + 1, kVocab});
  for (const SamplingMode mode : {SamplingMode::RANDOM, SamplingMode::MIXED}) {
    SamplingParameters params;
    params.sampling_mode() = mode;
    params.seeds() = torch::arange(kRows, options.dtype(torch::kInt64)).add(42);
    params.seed_offsets() = torch::arange(kRows, params.seeds().options());
    if (mode == SamplingMode::MIXED) {
      params.do_sample() = params.seeds().remainder(2).eq(0);
    }
    torch::Tensor expected =
        torch::empty({kRows, kSteps + 1}, params.seeds().options());
    for (int64_t column = 0; column <= kSteps; ++column) {
      SamplingParameters step_params = params;
      step_params.seed_offsets() = params.seed_offsets() + column;
      torch::Tensor logits = original.select(1, column).clone();
      expected.select(1, column).copy_(sample(logits, step_params).next_tokens);
    }
    const torch::Tensor tokens = expected.narrow(1, 0, kSteps).clone();
    const DraftProposal proposal(tokens);
    const SampleOutput accepted =
        rejection_sample(params, proposal, original.clone().view({-1, kVocab}));
    EXPECT_TRUE(torch::equal(accepted.next_tokens, expected));
    EXPECT_TRUE(torch::equal(
        accepted.num_accepted_tokens,
        torch::full({kRows, 1}, kSteps, options.dtype(torch::kInt32))));

    const SampleOutput bonus = rejection_sample(
        params,
        DraftProposal(torch::empty({kRows, 0}, params.seeds().options())),
        original.select(1, 0).clone());
    EXPECT_TRUE(torch::equal(bonus.next_tokens, expected.narrow(1, 0, 1)));
    EXPECT_TRUE(
        torch::equal(bonus.num_accepted_tokens,
                     torch::zeros({kRows, 1}, options.dtype(torch::kInt32))));

    const torch::Tensor invalid =
        torch::zeros({kRows, kSteps}, options.dtype(torch::kBool));
    invalid.select(1, 1).fill_(true);
    const SampleOutput rejected = rejection_sample(
        params, proposal, original.clone().view({-1, kVocab}), invalid);
    expected.narrow(1, 2, kSteps - 1).fill_(-1);
    EXPECT_TRUE(torch::equal(rejected.next_tokens, expected));
    EXPECT_TRUE(
        torch::equal(rejected.num_accepted_tokens,
                     torch::ones({kRows, 1}, options.dtype(torch::kInt32))));
  }
}

TEST(SeededSamplerTest, NpuVerificationLogprobsUseRawLogitsWithBiasAndTopK) {
  constexpr int64_t kRows = 64;
  constexpr int64_t kVocab = 65536;
  const auto options = torch::TensorOptions()
                           .device(torch::kPrivateUse1, 0)
                           .dtype(torch::kBFloat16);
  const torch::Tensor original =
      torch::arange(kVocab, options.dtype(torch::kFloat32))
          .remainder(17)
          .to(torch::kBFloat16)
          .repeat({kRows * 3, 1});
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::RANDOM;
  params.logprobs() = true;
  params.logits_bias() = torch::zeros({kRows, kVocab}, options);
  params.logits_bias().select(1, 0).fill_(100.0F);
  params.top_k() = torch::ones({kRows}, options.dtype(torch::kInt64));
  const DraftProposal proposal(
      torch::zeros({kRows, 2}, options.dtype(torch::kInt64)));
  const torch::Tensor expected =
      torch::log_softmax(original, -1, torch::kFloat32)
          .select(1, 0)
          .view({kRows, 3});
  for (const int64_t top_logprobs : {int64_t{0}, int64_t{5}}) {
    params.max_top_logprobs() = top_logprobs;
    const SampleOutput result =
        rejection_sample(params, proposal, original.clone());
    EXPECT_TRUE(
        torch::equal(result.next_tokens,
                     torch::zeros({kRows, 3}, options.dtype(torch::kInt64))));
    EXPECT_TRUE(torch::allclose(result.logprobs, expected, 1e-5, 5e-6));
    if (top_logprobs > 0) {
      const auto [values, ids] =
          torch::log_softmax(
              original.view({kRows, 3, kVocab}), -1, torch::kFloat32)
              .topk(top_logprobs, -1);
      EXPECT_TRUE(torch::equal(result.top_tokens, ids));
      EXPECT_TRUE(torch::equal(result.top_logprobs, values));
    }
  }
}

SamplingParameters make_seeded_params(const torch::Tensor& seeds,
                                      const torch::Tensor& offsets) {
  const int64_t rows = seeds.numel();
  SamplingParameters params;
  params.selected_token_idxes() = torch::arange(rows, torch::kInt);
  params.sampling_mode() = SamplingMode::RANDOM;
  params.seeds() = seeds;
  params.seed_offsets() = offsets;
  return params.to(torch::Device(torch::kPrivateUse1, 0), torch::kFloat32);
}

TEST(SeededSamplerTest, CategoricalEntryPreservesMaskedSupportAcrossLayouts) {
  const torch::Device device(torch::kPrivateUse1, 0);
  for (const int64_t rows : {int64_t{1}, int64_t{8}}) {
    torch::Tensor storage = torch::full({rows, 14},
                                        -std::numeric_limits<float>::infinity(),
                                        torch::TensorOptions().device(device));
    storage.select(/*dim=*/1, /*index=*/4).fill_(0.0F);
    const torch::Tensor contiguous_inner = storage.narrow(/*dim=*/1, 0, 7);
    const torch::Tensor strided_inner = storage.slice(/*dim=*/1, 0, 14, 2);
    EXPECT_TRUE(torch::equal(
        categorical_sample(contiguous_inner),
        torch::full({rows}, 4, storage.options().dtype(torch::kInt64))));
    EXPECT_TRUE(torch::equal(
        categorical_sample(strided_inner),
        torch::full({rows}, 2, storage.options().dtype(torch::kInt64))));
  }
}

TEST(SeededSamplerTest, NpuRequestsAreReproducibleAndIsolated) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto& global_generator = at_npu::detail::getDefaultNPUGenerator(0);
  const torch::Tensor before = global_generator.get_state().clone();
  for (int64_t offset = 0; offset < 8; ++offset) {
    const auto single =
        make_seeded_params(torch::tensor({42}, torch::kLong),
                           torch::tensor({offset}, torch::kLong));
    const auto batch =
        make_seeded_params(torch::tensor({7, 42}, torch::kLong),
                           torch::tensor({int64_t{1}, offset}, torch::kLong));
    torch::Tensor logits =
        torch::zeros({1, 64}, torch::TensorOptions().device(device));
    torch::Tensor batch_logits = torch::zeros({2, 64}, logits.options());
    const auto first = sample(logits, single);
    const auto repeated = sample(logits, single);
    const auto batched = sample(batch_logits, batch);
    EXPECT_EQ(first.next_tokens[0].item<int64_t>(),
              repeated.next_tokens[0].item<int64_t>());
    EXPECT_EQ(first.next_tokens[0].item<int64_t>(),
              batched.next_tokens[1].item<int64_t>());
  }
  EXPECT_TRUE(torch::equal(before, global_generator.get_state()));
}

#if defined(XLLM_NPU_SAMPLING_ABLATION)
TEST(SeededSamplerTest,
     NpuWeightsSamplingHandlesTailsZerosAndUniformEndpoints) {
  const torch::Device device(torch::kPrivateUse1, 0);
  for (const int64_t vocab : {int64_t{1},
                              int64_t{31},
                              int64_t{33},
                              int64_t{4095},
                              int64_t{4096},
                              int64_t{4097},
                              int64_t{248320}}) {
    constexpr int64_t kRows = 49;
    // Include padding so the kernel must honor row_stride, not just vocab.
    torch::Tensor storage = torch::zeros({kRows, vocab + 8});
    torch::Tensor weights = storage.narrow(1, 0, vocab);
    weights.select(1, 0).fill_(0.25F);
    weights.select(1, vocab - 1).fill_(0.75F);
    torch::Tensor uniforms = torch::linspace(0.0, 1.0, kRows);
    uniforms.select(0, kRows - 1).fill_(1.0F - 0x1p-24F);
    const torch::Tensor expected =
        vocab == 1
            ? torch::zeros({kRows}, torch::kInt64)
            : torch::where(uniforms.lt(0.25F), 0, vocab - 1).to(torch::kInt64);
    const torch::Tensor device_storage = storage.to(device);
    const torch::Tensor actual = kernel::npu::tilelang::seeded_sample(
        device_storage.narrow(1, 0, vocab),
        torch::full({kRows},
                    -1,
                    torch::TensorOptions().dtype(torch::kInt64).device(device)),
        torch::zeros(
            {kRows},
            torch::TensorOptions().dtype(torch::kInt64).device(device)),
        uniforms.to(device));
    EXPECT_TRUE(torch::equal(actual.cpu(), expected)) << "vocab=" << vocab;
  }
}

#endif

TEST(SeededSamplerTest, NpuLogitsSamplingHandlesDtypesTailsAndMissingSeeds) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 49;
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
    for (const int64_t vocab : {int64_t{1},
                                int64_t{31},
                                int64_t{33},
                                int64_t{4095},
                                int64_t{4096},
                                int64_t{4097},
                                int64_t{248320}}) {
      torch::Tensor storage = torch::full(
          {kRows, vocab + 8}, -std::numeric_limits<float>::infinity());
      torch::Tensor logits = storage.narrow(1, 0, vocab);
      logits.select(1, 0).fill_(0.0F);
      logits.select(1, vocab - 1).fill_(0.0F);
      const torch::Tensor device_logits =
          storage.to(device, dtype).narrow(1, 0, vocab);
      torch::Tensor uniforms = torch::linspace(0.0, 1.0, kRows);
      uniforms.select(0, kRows - 1).fill_(1.0F - 0x1p-24F);
      const torch::Tensor expected =
          vocab == 1
              ? torch::zeros({kRows}, torch::kInt64)
              : torch::where(uniforms.lt(0.5F), 0, vocab - 1).to(torch::kInt64);
      const torch::Tensor actual = kernel::npu::tilelang::sampling_logits(
          device_logits, {}, {}, uniforms.to(device));
      EXPECT_TRUE(torch::equal(actual.cpu(), expected))
          << "vocab=" << vocab << ", dtype=" << dtype;
    }
  }
}

TEST(SeededSamplerTest,
     NpuLogitsSelectionMatchesIndependentDoublePrecisionCdf) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 49;
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
    for (const int64_t vocab : {int64_t{1},
                                int64_t{33},
                                int64_t{4097},
                                int64_t{65539},
                                int64_t{248320}}) {
      // Deterministic nonuniform rows, padding, zero-mass chunks and tail IDs.
      const torch::Tensor columns = torch::arange(vocab, torch::kFloat64);
      const torch::Tensor rows =
          torch::arange(kRows, torch::kFloat64).unsqueeze(1);
      torch::Tensor values = (columns * 0.37 + rows * 0.73).sin().mul(3.0);
      values.masked_fill_(columns.remainder(11).eq(3), -INFINITY);
      if (vocab > 4096) {
        values.select(0, 0).narrow(0, 0, 4096).fill_(-INFINITY);
      }
      torch::Tensor storage = torch::empty({kRows, vocab + 8}, dtype);
      storage.narrow(1, 0, vocab).copy_(values);
      // The reference uses the actual quantized input, not its FP64 precursor.
      const torch::Tensor quantized =
          storage.narrow(1, 0, vocab).to(torch::kFloat64);
      const torch::Tensor cdf = torch::softmax(quantized, -1).cumsum(-1);
      torch::Tensor uniforms = torch::linspace(0.0, 1.0, kRows);
      uniforms.select(0, kRows - 1).fill_(1.0F - 0x1p-24F);
      const torch::Tensor selected = kernel::npu::tilelang::sampling_logits(
                                         storage.to(device).narrow(1, 0, vocab),
                                         {},
                                         {},
                                         uniforms.to(device))
                                         .cpu()
                                         .unsqueeze(1);
      ASSERT_TRUE(
          selected.ge(0).logical_and(selected.lt(vocab)).all().item<bool>());
      EXPECT_TRUE(
          torch::isfinite(quantized.gather(1, selected)).all().item<bool>());
      const torch::Tensor upper = cdf.gather(1, selected).squeeze(1);
      const torch::Tensor lower = cdf.gather(1, (selected - 1).clamp_min(0))
                                      .masked_fill_(selected.eq(0), 0)
                                      .squeeze(1);
      const torch::Tensor u = uniforms.to(torch::kFloat64);
      EXPECT_LE((lower - u).clamp_min(0).max().item<double>(), 2e-6)
          << "vocab=" << vocab << ", dtype=" << dtype;
      EXPECT_LE((u - upper).clamp_min(0).max().item<double>(), 2e-6)
          << "vocab=" << vocab << ", dtype=" << dtype;
    }
  }
}

TEST(SeededSamplerTest, NpuCompactLogprobsAreInvariantToLargeCommonOffsets) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 128;
  constexpr int64_t kVocab = 65536;
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::GREEDY;
  params.logprobs() = true;
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
    for (const float offset : {0.0F, 8192.0F, 60000.0F, 1e8F}) {
      if (dtype == torch::kFloat16 && offset > 65504.0F) {
        continue;
      }
      torch::Tensor logits =
          torch::full({kRows, kVocab},
                      offset,
                      torch::TensorOptions().device(device).dtype(dtype));
      const SampleOutput actual = sample(logits, params);
      EXPECT_TRUE(
          torch::allclose(actual.logprobs,
                          torch::full({kRows},
                                      -std::log(kVocab),
                                      logits.options().dtype(torch::kFloat32)),
                          /*rtol=*/1e-5,
                          /*atol=*/5e-6))
          << "offset=" << offset << ", dtype=" << dtype;
    }
  }
}

TEST(SeededSamplerTest, NpuCategoricalPreservesSupportAndDistribution) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 4096;
  const torch::Tensor weights =
      torch::tensor({0.1F, 0.3F, 0.0F, 0.6F}).repeat({kRows, 1}).to(device);
  const torch::Tensor seeds =
      torch::arange(kRows, torch::TensorOptions().dtype(torch::kInt64))
          .to(device);
  const torch::Tensor offsets = torch::zeros_like(seeds);
  const auto check_distribution = [](const torch::Tensor& samples) {
    const torch::Tensor counts = samples.cpu().bincount({}, 4);
    constexpr std::array<double, 4> kProbabilities = {0.1, 0.3, 0.0, 0.6};
    for (int64_t token = 0; token < 4; ++token) {
      const double observed = counts[token].item<double>() / kRows;
      EXPECT_NEAR(observed, kProbabilities[token], 0.025);
    }
    EXPECT_EQ(counts[2].item<int64_t>(), 0);
  };
  check_distribution(kernel::npu::tilelang::sampling_logits(
      weights.log(), seeds, offsets, {}));
#if defined(XLLM_NPU_SAMPLING_ABLATION)
  check_distribution(
      kernel::npu::tilelang::seeded_sample(weights, seeds, offsets, {}));
#endif
}

TEST(SeededSamplerTest,
     NpuFusedSeededSelectionIsReproducibleForNonuniformLogits) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 49;
  constexpr int64_t kVocab = 65539;
  const auto options = torch::TensorOptions().device(device);
  const torch::Tensor seeds =
      torch::arange(kRows, options.dtype(torch::kInt64)).add(100);
  const torch::Tensor offsets = torch::arange(kRows, seeds.options());
  const torch::Tensor order =
      torch::arange(kRows, seeds.options()).flip(/*dims=*/{0});
  for (const torch::ScalarType dtype :
       {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
    const torch::Tensor logits = torch::arange(kVocab, options)
                                     .remainder(37)
                                     .mul(0.125)
                                     .to(dtype)
                                     .unsqueeze(0)
                                     .expand({kRows, kVocab})
                                     .clone();
    const torch::Tensor expected =
        kernel::npu::tilelang::sampling_logits(logits, seeds, offsets, {});
    for (int64_t repeat = 0; repeat < 8; ++repeat) {
      // Keep another allocation live while sampling a fresh logits buffer.
      [[maybe_unused]] const torch::Tensor unrelated =
          torch::full_like(logits, repeat);
      const torch::Tensor actual = kernel::npu::tilelang::sampling_logits(
          logits.clone(), seeds, offsets, {});
      EXPECT_TRUE(torch::equal(actual, expected));
      const torch::Tensor reordered =
          kernel::npu::tilelang::sampling_logits(logits.index_select(0, order),
                                                 seeds.index_select(0, order),
                                                 offsets.index_select(0, order),
                                                 {});
      EXPECT_TRUE(torch::equal(reordered, expected.index_select(0, order)));
    }
  }
}

TEST(SeededSamplerTest, NpuCompactLogprobsPreserveRawModelDistribution) {
  const torch::Device device(torch::kPrivateUse1, 0);
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat16}) {
    SamplingParameters params = make_seeded_params(
        torch::arange(128, torch::kInt64), torch::zeros({128}, torch::kInt64));
    params.logprobs() = true;
    params.max_top_logprobs() = 0;
    params.temperatures() =
        torch::full({128}, 0.7, torch::TensorOptions().device(device));
    params.top_k() = torch::full({128}, 20, params.seeds().options());
    params.top_p() = torch::full({128}, 0.95, params.temperatures().options());
    const torch::Tensor logits = torch::randn(
        {128, 65536}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor compact_logits = logits.clone();
    const SampleOutput actual = sample(compact_logits, params);
    const torch::Tensor expected_logprobs =
        torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32)
            .gather(/*dim=*/-1, actual.next_tokens.view({-1, 1}))
            .view({-1});
    params.logprobs() = false;
    torch::Tensor sampling_logits = logits.clone();
    const SampleOutput sampled = sample(sampling_logits, params);
    EXPECT_TRUE(torch::equal(actual.next_tokens, sampled.next_tokens));
    EXPECT_TRUE(torch::allclose(actual.logprobs,
                                expected_logprobs,
                                /*rtol=*/1e-5,
                                /*atol=*/5e-6));
  }
}

TEST(SeededSamplerTest, NpuCompactNormalizationPreservesRoundedTopLogprobTies) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  torch::Tensor logits = torch::zeros({128, 65536}, options);
  logits.narrow(1, 0, 16).copy_(
      torch::arange(16, options.dtype(torch::kFloat32)).mul(1e-8));
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::GREEDY;
  params.logprobs() = true;
  params.max_top_logprobs() = 5;
  const auto [expected_values, expected_tokens] =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32)
          .topk(/*k=*/5, /*dim=*/-1);
  const SampleOutput actual = sample(logits, params);
  EXPECT_TRUE(torch::equal(actual.top_tokens, expected_tokens));
  EXPECT_TRUE(torch::equal(actual.top_logprobs, expected_values));
}

TEST(SeededSamplerTest, NpuMixedSamplingKeepsGreedyRowsAndAdvancesUnseededRng) {
  const torch::Device device(torch::kPrivateUse1, 0);
  SamplingParameters params =
      make_seeded_params(torch::tensor({42, -1, 7, -1}, torch::kLong),
                         torch::zeros({4}, torch::kLong));
  params.sampling_mode() = SamplingMode::MIXED;
  params.do_sample() = torch::tensor({true, true, false, false}).to(device);
  params.has_unseeded_sample() = true;
  torch::Tensor logits =
      torch::zeros({4, 33}, params.seeds().options().dtype(torch::kFloat32));
  logits.select(0, 2).select(0, 17).fill_(2.0F);
  logits.select(0, 3).select(0, 31).fill_(2.0F);
  const auto& generator = at_npu::detail::getDefaultNPUGenerator(0);
  const torch::Tensor before = generator.get_state().clone();
  const SampleOutput output = sample(logits, params);
  EXPECT_EQ(output.next_tokens[2].item<int64_t>(), 17);
  EXPECT_EQ(output.next_tokens[3].item<int64_t>(), 31);
  EXPECT_FALSE(torch::equal(before, generator.get_state()));
}

TEST(SeededSamplerTest, NpuMixedFilteringPreservesTieMasks) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kVocab = 49152;
  for (const int64_t rows : {int64_t{2}, int64_t{8}, int64_t{32}}) {
    for (const torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
      const auto options = torch::TensorOptions().device(device).dtype(dtype);
      SamplingParameters params;
      params.sampling_mode() = SamplingMode::MIXED;
      params.num_random_rows() = (rows + 3) / 4;
      params.do_sample() =
          torch::arange(rows, options).lt(params.num_random_rows());
      params.top_k() =
          torch::where(params.do_sample(), 20, 0).to(torch::kInt64);
      params.top_p() =
          torch::where(params.do_sample(), 0.95, 1).to(torch::kFloat32);
      params.temperatures() =
          torch::where(params.do_sample(), 0.7, 0).to(torch::kFloat32);
      for (const int64_t levels : {int64_t{1}, int64_t{5}}) {
        torch::Tensor original =
            torch::arange(kVocab, options.dtype(torch::kFloat32))
                .remainder(levels)
                .to(dtype)
                .unsqueeze(0)
                .expand({rows, kVocab})
                .clone();
        torch::Tensor reference = original.clone();
        SamplingParameters reference_params = params;
        reference_params.sampling_mode() = SamplingMode::RANDOM;
        process_logits(reference, reference_params);
        torch::Tensor compacted = original.clone();
        process_logits(compacted, params);
        EXPECT_TRUE(
            torch::equal(compacted.narrow(0, 0, params.num_random_rows()),
                         reference.narrow(0, 0, params.num_random_rows())))
            << "rows=" << rows << ", dtype=" << dtype << ", levels=" << levels;
        EXPECT_TRUE(torch::equal(compacted.argmax(-1), reference.argmax(-1)));
      }
    }
  }
}

TEST(SeededSamplerTest, NpuMixedFilterMatchesRandomRowsAndSkipsGreedyRows) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kRows = 128;
  constexpr int64_t kVocab = 65536;
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::MIXED;
  params.num_random_rows() = 32;
  const auto options = torch::TensorOptions().device(device);
  params.do_sample() = torch::arange(kRows, options).lt(32);
  params.top_k() = torch::full({kRows}, 20, options.dtype(torch::kInt64));
  torch::Tensor original = torch::arange(kVocab, options)
                               .unsqueeze(0)
                               .expand({kRows, kVocab})
                               .clone();
  torch::Tensor reference = original.clone();
  SamplingParameters reference_params = params;
  reference_params.sampling_mode() = SamplingMode::RANDOM;
  process_logits(reference, reference_params);
  torch::Tensor compacted = original.clone();
  process_logits(compacted, params);
  EXPECT_TRUE(
      torch::equal(compacted.narrow(0, 0, 32), reference.narrow(0, 0, 32)));
  EXPECT_TRUE(torch::equal(compacted.narrow(0, 32, kRows - 32),
                           original.narrow(0, 32, kRows - 32)));
  EXPECT_TRUE(torch::equal(compacted.argmax(-1), reference.argmax(-1)));
}

}  // namespace
}  // namespace xllm
