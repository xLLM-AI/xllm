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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/sampling/sampler.h"
#include "core/kernels/cuda/seeded_sample.h"

namespace xllm {
namespace {

SamplingParameters make_device_params(int64_t rows, int32_t device = 0) {
  SamplingParameters params;
  const auto options = torch::TensorOptions().device(torch::kCUDA, device);
  params.selected_token_idxes = torch::arange(rows, options.dtype(torch::kInt));
  params.seeds = torch::arange(rows, options.dtype(torch::kLong)) + 42;
  params.seed_offsets = torch::zeros({rows}, options.dtype(torch::kLong));
  params.do_sample = torch::ones({rows}, options.dtype(torch::kBool));
  params.sampling_mode = SamplingMode::RANDOM;
  return params;
}

torch::Tensor draw(const torch::Tensor& probs,
                   const SamplingParameters& params) {
  return kernel::cuda::seeded_sample(probs,
                                     params.seeds,
                                     params.seed_offsets,
                                     params.do_sample,
                                     params.has_unseeded_sample);
}

TEST(CudaSeededSamplerTest, ReplayPermutationSplitSeedsAndOffsets) {
  auto params = make_device_params(37);
  // Positions include sequence identity in the high 32 bits in serving.
  params.seed_offsets =
      torch::arange(37, params.seeds.options()) * (int64_t{1} << 32) + 7;
  params.seeds[0] = -2;
  const auto options = params.seeds.options().dtype(torch::kFloat);
  const auto probs = torch::arange(1, 1032, options).repeat({37, 1});
  const auto expected = draw(probs, params);
  EXPECT_TRUE(torch::equal(expected, draw(probs, params)));
  const auto order = torch::arange(36, -1, -1, params.seeds.options());
  auto permuted = params;
  permuted.seeds = params.seeds.index_select(0, order);
  permuted.seed_offsets = params.seed_offsets.index_select(0, order);
  EXPECT_TRUE(torch::equal(expected.index_select(0, order),
                           draw(probs.index_select(0, order), permuted)));
  std::vector<torch::Tensor> chunks;
  chunks.reserve(37);
  for (int64_t row = 0; row < 37; ++row) {
    auto single = make_device_params(1);
    single.seeds = params.seeds.slice(0, row, row + 1);
    single.seed_offsets = params.seed_offsets.slice(0, row, row + 1);
    chunks.emplace_back(draw(probs.slice(0, row, row + 1), single));
  }
  EXPECT_TRUE(torch::equal(expected, torch::cat(chunks)));
  params.seed_offsets.add_(1);
  EXPECT_FALSE(torch::equal(expected, draw(probs, params)));
  params.seed_offsets.sub_(1);
  params.seeds.add_(1000);
  EXPECT_FALSE(torch::equal(expected, draw(probs, params)));
}

TEST(CudaSeededSamplerTest, SparseStridedRowsAwkwardWidthsAndGreedyTies) {
  auto params = make_device_params(6);
  for (const int64_t width : {1, 17, 255, 257, 1031, 131073}) {
    SCOPED_TRACE(width);
    auto storage =
        torch::zeros({12, width}, params.seeds.options().dtype(torch::kFloat));
    auto probs = storage.slice(0, 0, 12, 2);
    probs.select(1, width - 1).fill_(1.0);
    EXPECT_TRUE(
        torch::equal(draw(probs, params),
                     torch::full({6}, width - 1, params.seeds.options())));
    probs.fill_(1.0);
    params.do_sample.zero_();
    EXPECT_TRUE(torch::equal(draw(probs, params),
                             torch::zeros({6}, params.seeds.options())));
    params.do_sample.fill_(true);
  }
  auto empty = make_device_params(0);
  EXPECT_EQ(
      draw(torch::empty({0, 17}, params.seeds.options().dtype(torch::kFloat)),
           empty)
          .numel(),
      0);
}

TEST(CudaSeededSamplerTest, DistributionAndZeroProbabilityExclusion) {
  constexpr int64_t kDraws = 100000;
  auto params = make_device_params(kDraws);
  auto probs = torch::tensor({0.0F, 0.1F, 0.2F, 0.0F, 0.7F},
                             params.seeds.options().dtype(torch::kFloat))
                   .repeat({kDraws, 1});
  const auto counts = torch::bincount(draw(probs, params), {}, 5).cpu();
  EXPECT_EQ(counts[0].item<int64_t>(), 0);
  EXPECT_EQ(counts[3].item<int64_t>(), 0);
  EXPECT_NEAR(counts[1].item<int64_t>() / double{kDraws}, 0.1, 0.005);
  EXPECT_NEAR(counts[2].item<int64_t>() / double{kDraws}, 0.2, 0.006);
  EXPECT_NEAR(counts[4].item<int64_t>() / double{kDraws}, 0.7, 0.007);

  // Three nonzero entries share a strided lane, exercising selection within
  // that lane rather than just selection between lanes.
  auto wide = torch::zeros({1031}, probs.options());
  wide[1] = 0.1F;
  wide[257] = 0.2F;
  wide[769] = 0.7F;
  const auto wide_counts =
      torch::bincount(draw(wide.expand({kDraws, -1}), params), {}, 1031).cpu();
  EXPECT_NEAR(wide_counts[1].item<int64_t>() / double{kDraws}, 0.1, 0.005);
  EXPECT_NEAR(wide_counts[257].item<int64_t>() / double{kDraws}, 0.2, 0.006);
  EXPECT_NEAR(wide_counts[769].item<int64_t>() / double{kDraws}, 0.7, 0.007);
  EXPECT_EQ(
      (wide_counts[1] + wide_counts[257] + wide_counts[769]).item<int64_t>(),
      kDraws);
}

TEST(CudaSeededSamplerTest, RoundedPrefixBoundariesAlwaysWritePositiveToken) {
  constexpr int64_t kRows = 128;
  auto params = make_device_params(kRows);
  // This seed produces a request uniform two FP64 steps below one.
  params.seeds.fill_(-1604402868668921676LL);
  for (const int64_t stride : {1, 256, 512}) {
    for (const float scale : {1.0F, 0x1p-70F, 0x1p120F}) {
      SCOPED_TRACE(stride);
      SCOPED_TRACE(scale);
      auto row = torch::zeros({256 * stride},
                              params.seeds.options().dtype(torch::kFloat));
      // At stride 1, the scan total rounds above every positive lane's end.
      // Larger strides exercise the same sparse boundary inside a lane.
      row[229 * stride] = scale;
      row[241 * stride] = scale * 0x1.8p-54F;
      row[242 * stride] = scale * 0x1.8p-54F;
      const auto probs = row.expand({kRows, -1});
      {
        // Recycled output memory must not hide a missing kernel write.
        const auto recycled = torch::full({kRows}, -1, params.seeds.options());
      }
      const auto expected = draw(probs, params);
      ASSERT_TRUE(expected.ge(0)
                      .logical_and(expected.lt(row.numel()))
                      .all()
                      .item<bool>());
      EXPECT_TRUE(
          probs.gather(1, expected.unsqueeze(1)).gt(0).all().item<bool>());
      EXPECT_TRUE(torch::equal(expected, draw(probs, params)));
    }
  }
}

TEST(CudaSeededSamplerTest, RoundedChunkBoundariesAlwaysWritePositiveToken) {
  constexpr int64_t kRows = 128;
  auto params = make_device_params(kRows);
  params.seeds.fill_(-1604402868668921676LL);
  auto row = torch::zeros({65536}, params.seeds.options().dtype(torch::kFloat));
  // All weights share lane zero. Its sequential mass rounds above the chunk
  // scan total, whose rounded interval ends leave the old selection empty.
  row[113 * 256] = 1.0F;
  for (const int32_t index : {127, 222, 224}) {
    row[index * 256] = 0x1.8p-53F;
  }
  for (const int32_t index : {146, 220}) {
    row[index * 256] = 0x1.8p-52F;
  }
  const auto probs = row.expand({kRows, -1});
  {
    const auto recycled = torch::full({kRows}, -1, params.seeds.options());
  }
  const auto expected = draw(probs, params);
  ASSERT_TRUE(
      expected.ge(0).logical_and(expected.lt(row.numel())).all().item<bool>());
  EXPECT_TRUE(probs.gather(1, expected.unsqueeze(1)).gt(0).all().item<bool>());
  EXPECT_TRUE(torch::equal(expected, draw(probs, params)));
}

TEST(CudaSeededSamplerTest, MixedRowsPreserveSeededAndGlobalRngContracts) {
  auto params = make_device_params(32);
  const auto probs =
      torch::ones({32, 1024}, params.seeds.options().dtype(torch::kFloat));
  const auto& generator =
      torch::globalContext().defaultGenerator(probs.device());
  const auto before = generator.get_state().clone();
  const auto expected = draw(probs, params);
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
  params.do_sample[1] = false;
  params.seeds[1] = -1;  // Greedy rows do not require unseeded randomness.
  EXPECT_EQ(draw(probs, params)[1].item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
  params.seeds[0] = -1;
  params.has_unseeded_sample = true;
  const auto mixed = draw(probs, params);
  EXPECT_FALSE(torch::equal(before, generator.get_state()));
  EXPECT_TRUE(torch::equal(expected.slice(0, 2), mixed.slice(0, 2)));
  EXPECT_GE(mixed[0].item<int64_t>(), 0);
  EXPECT_LT(mixed[0].item<int64_t>(), 1024);
  // Consecutive random draws may legitimately select the same token.
  const auto after = generator.get_state().clone();
  draw(probs, params);
  EXPECT_FALSE(torch::equal(after, generator.get_state()));
}

TEST(CudaSeededSamplerTest, CurrentStreamMultipleDevicesAndGraphReplay) {
  for (int32_t device = 0; device < torch::cuda::device_count(); ++device) {
    const auto stream = c10::cuda::getStreamFromPool(false, device);
    const c10::cuda::CUDAStreamGuard guard(stream);
    auto params = make_device_params(32, device);
    params.do_sample = torch::Tensor();
    auto probs =
        torch::ones({32, 1031}, params.seeds.options().dtype(torch::kFloat));
    const auto expected = draw(probs, params);
    stream.synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    const auto captured = draw(probs, params);
    graph.capture_end();
    graph.replay();
    EXPECT_TRUE(torch::equal(expected, captured));
    params.seed_offsets.add_(1);
    const auto changed = draw(probs, params);
    graph.replay();
    EXPECT_TRUE(torch::equal(changed, captured));
    EXPECT_FALSE(torch::equal(expected, captured));
  }
}

TEST(CudaSeededSamplerTest, SamplerAppliesFiltersAndReportsRawLogprobs) {
  auto params = make_device_params(32);
  params.do_sample = torch::Tensor();
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.top_k = torch::full({32}, 2, params.seeds.options());
  params.top_p =
      torch::full({32}, 0.9, params.seeds.options().dtype(torch::kFloat));
  params.min_p = torch::full({32}, 0.1, params.top_p.options());
  params.temperatures = torch::full({32}, 0.7, params.top_p.options());
  auto logits =
      torch::tensor({{0.0F, 1.0F, 3.5F, 4.0F}}, params.top_p.options())
          .repeat({32, 1});
  const auto raw = torch::log_softmax(logits, -1);
  auto processed = logits.clone();
  Sampler::process_logits(processed, params);
  EXPECT_TRUE(torch::equal(torch::isfinite(processed).sum(1),
                           torch::full({32}, 2, params.seeds.options())));
  const auto expected = draw(torch::softmax(processed, -1), params);
  const auto result = Sampler().forward(logits, params);
  EXPECT_TRUE(torch::equal(result.next_tokens, expected));
  EXPECT_TRUE(torch::allclose(result.logprobs,
                              raw.gather(1, expected.unsqueeze(1)).squeeze(1)));
  EXPECT_TRUE(torch::equal(result.top_tokens, std::get<1>(raw.topk(2))));
}

TEST(CudaSeededSamplerTest, SamplerRandomModeAcceptsUndefinedRowMask) {
  auto params = make_device_params(32);
  const auto logits =
      torch::zeros({32, 1031}, params.seeds.options().dtype(torch::kFloat));
  const auto expected = draw(torch::softmax(logits, -1), params);
  params.do_sample = torch::Tensor();
  auto input = logits.clone();
  EXPECT_TRUE(
      torch::equal(Sampler().forward(input, params).next_tokens, expected));
  params.seeds[0] = -1;
  params.has_unseeded_sample = true;
  input = logits.clone();
  const auto mixed = Sampler().forward(input, params).next_tokens;
  EXPECT_TRUE(torch::equal(expected.slice(0, 1), mixed.slice(0, 1)));
  EXPECT_GE(mixed[0].item<int64_t>(), 0);
  EXPECT_LT(mixed[0].item<int64_t>(), logits.size(1));
}

TEST(CudaSeededSamplerTest, SamplerGraphCaptureAndMixedRows) {
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  auto params = make_device_params(16);
  auto logits =
      torch::zeros({16, 257}, params.seeds.options().dtype(torch::kFloat));
  params.sampling_mode = SamplingMode::MIXED;
  params.do_sample[3] = false;
  params.seeds[0] = -1;
  params.has_unseeded_sample = true;
  const auto warmup = Sampler().forward(logits, params);
  stream.synchronize();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  const auto captured = Sampler().forward(logits, params);
  graph.capture_end();
  graph.replay();
  const auto first = captured.next_tokens.clone();
  params.seed_offsets.add_(1);
  graph.replay();
  const auto eager = Sampler().forward(logits, params);
  EXPECT_TRUE(torch::equal(captured.next_tokens.slice(0, 1),
                           eager.next_tokens.slice(0, 1)));
  EXPECT_EQ(captured.next_tokens[3].item<int64_t>(), 0);
  EXPECT_FALSE(torch::equal(first, captured.next_tokens));
}

TEST(CudaSeededSamplerTest, SamplerAcceptsHalfAndBfloat16Logits) {
  for (const torch::ScalarType dtype : {torch::kFloat16, torch::kBFloat16}) {
    auto params = make_device_params(8);
    params.do_sample = torch::Tensor();
    params.logprobs = true;
    auto logits =
        torch::full({8, 17}, -INFINITY, params.seeds.options().dtype(dtype));
    logits.select(1, 7).fill_(1.0);
    const auto result = Sampler().forward(logits, params);
    EXPECT_TRUE(torch::equal(result.next_tokens,
                             torch::full({8}, 7, params.seeds.options())));
    EXPECT_EQ(result.logprobs.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(
        torch::equal(result.logprobs, torch::zeros_like(result.logprobs)));
  }
}

// Fork before CUDA initialization so validation failures cannot poison a live
// CUDA context in another test. CHECKs are part of the host API contract.
TEST(CudaSeededSamplerDeathTest, RejectsInvalidMetadataAndProbabilityLayout) {
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs =
            torch::ones({2, 4}, params.seeds.options().dtype(torch::kFloat));
        kernel::cuda::seeded_sample(probs,
                                    params.seeds.cpu(),
                                    params.seed_offsets,
                                    params.do_sample,
                                    false);
      },
      "device");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs =
            torch::ones({2, 4}, params.seeds.options().dtype(torch::kFloat));
        kernel::cuda::seeded_sample(probs,
                                    params.seeds.to(torch::kInt),
                                    params.seed_offsets,
                                    params.do_sample,
                                    false);
      },
      "scalar_type");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs =
            torch::ones({2, 4}, params.seeds.options().dtype(torch::kFloat));
        kernel::cuda::seeded_sample(probs,
                                    params.seeds.slice(0, 0, 1),
                                    params.seed_offsets,
                                    params.do_sample,
                                    false);
      },
      "size");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs =
            torch::ones({4, 2}, params.seeds.options().dtype(torch::kFloat))
                .t();
        draw(probs, params);
      },
      "stride");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs = torch::ones({2, 4}, params.seeds.options());
        draw(probs, params);
      },
      "scalar_type");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        params.seeds =
            torch::arange(4, params.seeds.options()).slice(0, 0, 4, 2);
        auto probs =
            torch::ones({2, 4}, params.seeds.options().dtype(torch::kFloat));
        draw(probs, params);
      },
      "is_contiguous");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        auto probs =
            torch::zeros({2, 4}, params.seeds.options().dtype(torch::kFloat));
        draw(probs, params);
        CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
      },
      "total > 0");
  EXPECT_DEATH(
      {
        auto params = make_device_params(2);
        params.seeds.fill_(-1);
        auto probs =
            torch::ones({2, 4}, params.seeds.options().dtype(torch::kFloat));
        draw(probs, params);
        CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
      },
      "uniforms != nullptr");
}

}  // namespace
}  // namespace xllm
