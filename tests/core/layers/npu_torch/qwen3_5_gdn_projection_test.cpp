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

#include "core/layers/npu_torch/qwen3_5_gdn_projection.h"

#include <gtest/gtest.h>
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/framework/quantization/quant_args.h"
#include "core/layers/npu_torch/qwen3_5_gdn_prefill_layout.h"
namespace xllm::layer {
namespace {
const torch::Device kDevice(torch::kPrivateUse1, 0);
const std::vector<GdnProjectionMode> kModes = {GdnProjectionMode::SEPARATE,
                                               GdnProjectionMode::GROUPED};

TEST(GdnProjectionModeTest, UsesGroupedForUnquantizedBf16) {
  EXPECT_EQ(resolve_gdn_projection_mode(QuantArgs{}, torch::kBFloat16),
            GdnProjectionMode::GROUPED);
}

TEST(GdnProjectionModeTest, UsesSeparateForNonBf16Weights) {
  for (const torch::ScalarType dtype : {torch::kFloat16, torch::kFloat32}) {
    EXPECT_EQ(resolve_gdn_projection_mode(QuantArgs{}, dtype),
              GdnProjectionMode::SEPARATE);
  }
}

TEST(GdnProjectionModeTest, UsesSeparateForEachQuantizationMetadataSource) {
  QuantArgs method;
  method.quant_method() = "smoothquant";
  QuantArgs type;
  type.quantize_type() = "w8a8";
  QuantArgs descriptions;
  descriptions.quant_descs() = {{"in_proj_qkv.weight", "W8A8"}};
  QuantArgs compressed;
  compressed.is_compressed_tensors_w8a8_dynamic() = true;
  QuantArgs groups;
  groups.compressed_groups() = {CompressedQuantGroup{}};
  for (const auto& quant_args :
       {method, type, descriptions, compressed, groups}) {
    EXPECT_EQ(resolve_gdn_projection_mode(quant_args, torch::kBFloat16),
              GdnProjectionMode::SEPARATE);
  }
}

GdnProjectionTensors make_weights(int64_t hidden, int64_t tp) {
  const std::array<int64_t, 4> widths = {
      10240 / tp, 6144 / tp, 48 / tp, 48 / tp};
  GdnProjectionTensors weights;
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = (torch::randn({widths[i], hidden}) * 0.02)
                     .to(kDevice, torch::kBFloat16);
  }
  return weights;
}
void expect_close(const GdnProjectionTensors& actual,
                  const GdnProjectionTensors& expected) {
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].sizes(), expected[i].sizes());
    EXPECT_EQ(actual[i].scalar_type(), torch::kBFloat16);
    EXPECT_TRUE(actual[i].is_contiguous());
    auto result = actual[i].to(torch::kCPU, torch::kFloat32);
    auto reference = expected[i].to(torch::kCPU, torch::kFloat32);
    EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
    const double rms = reference.square().mean().sqrt().item<double>();
    const double error =
        (result - reference).square().mean().sqrt().item<double>();
    EXPECT_LE(error / std::max(rms, 1e-8), 0.005);
    EXPECT_TRUE(torch::allclose(result, reference, 0.02, 0.02));
  }
}
class GdnProjectionTest : public ::testing::TestWithParam<GdnProjectionMode> {};
TEST_P(GdnProjectionTest, MatchesIndependentBf16ForAllTpWidths) {
  torch::InferenceMode inference;
  torch::manual_seed(1327);
  for (const int64_t tp : {1, 2, 4, 8}) {
    SCOPED_TRACE(tp);
    auto weights = make_weights(5120, tp);
    GdnPrefillProjection reference(weights, GdnProjectionMode::SEPARATE);
    GdnPrefillProjection candidate(weights, GetParam());
    for (const int64_t tokens : {1, 17, 128}) {
      SCOPED_TRACE(tokens);
      auto input =
          (torch::randn({tokens, 5120}) * 0.2).to(kDevice, torch::kBFloat16);
      expect_close(candidate.forward(input), reference.forward(input));
    }
  }
}
TEST_P(GdnProjectionTest, PreservesOutputsAcrossQueuedAllocationReuse) {
  torch::InferenceMode inference;
  torch::manual_seed(20261024);
  for (const int64_t tp : {1, 2, 4, 8}) {
    SCOPED_TRACE(tp);
    auto weights = make_weights(5120, tp);
    GdnPrefillProjection reference(weights, GdnProjectionMode::SEPARATE);
    GdnPrefillProjection candidate(weights, GetParam());
    for (const int64_t tokens : {1, 17, 322}) {
      SCOPED_TRACE(tokens);
      auto input =
          (torch::randn({tokens, 5120}) * 0.2).to(kDevice, torch::kBFloat16);
      const auto expected = reference.forward(input);
      torch::npu::synchronize();
      // Keep only a reduction of each output so output allocations can be
      // reused while host commands remain queued.
      // No device synchronization occurs until every iteration is enqueued.
      std::vector<std::array<torch::Tensor, 4>> reductions;
      reductions.reserve(32);
      for (int32_t iteration = 0; iteration < 32; ++iteration) {
        const auto outputs = candidate.forward(input);
        std::array<torch::Tensor, 4> sums;
        for (size_t i = 0; i < outputs.size(); ++i) {
          sums[i] = outputs[i].to(torch::kFloat32).sum(0);
        }
        reductions.emplace_back(std::move(sums));
      }
      torch::npu::synchronize();
      for (const auto& sums : reductions) {
        for (size_t i = 0; i < sums.size(); ++i) {
          auto actual = sums[i].cpu();
          auto gold = expected[i].to(torch::kFloat32).sum(0).cpu();
          const double scale = gold.square().mean().sqrt().item<double>();
          const double error =
              (actual - gold).square().mean().sqrt().item<double>();
          EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
          EXPECT_LE(error / std::max(scale, 1e-8), 0.005);
        }
      }
    }
  }
}
TEST_P(GdnProjectionTest, PreservesRankLocalHeadOrdering) {
  torch::InferenceMode inference;
  torch::manual_seed(814);
  for (int64_t rank = 0; rank < 4; ++rank) {
    auto weights = make_weights(32, 4);
    int64_t offset = 0;
    const std::array<int64_t, 3> sizes = {512, 512, 1536};
    for (size_t part = 0; part < sizes.size(); ++part) {
      weights[0]
          .slice(0, offset, offset + sizes[part])
          .fill_(0.01 * (1 + rank + static_cast<int64_t>(part) * 4));
      offset += sizes[part];
    }
    auto input = torch::ones({7, 32}, weights[0].options());
    GdnPrefillProjection reference(weights, GdnProjectionMode::SEPARATE);
    GdnPrefillProjection candidate(weights, GetParam());
    expect_close(candidate.forward(input), reference.forward(input));
  }
}
TEST_P(GdnProjectionTest, PreparedWeightsOutliveOriginalOwnersAndServeDecode) {
  torch::InferenceMode inference;
  torch::manual_seed(20261025);
  for (const int64_t tp : {1, 2, 4, 8}) {
    SCOPED_TRACE(tp);
    auto weights = make_weights(32, tp);
    GdnProjectionTensors original;
    for (size_t index = 0; index < weights.size(); ++index) {
      original[index] = weights[index].cpu();
    }
    GdnPrefillProjection candidate(std::move(weights), GetParam());
    const auto& prepared = candidate.weights();
    for (size_t index = 0; index < prepared.size(); ++index) {
      EXPECT_TRUE(prepared[index].is_contiguous());
      EXPECT_TRUE(torch::equal(prepared[index].cpu(), original[index]));
      // Updating a Decode parameter must be visible to every prepared matmul,
      // including cached transposes; equal copies are not enough.
      prepared[index].fill_(0.01 * static_cast<double>(index + 1));
    }
    // Decode consumes the original weight allocations, independently of the
    // prefill plan. Both paths must retain valid storage after source owners
    // disappear.
    const auto input = torch::ones({1, 32}, prepared[0].options());
    GdnProjectionTensors decode;
    for (size_t index = 0; index < decode.size(); ++index) {
      decode[index] = torch::nn::functional::linear(input, prepared[index]);
    }
    expect_close(candidate.forward(input), decode);
  }
}

INSTANTIATE_TEST_SUITE_P(Strategies,
                         GdnProjectionTest,
                         ::testing::ValuesIn(kModes));
TEST(GdnPackedPrefillTest, ExcludesDecodeAndSpeculativeVerification) {
  EXPECT_TRUE(use_packed_gdn_prefill(true, false, false));
  EXPECT_TRUE(use_packed_gdn_prefill(false, true, false));
  EXPECT_TRUE(use_packed_gdn_prefill(true, true, false));
  EXPECT_FALSE(use_packed_gdn_prefill(false, false, false));
  EXPECT_FALSE(use_packed_gdn_prefill(false, true, true));
  EXPECT_FALSE(use_packed_gdn_prefill(true, false, true));
  EXPECT_FALSE(use_packed_gdn_prefill(false, false, true));
  EXPECT_FALSE(use_packed_gdn_prefill(true, true, true));
}
TEST(GdnProjectionTest, GroupedSupportsOtherProjectionWidths) {
  torch::InferenceMode inference;
  torch::manual_seed(20261026);
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  GdnProjectionTensors cpu_weights = {torch::randn({31, 32}, options),
                                      torch::randn({19, 32}, options),
                                      torch::randn({3, 32}, options),
                                      torch::randn({3, 32}, options)};
  GdnProjectionTensors weights;
  for (size_t index = 0; index < weights.size(); ++index) {
    weights[index] = cpu_weights[index].to(kDevice);
  }
  GdnPrefillProjection candidate(weights, GdnProjectionMode::GROUPED);
  for (const int64_t tokens : {1, 7, 33}) {
    const auto cpu_input = (torch::randn({tokens, 32}) * 0.2).to(options);
    GdnProjectionTensors reference;
    for (size_t index = 0; index < reference.size(); ++index) {
      reference[index] =
          torch::nn::functional::linear(cpu_input.to(torch::kFloat32),
                                        cpu_weights[index].to(torch::kFloat32));
    }
    expect_close(candidate.forward(cpu_input.to(kDevice)), reference);
  }
}

}  // namespace
}  // namespace xllm::layer
