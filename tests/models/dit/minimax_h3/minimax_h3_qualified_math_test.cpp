/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/minimax_h3_qualified_weights.h"

namespace xllm {
namespace {

TEST(QualifiedMath, DenseRuntimeQkvRecoversSeparateSourceProjections) {
  const torch::Tensor query =
      torch::arange(24, torch::kFloat32).reshape({6, 4});
  const torch::Tensor key = query + 100;
  const torch::Tensor value = query - 100;
  const torch::Tensor grouped = h3_qualified_group_qkv(query, key, value, 2, 3);
  const torch::Tensor runtime = minimax_h3_reorder_grouped_qkv(grouped, 2, 3);
  const torch::Tensor input = torch::tensor({{1.F, 2.F, -1.F, 0.5F}});
  const torch::Tensor actual = torch::nn::functional::linear(input, runtime);
  const torch::Tensor expected =
      torch::cat({torch::nn::functional::linear(input, query),
                  torch::nn::functional::linear(input, key),
                  torch::nn::functional::linear(input, value)},
                 1);
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(QualifiedMath, RmsNormUsesFp32AccumulationAndBfloat16Output) {
  const torch::Tensor normal =
      torch::tensor({{1.F, 2.F, 3.F, 4.F}}, torch::kBFloat16);
  const torch::Tensor actual =
      minimax_h3_rms_norm(normal, torch::ones({4}, torch::kBFloat16), 1e-5);
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::equal(
      actual,
      torch::tensor({{0.365234375F, 0.73046875F, 1.09375F, 1.4609375F}},
                    torch::kBFloat16)));
}

TEST(QualifiedMath, SegmentPaddingCannotContaminateUsedAttention) {
  const torch::Tensor query = torch::zeros({4, 1, 2}, torch::kFloat32);
  const torch::Tensor key = torch::zeros_like(query);
  const torch::Tensor value = torch::tensor(
      {{{2.F, 4.F}}, {{6.F, 8.F}}, {{1000.F, -1000.F}}, {{3000.F, -3000.F}}});
  const torch::Tensor output = minimax_h3_segmented_sdpa(
      query, key, value, torch::tensor({0, 2, 4}, torch::kInt32));
  EXPECT_TRUE(torch::equal(output.slice(0, 0, 2),
                           torch::tensor({{{4.F, 6.F}}, {{4.F, 6.F}}})));
  EXPECT_TRUE(
      torch::equal(output.slice(0, 2, 4),
                   torch::tensor({{{2000.F, -2000.F}}, {{2000.F, -2000.F}}})));
}

TEST(QualifiedMath, RopeRotatesOnlyTheConfiguredPrefix) {
  const torch::Tensor input =
      torch::tensor({{{1.F, 2.F, 3.F, 4.F, 99.F, 100.F}}});
  const torch::Tensor frequencies = torch::zeros({1, 4}, torch::kFloat32);
  EXPECT_TRUE(torch::equal(minimax_h3_apply_rope(input, frequencies), input));
  const torch::Tensor quarter =
      torch::full({1, 4}, 1.5707963267948966, torch::kFloat32);
  const torch::Tensor rotated = minimax_h3_apply_rope(input, quarter);
  EXPECT_TRUE(
      torch::allclose(rotated,
                      torch::tensor({{{-3.F, -4.F, 1.F, 2.F, 99.F, 100.F}}}),
                      1e-6,
                      1e-6));
}

TEST(QualifiedMath, FixedAndStreamingParametersKeepMixedPrecision) {
  const auto options = torch::TensorOptions().device(torch::kMeta);
  MiniMaxH3StreamingDenoiser denoiser(options);
  const auto parameters = denoiser->named_parameters(true);
  const auto buffers = denoiser->named_buffers(true);
  EXPECT_EQ(parameters.size(), 44);
  EXPECT_EQ(buffers.size(), 1);
  EXPECT_EQ(parameters["video_patch_proj.weight"].scalar_type(),
            torch::kFloat32);
  EXPECT_EQ(parameters["condition_proj.weight"].scalar_type(),
            torch::kBFloat16);
  EXPECT_EQ(parameters["final_layer.video_out.weight"].scalar_type(),
            torch::kFloat32);
  EXPECT_EQ(parameters["final_layer.audio_out.weight"].scalar_type(),
            torch::kFloat32);
  EXPECT_EQ(parameters["final_layer.norm.weight"].scalar_type(),
            torch::kBFloat16);
  EXPECT_EQ(parameters["final_layer.adaln_proj.linear.weight"].scalar_type(),
            torch::kBFloat16);
  EXPECT_EQ(parameters["streaming_block.attn.qkv_proj.weight"].sizes().vec(),
            std::vector<int64_t>({21504, 5376}));
  EXPECT_EQ(parameters["streaming_block.attn.qkv_proj.weight"].scalar_type(),
            torch::kBFloat16);
  EXPECT_EQ(buffers["rope.inv_freq"].scalar_type(), torch::kFloat32);
}

}  // namespace
}  // namespace xllm
