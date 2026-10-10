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

#include "core/layers/common/quant_utils.h"

namespace xllm::layer {
namespace {

QuantArgs ct_args() {
  QuantArgs args;
  args.quant_method() = "compressed-tensors";
  args.bits() = 8;
  args.is_compressed_tensors_w8a8_dynamic() = true;
  args.ignored_modules() = {"re:.*gate$"};
  return args;
}

TEST(CompressedTensorsTest, ResolvesQuantizationIndependentlyOfShardOrder) {
  auto args = ct_args();
  const std::vector<std::string> prefixes = {"q.", "k.", "v."};
  const std::vector<StateDict> shards = {
      StateDict({{"q.weight", torch::ones({4, 8}, torch::kInt8)}}),
      StateDict({{"v.weight_scale", torch::ones({4, 1})}}),
      StateDict({{"k.weight_scale", torch::ones({4})}})};
  std::optional<std::string> resolved;
  for (const auto& shard : shards) {
    resolve_weight_quant_method_for_linear_load(
        args, shard, &prefixes, resolved);
    EXPECT_EQ(resolved, "w8a8_dynamic");
  }
  for (auto it = shards.rbegin(); it != shards.rend(); ++it) {
    resolve_weight_quant_method_for_linear_load(args, *it, &prefixes, resolved);
    EXPECT_EQ(resolved, "w8a8_dynamic");
  }
}

TEST(CompressedTensorsTest, KeepsIgnoredModulesFloatingPoint) {
  auto args = ct_args();
  const StateDict shard({{"weight", torch::ones({4, 8})}}, "model.mlp.gate.");
  std::optional<std::string> resolved = "w8a8_dynamic";
  resolve_weight_quant_method_for_linear_load(args, shard, nullptr, resolved);
  EXPECT_FALSE(resolved.has_value());
}

TEST(CompressedTensorsTest, PreservesExplicitQuantDescription) {
  auto args = ct_args();
  args.quant_descs() = {{"model.proj.weight", "w8a8"}};
  const StateDict shard({{"weight", torch::ones({4, 8}, torch::kInt8)}},
                        "model.proj.");
  std::optional<std::string> resolved;
  resolve_weight_quant_method_for_linear_load(args, shard, nullptr, resolved);
  EXPECT_EQ(resolved, "w8a8");
}

TEST(CompressedTensorsDeathTest, RejectsMixedFusionAndInvalidWeights) {
  auto args = ct_args();
  const std::vector<std::string> prefixes = {"gate.", "up."};
  std::optional<std::string> resolved;
  const StateDict mixed({{"up.weight", torch::ones({4, 8}, torch::kInt8)}},
                        "model.mlp.");
  EXPECT_DEATH(resolve_weight_quant_method_for_linear_load(
                   args, mixed, &prefixes, resolved),
               "Cannot fuse quantized and ignored");
  const StateDict floating({{"weight", torch::ones({4, 8})}}, "model.proj.");
  EXPECT_DEATH(resolve_weight_quant_method_for_linear_load(
                   args, floating, nullptr, resolved),
               "Weight dtype disagrees");
  const StateDict invalid_scale({{"weight_scale", torch::zeros({4, 1})}},
                                "model.proj.");
  EXPECT_DEATH(resolve_weight_quant_method_for_linear_load(
                   args, invalid_scale, nullptr, resolved),
               "finite and positive");
}

}  // namespace
}  // namespace xllm::layer
