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

#include "core/framework/speculative/adaptive_speculative_controller.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/speculative/speculative_profile_registry.h"
#include "runtime/options.h"

namespace xllm {
namespace {

runtime::Options make_options() {
  runtime::Options options;
  options.enable_adaptive_speculative_decode(true)
      .num_speculative_tokens(4)
      .speculative_algorithm("MTP")
      .enable_graph(false)
      .adaptive_speculative_min_gain(0.0);
  return options;
}

void setup_registry() {
  SpeculativeProfileRegistry::ValidateTimePredictor predictor;
  predictor.intercept_ms = 1.0;
  predictor.batch_ms = 0.1;
  predictor.query_token_ms = 0.5;
  predictor.query_prefix_ms = 0.001;
  SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
      predictor);
}

TEST(AdaptiveSpeculativeControllerTest, EnablesOnlyForMtpWithoutGraph) {
  runtime::Options options = make_options();
  AdaptiveSpeculativeController controller(options);
  EXPECT_TRUE(controller.enabled());

  options.speculative_algorithm("Eagle3");
  AdaptiveSpeculativeController eagle_controller(options);
  EXPECT_FALSE(eagle_controller.enabled());

  options = make_options();
  options.enable_graph(true);
  AdaptiveSpeculativeController graph_controller(options);
  EXPECT_FALSE(graph_controller.enabled());
}

TEST(AdaptiveSpeculativeControllerTest, SelectsPrunedPrefixesByPathProb) {
  setup_registry();
  AdaptiveSpeculativeController controller(make_options());
  torch::Tensor probs = torch::tensor({
      {0.95, 0.90, 0.20},
      {0.80, 0.00, 0.10},
  });
  std::vector<double> per_seq_kv_lens = {100.0, 100.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/1.0,
                                              /*target_step_time_ms=*/10.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_GE(prefix_lengths[0], 1);
  EXPECT_GE(prefix_lengths[1], 0);
}

TEST(AdaptiveSpeculativeControllerTest, AllowsZeroPrefixWhenNoGain) {
  setup_registry();
  AdaptiveSpeculativeController controller(make_options());
  torch::Tensor probs = torch::tensor({
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  });
  std::vector<double> per_seq_kv_lens = {100.0, 100.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/1.0,
                                              /*target_step_time_ms=*/100.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_EQ(prefix_lengths[0], 0);
  EXPECT_EQ(prefix_lengths[1], 0);
}

TEST(AdaptiveSpeculativeControllerTest, ContinuesAfterNonImprovingCandidate) {
  setup_registry();
  AdaptiveSpeculativeController controller(make_options());
  torch::Tensor probs = torch::tensor({
      {0.99, 0.98},
      {0.90, 0.00},
  });
  std::vector<double> per_seq_kv_lens = {100.0, 100.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/1.0,
                                              /*target_step_time_ms=*/100.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_GE(prefix_lengths[0], 1);
  EXPECT_GE(prefix_lengths[1], 0);
}

}  // namespace
}  // namespace xllm
