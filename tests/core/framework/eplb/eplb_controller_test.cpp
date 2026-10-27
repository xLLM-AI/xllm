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

#include "core/framework/eplb/eplb_controller.h"

#include <folly/Try.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/model/model_args.h"
#include "core/runtime/options.h"

namespace xllm {
namespace {

class EplbControllerTest : public testing::Test {
 protected:
  ~EplbControllerTest() override {
    EPLBConfig::get_instance() = previous_config_;
  }

 private:
  const EPLBConfig previous_config_ = EPLBConfig::get_instance();
};

ModelArgs make_model_args() {
  ModelArgs args;
  args.n_layers(1).n_routed_experts(2);
  return args;
}

TEST_F(EplbControllerTest, ReturnsNullWhenDisabled) {
  EPLBConfig::get_instance().enable_eplb(true);
  EplbControllerConfig config;
  config.policy.redundant_experts_num = -1;
  config.policy.eplb_update_interval = -1;
  config.policy.eplb_min_peak_load_improvement =
      std::numeric_limits<double>::quiet_NaN();
  config.policy.eplb_prepare_timeout_seconds = 0;

  EXPECT_EQ(EplbController::create(ModelArgs{},
                                   /*worker_num=*/0,
                                   /*ep_size=*/0,
                                   std::move(config)),
            nullptr);
  EXPECT_TRUE(EPLBConfig::get_instance().enable_eplb());
}

TEST_F(EplbControllerTest, AnnotatesInputsAndSubmitsWorkerResults) {
  EPLBConfig::get_instance().enable_eplb(false);
  std::unique_ptr<EplbController> controller =
      EplbController::create(make_model_args(),
                             /*worker_num=*/2,
                             /*ep_size=*/2,
                             EplbControllerConfig{.enabled = true});
  ASSERT_NE(controller, nullptr);

  std::vector<LlmForwardInput> inputs;
  inputs.resize(2);
  inputs[0].token_ids_host = torch::tensor({1, 2}, torch::kInt64);
  inputs[1].token_ids_host = torch::tensor({3}, torch::kInt64);
  inputs[0].input_params.expert.eplb_decode_token_mask =
      torch::tensor({true, false}, torch::kBool);
  inputs[1].input_params.expert.eplb_decode_token_mask =
      torch::tensor({true}, torch::kBool);

  controller->annotate_inputs(inputs,
                              /*dp_token_counts=*/{2, 1},
                              /*allow_eplb_command=*/false);

  EXPECT_TRUE(torch::equal(inputs[0].input_params.expert.eplb_decode_token_mask,
                           torch::tensor({true, false, true}, torch::kBool)));
  EXPECT_TRUE(torch::equal(inputs[1].input_params.expert.eplb_decode_token_mask,
                           torch::tensor({true, false, true}, torch::kBool)));
  EXPECT_EQ(inputs[0].input_params.expert.eplb_info.activation_token, -1);
  EXPECT_EQ(inputs[1].input_params.expert.eplb_info.activation_token, -1);

  controller->on_step_dispatched(inputs, /*is_graph_warmup=*/false);
  std::vector<folly::Try<std::optional<RawForwardOutput>>> results;
  results.reserve(2);
  for (const std::vector<int64_t>& expert_load_data :
       std::vector<std::vector<int64_t>>{{1, 2}, {3, 4}}) {
    RawForwardOutput output;
    output.expert_load_data = expert_load_data;
    results.emplace_back(std::optional<RawForwardOutput>(std::move(output)));
  }
  controller->on_step_completed(results, /*is_graph_warmup=*/false);
}

TEST_F(EplbControllerTest, AnnotatesGlobalMaskAcrossEmptyDpRank) {
  EPLBConfig::get_instance().enable_eplb(false);
  std::unique_ptr<EplbController> controller =
      EplbController::create(make_model_args(),
                             /*worker_num=*/2,
                             /*ep_size=*/2,
                             EplbControllerConfig{.enabled = true});
  ASSERT_NE(controller, nullptr);

  std::vector<LlmForwardInput> inputs(2);
  inputs[0].token_ids_host = torch::tensor({1, 2}, torch::kInt64);
  inputs[0].input_params.expert.eplb_decode_token_mask =
      torch::tensor({false, true}, torch::kBool);
  // An empty DP rank has no local mask. The controller must materialize an
  // empty one before concatenating the global mask used by every rank.
  inputs[1].token_ids_host = torch::empty({0}, torch::kInt64);

  controller->annotate_inputs(inputs,
                              /*dp_token_counts=*/{2, 0},
                              /*allow_eplb_command=*/false);

  const torch::Tensor expected_mask =
      torch::tensor({false, true}, torch::kBool);
  for (const LlmForwardInput& input : inputs) {
    EXPECT_TRUE(torch::equal(input.input_params.expert.eplb_decode_token_mask,
                             expected_mask));
    // The false eligibility bit must prevent a command from being handed out,
    // even while the input mask is still annotated.
    EXPECT_EQ(input.input_params.expert.eplb_info.activation_token, -1);
    EXPECT_EQ(input.input_params.expert.eplb_info.prepare_token, -1);
    EXPECT_EQ(input.input_params.expert.eplb_info.update_layer_id, -1);
  }
}

TEST_F(EplbControllerTest, InjectedPolicyRemainsStableWhenGlobalsDisagree) {
  EPLBConfig& global = EPLBConfig::get_instance();
  global.enable_eplb(false)
      .redundant_experts_num(3)
      .eplb_update_interval(17)
      .eplb_min_peak_load_improvement(0.75)
      .eplb_prepare_timeout_seconds(19)
      .eplb_policy_kind("balanced");
  EplbControllerConfig config;
  config.enabled = true;
  config.policy.redundant_experts_num = 0;
  config.policy.eplb_update_interval = 0;
  config.policy.eplb_min_peak_load_improvement = 0.0;
  config.policy.eplb_prepare_timeout_seconds = 2;
  config.policy.eplb_policy_kind = "greedy";
  std::unique_ptr<EplbController> controller =
      EplbController::create(make_model_args(),
                             /*worker_num=*/2,
                             /*ep_size=*/2,
                             config);
  ASSERT_NE(controller, nullptr);
  config.policy.redundant_experts_num = 9;

  std::vector<LlmForwardInput> inputs(1);
  controller->on_step_dispatched(inputs, /*is_graph_warmup=*/false);
  std::vector<folly::Try<std::optional<RawForwardOutput>>> results;
  results.reserve(2);
  for (int32_t rank = 0; rank < 2; ++rank) {
    RawForwardOutput output;
    // Two logical experts on two workers with no redundancy produce one
    // physical expert per worker, despite the conflicting global value.
    output.expert_load_data = {rank + 1};
    results.emplace_back(std::optional<RawForwardOutput>(std::move(output)));
  }
  controller->on_step_completed(results, /*is_graph_warmup=*/false);

  EXPECT_FALSE(global.enable_eplb());
  EXPECT_EQ(global.redundant_experts_num(), 3);
  EXPECT_EQ(global.eplb_update_interval(), 17);
  EXPECT_DOUBLE_EQ(global.eplb_min_peak_load_improvement(), 0.75);
  EXPECT_EQ(global.eplb_prepare_timeout_seconds(), 19);
  EXPECT_EQ(global.eplb_policy_kind(), "balanced");
}

TEST_F(EplbControllerTest, RejectsInvalidEnabledTopologyBeforeRuntime) {
  const EplbControllerConfig config{.enabled = true};
  EXPECT_DEATH(EplbController::create(make_model_args(),
                                      /*worker_num=*/2,
                                      /*ep_size=*/1,
                                      config),
               "worker_num.*ep_size");
  EXPECT_DEATH(EplbController::create(make_model_args(),
                                      /*worker_num=*/3,
                                      /*ep_size=*/3,
                                      config),
               "divisible");
}

TEST_F(EplbControllerTest,
       RuntimeOptionsCopiesRetainIndependentControllerConfig) {
  EplbControllerConfig config;
  config.enabled = true;
  config.policy.redundant_experts_num = 0;
  config.policy.eplb_update_interval = 7;
  config.policy.eplb_min_peak_load_improvement = 0.25;
  config.policy.eplb_prepare_timeout_seconds = 11;
  config.policy.eplb_policy_kind = "deepseek-hier";
  runtime::Options original;
  original.eplb_controller_config(std::move(config));

  runtime::Options target = original;
  runtime::Options draft = original;
  draft.is_draft_model(true);
  original.eplb_controller_config().enabled = false;
  original.eplb_controller_config().policy.eplb_policy_kind = "greedy";

  for (const runtime::Options* copy : {&target, &draft}) {
    const EplbControllerConfig& copied = copy->eplb_controller_config();
    EXPECT_TRUE(copied.enabled);
    EXPECT_EQ(copied.policy.redundant_experts_num, 0);
    EXPECT_EQ(copied.policy.eplb_update_interval, 7);
    EXPECT_DOUBLE_EQ(copied.policy.eplb_min_peak_load_improvement, 0.25);
    EXPECT_EQ(copied.policy.eplb_prepare_timeout_seconds, 11);
    EXPECT_EQ(copied.policy.eplb_policy_kind, "deepseek-hier");
  }
}

}  // namespace
}  // namespace xllm
