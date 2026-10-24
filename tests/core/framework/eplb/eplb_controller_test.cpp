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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/model/model_args.h"

namespace xllm {
namespace {

class ScopedEplbEnabled final {
 public:
  explicit ScopedEplbEnabled(bool enabled)
      : previous_(EPLBConfig::get_instance().enable_eplb()) {
    EPLBConfig::get_instance().enable_eplb(enabled);
  }

  ~ScopedEplbEnabled() { EPLBConfig::get_instance().enable_eplb(previous_); }

 private:
  const bool previous_;
};

ModelArgs make_model_args() {
  ModelArgs args;
  args.n_layers(1).n_routed_experts(2);
  return args;
}

TEST(EplbControllerTest, ReturnsNullWhenDisabled) {
  ScopedEplbEnabled eplb_enabled(/*enabled=*/false);

  EXPECT_EQ(EplbController::create(make_model_args(),
                                   /*worker_num=*/2,
                                   /*ep_size=*/2),
            nullptr);
}

TEST(EplbControllerTest, AnnotatesInputsAndSubmitsWorkerResults) {
  ScopedEplbEnabled eplb_enabled(/*enabled=*/true);
  std::unique_ptr<EplbController> controller =
      EplbController::create(make_model_args(),
                             /*worker_num=*/2,
                             /*ep_size=*/2);
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

TEST(EplbControllerTest, AnnotatesGlobalMaskAcrossEmptyDpRank) {
  ScopedEplbEnabled eplb_enabled(/*enabled=*/true);
  std::unique_ptr<EplbController> controller =
      EplbController::create(make_model_args(),
                             /*worker_num=*/2,
                             /*ep_size=*/2);
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

}  // namespace
}  // namespace xllm
