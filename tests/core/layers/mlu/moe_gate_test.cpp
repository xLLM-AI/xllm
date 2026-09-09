/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/mlu/moe_gate.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

class MoEGateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_args_ = test::create_default_model_args();
    quant_args_ = QuantArgs();  // use empty quant
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Set MoE gate related fields on model_args_. Call before constructing
  // MoEGateImpl in each test.
  void set_moe_gate_params(int64_t num_experts,
                           int64_t top_k,
                           int64_t num_expert_group,
                           int64_t topk_group,
                           double route_scale,
                           int64_t hidden_size,
                           bool renormalize,
                           const std::string& scoring_func,
                           const std::string& topk_method) {
    model_args_.n_routed_experts() = static_cast<int32_t>(num_experts);
    model_args_.num_experts_per_tok() = static_cast<int32_t>(top_k);
    model_args_.n_group() = static_cast<int32_t>(num_expert_group);
    model_args_.topk_group() = static_cast<int32_t>(topk_group);
    model_args_.routed_scaling_factor() = static_cast<float>(route_scale);
    model_args_.hidden_size() = hidden_size;
    model_args_.norm_topk_prob() = renormalize;
    model_args_.scoring_func() = scoring_func;
    model_args_.topk_method() = topk_method;
  }

  // Build state dict with seeded tensors for gate (bfloat16, no quant).
  std::unordered_map<std::string, torch::Tensor> create_gate_weights_seeded(
      int64_t num_experts,
      int64_t hidden_size,
      const std::string& topk_method) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    // Gate projection: ReplicatedLinear(hidden_size, num_experts, false, ...)
    // Weight shape [num_experts, hidden_size] (out_features, in_features).
    auto gate_weight = test::seeded_tensor("moe_gate_tests.gate_proj.weight",
                                           {num_experts, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
    weight_dict["weight"] = gate_weight;

    if (topk_method == "noaux_tc") {
      auto e_bias =
          test::seeded_tensor("moe_gate_tests.e_score_correction_bias",
                              {num_experts},
                              torch::kBFloat16,
                              options_.device());
      weight_dict["e_score_correction_bias"] = e_bias;
    }
    return weight_dict;
  }

  ModelArgs model_args_;
  // Run forward, assert shapes, and verify min/max/sum against expected values.
  void run_forward_and_expect(MoEGateImpl* moe_gate,
                              int64_t num_tokens,
                              int64_t hidden_size,
                              int64_t top_k,
                              const std::string& seed_key_prefix,
                              double expected_rw_min,
                              double expected_rw_max,
                              double expected_rw_sum,
                              double expected_eid_min,
                              double expected_eid_max,
                              double expected_eid_sum) {
    auto hidden_states = test::seeded_tensor(seed_key_prefix + ".hidden_states",
                                             {num_tokens, hidden_size},
                                             torch::kBFloat16,
                                             options_.device());
    auto [reduce_weight, expert_id] = moe_gate->forward(hidden_states);

    Device device(options_.device());
    device.synchronize_default_stream();

    ASSERT_TRUE(reduce_weight.defined()) << "reduce_weight should be defined";
    ASSERT_TRUE(expert_id.defined()) << "expert_id should be defined";
    ASSERT_EQ(reduce_weight.dim(), 2);
    ASSERT_EQ(expert_id.dim(), 2);
    ASSERT_EQ(reduce_weight.size(0), num_tokens);
    ASSERT_EQ(reduce_weight.size(1), top_k);
    ASSERT_EQ(expert_id.size(0), num_tokens);
    ASSERT_EQ(expert_id.size(1), top_k);
    double rw_sum =
        torch::sum(reduce_weight.flatten().to(torch::kFloat64)).item<double>();
    ASSERT_NE(rw_sum, 0.0) << "reduce_weight sum should not be zero";

    test::expect_tensor_stats(
        reduce_weight, expected_rw_min, expected_rw_max, expected_rw_sum);
    test::expect_tensor_stats(
        expert_id, expected_eid_min, expected_eid_max, expected_eid_sum);
  }

  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
};

// Sigmoid scoring: typical MoE config (multi-expert, top_k=2, groups).
TEST_F(MoEGateTest, Sigmoid) {
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool renormalize = true;
  const std::string scoring_func = "sigmoid";
  const std::string topk_method = "noaux_tc";

  set_moe_gate_params(num_experts,
                      top_k,
                      num_expert_group,
                      topk_group,
                      route_scale,
                      hidden_size,
                      renormalize,
                      scoring_func,
                      topk_method);
  MoEGateImpl moe_gate(model_args_, quant_args_, options_);
  auto weight_dict =
      create_gate_weights_seeded(num_experts, hidden_size, topk_method);
  StateDict state_dict(weight_dict);
  moe_gate.load_state_dict(state_dict);

  int64_t num_tokens = batch_size * seq_len;
  run_forward_and_expect(&moe_gate,
                         num_tokens,
                         hidden_size,
                         top_k,
                         "moe_gate_tests.sigmoid",
                         /*reduce_weight*/ 1.25,
                         1.25,
                         1280.0,
                         /*expert_id*/ 6.0,
                         7.0,
                         6656.0);
}

// Softmax scoring: same shape as sigmoid, different scoring path.
TEST_F(MoEGateTest, Softmax) {
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool renormalize = true;
  const std::string scoring_func = "softmax";
  const std::string topk_method = "";

  set_moe_gate_params(num_experts,
                      top_k,
                      num_expert_group,
                      topk_group,
                      route_scale,
                      hidden_size,
                      renormalize,
                      scoring_func,
                      topk_method);
  MoEGateImpl moe_gate(model_args_, quant_args_, options_);
  auto weight_dict =
      create_gate_weights_seeded(num_experts, hidden_size, topk_method);
  StateDict state_dict(weight_dict);
  moe_gate.load_state_dict(state_dict);

  int64_t num_tokens = batch_size * seq_len;
  run_forward_and_expect(&moe_gate,
                         num_tokens,
                         hidden_size,
                         top_k,
                         "moe_gate_tests.softmax",
                         /*reduce_weight*/ 3.16604e-14,
                         2.5,
                         1280.0,
                         /*expert_id*/ 0.0,
                         14.0,
                         4413.0);
}

// Sigmoid with topk_group=1
TEST_F(MoEGateTest, SigmoidTopkGroup1) {
  const int64_t batch_size = 8;
  const int64_t seq_len = 16;
  const int64_t hidden_size = 1024;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 1;
  const int64_t topk_group = 1;
  const int64_t top_k = 2;
  const double route_scale = 1.0;
  const bool renormalize = true;
  const std::string scoring_func = "sigmoid";
  const std::string topk_method = "";

  set_moe_gate_params(num_experts,
                      top_k,
                      num_expert_group,
                      topk_group,
                      route_scale,
                      hidden_size,
                      renormalize,
                      scoring_func,
                      topk_method);
  MoEGateImpl moe_gate(model_args_, quant_args_, options_);
  auto weight_dict =
      create_gate_weights_seeded(num_experts, hidden_size, topk_method);
  StateDict state_dict(weight_dict);
  moe_gate.load_state_dict(state_dict);

  int64_t num_tokens = batch_size * seq_len;
  run_forward_and_expect(&moe_gate,
                         num_tokens,
                         hidden_size,
                         top_k,
                         "moe_gate_tests.sigmoid_topk1",
                         /*reduce_weight*/ 0.5,
                         0.5,
                         128.0,
                         /*expert_id*/ 0.0,
                         1.0,
                         128.0);
}

}  // namespace layer
}  // namespace xllm
