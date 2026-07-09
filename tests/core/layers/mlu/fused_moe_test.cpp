/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/mlu/fused_moe.h"

#include <glog/logging.h>
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

class FusedMoETest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize default model arguments for testing
    model_args_ = test::create_default_model_args();

    // Initialize w8a8 quantization arguments
    quant_args_ = test::create_default_quant_args();

    // Initialize tensor options
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);

    // Create mock ProcessGroup and initialize ParallelArgs
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);

    // Note: FusedMoE will be created by individual test cases with their
    // desired dimensions
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Helper function to create router logits tensor
  torch::Tensor create_router_logits(const std::vector<int64_t>& shape,
                                     const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int64_t num_experts,
      int64_t hidden_size,
      int64_t intermediate_size) {
    // Create test weights for each expert
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";

      // Create gate_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto gate_weight =
          torch::full({intermediate_size, hidden_size}, 2.0f, options_);
      auto gate_qweight = gate_weight.to(torch::kInt8);
      auto gate_scale = torch::full({intermediate_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto gate_smooth = torch::full({hidden_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      // Create up_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto up_weight =
          torch::full({intermediate_size, hidden_size}, 2.0f, options_);
      auto up_qweight = up_weight.to(torch::kInt8);
      auto up_scale = torch::full({intermediate_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));
      auto up_smooth = torch::full({hidden_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      // Create down_proj weights (RowParallelLinear)
      // Shape: [hidden_size, intermediate_size]
      auto down_weight =
          torch::full({hidden_size, intermediate_size}, 3.0f, options_);
      auto down_qweight = down_weight.to(torch::kInt8);
      auto down_scale = torch::full({hidden_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto down_smooth = torch::full({intermediate_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      // Add weights to dictionary
      // expert
      weight_dict[expert_prefix + "gate_proj.qweight"] = gate_qweight;
      weight_dict[expert_prefix + "gate_proj.per_channel_scale"] = gate_scale;
      weight_dict[expert_prefix + "gate_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "up_proj.qweight"] = up_qweight;
      weight_dict[expert_prefix + "up_proj.per_channel_scale"] = up_scale;
      weight_dict[expert_prefix + "up_proj.smooth"] = up_smooth;

      weight_dict[expert_prefix + "down_proj.qweight"] = down_qweight;
      weight_dict[expert_prefix + "down_proj.per_channel_scale"] = down_scale;
      weight_dict[expert_prefix + "down_proj.smooth"] = down_smooth;
    }

    // gate weight generation
    auto gate_weight = torch::full({num_experts, hidden_size}, 5.0f, options_);
    auto e_score_correction_bias = torch::full({num_experts}, 0.1f, options_);

    // Create shared experts weights
    auto shared_expert_up_weight =
        torch::full({intermediate_size, hidden_size}, 1.5f, options_);
    auto shared_expert_up_qweight = shared_expert_up_weight.to(torch::kInt8);
    auto shared_expert_up_scale = torch::full({intermediate_size},
                                              0.1f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));
    auto shared_expert_up_smooth = torch::full({hidden_size},
                                               0.05f,
                                               torch::TensorOptions()
                                                   .dtype(torch::kFloat32)
                                                   .device(options_.device()));
    auto shared_expert_gate_weight =
        torch::full({intermediate_size, hidden_size}, 1.5f, options_);
    auto shared_expert_gate_qweight =
        shared_expert_gate_weight.to(torch::kInt8);
    auto shared_expert_gate_scale = torch::full({intermediate_size},
                                                0.1f,
                                                torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(options_.device()));
    auto shared_expert_gate_smooth =
        torch::full({hidden_size},
                    0.05f,
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(options_.device()));
    auto shared_expert_down_weight =
        torch::full({hidden_size, intermediate_size}, 1.3f, options_);
    auto shared_expert_down_qweight =
        shared_expert_down_weight.to(torch::kInt8);
    auto shared_expert_down_scale = torch::full({hidden_size},
                                                0.1f,
                                                torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(options_.device()));
    auto shared_expert_down_smooth =
        torch::full({intermediate_size},
                    0.05f,
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(options_.device()));

    // gate
    weight_dict["gate.weight"] = gate_weight;
    weight_dict["gate.e_score_correction_bias"] = e_score_correction_bias;

    // shared experts
    weight_dict["shared_experts.up_proj.qweight"] = shared_expert_up_qweight;
    weight_dict["shared_experts.up_proj.per_channel_scale"] =
        shared_expert_up_scale;
    weight_dict["shared_experts.up_proj.smooth"] = shared_expert_up_smooth;
    weight_dict["shared_experts.gate_proj.qweight"] =
        shared_expert_gate_qweight;
    weight_dict["shared_experts.gate_proj.per_channel_scale"] =
        shared_expert_gate_scale;
    weight_dict["shared_experts.gate_proj.smooth"] = shared_expert_gate_smooth;
    weight_dict["shared_experts.down_proj.qweight"] =
        shared_expert_down_qweight;
    weight_dict["shared_experts.down_proj.per_channel_scale"] =
        shared_expert_down_scale;
    weight_dict["shared_experts.down_proj.smooth"] = shared_expert_down_smooth;

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully for "
              << num_experts << " experts";
    LOG(INFO) << "Hidden size: " << hidden_size
              << ", Intermediate size: " << intermediate_size;

    return weight_dict;
  }

  std::unordered_map<std::string, torch::Tensor>
  create_w4a8_groupwise_test_weights(int64_t num_experts,
                                     int64_t hidden_size,
                                     int64_t intermediate_size,
                                     int64_t group_size) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";
      std::string seed_prefix =
          "fused_moe_tests.expert_" + std::to_string(expert_id);
      test::append_w4a8_expert_weights(weight_dict,
                                       expert_prefix,
                                       seed_prefix,
                                       hidden_size,
                                       intermediate_size,
                                       intermediate_size,
                                       intermediate_size,
                                       group_size,
                                       options_.device());
    }

    weight_dict["gate.weight"] =
        torch::full({num_experts, hidden_size}, 5.0f, options_);
    weight_dict["gate.e_score_correction_bias"] =
        torch::full({num_experts}, 0.1f, options_);

    return weight_dict;
  }

  FusedMoE create_fused_moe(int64_t num_experts,
                            int64_t top_k,
                            int64_t num_expert_group,
                            int64_t topk_group,
                            double route_scale,
                            int64_t hidden_size,
                            int64_t intermediate_size,
                            int64_t n_shared_experts = 1,
                            bool is_gated = true,
                            int64_t renormalize = 0,
                            bool enable_result_reduction = true,
                            const std::string& hidden_act = "silu",
                            const std::string& scoring_func = "sigmoid",
                            const std::string& topk_method = "noaux_tc") {
    ModelArgs args = model_args_;
    args.n_routed_experts() = static_cast<int32_t>(num_experts);
    args.num_experts_per_tok() = static_cast<int32_t>(top_k);
    args.n_group() = static_cast<int32_t>(num_expert_group);
    args.topk_group() = static_cast<int32_t>(topk_group);
    args.routed_scaling_factor() = static_cast<float>(route_scale);
    args.hidden_size() = hidden_size;
    args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
    args.n_shared_experts() = static_cast<int32_t>(n_shared_experts);
    args.norm_topk_prob() = (renormalize != 0);
    args.hidden_act() = hidden_act;
    args.scoring_func() = scoring_func;
    args.topk_method() = topk_method;
    const FusedMoEArgs moe_args{
        .is_gated = is_gated,
        .enable_result_reduction = enable_result_reduction};
    return FusedMoE(
        FusedMoEImpl(args, moe_args, quant_args_, parallel_args_, options_));
  }

  void set_tp_ctx(int64_t world_size) {
    mock_process_group_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    parallel_args_ = ParallelArgs(/*rank=*/0,
                                  world_size,
                                  /*dp_size=*/1,
                                  mock_process_group_.get());
    parallel_args_.process_group_ = mock_process_group_.get();
    parallel_args_.tp_group_ = mock_process_group_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
  }

  // Helper function to create test weights for the FusedMoE (w8a8 smoothquant
  // format)
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int64_t num_experts,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1) {
    // Use custom sizes if provided, otherwise use model_args_ values
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(
        num_experts, test_hidden_size, test_intermediate_size);
  }

  // Helper function to verify tensor values are close to expected
  void verify_tensor_close(const torch::Tensor& actual,
                           const torch::Tensor& expected,
                           double rtol = 1e-5,
                           double atol = 1e-8) {
    test::verify_tensor_close(actual, expected, rtol, atol);
  }

  // Helper function to create custom input tensor for precision testing
  torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                    const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  // Helper function to set expected output for precision verification
  void set_expected_output(const std::vector<float>& expected_values) {
    expected_output_ = expected_values;
  }

  // Helper function to verify precision against expected output
  void verify_precision(const torch::Tensor& actual_output,
                        double rtol = 1e-3,
                        double atol = 1e-4) {
    test::verify_precision(actual_output, expected_output_, rtol, atol);
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;

  // Helper to create a mock ProcessGroup for testing
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::unique_ptr<xllm::ProcessGroup> single_rank_pg_;

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(FusedMoETest, LoadStateDictTest) {
  // Test loading weights into the FusedMoE
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t intermediate_size = 2048;
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const int64_t renormalize = 1;
  const int64_t n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = create_fused_moe(num_experts,
                                    top_k,
                                    num_expert_group,
                                    topk_group,
                                    route_scale,
                                    hidden_size,
                                    intermediate_size,
                                    n_shared_experts,
                                    gated,
                                    renormalize);

  // Create test weights and load them
  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);

  CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  CHECK_EQ(output.size(0), batch_size * seq_len)
      << "The number of tokens should match";
  CHECK_EQ(output.size(1), hidden_size) << "The hidden size should match";
  auto output_sum = torch::sum(output).item<float>();
  CHECK_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights";

  LOG(INFO) << "State dict loading test passed - output sum: " << output_sum;
}

TEST_F(FusedMoETest, PrecisionVerificationTest) {
  // Test loading weights into the FusedMoE
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t intermediate_size = 2048;
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const int64_t renormalize = 1;
  const int64_t n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = create_fused_moe(num_experts,
                                    top_k,
                                    num_expert_group,
                                    topk_group,
                                    route_scale,
                                    hidden_size,
                                    intermediate_size,
                                    n_shared_experts,
                                    gated,
                                    renormalize);

  // Create test weights and load them
  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  CHECK_EQ(output.size(0), batch_size * seq_len)
      << "Batch size * seq_len should match";
  CHECK_EQ(output.size(1), hidden_size) << "Hidden size should match";

  // Set expected output values for precision verification
  // The expected values should be calculated based on your specific test case
  std::vector<float> expected_values;

  // Fill expected_values with placeholder data using custom dimensions
  expected_values.reserve(batch_size * seq_len * hidden_size);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < seq_len; ++j) {
      for (size_t k = 0; k < hidden_size; ++k) {
        expected_values.push_back(992.0f);  // calculated via vLLM MLU
      }
    }
  }

  set_expected_output(expected_values);

  verify_precision(output, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, W4A8GroupwiseSmokeTest) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t group_size = 128;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  quant_args_.moe_weight_bits() = 4;
  quant_args_.group_size() = group_size;

  auto fused_moe = create_fused_moe(num_experts,
                                    top_k,
                                    num_expert_group,
                                    topk_group,
                                    route_scale,
                                    hidden_size,
                                    intermediate_size,
                                    /*n_shared_experts=*/0);

  auto weight_dict = create_w4a8_groupwise_test_weights(
      num_experts, hidden_size, intermediate_size, group_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  EXPECT_EQ(output.dim(), 2);
  EXPECT_EQ(output.size(0), batch_size * seq_len);
  EXPECT_EQ(output.size(1), hidden_size);
  EXPECT_NE(torch::sum(output).item<float>(), 0.0f);
}

TEST_F(FusedMoETest, PrepRouteMatchesBaseForward) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  auto fused_moe = create_fused_moe(num_experts,
                                    top_k,
                                    num_expert_group,
                                    topk_group,
                                    route_scale,
                                    hidden_size,
                                    intermediate_size,
                                    /*n_shared_experts=*/0);

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto expected =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);
  auto route_info = fused_moe->prep_route(hidden_states);
  auto actual =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 route_info);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, SelectedRouteMatchesGateRouteWith3DInput) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  FusedMoE fused_moe = create_fused_moe(num_experts,
                                        top_k,
                                        num_expert_group,
                                        topk_group,
                                        route_scale,
                                        hidden_size,
                                        intermediate_size,
                                        /*n_shared_experts=*/0);

  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  torch::Tensor hidden_states = create_custom_input(
      {batch_size, seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({batch_size * seq_len, hidden_size});
  FusedMoEImpl::RouteInfo route_info = fused_moe->prep_route(hidden_states_2d);
  route_info.expert_id = route_info.expert_id.to(torch::kLong);

  torch::Tensor expected =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);
  torch::Tensor actual = fused_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false, route_info);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  EXPECT_EQ(actual.dim(), hidden_states.dim());
  EXPECT_EQ(actual.size(0), batch_size);
  EXPECT_EQ(actual.size(1), seq_len);
  EXPECT_EQ(actual.size(2), hidden_size);
  verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, SelectedRouteRejectsInvalidRouteShape) {
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  FusedMoE fused_moe = create_fused_moe(num_experts,
                                        top_k,
                                        num_expert_group,
                                        topk_group,
                                        route_scale,
                                        hidden_size,
                                        intermediate_size,
                                        /*n_shared_experts=*/0);

  torch::Tensor hidden_states = create_custom_input(
      {3, hidden_size}, std::vector<float>(3 * hidden_size, 0.05f));
  FusedMoEImpl::RouteInfo bad_topk_route{
      .reduce_weight = torch::ones({3, top_k + 1}, options_),
      .expert_id = torch::zeros({3, top_k + 1}, options_.dtype(torch::kInt32))};
  FusedMoEImpl::RouteInfo bad_token_route{
      .reduce_weight = torch::ones({2, top_k}, options_),
      .expert_id = torch::zeros({2, top_k}, options_.dtype(torch::kInt32))};

  EXPECT_DEATH(
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 bad_topk_route),
      "topk mismatch");

  EXPECT_DEATH(
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 bad_token_route),
      "token count mismatch");
}

TEST_F(FusedMoETest, NoReductionModeMatchesExternalReduce) {
  set_tp_ctx(/*world_size=*/2);

  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  auto reduced_moe = create_fused_moe(num_experts,
                                      top_k,
                                      num_expert_group,
                                      topk_group,
                                      route_scale,
                                      hidden_size,
                                      intermediate_size,
                                      /*n_shared_experts=*/1,
                                      /*is_gated=*/true,
                                      /*renormalize=*/0,
                                      /*enable_result_reduction=*/true);
  auto raw_moe = create_fused_moe(num_experts,
                                  top_k,
                                  num_expert_group,
                                  topk_group,
                                  route_scale,
                                  hidden_size,
                                  intermediate_size,
                                  /*n_shared_experts=*/1,
                                  /*is_gated=*/true,
                                  /*renormalize=*/0,
                                  /*enable_result_reduction=*/false);

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  reduced_moe->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto expected = reduced_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto actual = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  actual = parallel_state::reduce(actual, parallel_args_.tp_group_);

  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());
  ASSERT_EQ(raw_moe->shared_pg(), parallel_args_.single_rank_group_);
  shared = parallel_state::reduce(shared, raw_moe->shared_pg());
  actual = actual + shared;

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, DeepSeekV4MoETest) {
  const int64_t batch_size = 2;
  const int64_t seq_len = 4;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 4;
  const int64_t num_expert_group = 2;
  const int64_t topk_group = 2;
  const int64_t top_k = 2;
  const double route_scale = 1.0;
  const float swiglu_limit = 10.0f;

  ModelArgs args = model_args_;
  args.n_routed_experts() = static_cast<int32_t>(num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(top_k);
  args.n_group() = static_cast<int32_t>(num_expert_group);
  args.topk_group() = static_cast<int32_t>(topk_group);
  args.routed_scaling_factor() = static_cast<float>(route_scale);
  args.hidden_size() = hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
  args.n_shared_experts() = 1;
  args.hidden_act() = "silu";
  args.model_type() = "deepseek_v4";
  args.scoring_func() = "sqrtsoftplus";
  args.swiglu_limit() = swiglu_limit;

  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = true, .use_hash = false};

  auto fused_moe = FusedMoE(
      FusedMoEImpl(args, moe_args, quant_args_, parallel_args_, options_));

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 1.0f));

  auto output = fused_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.dim(), 2);
  ASSERT_EQ(output.size(0), batch_size * seq_len);
  ASSERT_EQ(output.size(1), hidden_size);

  auto output_sum = torch::sum(output).item<float>();
  ASSERT_TRUE(std::isfinite(output_sum))
      << "Output should be finite with swiglu_limit";

  LOG(INFO) << "DeepSeek V4 MoE forward test passed, output sum: "
            << output_sum;
}

TEST_F(FusedMoETest, DeepSeekV4HashMoETest) {
  const int64_t batch_size = 2;
  const int64_t seq_len = 4;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 4;
  const int64_t num_expert_group = 2;
  const int64_t topk_group = 2;
  const int64_t top_k = 2;
  const int64_t vocab_size = 1024;
  const double route_scale = 1.0;
  const float swiglu_limit = 10.0f;

  ModelArgs args = model_args_;
  args.n_routed_experts() = static_cast<int32_t>(num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(top_k);
  args.n_group() = static_cast<int32_t>(num_expert_group);
  args.topk_group() = static_cast<int32_t>(topk_group);
  args.routed_scaling_factor() = static_cast<float>(route_scale);
  args.hidden_size() = hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
  args.n_shared_experts() = 1;
  args.hidden_act() = "silu";
  args.model_type() = "deepseek_v4";
  args.scoring_func() = "sqrtsoftplus";
  args.swiglu_limit() = swiglu_limit;
  args.vocab_size() = vocab_size;

  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = true, .use_hash = true};

  auto fused_moe = FusedMoE(
      FusedMoEImpl(args, moe_args, quant_args_, parallel_args_, options_));

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  weight_dict["gate.tid2eid"] =
      torch::full({vocab_size, top_k}, 0.1f, options_);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 1.0f));
  auto input_ids = torch::randint(
      0, vocab_size, {batch_size * seq_len}, options_.dtype(torch::kInt32));
  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 /*route_info=*/std::nullopt,
                                 input_ids);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.dim(), 2);
  ASSERT_EQ(output.size(0), batch_size * seq_len);
  ASSERT_EQ(output.size(1), hidden_size);

  auto output_sum = torch::sum(output).item<float>();
  ASSERT_TRUE(std::isfinite(output_sum))
      << "Output should be finite with swiglu_limit";

  LOG(INFO) << "DeepSeek V4 hash MoE forward test passed, output sum: "
            << output_sum;
}
}  // namespace layer
}  // namespace xllm
