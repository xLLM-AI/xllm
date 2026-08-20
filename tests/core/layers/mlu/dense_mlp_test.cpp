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

#include "layers/common/dense_mlp.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

class DenseMLPTest : public ::testing::Test {
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

    // Note: MLP will be created by individual test cases with their desired
    // dimensions
  }

  void TearDown() override {
    // Clean up if needed
  }

  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int64_t hidden_size,
      int64_t intermediate_size) {
    // Create test weights for gate_up_proj (gate + up projection)
    // Shape: [intermediate_size * 2, hidden_size]
    auto gate_up_weight =
        torch::full({intermediate_size * 2, hidden_size}, 5.0f, options_);

    // Create test weights for down_proj (down projection)
    // Shape: [hidden_size, intermediate_size] for RowParallelLinear
    auto down_weight =
        torch::full({hidden_size, intermediate_size}, 3.0f, options_);

    // For w8a8 smoothquant, we need to create quantized weights and scales
    // Create qweight (int8 quantized weights)
    auto gate_up_qweight = gate_up_weight.to(torch::kInt8);
    auto down_qweight = down_weight.to(torch::kInt8);

    // Create per_channel_scale (float32 scales for each channel)
    // For ColumnParallelLinear: per_channel_scale shape is
    // [out_features_per_partition]
    auto gate_up_scale = torch::full({intermediate_size},
                                     0.1f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));
    auto up_scale = torch::full({intermediate_size},
                                0.1f,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(options_.device()));
    auto down_scale = torch::full({hidden_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));

    // Create smooth quantization scales
    // For ColumnParallelLinear: smooth shape is [in_features] = [hidden_size]
    auto gate_up_smooth = torch::full({hidden_size},
                                      0.05f,
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(options_.device()));
    auto up_smooth = torch::full({hidden_size},
                                 0.05f,
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));
    // For RowParallelLinear: smooth shape is [in_features_per_partition] =
    // [intermediate_size]
    auto down_smooth = torch::full({intermediate_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

    // Create StateDict with w8a8 smoothquant weights
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    // Gate projection weights (ColumnParallelLinear)
    // gate_up_qweight shape: [intermediate_size * 2, hidden_size]
    // gate_proj.qweight should be first intermediate_size rows:
    // [intermediate_size, hidden_size]
    weight_dict["gate_proj.qweight"] =
        gate_up_qweight.slice(0, 0, intermediate_size);
    weight_dict["gate_proj.per_channel_scale"] = gate_up_scale;
    weight_dict["gate_proj.smooth"] = gate_up_smooth;

    // Up projection weights (ColumnParallelLinear)
    // up_proj.qweight should be last intermediate_size rows:
    // [intermediate_size, hidden_size]
    weight_dict["up_proj.qweight"] =
        gate_up_qweight.slice(0, intermediate_size, intermediate_size * 2);
    weight_dict["up_proj.per_channel_scale"] = up_scale;
    weight_dict["up_proj.smooth"] = up_smooth;

    // Down projection weights (RowParallelLinear)
    weight_dict["down_proj.qweight"] = down_qweight;
    weight_dict["down_proj.per_channel_scale"] = down_scale;
    weight_dict["down_proj.smooth"] = down_smooth;

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully";
    LOG(INFO) << "Gate qweight shape: "
              << weight_dict["gate_proj.qweight"].sizes();
    LOG(INFO) << "Gate per_channel_scale shape: "
              << weight_dict["gate_proj.per_channel_scale"].sizes();
    LOG(INFO) << "Gate smooth shape: "
              << weight_dict["gate_proj.smooth"].sizes();
    LOG(INFO) << "Up qweight shape: " << weight_dict["up_proj.qweight"].sizes();
    LOG(INFO) << "Up per_channel_scale shape: "
              << weight_dict["up_proj.per_channel_scale"].sizes();
    LOG(INFO) << "Up smooth shape: " << weight_dict["up_proj.smooth"].sizes();
    LOG(INFO) << "Down qweight shape: "
              << weight_dict["down_proj.qweight"].sizes();
    LOG(INFO) << "Down per_channel_scale shape: "
              << weight_dict["down_proj.per_channel_scale"].sizes();
    LOG(INFO) << "Down smooth shape: "
              << weight_dict["down_proj.smooth"].sizes();

    return weight_dict;
  }

  // Helper function to create MLP with custom dimensions
  DenseMLP create_mlp(int64_t hidden_size, int64_t intermediate_size) {
    // Create MLP with specified dimensions using the new constructor
    return DenseMLP(DenseMLPImpl(hidden_size,
                                 intermediate_size,
                                 /*is_gated=*/true,
                                 /*has_bias=*/false,
                                 /*hidden_act=*/"silu",
                                 /*enable_result_reduction=*/true,
                                 quant_args_,
                                 parallel_args_.tp_group_,
                                 options_));
  }

  // Helper function to create test weights for the MLP (w8a8 smoothquant
  // format)
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1) {
    // Use custom sizes if provided, otherwise use model_args_ values
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(test_hidden_size,
                                       test_intermediate_size);
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

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(DenseMLPTest, Bfloat16LoadStateDictTest) {
  // Test bfloat16 mode (non-quantized)
  const int64_t batch_size = 8;
  const int64_t hidden_size = 1024;
  const int64_t intermediate_size = 2048;

  // Create non-quantized quant_args for bfloat16 mode
  QuantArgs bfloat16_quant_args;  // Empty means no quantization

  // Create MLP in bfloat16 mode
  auto mlp = DenseMLP(DenseMLPImpl(hidden_size,
                                   intermediate_size,
                                   /*is_gated=*/true,
                                   /*has_bias=*/false,
                                   /*hidden_act=*/"silu",
                                   /*enable_result_reduction=*/true,
                                   bfloat16_quant_args,
                                   parallel_args_.tp_group_,
                                   options_));

  // Create simple test weights for bfloat16 mode
  std::unordered_map<std::string, torch::Tensor> weight_dict;

  // For bfloat16 mode, we need regular weight tensors
  auto gate_up_weight =
      torch::full({intermediate_size * 2, hidden_size}, 0.1f, options_);
  auto down_weight =
      torch::full({hidden_size, intermediate_size}, 0.1f, options_);

  weight_dict["gate_proj.weight"] =
      gate_up_weight.slice(0, 0, intermediate_size);
  weight_dict["up_proj.weight"] =
      gate_up_weight.slice(0, intermediate_size, intermediate_size * 2);
  weight_dict["down_proj.weight"] = down_weight;

  // Load weights
  StateDict state_dict(weight_dict);
  mlp->load_state_dict(state_dict);

  // Test forward pass
  auto hidden_states = torch::ones({batch_size, hidden_size}, options_);
  auto output = mlp->forward(hidden_states);

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), hidden_size) << "Hidden size should match";

  // Verify output is not all zeros
  auto output_sum = torch::sum(output).item<float>();
  ASSERT_NE(output_sum, 0.0f) << "Output should not be all zeros";

  LOG(INFO) << "Bfloat16 mode test passed - output sum: " << output_sum;
}

TEST_F(DenseMLPTest, SmoothquantLoadStateDictTest) {
  // Test loading weights into the MLP
  const int64_t batch_size = 16;
  const int64_t hidden_size = model_args_.hidden_size();
  const int64_t intermediate_size = model_args_.intermediate_size();

  // Create MLP with default dimensions
  auto mlp = create_mlp(hidden_size, intermediate_size);

  // Create test weights and load them (using default model_args_ dimensions)
  auto weight_dict = create_test_weights();

  // Load weights into the MLP
  StateDict state_dict(weight_dict);
  mlp->load_state_dict(state_dict);

  // Test forward pass with loaded weights
  auto hidden_states = torch::ones({batch_size, hidden_size}, options_);

  LOG(INFO) << "Testing forward pass with loaded weights";
  auto output = mlp->forward(hidden_states);

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), hidden_size) << "Hidden size should match";

  // Verify output is not all zeros (weights were loaded)
  auto output_sum = torch::sum(output).item<float>();
  ASSERT_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights";

  LOG(INFO) << "State dict loading test passed - output sum: " << output_sum;
}

#if defined(USE_CUDA)
TEST_F(DenseMLPTest, Fp8IgnoredDownProjLoadsAsUnquantized) {
  QuantArgs fp8_quant_args;
  fp8_quant_args.quant_method() = kQuantMethodFp8;
  fp8_quant_args.bits() = 8;
  fp8_quant_args.activation_dynamic() = false;
  fp8_quant_args.ignored_modules() = {"model.layers.1.mlp.down_proj"};

  const int64_t hidden_size = 16;
  const int64_t intermediate_size = 32;
  auto mlp = DenseMLP(DenseMLPImpl(hidden_size,
                                   intermediate_size,
                                   /*is_gated=*/true,
                                   /*has_bias=*/false,
                                   /*hidden_act=*/"silu",
                                   /*enable_result_reduction=*/true,
                                   fp8_quant_args,
                                   parallel_args_.tp_group_,
                                   options_,
                                   "model.layers.1.mlp"));

  std::unordered_map<std::string, torch::Tensor> weight_dict;
  auto fp8_weight_options = options_.dtype(torch::kFloat8_e4m3fn);
  auto scale_options = options_.dtype(torch::kFloat32);

  weight_dict["gate_proj.weight"] =
      torch::zeros({intermediate_size, hidden_size}, fp8_weight_options);
  weight_dict["gate_proj.weight_scale"] = torch::ones({1}, scale_options);
  weight_dict["gate_proj.input_scale"] = torch::ones({1}, scale_options);

  weight_dict["up_proj.weight"] =
      torch::zeros({intermediate_size, hidden_size}, fp8_weight_options);
  weight_dict["up_proj.weight_scale"] = torch::ones({1}, scale_options);
  weight_dict["up_proj.input_scale"] = torch::ones({1}, scale_options);

  weight_dict["down_proj.weight"] =
      torch::zeros({hidden_size, intermediate_size}, options_);

  StateDict state_dict(weight_dict);
  mlp->load_state_dict(state_dict);

  const auto params = mlp->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(params.contains("gate_up_proj.weight_scale"));
  EXPECT_TRUE(params.contains("gate_up_proj.input_scale"));
  EXPECT_TRUE(params.contains("down_proj.weight"));
  EXPECT_FALSE(params.contains("down_proj.weight_scale"));
  EXPECT_FALSE(params.contains("down_proj.input_scale"));
}
#endif

TEST_F(DenseMLPTest, SmoothquantPrecisionVerificationTest) {
  // Test precision verification with custom input and expected output
  const int64_t batch_size = 16;

  // Use custom hidden and intermediate sizes for more controllable output
  const int64_t custom_hidden_size = 7168;
  const int64_t custom_intermediate_size = 9216;

  // Create custom MLP with smaller dimensions
  auto custom_mlp = create_mlp(custom_hidden_size, custom_intermediate_size);

  // Create test weights and load them with custom dimensions
  auto weight_dict =
      create_test_weights(custom_hidden_size, custom_intermediate_size);

  // Load weights into the MLP
  StateDict state_dict(weight_dict);
  custom_mlp->load_state_dict(state_dict);

  // Create custom input tensor for precision testing
  // You can modify these values as needed for your specific test case
  std::vector<float> input_values;

  // Fill input_values with test data using custom dimensions
  input_values.reserve(batch_size * custom_hidden_size);
  // Populate the input_values vector:
  for (size_t i = 0; i < batch_size; ++i) {
    float value = 0.5f;
    for (size_t j = 0; j < custom_hidden_size; ++j) {
      input_values.push_back(value);
    }
  }

  auto hidden_states =
      create_custom_input({batch_size, custom_hidden_size}, input_values);

  LOG(INFO) << "Testing precision verification with custom input (hidden_size="
            << custom_hidden_size << ")";
  auto output = custom_mlp->forward(hidden_states);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), custom_hidden_size) << "Hidden size should match";

  // Set expected output values for precision verification
  // TODO: Replace these placeholder values with your expected output
  // The expected values should be calculated based on your specific test case
  std::vector<float> expected_values;

  // Fill expected_values with placeholder data using custom dimensions
  expected_values.reserve(batch_size * custom_hidden_size);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < custom_hidden_size; ++j) {
      expected_values.push_back(1105920.0f);  // calculated via vLLM MLU
    }
  }

  set_expected_output(expected_values);

  // Note: The precision verification is commented out until you set the
  // expected values Uncomment the following line after setting the correct
  verify_precision(output, 1e-3, 1e-4);
}

}  // namespace layer
}  // namespace xllm
