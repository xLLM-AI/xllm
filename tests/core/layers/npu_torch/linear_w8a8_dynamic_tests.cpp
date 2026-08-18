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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/flash_comm1_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/ops_api.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/linear.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class NpuLinearW8A8TestBase : public ::testing::Test {
 protected:
  void init_quant_args(const std::string& quantize_type,
                       bool activation_dynamic) {
    quant_args_.quantize_type() = quantize_type;
    quant_args_.quant_method() = "";
    quant_args_.activation_dynamic() = activation_dynamic;

    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  }

  torch::Tensor make_input(const std::string& key,
                           const std::vector<int64_t>& shape) const {
    auto input =
        test::seeded_tensor(key, shape, torch::kFloat32, options_.device());
    return (input * 0.1f).to(options_);
  }

  torch::Tensor make_input(const std::string& key,
                           int64_t batch_size,
                           int64_t in_features) const {
    return make_input(key, {batch_size, in_features});
  }

  torch::Tensor make_qweight(const std::string& key,
                             int64_t out_features,
                             int64_t in_features) const {
    return test::seeded_tensor(
        key, {out_features, in_features}, torch::kInt8, options_.device());
  }

  torch::Tensor make_weight_scale(const std::string& key,
                                  int64_t out_features) const {
    auto scale = test::seeded_tensor(
        key, {out_features}, torch::kFloat32, options_.device());
    return scale * 0.02f + 0.01f;
  }

  torch::Tensor make_weight_offset(const std::string& key,
                                   int64_t out_features) const {
    auto offset = test::seeded_tensor(
        key, {out_features}, torch::kFloat32, options_.device());
    return offset * 0.2f - 0.1f;
  }

  void add_quant_desc(const std::string& prefix) {
    quant_args_.quant_descs()[prefix + ".weight"] = quant_args_.quantize_type();
  }

  StateDict make_quant_state_dict(
      std::unordered_map<std::string, torch::Tensor> weight_dict,
      const std::string& prefix) {
    return StateDict(std::move(weight_dict), prefix + ".");
  }

  // weight_offset is still loaded to verify checkpoint compatibility, but the
  // aligned dynamic forward path now follows the Python reference and ignores
  // it when building the math expectation.
  torch::Tensor make_bias(const std::string& key, int64_t out_features) const {
    auto bias = test::seeded_tensor(
        key, {out_features}, torch::kFloat32, options_.device());
    return (bias * 0.1f).to(options_);
  }

  void expect_output_close(const torch::Tensor& actual,
                           const torch::Tensor& expected) const {
    auto actual_fp32 = actual.to(torch::kFloat32).cpu();
    auto expected_fp32 = expected.to(torch::kFloat32).cpu();
    ASSERT_TRUE(torch::allclose(actual_fp32, expected_fp32, 5e-2, 5e-2));
    ASSERT_TRUE(torch::isfinite(actual_fp32).all().item<bool>())
        << "Output has non-finite values";
  }

  QuantArgs quant_args_;
  torch::TensorOptions options_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
};

class NpuLinearW8A8DynamicTest : public NpuLinearW8A8TestBase {
 protected:
  void SetUp() override { init_quant_args("w8a8_dynamic", true); }

  torch::Tensor make_reference_output(const torch::Tensor& input,
                                      const torch::Tensor& qweight,
                                      const torch::Tensor& weight_scale,
                                      const torch::Tensor& bias) const {
    auto weight = qweight.to(torch::kFloat32) * weight_scale.view({-1, 1});
    auto output = torch::matmul(input, weight.to(input.scalar_type()).t());
    if (bias.defined()) {
      output = output + bias;
    }
    return output;
  }

  torch::Tensor make_unflattened_kernel_reference_output(
      const torch::Tensor& input,
      const torch::Tensor& qweight,
      const torch::Tensor& weight_scale,
      const std::optional<torch::Tensor>& bias) const {
    xllm::kernel::NpuQuantizeParams quant_params;
    quant_params.input = input;

    torch::Tensor quantized_input;
    std::optional<torch::Tensor> pertoken_scale;
    std::tie(quantized_input, pertoken_scale) =
        xllm::kernel::dynamic_quant(quant_params);
    CHECK(pertoken_scale.has_value() && pertoken_scale->defined());

    xllm::kernel::QuantMatmulParams matmul_params;
    matmul_params.x1 = quantized_input;
    matmul_params.x2 = qweight;
    matmul_params.transpose2 = true;
    matmul_params.scale = weight_scale;
    matmul_params.pertoken_scale = pertoken_scale;
    matmul_params.output_dtype = input.scalar_type();

    auto output = xllm::kernel::quant_matmul(matmul_params);
    if (bias.has_value() && bias->defined()) {
      output = output + bias.value();
    }
    return output;
  }
};

class NpuLinearW8A8StaticTest : public NpuLinearW8A8TestBase {
 protected:
  void SetUp() override { init_quant_args("w8a8", false); }

  torch::Tensor make_input_scale() const {
    return torch::full({1},
                       0.02f,
                       torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(options_.device()));
  }

  torch::Tensor make_input_offset() const {
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kInt8).device(options_.device()));
  }

  torch::Tensor make_deq_scale(const std::string& key,
                               int64_t out_features) const {
    return make_weight_scale(key, out_features);
  }

  torch::Tensor make_quant_bias(const std::string& key,
                                int64_t out_features) const {
    auto bias = test::seeded_tensor(
        key, {out_features}, torch::kFloat32, options_.device());
    return (bias * 8.0f).to(torch::kInt32);
  }

  torch::Tensor make_unflattened_kernel_reference_output(
      const torch::Tensor& input,
      const torch::Tensor& qweight,
      const torch::Tensor& input_scale,
      const torch::Tensor& input_offset,
      const torch::Tensor& deq_scale,
      const std::optional<torch::Tensor>& quant_bias) const {
    xllm::kernel::NpuQuantizeParams quant_params;
    quant_params.input = input;
    quant_params.scale = input_scale;
    quant_params.zero_point = input_offset;
    quant_params.axis = -1;

    auto quantized_input = xllm::kernel::quantize(quant_params);

    xllm::kernel::QuantMatmulParams matmul_params;
    matmul_params.x1 = quantized_input;
    matmul_params.x2 = qweight;
    matmul_params.transpose2 = true;
    matmul_params.scale = deq_scale;
    matmul_params.bias = quant_bias;
    matmul_params.output_dtype = input.scalar_type();

    return xllm::kernel::quant_matmul(matmul_params);
  }
};

TEST_F(NpuLinearW8A8DynamicTest, ColumnParallelLinearLoadAndForward) {
  const int64_t batch_size = 3;
  const int64_t in_features = 16;
  const int64_t out_features = 12;
  const std::string prefix = "npu.linear.column";
  add_quant_desc(prefix);

  auto linear =
      ColumnParallelLinear(ColumnParallelLinearImpl(in_features,
                                                    out_features,
                                                    /*bias=*/true,
                                                    /*gather=*/false,
                                                    quant_args_,
                                                    parallel_args_.tp_group_,
                                                    options_));

  auto weight =
      make_qweight("npu.linear.column.weight", out_features, in_features);
  auto weight_scale =
      make_weight_scale("npu.linear.column.scale", out_features);
  auto weight_offset =
      make_weight_offset("npu.linear.column.offset", out_features);
  auto bias = make_bias("npu.linear.column.bias", out_features);

  std::unordered_map<std::string, torch::Tensor> weight_dict = {
      {"weight", weight},
      {"weight_scale", weight_scale},
      {"weight_offset", weight_offset},
      {"bias", bias},
  };
  StateDict state_dict = make_quant_state_dict(std::move(weight_dict), prefix);
  linear->load_state_dict(state_dict);

  EXPECT_EQ(linear->weight().scalar_type(), torch::kInt8);

  auto input = make_input("npu.linear.column.input", batch_size, in_features);
  auto output = linear->forward(input);
  Device(options_.device()).synchronize_default_stream();

  auto expected = make_reference_output(input, weight, weight_scale, bias);
  ASSERT_TRUE(output.sizes() == expected.sizes());
  expect_output_close(output, expected);
}

TEST_F(NpuLinearW8A8DynamicTest, DenseMlpGathersQuantizedActivation) {
  const int32_t num_tokens = 4;
  const int64_t hidden_size = 16;
  const int64_t intermediate_size = 20;
  const std::string prefix = "npu.mlp.quantized_gather";
  add_quant_desc(prefix + ".gate_proj");
  add_quant_desc(prefix + ".up_proj");
  add_quant_desc(prefix + ".down_proj");

  mock_process_group_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  test::MockProcessGroup* mock_group =
      static_cast<test::MockProcessGroup*>(mock_process_group_.get());
  auto mlp = DenseMLP(DenseMLPImpl(hidden_size,
                                   intermediate_size,
                                   /*is_gated=*/true,
                                   /*has_bias=*/false,
                                   /*hidden_act=*/"silu",
                                   /*enable_result_reduction=*/true,
                                   quant_args_,
                                   mock_group,
                                   options_,
                                   prefix));

  std::unordered_map<std::string, torch::Tensor> weight_dict = {
      {"gate_proj.weight",
       make_qweight("npu.mlp.quantized_gather.gate.weight",
                    intermediate_size,
                    hidden_size)},
      {"gate_proj.weight_scale",
       make_weight_scale("npu.mlp.quantized_gather.gate.scale",
                         intermediate_size)},
      {"gate_proj.weight_offset",
       torch::zeros({intermediate_size}, options_.dtype(torch::kFloat32))},
      {"up_proj.weight",
       make_qweight("npu.mlp.quantized_gather.up.weight",
                    intermediate_size,
                    hidden_size)},
      {"up_proj.weight_scale",
       make_weight_scale("npu.mlp.quantized_gather.up.scale",
                         intermediate_size)},
      {"up_proj.weight_offset",
       torch::zeros({intermediate_size}, options_.dtype(torch::kFloat32))},
      {"down_proj.weight",
       make_qweight("npu.mlp.quantized_gather.down.weight",
                    hidden_size,
                    intermediate_size)},
      {"down_proj.weight_scale",
       make_weight_scale("npu.mlp.quantized_gather.down.scale", hidden_size)},
      {"down_proj.weight_offset",
       torch::zeros({hidden_size}, options_.dtype(torch::kFloat32))},
  };
  mlp->load_state_dict(StateDict(std::move(weight_dict), prefix + "."));

  torch::Tensor local_input =
      make_input("npu.mlp.quantized_gather.input", num_tokens / 2, hidden_size);
  torch::Tensor expected = mlp->forward(local_input);
  Device(options_.device()).synchronize_default_stream();

  FlashComm1Context context;
  context.enabled = true;
  context.tp_rank = 0;
  context.tp_world_size = 2;
  context.original_num_tokens = num_tokens;
  context.padded_num_tokens = num_tokens;
  context.padded_local_num_tokens = num_tokens / context.tp_world_size;
  context.tp_group = mock_group;
  mock_group->clear_allgather_input_dtypes();
  torch::Tensor output;
  {
    FlashComm1ContextScope context_scope(&context);
    output = mlp->forward(local_input);
  }
  Device(options_.device()).synchronize_default_stream();

  const std::vector<torch::ScalarType>& gathered_dtypes =
      mock_group->allgather_input_dtypes();
  ASSERT_EQ(gathered_dtypes.size(), 2);
  EXPECT_EQ(gathered_dtypes[0], torch::kInt8);
  EXPECT_EQ(gathered_dtypes[1], torch::kFloat32);
  ASSERT_TRUE(output.sizes() == expected.sizes());
  expect_output_close(output, expected);
}

TEST_F(NpuLinearW8A8DynamicTest, RowParallelLinearLoadAndForward) {
  const int64_t batch_size = 4;
  const int64_t in_features = 20;
  const int64_t out_features = 10;
  const std::string prefix = "npu.linear.row";
  add_quant_desc(prefix);

  auto linear =
      RowParallelLinear(RowParallelLinearImpl(in_features,
                                              out_features,
                                              /*bias=*/true,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/false,
                                              quant_args_,
                                              parallel_args_.tp_group_,
                                              options_));

  auto weight =
      make_qweight("npu.linear.row.weight", out_features, in_features);
  auto weight_scale = make_weight_scale("npu.linear.row.scale", out_features);
  auto weight_offset =
      make_weight_offset("npu.linear.row.offset", out_features);
  auto bias = make_bias("npu.linear.row.bias", out_features);

  std::unordered_map<std::string, torch::Tensor> weight_dict = {
      {"weight", weight},
      {"weight_scale", weight_scale},
      {"weight_offset", weight_offset},
      {"bias", bias},
  };
  StateDict state_dict = make_quant_state_dict(std::move(weight_dict), prefix);
  linear->load_state_dict(state_dict);

  EXPECT_EQ(linear->weight().scalar_type(), torch::kInt8);

  auto input = make_input("npu.linear.row.input", batch_size, in_features);
  auto output = linear->forward(input);
  Device(options_.device()).synchronize_default_stream();

  auto expected = make_reference_output(input, weight, weight_scale, bias);
  ASSERT_TRUE(output.sizes() == expected.sizes());
  expect_output_close(output, expected);
}

TEST_F(NpuLinearW8A8DynamicTest, ReplicatedLinearLoadAndForward) {
  const int64_t batch_size = 2;
  const int64_t in_features = 14;
  const int64_t out_features = 9;
  const std::string prefix = "npu.linear.rep";
  add_quant_desc(prefix);

  auto linear = ReplicatedLinear(ReplicatedLinearImpl(
      in_features, out_features, /*bias=*/true, quant_args_, options_));

  auto weight =
      make_qweight("npu.linear.rep.weight", out_features, in_features);
  auto weight_scale = make_weight_scale("npu.linear.rep.scale", out_features);
  auto weight_offset =
      make_weight_offset("npu.linear.rep.offset", out_features);
  auto bias = make_bias("npu.linear.rep.bias", out_features);

  std::unordered_map<std::string, torch::Tensor> weight_dict = {
      {"weight", weight},
      {"weight_scale", weight_scale},
      {"weight_offset", weight_offset},
      {"bias", bias},
  };
  StateDict state_dict = make_quant_state_dict(std::move(weight_dict), prefix);
  linear->load_state_dict(state_dict);

  EXPECT_EQ(linear->weight().scalar_type(), torch::kInt8);

  auto input = make_input("npu.linear.rep.input", batch_size, in_features);
  auto output = linear->forward(input);
  Device(options_.device()).synchronize_default_stream();

  auto expected = make_reference_output(input, weight, weight_scale, bias);
  ASSERT_TRUE(output.sizes() == expected.sizes());
  expect_output_close(output, expected);
}

TEST_F(NpuLinearW8A8DynamicTest, QKVParallelLinearLoadAndForward) {
  const int64_t batch_size = 3;
  const int64_t hidden_size = 16;
  const int64_t num_heads = 2;
  const int64_t num_kv_heads = 2;
  const int64_t head_size = 4;
  const int64_t num_kv_head_replicas = 1;
  const int64_t out_features = (num_heads + num_kv_heads * 2) * head_size;
  const std::string prefix = "npu.linear.qkv";
  add_quant_desc(prefix);

  auto linear = QKVParallelLinear(QKVParallelLinearImpl(hidden_size,
                                                        num_heads,
                                                        num_kv_heads,
                                                        head_size,
                                                        num_kv_head_replicas,
                                                        /*bias=*/true,
                                                        /*gather=*/false,
                                                        parallel_args_,
                                                        options_,
                                                        quant_args_));

  auto weight =
      make_qweight("npu.linear.qkv.weight", out_features, hidden_size);
  auto weight_scale = make_weight_scale("npu.linear.qkv.scale", out_features);
  auto weight_offset =
      make_weight_offset("npu.linear.qkv.offset", out_features);
  auto bias = make_bias("npu.linear.qkv.bias", out_features);

  std::unordered_map<std::string, torch::Tensor> weight_dict = {
      {"weight", weight},
      {"weight_scale", weight_scale},
      {"weight_offset", weight_offset},
      {"bias", bias},
  };
  StateDict state_dict = make_quant_state_dict(std::move(weight_dict), prefix);
  linear->load_state_dict(state_dict);

  EXPECT_EQ(linear->weight().scalar_type(), torch::kInt8);

  auto input = make_input("npu.linear.qkv.input", batch_size, hidden_size);
  auto output = linear->forward(input);
  Device(options_.device()).synchronize_default_stream();

  auto expected = make_reference_output(input, weight, weight_scale, bias);
  ASSERT_TRUE(output.sizes() == expected.sizes());
  expect_output_close(output, expected);
}

}  // namespace layer
}  // namespace xllm
