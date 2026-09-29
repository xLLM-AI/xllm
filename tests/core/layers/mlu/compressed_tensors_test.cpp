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
#include <torch/torch.h>

#include "core/framework/model_context.h"
#include "core/layers/common/linear.h"
#include "core/layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

QuantArgs ct_args() {
  QuantArgs args;
  args.quant_method() = "compressed-tensors";
  args.bits() = 8;
  args.is_sym() = true;
  args.is_compressed_tensors_w8a8_dynamic() = true;
  args.ignored_modules() = {"re:.*gate$"};
  return args;
}

torch::TensorOptions mlu_options() {
  return torch::TensorOptions()
      .device(torch::kPrivateUse1, 0)
      .dtype(torch::kBFloat16);
}

using Weights = std::unordered_map<std::string, torch::Tensor>;

Weights ct_weights(int64_t out, int64_t in, bool smooth) {
  Weights result;
  result["weight"] =
      torch::arange(out * in).remainder(15).sub(7).reshape({out, in}).to(
          torch::kInt8);
  result["weight_scale"] =
      torch::linspace(0.01, 0.03, out).reshape({out, 1}).to(torch::kBFloat16);
  if (smooth) {
    result["smooth"] = torch::linspace(0.5, 1.5, in).to(torch::kBFloat16);
  }
  return result;
}

// Independent per-token INT8 reference using the checkpoint's dequant scales.
torch::Tensor reference(const torch::Tensor& input, const Weights& weights) {
  torch::Tensor x = input.cpu().to(torch::kFloat32);
  if (weights.contains("smooth")) {
    x = x * weights.at("smooth").to(torch::kFloat32);
  }
  torch::Tensor scale = std::get<0>(x.abs().max(-1, true)) / 127.0;
  scale = scale.clamp_min(1e-12);
  torch::Tensor quant = (x / scale).round().clamp(-127, 127);
  torch::Tensor weight = weights.at("weight").to(torch::kFloat32) *
                         weights.at("weight_scale").to(torch::kFloat32);
  return torch::matmul(quant, weight.t()) * scale;
}

void expect_close(const torch::Tensor& actual, const torch::Tensor& expected) {
  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/0.02,
                              /*atol=*/0.025))
      << "max error "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

ModelArgs moe_model_args() {
  auto model = test::create_default_model_args();
  model.hidden_size() = 256;
  model.moe_intermediate_size() = 256;
  model.n_routed_experts() = 4;
  model.num_experts_per_tok() = 2;
  model.n_shared_experts() = 0;
  model.n_group() = 1;
  model.topk_group() = 1;
  model.scoring_func() = "softmax";
  model.topk_method() = "greedy";
  model.hidden_act() = "silu";
  return model;
}

TEST(CompressedTensorsMluTest, LinearLoadsWeightAndScaleInEitherOrder) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/false);
  for (bool scale_first : {false, true}) {
    ColumnParallelLinear linear(128,
                                64,
                                /*bias=*/false,
                                /*gather_output=*/false,
                                ct_args(),
                                &pg,
                                mlu_options());
    const std::string first = scale_first ? "weight_scale" : "weight";
    const std::string second = scale_first ? "weight" : "weight_scale";
    linear->load_state_dict(StateDict({{first, weights.at(first)}}));
    EXPECT_FALSE(linear->is_weight_loaded());
    linear->load_state_dict(StateDict({{second, weights.at(second)}}));
    ASSERT_TRUE(linear->is_weight_loaded());
    const auto input =
        torch::linspace(-1.0, 2.0, 3 * 128).reshape({3, 128}).to(mlu_options());
    expect_close(linear->forward(input), reference(input, weights));
    // Exercise the 3D input path and repeated decode-style calls.
    expect_close(linear->forward(input.reshape({1, 3, 128})).reshape({3, 64}),
                 reference(input, weights));
  }
}

TEST(CompressedTensorsMluTest, ReplicatedRouterRemainsBfloat16) {
  auto weights = torch::ones({4, 128}, torch::kBFloat16);
  ReplicatedLinear router(128, 4, false, ct_args(), mlu_options());
  router->load_state_dict(
      StateDict({{"weight", weights}}, "model.layers.0.mlp.gate."));
  EXPECT_EQ(router->weight().scalar_type(), torch::kBFloat16);
  expect_close(router->forward(torch::ones({2, 128}, mlu_options())),
               torch::full({2, 4}, 128.0));
}

TEST(CompressedTensorsMluDeathTest, IgnoredRouterRejectsSmoothInLaterShard) {
  ReplicatedLinear router(128, 4, /*bias=*/false, ct_args(), mlu_options());
  router->load_state_dict(
      StateDict({{"weight", torch::ones({4, 128}, torch::kBFloat16)}},
                "model.mlp.gate."));
  EXPECT_DEATH(router->load_state_dict(StateDict(
                   {{"smooth", torch::ones({128})}}, "model.mlp.gate.")),
               "smooth");
}

TEST(CompressedTensorsMluTest, RowShardsWeightWithoutSmooth) {
  auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/false);
  torch::Tensor input =
      torch::linspace(-1, 1, 256).reshape({2, 128}).to(mlu_options());
  for (int64_t rank = 0; rank < 2; ++rank) {
    test::MockProcessGroup pg(mlu_options().device(), rank, 2);
    RowParallelLinear linear(
        128, 64, false, true, false, ct_args(), &pg, mlu_options());
    linear->load_state_dict(StateDict(weights));
    Weights local = weights;
    local["weight"] = weights.at("weight").narrow(1, rank * 64, 64);
    torch::Tensor local_input = input.narrow(1, rank * 64, 64).contiguous();
    expect_close(linear->forward(local_input), reference(local_input, local));
  }
}

TEST(CompressedTensorsMluTest, QkvReplicatesKvHeadsAndFusesScales) {
  for (int64_t rank = 0; rank < 2; ++rank) {
    test::MockProcessGroup pg(mlu_options().device(), rank, 2);
    ParallelArgs parallel(rank, 2, &pg);
    parallel.tp_group_ = &pg;
    QKVParallelLinear linear(
        128, 1, 1, 64, 2, false, false, parallel, mlu_options(), ct_args());
    Weights all;
    auto q = ct_weights(/*out=*/128, /*in=*/128, /*smooth=*/false);
    auto k = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/false);
    auto v = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/false);
    for (const auto& [name, tensor] : q) {
      all["q_proj." + name] = tensor;
    }
    for (const auto& [name, tensor] : k) {
      all["k_proj." + name] = tensor;
    }
    for (const auto& [name, tensor] : v) {
      all["v_proj." + name] = tensor;
    }
    for (const std::string& prefix : {"q_proj.", "k_proj.", "v_proj."}) {
      Weights shard;
      for (const auto& [name, tensor] : all) {
        if (name.starts_with(prefix)) {
          shard[name] = tensor;
        }
      }
      linear->load_state_dict(StateDict(shard),
                              {"q_proj.", "k_proj.", "v_proj."});
    }
    q["weight"] = q.at("weight").narrow(0, rank * 64, 64);
    q["weight_scale"] = q.at("weight_scale").narrow(0, rank * 64, 64);
    torch::Tensor input =
        torch::linspace(-1, 1, 256).reshape({2, 128}).to(mlu_options());
    expect_close(
        linear->forward(input),
        torch::cat(
            {reference(input, q), reference(input, k), reference(input, v)},
            1));
  }
}

TEST(CompressedTensorsMluDeathTest, LinearRejectsSmoothAfterWeightsAreLoaded) {
  test::MockProcessGroup pg(mlu_options().device());
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/false);
  const StateDict late_smooth({{"smooth", torch::ones({128})}});
  ColumnParallelLinear column(128,
                              64,
                              /*bias=*/false,
                              /*gather_output=*/false,
                              ct_args(),
                              &pg,
                              mlu_options());
  column->load_state_dict(StateDict(weights));
  ASSERT_TRUE(column->is_weight_loaded());
  EXPECT_DEATH(column->load_state_dict(late_smooth), "smooth");

  RowParallelLinear row(128,
                        64,
                        /*bias=*/false,
                        /*input_is_parallelized=*/true,
                        /*enable_result_reduction=*/false,
                        ct_args(),
                        &pg,
                        mlu_options());
  row->load_state_dict(StateDict(weights));
  ASSERT_TRUE(row->is_weight_loaded());
  EXPECT_DEATH(row->load_state_dict(late_smooth), "smooth");

  ReplicatedLinear replicated(
      128, 64, /*bias=*/false, ct_args(), mlu_options());
  replicated->load_state_dict(StateDict(weights));
  EXPECT_DEATH(replicated->load_state_dict(late_smooth), "smooth");
}

TEST(CompressedTensorsMluDeathTest,
     FusedColumnRejectsSmoothFromEitherProjection) {
  test::MockProcessGroup pg(mlu_options().device());
  ColumnParallelLinear linear(128,
                              128,
                              /*bias=*/false,
                              /*gather_output=*/false,
                              ct_args(),
                              &pg,
                              mlu_options());
  const std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  linear->load_state_dict(
      StateDict({{"gate_proj.weight", torch::ones({64, 128}, torch::kInt8)}}),
      prefixes);
  for (const std::string& prefix : prefixes) {
    SCOPED_TRACE(prefix);
    EXPECT_DEATH(
        linear->load_state_dict(
            StateDict({{prefix + "smooth", torch::ones({128})}}), prefixes),
        "smooth");
  }
}

TEST(CompressedTensorsMluDeathTest, QkvRejectsSmoothFromEveryProjection) {
  test::MockProcessGroup pg(
      mlu_options().device(), /*rank=*/0, /*world_size=*/2);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/2, &pg);
  parallel.tp_group_ = &pg;
  QKVParallelLinear linear(128,
                           1,
                           1,
                           64,
                           2,
                           /*bias=*/false,
                           /*gather_output=*/false,
                           parallel,
                           mlu_options(),
                           ct_args());
  const std::vector<std::string> prefixes = {"q_proj.", "k_proj.", "v_proj."};
  linear->load_state_dict(
      StateDict({{"q_proj.weight", torch::ones({128, 128}, torch::kInt8)}}),
      prefixes);
  for (const std::string& prefix : prefixes) {
    SCOPED_TRACE(prefix);
    EXPECT_DEATH(
        linear->load_state_dict(
            StateDict({{prefix + "smooth", torch::ones({128})}}), prefixes),
        "smooth");
  }
}

TEST(CompressedTensorsMluDeathTest, MoeRejectsSmoothFromEveryProjection) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  const auto model = moe_model_args();
  ModelContext ctx(parallel, model, ct_args(), mlu_options());
  FusedMoE moe(ctx, FusedMoEArgs{.is_gated = true});
  moe->load_state_dict(
      StateDict({{"gate.weight", torch::ones({4, 256}, torch::kBFloat16)}},
                "model.mlp."));
  for (const std::string& projection :
       {"gate_proj.", "up_proj.", "down_proj."}) {
    const std::string name = "experts.3." + projection + "smooth";
    SCOPED_TRACE(name);
    EXPECT_DEATH(moe->load_state_dict(
                     StateDict({{name, torch::ones({256})}}, "model.mlp.")),
                 "smooth");
  }
}

TEST(CompressedTensorsMluTest, LegacyLinearUsesSharedW8a8Execution) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  auto args = ct_args();
  args.quant_method() = kQuantMethodSmoothquant;
  args.is_compressed_tensors_w8a8_dynamic() = false;
  const auto bias = torch::linspace(-0.1, 0.1, 64).to(torch::kBFloat16);
  const StateDict legacy(
      {{"qweight", weights.at("weight")},
       {"per_channel_scale", weights.at("weight_scale").flatten()},
       {"smooth", weights.at("smooth")},
       {"bias", bias}});
  ColumnParallelLinear column(128,
                              64,
                              /*bias=*/true,
                              /*gather_output=*/false,
                              args,
                              &pg,
                              mlu_options());
  column->load_state_dict(legacy);
  const auto input =
      torch::linspace(-1, 1, 384).reshape({1, 3, 128}).to(mlu_options());
  expect_close(column->forward(input), reference(input, weights) + bias);

  RowParallelLinear row(128,
                        64,
                        /*bias=*/true,
                        /*input_is_parallelized=*/true,
                        /*enable_result_reduction=*/false,
                        args,
                        &pg,
                        mlu_options(),
                        LinearExtraArgs("silu", /*is_gated=*/true));
  row->load_state_dict(legacy);
  const auto gated_input = torch::cat({input, input}, -1);
  const auto activated =
      (torch::silu(input.to(torch::kFloat32)) * input).to(mlu_options());
  expect_close(row->forward(gated_input), reference(activated, weights) + bias);
}

TEST(CompressedTensorsMluTest, MoeWithoutSmoothMatchesLegacyWithUnitSmooth) {
  std::unique_ptr<ProcessGroup> pg;
  auto parallel = test::create_default_parallel_args(pg);
  const auto model = moe_model_args();
  auto args = ct_args();
  auto legacy_args = args;
  legacy_args.quant_method() = kQuantMethodSmoothquant;
  legacy_args.is_compressed_tensors_w8a8_dynamic() = false;
  ModelContext ctx(parallel, model, args, mlu_options());
  ModelContext legacy_ctx(parallel, model, legacy_args, mlu_options());
  FusedMoE ct(ctx, FusedMoEArgs{.is_gated = true});
  FusedMoE legacy(legacy_ctx, FusedMoEArgs{.is_gated = true});
  Weights checkpoint;
  Weights old;
  checkpoint["gate.weight"] = torch::ones({4, 256}, torch::kBFloat16);
  old["gate.weight"] = checkpoint.at("gate.weight");
  for (int64_t expert = 0; expert < 4; ++expert) {
    for (const std::string& projection :
         {"gate_proj.", "up_proj.", "down_proj."}) {
      auto w = ct_weights(/*out=*/256, /*in=*/256, /*smooth=*/false);
      const std::string prefix =
          "experts." + std::to_string(expert) + "." + projection;
      for (const auto& [name, tensor] : w) {
        checkpoint[prefix + name] = tensor;
      }
      old[prefix + "qweight"] = w.at("weight");
      old[prefix + "per_channel_scale"] =
          w.at("weight_scale").flatten().to(torch::kFloat32);
      old[prefix + "smooth"] = torch::ones({256});
    }
  }
  ct->load_state_dict(StateDict(checkpoint, "model.layers.0.mlp."));
  legacy->load_state_dict(StateDict(old));
  ct->verify_loaded_weights();
  torch::Tensor input =
      torch::linspace(-0.1, 0.2, 1024).reshape({4, 256}).to(mlu_options());
  expect_close(ct->forward_experts(input, false),
               legacy->forward_experts(input, false).cpu().to(torch::kFloat32));
}

}  // namespace
}  // namespace xllm::layer
