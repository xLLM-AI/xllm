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

#include <algorithm>

#include "core/framework/model_context/model_context.h"
#include "core/layers/common/linear.h"
#include "core/layers/mlu/deepseek_v2_attention.h"
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

ModelArgs attention_args(bool nope) {
  ModelArgs args;
  args.hidden_size() = 128;
  args.n_heads() = 2;
  args.q_lora_rank() = 128;
  args.kv_lora_rank() = 128;
  args.qk_nope_head_dim() = 64;
  args.qk_rope_head_dim() = nope ? 0 : 64;
  args.v_head_dim() = 64;
  args.max_position_embeddings() = 128;
  args.rms_norm_eps() = 1e-5;
  args.rope_theta() = 10000;
  args.enable_mla() = true;
  return args;
}

QuantArgs attention_quant(bool compressed) {
  auto quant = ct_args();
  if (compressed) {
    quant.compressed_groups() = {CompressedQuantGroup{
        .targets = {"re:.*layers\\.(3|45)\\.self_attn\\.(q_a_proj|q_b_proj|kv_"
                    "a_proj_with_mqa|o_proj)$"},
        .preserve_smooth = true}};
  } else {
    quant.quant_method() = kQuantMethodSmoothquant;
    quant.is_compressed_tensors_w8a8_dynamic() = false;
    quant.only_expert_per_group() = true;
    quant.moe_weight_bits() = 4;
  }
  return quant;
}

Weights attention_weights(const ModelArgs& args, bool compressed) {
  Weights weights;
  auto add =
      [&](const std::string& name, int64_t out, int64_t in, bool quantized) {
        if (!quantized) {
          weights.emplace(name + ".weight",
                          torch::full({out, in}, 0.01, torch::kBFloat16));
          return;
        }
        for (const auto& [key, tensor] : ct_weights(out, in, true)) {
          std::string suffix = key;
          auto value = tensor;
          if (!compressed && key == "weight") {
            suffix = "qweight";
          } else if (!compressed && key == "weight_scale") {
            suffix = "per_channel_scale";
            value = tensor.flatten();
          }
          weights.emplace(name + "." + suffix, value);
        }
      };
  add("q_a_proj", args.q_lora_rank(), args.hidden_size(), compressed);
  add("q_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.qk_rope_head_dim()),
      args.q_lora_rank(),
      true);
  add("kv_a_proj_with_mqa",
      args.kv_lora_rank() + args.qk_rope_head_dim(),
      args.hidden_size(),
      compressed);
  add("kv_b_proj",
      args.n_heads() * (args.qk_nope_head_dim() + args.v_head_dim()),
      args.kv_lora_rank(),
      false);
  add("o_proj", args.hidden_size(), args.n_heads() * args.v_head_dim(), true);
  weights.emplace("q_a_layernorm.weight",
                  torch::ones({args.q_lora_rank()}, torch::kBFloat16));
  weights.emplace("kv_a_layernorm.weight",
                  torch::ones({args.kv_lora_rank()}, torch::kBFloat16));
  return weights;
}

TEST(CompressedTensorsMluTest, AttentionPreservesBothCheckpointFormats) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  ParallelArgs parallel(0, 1, &pg);
  parallel.tp_group_ = &pg;
  for (bool compressed : {false, true}) {
    const auto args = attention_args(compressed);
    const auto quant = attention_quant(compressed);
    const std::string prefix = compressed
                                   ? "model.language_model.layers.45.self_attn."
                                   : "model.layers.0.self_attn.";
    OptimizationConfig optimization;
    optimization.enable_fused_mla_kernel = true;
    DeepseekV2Attention attention(args,
                                  quant,
                                  parallel,
                                  mlu_options(),
                                  optimization,
                                  /*enable_indexer=*/false);
    const auto weights = attention_weights(args, compressed);
    Weights tensors;
    Weights scales;
    for (const auto& [key, tensor] : weights) {
      if (key.ends_with("scale") || key.ends_with("smooth")) {
        scales.emplace(key, tensor);
      } else {
        tensors.emplace(key, tensor);
      }
    }
    attention->load_state_dict(StateDict(std::move(scales), prefix));
    attention->load_state_dict(StateDict(std::move(tensors), prefix));
    attention->verify_loaded_weights();
    const auto children = attention->named_children();
    const auto q_b = children["q_b_proj"]->as<ColumnParallelLinearImpl>();
    ASSERT_NE(q_b, nullptr);
    EXPECT_EQ(q_b->weight().scalar_type(), torch::kInt8);
    const auto q_a = children["q_a_proj"]->as<ReplicatedLinearImpl>();
    ASSERT_NE(q_a, nullptr);
    EXPECT_EQ(q_a->weight().scalar_type(),
              compressed ? torch::kInt8 : torch::kBFloat16);
    const auto kv_a =
        children["kv_a_proj_with_mqa"]->as<ReplicatedLinearImpl>();
    ASSERT_NE(kv_a, nullptr);
    EXPECT_EQ(kv_a->weight().scalar_type(),
              compressed ? torch::kInt8 : torch::kBFloat16);
    const auto out = children["o_proj"]->as<RowParallelLinearImpl>();
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->weight().scalar_type(), torch::kInt8);
    EXPECT_EQ(children["kv_b_proj"]
                  ->as<ColumnParallelLinearImpl>()
                  ->weight()
                  .scalar_type(),
              torch::kBFloat16);
  }
}

TEST(CompressedTensorsMluTest, KpoolProjectionRemainsBfloat16) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  ParallelArgs parallel(0, 1, &pg);
  parallel.tp_group_ = &pg;
  auto args = attention_args(/*nope=*/true);
  args.index_n_heads() = 2;
  args.index_head_dim() = 64;
  args.index_topk() = 16;
  args.index_kpool() = 4;
  Glm5NextKPoolIndexer indexer(args,
                               attention_quant(/*compressed=*/true),
                               parallel,
                               /*rotary_emb=*/nullptr,
                               mlu_options());
  const auto weight = torch::full({128, 128}, 0.01, torch::kBFloat16);
  indexer->load_state_dict(
      StateDict({{"wq_b.weight", weight}},
                "model.language_model.layers.45.self_attn.indexer."));
  const auto projection =
      indexer->named_children()["wq_b"]->as<ReplicatedLinearImpl>();
  ASSERT_NE(projection, nullptr);
  EXPECT_TRUE(projection->is_weight_loaded());
  EXPECT_EQ(projection->weight().scalar_type(), torch::kBFloat16);
}

TEST(CompressedTensorsMluDeathTest, AttentionRejectsIncompleteCheckpoint) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  ParallelArgs parallel(0, 1, &pg);
  parallel.tp_group_ = &pg;
  for (bool compressed : {false, true}) {
    const auto args = attention_args(compressed);
    for (const std::string& missing :
         {compressed ? "q_a_proj.weight_scale" : "q_b_proj.qweight",
          compressed ? "q_a_proj.smooth" : "o_proj.smooth"}) {
      DeepseekV2Attention attention(args,
                                    attention_quant(compressed),
                                    parallel,
                                    mlu_options(),
                                    OptimizationConfig(),
                                    /*enable_indexer=*/false);
      auto weights = attention_weights(args, compressed);
      weights.erase(missing);
      attention->load_state_dict(
          StateDict(std::move(weights), "model.layers.3.self_attn."));
      EXPECT_DEATH(attention->verify_loaded_weights(),
                   "model.layers.3.self_attn.");
    }
  }
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
    expect_close(linear->forward(input.reshape({1, 3, 128})).reshape({3, 64}),
                 reference(input, weights));
  }
}

TEST(CompressedTensorsMluTest, LinearLoadsPreservedSmooth) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  auto args = ct_args();
  args.compressed_groups() = {CompressedQuantGroup{
      .targets = {"re:.*self_attn\\.o_proj$"}, .preserve_smooth = true}};
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  ColumnParallelLinear linear(128,
                              64,
                              /*bias=*/false,
                              /*gather_output=*/false,
                              args,
                              &pg,
                              mlu_options());

  StateDict weight_only({{"weight", weights.at("weight")},
                         {"weight_scale", weights.at("weight_scale")}},
                        "model.layers.3.self_attn.o_proj.");
  linear->load_state_dict(weight_only);
  EXPECT_FALSE(linear->is_weight_loaded());

  linear->load_state_dict(StateDict({{"smooth", weights.at("smooth")}},
                                    "model.layers.3.self_attn.o_proj."));
  EXPECT_TRUE(linear->is_weight_loaded());
  const auto input =
      torch::linspace(-1.0, 2.0, 3 * 128).reshape({3, 128}).to(mlu_options());
  expect_close(linear->forward(input), reference(input, weights));
}

QuantArgs mixed_linear_args() {
  auto args = ct_args();
  args.compressed_groups() = {
      CompressedQuantGroup{.targets = {"re:.*proj"}, .preserve_smooth = true}};
  return args;
}

TEST(CompressedTensorsMluTest, FusedSmoothLoadsInEitherOrder) {
  torch::DeviceGuard guard(mlu_options().device());
  test::MockProcessGroup pg(mlu_options().device());
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &pg);
  parallel.tp_group_ = &pg;
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  const auto input =
      torch::linspace(-1, 1, 256).reshape({2, 128}).to(mlu_options());
  for (bool qkv : {false, true}) {
    const std::vector<std::string> prefixes =
        qkv ? std::vector<std::string>{"q_proj.", "k_proj.", "v_proj."}
            : std::vector<std::string>{"gate_proj.", "up_proj."};
    for (bool reverse : {false, true}) {
      ColumnParallelLinear column(
          128, 128, false, false, mixed_linear_args(), &pg, mlu_options());
      QKVParallelLinear attention(128,
                                  1,
                                  1,
                                  64,
                                  1,
                                  false,
                                  false,
                                  parallel,
                                  mlu_options(),
                                  mixed_linear_args());
      auto load = [&](const StateDict& dict) {
        if (qkv) {
          attention->load_state_dict(dict, prefixes);
        } else {
          column->load_state_dict(dict, prefixes);
        }
      };
      Weights checkpoint;
      for (const auto& prefix : prefixes) {
        checkpoint[prefix + "weight"] = weights.at("weight");
        checkpoint[prefix + "weight_scale"] = weights.at("weight_scale");
      }
      load(StateDict(checkpoint));
      for (size_t i = 0; i < prefixes.size(); ++i) {
        const size_t index = reverse ? prefixes.size() - 1 - i : i;
        load(StateDict({{prefixes[index] + "smooth", weights.at("smooth")}}));
        EXPECT_EQ(
            qkv ? attention->is_weight_loaded() : column->is_weight_loaded(),
            i + 1 == prefixes.size());
      }
      std::vector<torch::Tensor> expected(prefixes.size(),
                                          reference(input, weights));
      expect_close(qkv ? attention->forward(input) : column->forward(input),
                   torch::cat(expected, /*dim=*/1));
    }
  }
}

TEST(CompressedTensorsMluDeathTest, FusedSmoothRequiresEveryProjection) {
  test::MockProcessGroup pg(mlu_options().device());
  ColumnParallelLinear linear(
      128, 128, false, false, mixed_linear_args(), &pg, mlu_options());
  const std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  Weights checkpoint;
  for (const auto& prefix : prefixes) {
    checkpoint[prefix + "weight"] = weights.at("weight");
    checkpoint[prefix + "weight_scale"] = weights.at("weight_scale");
  }
  checkpoint["gate_proj.smooth"] = weights.at("smooth");
  linear->load_state_dict(StateDict(checkpoint), prefixes);
  EXPECT_FALSE(linear->is_weight_loaded());
  const auto input = torch::ones({2, 128}, mlu_options());
  EXPECT_DEATH(linear->forward(input), "Missing.*smooth");
  EXPECT_DEATH(
      linear->load_state_dict(
          StateDict({{"up_proj.smooth", weights.at("smooth") + 1}}), prefixes),
      "Smooth values differ");
  EXPECT_DEATH(
      linear->load_state_dict(
          StateDict({{"up_proj.smooth", torch::ones({64})}}), prefixes),
      "Smooth size mismatch");
}

TEST(CompressedTensorsMluDeathTest, ReplicatedPreservedSmoothIsRequired) {
  ReplicatedLinear linear(128, 64, false, mixed_linear_args(), mlu_options());
  auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  const auto smooth = weights.at("smooth");
  weights.erase("smooth");
  linear->load_state_dict(StateDict(weights, "model.q_proj."));
  const auto input =
      torch::linspace(-1, 1, 256).reshape({2, 128}).to(mlu_options());
  EXPECT_DEATH(linear->forward(input), "Missing.*smooth");
  linear->load_state_dict(StateDict({{"smooth", smooth}}, "model.q_proj."));
  weights["smooth"] = smooth;
  expect_close(linear->forward(input), reference(input, weights));
}

TEST(CompressedTensorsMluDeathTest, LinearRejectsW4a8Target) {
  auto args = mixed_linear_args();
  args.compressed_groups() = {CompressedQuantGroup{
      .targets = {"model.q_proj"}, .bits = 4, .group_size = 128}};
  ReplicatedLinear linear(128, 64, false, args, mlu_options());
  EXPECT_DEATH(linear->load_state_dict(StateDict(
                   ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true),
                   "model.q_proj.")),
               "Linear only supports.*W8A8");
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

TEST(CompressedTensorsMluTest, IgnoredRouterIgnoresSmoothInLaterShard) {
  ReplicatedLinear router(128, 4, /*bias=*/false, ct_args(), mlu_options());
  router->load_state_dict(
      StateDict({{"weight", torch::ones({4, 128}, torch::kBFloat16)}},
                "model.mlp.gate."));
  router->load_state_dict(
      StateDict({{"smooth", torch::ones({128})}}, "model.mlp.gate."));
  EXPECT_FALSE(router->named_parameters().contains("smooth"));
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

TEST(CompressedTensorsMluTest, RowShardsPreservedSmooth) {
  const auto weights = ct_weights(/*out=*/64, /*in=*/128, /*smooth=*/true);
  for (int64_t rank = 0; rank < 2; ++rank) {
    torch::DeviceGuard guard(mlu_options().device());
    test::MockProcessGroup pg(mlu_options().device(), rank, 2);
    auto args = ct_args();
    args.compressed_groups() = {CompressedQuantGroup{
        .targets = {"re:.*mlp\\.down_proj$"}, .preserve_smooth = true}};
    RowParallelLinear linear(
        128, 64, false, true, false, args, &pg, mlu_options());
    linear->load_state_dict(
        StateDict(weights, "model.layers.3.mlp.down_proj."));
    Weights local = weights;
    local["weight"] = weights.at("weight").narrow(1, rank * 64, 64);
    local["smooth"] = weights.at("smooth").narrow(0, rank * 64, 64);
    torch::Tensor local_input = torch::linspace(-1, 1, 256)
                                    .reshape({2, 128})
                                    .narrow(1, rank * 64, 64)
                                    .contiguous()
                                    .to(mlu_options());
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

TEST(CompressedTensorsMluTest, LinearIgnoresSmoothAfterWeightsAreLoaded) {
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
  column->load_state_dict(late_smooth);
  EXPECT_FALSE(column->named_parameters().contains("smooth"));
  const auto column_input =
      torch::linspace(-1.0, 1.0, 256).reshape({2, 128}).to(mlu_options());
  expect_close(column->forward(column_input), reference(column_input, weights));

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
  row->load_state_dict(late_smooth);
  EXPECT_FALSE(row->named_parameters().contains("smooth"));
  const auto row_input =
      torch::linspace(-1.0, 1.0, 256).reshape({2, 128}).to(mlu_options());
  expect_close(row->forward(row_input), reference(row_input, weights));

  ReplicatedLinear replicated(
      128, 64, /*bias=*/false, ct_args(), mlu_options());
  replicated->load_state_dict(StateDict(weights));
  replicated->load_state_dict(late_smooth);
  EXPECT_FALSE(replicated->named_parameters().contains("smooth"));
  const auto replicated_input =
      torch::linspace(-1.0, 1.0, 256).reshape({2, 128}).to(mlu_options());
  expect_close(replicated->forward(replicated_input),
               reference(replicated_input, weights));
}

TEST(CompressedTensorsMluTest, FusedColumnIgnoresSmoothFromEitherProjection) {
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
    linear->load_state_dict(
        StateDict({{prefix + "smooth", torch::ones({128})}}), prefixes);
    EXPECT_FALSE(linear->named_parameters().contains("smooth"));
  }
}

TEST(CompressedTensorsMluTest, QkvIgnoresSmoothFromEveryProjection) {
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
    linear->load_state_dict(
        StateDict({{prefix + "smooth", torch::ones({128})}}), prefixes);
    EXPECT_FALSE(linear->named_parameters().contains("smooth"));
  }
}

TEST(CompressedTensorsMluTest, MoeIgnoresSmoothFromEveryProjection) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  const auto model = moe_model_args();
  ModelContext ctx(parallel, model, ct_args(), mlu_options());
  FusedMoE moe(
      ctx,
      FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
  moe->load_state_dict(
      StateDict({{"gate.weight", torch::ones({4, 256}, torch::kBFloat16)}},
                "model.mlp."));
  for (const std::string& projection :
       {"gate_proj.", "up_proj.", "down_proj."}) {
    const std::string name = "experts.3." + projection + "smooth";
    SCOPED_TRACE(name);
    moe->load_state_dict(StateDict({{name, torch::ones({256})}}, "model.mlp."));
    EXPECT_FALSE(moe->named_parameters().contains("input_smooth"));
    EXPECT_FALSE(moe->named_parameters().contains("act_smooth"));
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
  FusedMoE ct(
      ctx,
      FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
  FusedMoE legacy(
      legacy_ctx,
      FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
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

QuantArgs mixed_moe_args() {
  auto args = ct_args();
  args.moe_weight_bits() = 4;
  args.group_size() = 128;
  args.compressed_groups() = {CompressedQuantGroup{
      .targets =
          {R"(re:.*layers\.[0-9]+\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$)"},
      .bits = 4,
      .group_size = 128,
      .preserve_smooth = true}};
  return args;
}

Weights mixed_moe_weights() {
  Weights checkpoint;
  checkpoint["gate.weight"] = torch::ones({4, 256}, torch::kBFloat16);
  for (int64_t expert = 0; expert < 4; ++expert) {
    for (const std::string& projection :
         {"gate_proj.", "up_proj.", "down_proj."}) {
      const std::string prefix =
          "experts." + std::to_string(expert) + "." + projection;
      checkpoint[prefix + "weight"] =
          torch::arange(256 * 128)
              .add(expert + (projection == "down_proj." ? 3 : 1))
              .remainder(119)
              .reshape({256, 128})
              .to(torch::kInt8);
      checkpoint[prefix + "weight_scale"] =
          torch::linspace(0.01, 0.03, 512).reshape({256, 2});
      checkpoint[prefix + "smooth"] =
          torch::linspace(0.5, 1.5, 256) + expert * 0.1;
    }
  }
  return checkpoint;
}

TEST(CompressedTensorsMluTest, MixedMoeMatchesLegacyAndShardsTpWeights) {
  const auto checkpoint = mixed_moe_weights();
  Weights legacy_weights;
  for (const auto& [name, tensor] : checkpoint) {
    std::string key = name;
    if (name.starts_with("experts.") && name.ends_with(".weight")) {
      key = name.substr(0, name.size() - 6) + "qweight";
    } else if (name.ends_with(".weight_scale")) {
      key = name.substr(0, name.size() - 12) + "per_channel_scale";
    }
    legacy_weights[key] = tensor;
  }
  for (int64_t rank = 0; rank < 2; ++rank) {
    test::MockProcessGroup pg(mlu_options().device(), rank, 2);
    ParallelArgs parallel(rank, 2, &pg);
    parallel.tp_group_ = &pg;
    auto args = mixed_moe_args();
    auto legacy_args = args;
    legacy_args.quant_method() = kQuantMethodSmoothquant;
    legacy_args.is_compressed_tensors_w8a8_dynamic() = false;
    legacy_args.compressed_groups().clear();
    ModelContext ctx(parallel, moe_model_args(), args, mlu_options());
    ModelContext old_ctx(
        parallel, moe_model_args(), legacy_args, mlu_options());
    FusedMoE mixed(ctx,
                   FusedMoEArgs{.is_gated = true,
                                .enable_result_reduction = false,
                                .module_prefix = "model.layers.3.mlp"});
    FusedMoE legacy(old_ctx,
                    FusedMoEArgs{.is_gated = true,
                                 .enable_result_reduction = false,
                                 .module_prefix = "model.layers.3.mlp"});
    mixed->load_state_dict(StateDict(checkpoint, "model.layers.3.mlp."));
    legacy->load_state_dict(StateDict(legacy_weights));
    mixed->verify_loaded_weights();
    const auto params = mixed->named_parameters();
    expect_close(params["w2"][0],
                 checkpoint.at("experts.0.down_proj.weight")
                     .narrow(1, rank * 64, 64)
                     .to(torch::kFloat32));
    expect_close(
        params["act_smooth"][0],
        checkpoint.at("experts.0.down_proj.smooth").narrow(0, rank * 128, 128));
    expect_close(
        params["w2_scale"].select(/*dim=*/1, /*index=*/0).transpose(0, 1),
        checkpoint.at("experts.0.down_proj.weight_scale").narrow(1, rank, 1));
    const auto input =
        torch::linspace(-0.1, 0.2, 1024).reshape({4, 256}).to(mlu_options());
    expect_close(
        mixed->forward_experts(input, false),
        legacy->forward_experts(input, false).cpu().to(torch::kFloat32));
  }
}

TEST(CompressedTensorsMluTest, MixedPrecisionMoeLoadsInEitherShardOrder) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  ModelContext ctx(parallel, moe_model_args(), mixed_moe_args(), mlu_options());
  const auto checkpoint = mixed_moe_weights();
  FusedMoE complete(
      ctx,
      FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
  complete->load_state_dict(StateDict(checkpoint, "model.layers.3.mlp."));
  complete->verify_loaded_weights();
  const torch::Tensor input =
      torch::linspace(-0.1, 0.2, 1024).reshape({4, 256}).to(mlu_options());
  const torch::Tensor expected = complete->forward_experts(input, false);
  ASSERT_EQ(expected.sizes(), (std::vector<int64_t>{4, 256}));

  std::vector<std::string> names;
  names.reserve(checkpoint.size());
  for (const auto& [name, tensor] : checkpoint) {
    names.emplace_back(name);
  }
  std::sort(names.begin(), names.end());
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    if (reverse) {
      std::reverse(names.begin(), names.end());
    }
    FusedMoE sharded(
        ctx,
        FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
    for (const std::string& name : names) {
      sharded->load_state_dict(
          StateDict({{name, checkpoint.at(name)}}, "model.layers.3.mlp."));
    }
    sharded->verify_loaded_weights();
    expect_close(sharded->forward_experts(input, false),
                 expected.cpu().to(torch::kFloat32));
  }
}

TEST(CompressedTensorsMluDeathTest, MixedPrecisionMoeValidatesLateSmooth) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  ModelContext ctx(parallel, moe_model_args(), mixed_moe_args(), mlu_options());
  FusedMoE moe(
      ctx,
      FusedMoEArgs{.is_gated = true, .module_prefix = "model.layers.3.mlp"});
  auto checkpoint = mixed_moe_weights();
  const std::string name = "experts.3.up_proj.smooth";
  const torch::Tensor smooth = checkpoint.at(name);
  checkpoint.erase(name);
  moe->load_state_dict(StateDict(checkpoint, "model.layers.3.mlp."));
  EXPECT_DEATH(moe->verify_loaded_weights(), "Missing.*smooth");
  EXPECT_DEATH(moe->load_state_dict(
                   StateDict({{name, smooth + 1}}, "model.layers.3.mlp.")),
               "gate_proj and up_proj smooth must match");
  moe->load_state_dict(StateDict({{name, smooth}}, "model.layers.3.mlp."));
  moe->verify_loaded_weights();
}

TEST(CompressedTensorsMluDeathTest, FusedProjectionsRequireCompatibleSmooth) {
  test::MockProcessGroup pg(mlu_options().device());
  auto args = ct_args();
  args.compressed_groups() = {
      {.targets = {"gate_proj"}, .preserve_smooth = true},
      {.targets = {"up_proj"}, .preserve_smooth = false}};
  ColumnParallelLinear linear(128, 128, false, false, args, &pg, mlu_options());
  EXPECT_DEATH(linear->load_state_dict(
                   StateDict({{"gate_proj.weight",
                               torch::ones({64, 128}, torch::kInt8)}}),
                   std::vector<std::string>{"gate_proj.", "up_proj."}),
               "different preserve_smooth");
}

TEST(CompressedTensorsMluTest, MoeSmoothIsIndependentOfWeightBits) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  for (int64_t bits : {4, 8}) {
    for (bool smooth : {false, true}) {
      SCOPED_TRACE(std::to_string(bits) + ":" + std::to_string(smooth));
      auto args = mixed_moe_args();
      auto& scheme = args.compressed_groups().front();
      scheme.bits = bits;
      scheme.group_size = bits == 4 ? 128 : 0;
      scheme.preserve_smooth = smooth;
      auto legacy_args = ct_args();
      legacy_args.quant_method() = kQuantMethodSmoothquant;
      legacy_args.is_compressed_tensors_w8a8_dynamic() = false;
      legacy_args.moe_weight_bits() = bits;
      legacy_args.group_size() = scheme.group_size;
      auto checkpoint = mixed_moe_weights();
      Weights old;
      old["gate.weight"] = checkpoint.at("gate.weight");
      for (int64_t expert = 0; expert < 4; ++expert) {
        for (const std::string& projection :
             {"gate_proj.", "up_proj.", "down_proj."}) {
          const std::string prefix =
              "experts." + std::to_string(expert) + "." + projection;
          if (bits == 8) {
            const auto weights = ct_weights(256, 256, false);
            checkpoint[prefix + "weight"] = weights.at("weight");
            checkpoint[prefix + "weight_scale"] =
                weights.at("weight_scale").flatten();
          }
          old[prefix + "qweight"] = checkpoint.at(prefix + "weight");
          old[prefix + "per_channel_scale"] =
              checkpoint.at(prefix + "weight_scale");
          old[prefix + "smooth"] =
              smooth ? checkpoint.at(prefix + "smooth") : torch::ones({256});
        }
      }
      ModelContext ctx(parallel, moe_model_args(), args, mlu_options());
      ModelContext old_ctx(
          parallel, moe_model_args(), legacy_args, mlu_options());
      const FusedMoEArgs moe_args{.module_prefix = "model.layers.3.mlp"};
      FusedMoE ct(ctx, moe_args);
      FusedMoE legacy(old_ctx, moe_args);
      ct->load_state_dict(StateDict(checkpoint, "model.layers.3.mlp."));
      legacy->load_state_dict(StateDict(old));
      ct->verify_loaded_weights();
      EXPECT_EQ(ct->named_parameters().contains("input_smooth"), smooth);
      EXPECT_EQ(ct->named_parameters().contains("act_smooth"), smooth);
      const auto input =
          torch::linspace(-0.1, 0.2, 1024).reshape({4, 256}).to(mlu_options());
      expect_close(
          ct->forward_experts(input, false),
          legacy->forward_experts(input, false).cpu().to(torch::kFloat32));
      if (!smooth) {
        std::erase_if(checkpoint, [](const auto& item) {
          return item.first.ends_with("smooth");
        });
        FusedMoE without_smooth(ctx, moe_args);
        without_smooth->load_state_dict(
            StateDict(checkpoint, "model.layers.3.mlp."));
        without_smooth->verify_loaded_weights();
        expect_close(
            without_smooth->forward_experts(input, false),
            ct->forward_experts(input, false).cpu().to(torch::kFloat32));
      }
    }
  }
}

TEST(CompressedTensorsMluTest, MoeResolvesEachLayerBeforeAllocation) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  auto args = ct_args();
  args.compressed_groups() = {{.targets = {"re:model.layers.0.mlp.experts.*"},
                               .bits = 4,
                               .group_size = 128},
                              {.targets = {"re:model.layers.1.mlp.experts.*"},
                               .preserve_smooth = true}};
  ModelContext ctx(parallel, moe_model_args(), args, mlu_options());
  FusedMoE w4(ctx, FusedMoEArgs{.module_prefix = "model.layers.0.mlp"});
  FusedMoE w8(ctx, FusedMoEArgs{.module_prefix = "model.layers.1.mlp"});
  FusedMoE fp(ctx, FusedMoEArgs{.module_prefix = "model.layers.2.mlp"});
  EXPECT_EQ(w4->named_parameters()["w13"].size(2), 128);
  EXPECT_FALSE(w4->named_parameters().contains("input_smooth"));
  EXPECT_EQ(w8->named_parameters()["w13"].size(2), 256);
  EXPECT_TRUE(w8->named_parameters().contains("input_smooth"));
  EXPECT_EQ(fp->named_parameters()["w13"].scalar_type(), torch::kBFloat16);
}

TEST(CompressedTensorsMluDeathTest, MoeRejectsIncompatibleExpertSchemes) {
  std::unique_ptr<ProcessGroup> pg;
  const auto parallel = test::create_default_parallel_args(pg);
  auto args = ct_args();
  args.compressed_groups() = {
      {.targets = {"re:.*gate_proj"}, .preserve_smooth = true},
      {.targets = {"re:.*(up_proj|down_proj)"}, .preserve_smooth = false}};
  ModelContext ctx(parallel, moe_model_args(), args, mlu_options());
  EXPECT_DEATH(
      (FusedMoE(ctx, FusedMoEArgs{.module_prefix = "model.layers.0.mlp"})),
      "Incompatible.*schemes");
}

}  // namespace
}  // namespace xllm::layer
