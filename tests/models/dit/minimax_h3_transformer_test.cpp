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

#include "models/dit/pipelines/pipeline_minimax_h3.h"

namespace xllm {
namespace {

class SequenceParallelProbe final : public dit::SequenceParallelMixin {
 public:
  explicit SequenceParallelProbe(ProcessGroup* group)
      : dit::SequenceParallelMixin(group) {}
  using dit::SequenceParallelMixin::sequence_parallel_forward;
};

ModelContext make_cpu_context(ProcessGroup* tp_group = nullptr) {
  static ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &process_group);
  parallel_args.dit_tp_group_ = tp_group == nullptr ? &process_group : tp_group;
  parallel_args.tp_size(parallel_args.dit_tp_group_->world_size());
  const auto options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  return ModelContext(
      parallel_args, ModelArgs(), QuantArgs(), options, /*context=*/nullptr);
}

TEST(MiniMaxH3TransformerTest, TimestepEmbeddingRetainsBiasChecksAcrossShards) {
  torch::NoGradGuard no_grad;
  const ModelContext context = make_cpu_context();
  minimax_h3::H3TimestepEmbedding embedding(
      /*in_dim=*/2, /*hidden_dim=*/4, /*out_dim=*/2, context);
  embedding->load_state_dict(StateDict({
      {"linear_1.weight", torch::zeros({4, 2})},
      {"linear_2.weight", torch::ones({2, 4})},
  }));
  ASSERT_DEATH(embedding->verify_loaded_weights("time."), "linear_1.bias");
  embedding->load_state_dict(
      StateDict({{"linear_1.bias", torch::arange(4, torch::kFloat32)}}));
  ASSERT_DEATH(embedding->verify_loaded_weights("time."), "linear_2.bias");
  embedding->load_state_dict(
      StateDict({{"linear_2.bias", torch::tensor({0.5F, -0.5F})}}));
  embedding->verify_loaded_weights("time.");
  const torch::Tensor actual = embedding->forward(torch::zeros({1, 2}));
  const torch::Tensor expected = torch::tensor({{5.850375F, 4.850375F}});
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
}

TEST(MiniMaxH3TransformerTest, FinalModulationRejectsMissingBias) {
  torch::NoGradGuard no_grad;
  const ModelContext context = make_cpu_context();
  minimax_h3::H3AdaLayerNormOut norm(
      /*hidden_size=*/2, /*time_embed_dim=*/2, /*eps=*/1e-5, context);
  norm->load_state_dict(StateDict({
      {"norm.weight", torch::ones({2})},
      {"linear.weight", torch::zeros({4, 2})},
  }));
  ASSERT_DEATH(norm->verify_loaded_weights("out."), "linear.bias");
  norm->load_state_dict(
      StateDict({{"linear.bias", torch::tensor({2.0F, 3.0F, 0.0F, 0.0F})}}));
  norm->verify_loaded_weights("out.");
  const torch::Tensor actual = norm->forward(
      torch::zeros({1, 1, 2}),
      torch::zeros({1, 2}),
      torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong)));
  EXPECT_TRUE(torch::equal(actual, torch::tensor({{{2.0F, 3.0F}}})));
}

TEST(MiniMaxH3TransformerTest,
     ConvertedSwiGluWeightsPreserveGateAndBatchLayout) {
  torch::NoGradGuard no_grad;
  const ModelContext context = make_cpu_context();
  minimax_h3::H3SwiGLUFFN ffn(
      /*hidden_size=*/2, /*ffn_dim=*/2, /*bias=*/false, context);
  ffn->load_state_dict(StateDict({
      {"net.0.proj.weight",
       torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}, {2.0F, 0.0F}, {0.0F, 3.0F}})},
      {"net.2.weight", torch::eye(2)},
  }));
  ffn->verify_loaded_weights("ff.");
  const torch::Tensor input = torch::tensor({{{1.0F, 1.0F}, {1.0F, -1.0F}}});
  const torch::Tensor actual = ffn->forward(input);
  const torch::Tensor expected =
      torch::tensor({{{1.761594F, 2.857722F}, {1.761594F, 0.142278F}}});
  ASSERT_EQ(actual.sizes(), input.sizes());
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
}

TEST(MiniMaxH3TransformerTest, SwiGluLoaderOwnsCompleteShardedWeightState) {
  torch::NoGradGuard no_grad;
  const torch::Tensor weight = torch::arange(16, torch::kFloat32).view({8, 2});
  const torch::Tensor expected[] = {
      torch::tensor({{8.0F, 9.0F}, {10.0F, 11.0F}, {0.0F, 1.0F}, {2.0F, 3.0F}}),
      torch::tensor(
          {{12.0F, 13.0F}, {14.0F, 15.0F}, {4.0F, 5.0F}, {6.0F, 7.0F}}),
  };
  for (int32_t rank : {0, 1}) {
    SCOPED_TRACE(rank);
    ProcessGroup group(rank, /*world_size=*/2, torch::Device(torch::kCPU));
    minimax_h3::H3SwiGLUFFN ffn(
        /*hidden_size=*/2,
        /*ffn_dim=*/4,
        /*bias=*/false,
        make_cpu_context(&group));
    ffn->load_state_dict(StateDict({{"net.0.proj.weight", weight}}));
    const auto children = ffn->named_children();
    auto* projection =
        children["proj_in"]->as<layer::ColumnParallelLinearImpl>();
    ASSERT_NE(projection, nullptr);
    EXPECT_TRUE(projection->is_weight_loaded());
    EXPECT_TRUE(torch::equal(projection->weight(), expected[rank]));
    ASSERT_DEATH(ffn->verify_loaded_weights("ff."), "net.2.weight");
    ffn->load_state_dict(StateDict({{"net.2.weight", torch::ones({2, 4})}}));
    ffn->verify_loaded_weights("ff.");
    EXPECT_TRUE(torch::equal(projection->weight(), expected[rank]));
  }
}

TEST(MiniMaxH3TransformerTest,
     FusedQkvRetainsCheckpointOrderAcrossFilesAndRanks) {
  torch::NoGradGuard no_grad;
  for (int32_t rank : {0, 1}) {
    SCOPED_TRACE(rank);
    ProcessGroup group(rank, /*world_size=*/2, torch::Device(torch::kCPU));
    minimax_h3::H3Attention attention(/*hidden_size=*/2,
                                      /*heads=*/2,
                                      /*dim_head=*/2,
                                      /*qk_norm_eps=*/1e-5,
                                      make_cpu_context(&group),
                                      /*sp_group=*/nullptr);
    attention->load_state_dict(StateDict({
        {"to_k.weight", torch::arange(8, torch::kFloat32).view({4, 2}) + 10},
        {"to_out.0.weight", torch::ones({2, 4})},
        {"norm_q.weight", torch::ones({2})},
        {"norm_k.weight", torch::ones({2})},
    }));
    ASSERT_DEATH(attention->verify_loaded_weights("attn."), "Q/K/V weights");
    attention->load_state_dict(StateDict({
        {"to_q.weight", torch::arange(8, torch::kFloat32).view({4, 2})},
    }));
    ASSERT_DEATH(attention->verify_loaded_weights("attn."), "Q/K/V weights");
    attention->load_state_dict(StateDict({
        {"to_v.weight", torch::arange(8, torch::kFloat32).view({4, 2}) + 20},
    }));
    attention->verify_loaded_weights("attn.");
    const auto children = attention->named_children();
    auto* projection =
        children["to_qkv"]->as<layer::ColumnParallelLinearImpl>();
    ASSERT_NE(projection, nullptr);
    const torch::Tensor expected = torch::tensor({{0.0F, 1.0F},
                                                  {2.0F, 3.0F},
                                                  {10.0F, 11.0F},
                                                  {12.0F, 13.0F},
                                                  {20.0F, 21.0F},
                                                  {22.0F, 23.0F}}) +
                                   4 * rank;
    EXPECT_TRUE(torch::equal(projection->weight(), expected));
  }
}

TEST(MiniMaxH3TransformerTest,
     NoisePreservesModalityStreamsAndPackedAudioOrder) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto noise = minimax_h3::prepare_noise(
      {1, 1, 2, 2, 2}, {2, 2, 2}, /*seed=*/42, options);
  const torch::Tensor expected_video = torch::tensor({0.336690F,
                                                      0.128809F,
                                                      0.234462F,
                                                      0.230333F,
                                                      -1.122856F,
                                                      -0.186328F,
                                                      2.208201F,
                                                      -0.637997F})
                                           .view({1, 1, 2, 2, 2});
  const torch::Tensor expected_audio =
      torch::tensor({{{0.336690F, 0.234462F}, {0.128809F, 0.230333F}},
                     {{-1.122856F, 2.208201F}, {-0.186328F, -0.637997F}}});
  EXPECT_TRUE(torch::allclose(noise.first, expected_video, 1e-6, 1e-6));
  EXPECT_TRUE(torch::allclose(noise.second, expected_audio, 1e-6, 1e-6));
  const auto resized = minimax_h3::prepare_noise(
      {1, 1, 4, 2, 2}, {2, 2, 2}, /*seed=*/42, options);
  EXPECT_TRUE(torch::equal(noise.second, resized.second));
  const auto reseeded = minimax_h3::prepare_noise(
      {1, 1, 2, 2, 2}, {2, 2, 2}, /*seed=*/43, options);
  EXPECT_FALSE(torch::equal(noise.first, reseeded.first));
  EXPECT_FALSE(torch::equal(noise.second, reseeded.second));
}

TEST(MiniMaxH3TransformerTest, SequenceAlignmentPreservesRowsAndResetsPadding) {
  for (int32_t rank : {0, 1}) {
    ProcessGroup group(rank, /*world_size=*/2, torch::Device(torch::kCPU));
    SequenceParallelProbe parallel(&group);
    for (int64_t length : {65, 64, 63}) {
      SCOPED_TRACE(::testing::Message()
                   << "rank=" << rank << " length=" << length);
      const int64_t padded_length = length == 65 ? 128 : 64;
      const torch::Tensor input = torch::arange(length).view({1, length, 1});
      parallel.sequence_parallel_forward(
          {{"rows", {input, 1}}},
          [&](const dit::SequenceParallelTensorMap& inputs) {
            const torch::Tensor padded = parallel.pad_tensor(input, "rows", 1);
            EXPECT_EQ(padded.size(1), padded_length);
            EXPECT_TRUE(
                torch::equal(parallel.unpad_tensor(padded, "rows", 1), input));
            const torch::Tensor local = inputs.at("rows").first;
            EXPECT_EQ(local.size(1), padded_length / 2);
            EXPECT_TRUE(
                torch::equal(local,
                             padded.slice(1,
                                          rank * padded_length / 2,
                                          (rank + 1) * padded_length / 2)));
            EXPECT_EQ(padded.slice(1, length).count_nonzero().item<int64_t>(),
                      0);
            // The real scatter needs no collective backend. Gathering is
            // exercised by the two-rank NPU regression.
            return dit::SequenceParallelTensorMap{};
          },
          minimax_h3::kPackedSequenceAlignment);
    }
    parallel.sequence_parallel_forward(
        {{"rows", {torch::ones({1, 65, 1}), 1}}},
        [&](const dit::SequenceParallelTensorMap& inputs) {
          EXPECT_EQ(inputs.at("rows").first.size(1), 33);
          return dit::SequenceParallelTensorMap{};
        });
  }
}

}  // namespace
}  // namespace xllm
