/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/minimax_h3_qualified_weights.h"
#include "models/dit/utils/minimax_h3_qualified_input.h"

namespace xllm {
namespace {

H3QualifiedShapeConfig small_shape() {
  H3QualifiedShapeConfig shape;
  shape.hidden_size = 4;
  shape.heads = 2;
  shape.head_dim = 2;
  shape.ffn_dim = 4;
  shape.layers = 2;
  shape.refiner_layers = 2;
  shape.video_patch_dim = 4;
  shape.audio_dim = 2;
  shape.text_dim = 6;
  shape.time_input_dim = 2;
  shape.time_hidden_dim = 4;
  shape.time_dim = 2;
  shape.rope_freq_dim = 1;
  return shape;
}

using TensorMap = std::unordered_map<std::string, torch::Tensor>;
TensorMap source_tensors(const H3QualifiedShapeConfig& shape) {
  TensorMap tensors;
  for (const auto& spec : h3_qualified_source_inventory(shape)) {
    tensors.emplace(spec.source,
                    torch::zeros(spec.shape, torch::dtype(spec.dtype)));
  }
  return tensors;
}

std::vector<std::unique_ptr<StateDict>> split_shards(TensorMap tensors) {
  std::array<TensorMap, 3> maps;
  size_t index = 0;
  for (auto& [name, tensor] : tensors) {
    maps[index++ % maps.size()].emplace(name, std::move(tensor));
  }
  std::vector<std::unique_ptr<StateDict>> shards;
  shards.reserve(maps.size());
  for (auto& map : maps) {
    shards.emplace_back(std::make_unique<StateDict>(std::move(map)));
  }
  return shards;
}

TEST(QualifiedWeights, HeadMajorQkvAndGateUpAreExact) {
  const auto options = torch::dtype(torch::kBFloat16);
  auto query = torch::tensor({0, 1, 2, 3}, options).reshape({4, 1});
  auto key = torch::tensor({10, 11, 12, 13}, options).reshape({4, 1});
  auto value = torch::tensor({20, 21, 22, 23}, options).reshape({4, 1});
  auto grouped = h3_qualified_group_qkv(query, key, value, 2, 2);
  auto expected =
      torch::tensor({0, 1, 10, 11, 20, 21, 2, 3, 12, 13, 22, 23}, options)
          .reshape({12, 1});
  EXPECT_TRUE(torch::equal(grouped, expected));
  auto up_gate =
      torch::tensor({1, 2, 3, 4, 11, 12, 13, 14}, options).reshape({8, 1});
  auto gate_up = h3_qualified_up_gate_to_gate_up(up_gate);
  EXPECT_TRUE(torch::equal(
      gate_up,
      torch::tensor({11, 12, 13, 14, 1, 2, 3, 4}, options).reshape({8, 1})));
}

TEST(QualifiedWeights, CrossShardMappingOwnsSourceAndDoesNotMutateIt) {
  const auto shape = small_shape();
  auto tensors = source_tensors(shape);
  const auto options = torch::dtype(torch::kBFloat16);
  tensors["transformer_blocks.0.attn.to_q.weight"] =
      torch::arange(16, options).reshape({4, 4});
  tensors["transformer_blocks.0.attn.to_k.weight"] =
      (100 + torch::arange(16, options)).reshape({4, 4});
  tensors["transformer_blocks.0.attn.to_v.weight"] =
      (200 + torch::arange(16, options)).reshape({4, 4});
  tensors["transformer_blocks.0.ff.net.0.proj.weight"] =
      torch::arange(32, options).reshape({8, 4});
  auto original_q = tensors.at("transformer_blocks.0.attn.to_q.weight").clone();
  auto borrowed_q = tensors.at("transformer_blocks.0.attn.to_q.weight");
  auto shards = split_shards(std::move(tensors));
  H3QualifiedWeightSource source(std::move(shards), shape);
  auto fixed = source.fixed_weights();
  auto block = source.block_weights(0);
  EXPECT_EQ(fixed.size(), 35);
  EXPECT_EQ(block.size(), 10);
  EXPECT_TRUE(torch::equal(borrowed_q, original_q));
  auto qkv_first_column = block.get_tensor("attn.qkv_proj.weight").select(1, 0);
  EXPECT_TRUE(torch::equal(
      qkv_first_column,
      torch::tensor({0, 4, 100, 104, 200, 204, 8, 12, 108, 112, 208, 212},
                    options)));
  EXPECT_TRUE(
      torch::equal(block.get_tensor("mlp.fc1.weight").select(1, 0),
                   torch::tensor({16, 20, 24, 28, 0, 4, 8, 12}, options)));
  EXPECT_EQ(source.block_weights(1).size(), 10);
  EXPECT_THROW(source.block_weights(2), std::invalid_argument);
}

TEST(QualifiedWeights, RejectsMissingDuplicateAndWrongDtype) {
  const auto shape = small_shape();
  auto missing = source_tensors(shape);
  missing.erase("transformer_blocks.1.attn.to_k.weight");
  EXPECT_THROW(H3QualifiedWeightSource(split_shards(std::move(missing)), shape),
               std::invalid_argument);
  auto duplicate = split_shards(source_tensors(shape));
  duplicate.emplace_back(std::make_unique<StateDict>(
      TensorMap{{"proj_in.bias", torch::zeros({4})}}));
  EXPECT_THROW(H3QualifiedWeightSource(std::move(duplicate), shape),
               std::invalid_argument);
  auto wrong_dtype = source_tensors(shape);
  wrong_dtype["proj_in.bias"] = torch::zeros({4}, torch::kBFloat16);
  EXPECT_THROW(
      H3QualifiedWeightSource(split_shards(std::move(wrong_dtype)), shape),
      std::invalid_argument);
}

TEST(QualifiedWeights, ProductionInventoryAndGeneratedRope) {
  const H3QualifiedShapeConfig shape;
  const auto inventory = h3_qualified_source_inventory(shape);
  EXPECT_EQ(inventory.size(), 638);
  int64_t fp32 = 0;
  for (const auto& spec : inventory) {
    fp32 += spec.dtype == torch::kFloat32;
  }
  EXPECT_EQ(fp32, 12);
  auto inv = h3_qualified_rope_inv_freq(shape);
  EXPECT_EQ(inv.sizes().vec(), (std::vector<int64_t>{16}));
  EXPECT_EQ(inv.scalar_type(), torch::kFloat32);
  EXPECT_EQ(inv[0].item<float>(), 1.0F);
  EXPECT_TRUE(torch::all(inv.slice(0, 1) < inv.slice(0, 0, 15)).item<bool>());
}

TEST(QualifiedPacking, PreservesRowsMasksAndAlignment) {
  auto condition = torch::zeros({1, 3, 5120}, torch::kBFloat16);
  auto tags = torch::tensor({{1, 0, 1}}, torch::kInt64);
  const H3TargetLatents target{.audio_t = 2,
                               .audio_channels = 2,
                               .latent_t = 1,
                               .latent_h = 2,
                               .latent_w = 4};
  const std::vector<H3ReferenceBlock> refs{{.kind = H3ReferenceBlockKind::IMAGE,
                                            .latent_t = 1,
                                            .latent_h = 2,
                                            .latent_w = 2}};
  auto layout =
      minimax_h3_build_ref2va_packed_layout(condition, tags, target, refs);
  EXPECT_EQ(layout.used_length, 10);
  EXPECT_EQ(layout.aligned_length, 64);
  EXPECT_TRUE(
      torch::equal(layout.img_pos, torch::tensor({3, 8, 9}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(layout.audio_pos,
                           torch::tensor({4, 5, 6, 7}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(layout.update_mask,
                           torch::tensor({false, true, true}, torch::kBool)));
  EXPECT_TRUE(torch::equal(layout.cu_seqlens,
                           torch::tensor({0, 10, 64}, torch::kInt32)));
  EXPECT_TRUE(torch::all(layout.token_tags.slice(0, 10) == -1).item<bool>());
}

TEST(QualifiedPacking, VideoAndAudioRoundTripKeepRowOrder) {
  auto video = torch::arange(32, torch::kFloat32).reshape({1, 2, 2, 2, 4});
  auto rows = minimax_h3_patchify_video_latent(video);
  auto restored = minimax_h3_unpatchify_video_tokens(
      rows, {.channels = 2, .temporal = 2, .height = 2, .width = 4});
  EXPECT_TRUE(torch::equal(video, restored));
  auto audio = torch::arange(24, torch::kFloat32).reshape({2, 3, 4});
  EXPECT_TRUE(torch::equal(audio,
                           minimax_h3_unpack_audio_tokens(
                               minimax_h3_pack_audio_latent(audio), 2, 4)));
}

TEST(QualifiedTimestep, GeneratedAudioUsesItsOwnClock) {
  H3PackedLayout layout;
  layout.used_length = 8;
  layout.aligned_length = 8;
  layout.img_pos = torch::tensor({2, 7}, torch::kInt64);
  layout.audio_pos = torch::tensor({3, 4, 5, 6}, torch::kInt64);
  layout.update_mask = torch::tensor({false, true}, torch::kBool);
  layout.audio_update_mask =
      torch::tensor({false, false, true, true}, torch::kBool);
  const auto plan = minimax_h3_build_row_timestep_plan(layout, 0.2F, 0.4F);
  EXPECT_TRUE(torch::equal(
      plan.row_timesteps,
      torch::tensor({0.2F, 0.2F, 0.999F, 1.0F, 1.0F, 0.4F, 0.4F, 0.2F})));
  EXPECT_TRUE(torch::equal(plan.unique_timesteps,
                           torch::tensor({0.2F, 0.4F, 0.999F, 1.0F})));
  EXPECT_TRUE(
      torch::equal(plan.inverse_indices,
                   torch::tensor({0, 0, 2, 3, 3, 1, 1, 0}, torch::kInt64)));
}

TEST(QualifiedSchedule, UpstreamFortyNineStepsMatchFrozenFiftyPointClock) {
  for (double shift : {12.0, 3.0}) {
    MiniMaxH3Scheduler scheduler(shift);
    scheduler.set_timesteps(49, torch::Device(torch::kCPU));
    auto sigmas = h3_qualified_sigmas(static_cast<float>(shift));
    EXPECT_EQ(sigmas.numel(), 50);
    EXPECT_TRUE(
        torch::equal(scheduler.timesteps(), 1.0F - sigmas.slice(0, 0, 49)));
  }
}

TEST(QualifiedSchedule, AppliesVelocityOnceAndResetsNextRequest) {
  MiniMaxH3Scheduler scheduler(/*shift=*/1.0);
  scheduler.set_timesteps(2, torch::Device(torch::kCPU));
  auto velocity = torch::tensor({4.0F});
  auto first =
      scheduler.step(velocity, scheduler.timesteps()[0], torch::tensor({2.0F}));
  auto second = scheduler.step(velocity, scheduler.timesteps()[1], first);
  EXPECT_EQ(first.item<float>(), 4.0F);
  EXPECT_EQ(second.item<float>(), 6.0F);
  scheduler.set_timesteps(2, torch::Device(torch::kCPU));
  EXPECT_EQ(
      scheduler.step(velocity, scheduler.timesteps()[0], torch::tensor({2.0F}))
          .item<float>(),
      4.0F);
}
}  // namespace
}  // namespace xllm
