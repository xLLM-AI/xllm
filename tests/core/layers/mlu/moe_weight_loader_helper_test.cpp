/* Copyright 2026 The xLLM Authors.

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

#include "layers/common/moe_weight_loader_helper.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor make_arange(const std::vector<int64_t>& sizes,
                          torch::ScalarType dtype) {
  int64_t numel = 1;
  for (int64_t size : sizes) {
    numel *= size;
  }
  torch::Tensor tensor =
      torch::arange(numel, torch::TensorOptions().dtype(torch::kInt64));
  return tensor.to(dtype).reshape(sizes);
}

TEST(MoEWeightHelperTest, LoadsGateUpSmoothquantShard) {
  const int64_t rank = 1;
  const int64_t world_size = 2;
  const int64_t start_expert_id = 1;
  const int64_t num_experts_per_rank = 2;
  const int64_t hidden_size = 3;
  const int64_t full_intermediate = 4;
  const int64_t local_intermediate = full_intermediate / world_size;
  const int64_t num_total_experts = 4;

  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors["gate_up_proj.qweight"] = make_arange(
      {num_total_experts, full_intermediate * 2, hidden_size}, torch::kInt8);
  tensors["gate_up_proj.per_channel_scale"] =
      make_arange({num_total_experts, full_intermediate * 2}, torch::kFloat32);
  tensors["gate_up_proj.smooth"] =
      torch::tensor({0.25f, 0.5f, 0.75f}, torch::kFloat32);
  StateDict state_dict(std::move(tensors));

  torch::Tensor w13 =
      torch::empty({num_experts_per_rank, local_intermediate * 2, hidden_size},
                   torch::kInt8);
  torch::Tensor w13_scale = torch::empty(
      {num_experts_per_rank, local_intermediate * 2}, torch::kFloat32);
  torch::Tensor input_smooth =
      torch::empty({num_total_experts, hidden_size}, torch::kFloat32);
  bool w13_loaded = false;
  bool scale_loaded = false;
  bool smooth_loaded = false;

  bool loaded = moe_weight::try_load_gate_up_sq(state_dict,
                                                rank,
                                                world_size,
                                                start_expert_id,
                                                num_experts_per_rank,
                                                w13,
                                                w13_scale,
                                                input_smooth,
                                                w13_loaded,
                                                scale_loaded,
                                                smooth_loaded);

  torch::Tensor fused_qweight = state_dict.get_tensor("gate_up_proj.qweight");
  torch::Tensor gate_qweight = fused_qweight.slice(
      /*dim=*/1, rank * local_intermediate, (rank + 1) * local_intermediate);
  torch::Tensor up_qweight =
      fused_qweight.slice(/*dim=*/1,
                          full_intermediate + rank * local_intermediate,
                          full_intermediate + (rank + 1) * local_intermediate);
  torch::Tensor expected_w13 =
      torch::cat({gate_qweight, up_qweight}, /*dim=*/1)
          .slice(/*dim=*/0,
                 start_expert_id,
                 start_expert_id + num_experts_per_rank)
          .contiguous();

  torch::Tensor fused_scale =
      state_dict.get_tensor("gate_up_proj.per_channel_scale");
  torch::Tensor gate_scale = fused_scale.slice(
      /*dim=*/1, rank * local_intermediate, (rank + 1) * local_intermediate);
  torch::Tensor up_scale =
      fused_scale.slice(/*dim=*/1,
                        full_intermediate + rank * local_intermediate,
                        full_intermediate + (rank + 1) * local_intermediate);
  torch::Tensor expected_scale =
      torch::cat({gate_scale, up_scale}, /*dim=*/1)
          .slice(/*dim=*/0,
                 start_expert_id,
                 start_expert_id + num_experts_per_rank)
          .contiguous();
  torch::Tensor expected_smooth = state_dict.get_tensor("gate_up_proj.smooth")
                                      .reshape({1, -1})
                                      .expand({num_total_experts, hidden_size});

  EXPECT_TRUE(loaded);
  EXPECT_TRUE(w13_loaded);
  EXPECT_TRUE(scale_loaded);
  EXPECT_TRUE(smooth_loaded);
  EXPECT_TRUE(torch::equal(w13.cpu(), expected_w13.cpu()));
  EXPECT_TRUE(torch::equal(w13_scale.cpu(), expected_scale.cpu()));
  EXPECT_TRUE(torch::equal(input_smooth.cpu(), expected_smooth.cpu()));
}

TEST(MoEWeightHelperTest, LoadsDownSmoothquantShard) {
  const int64_t rank = 1;
  const int64_t world_size = 2;
  const int64_t start_expert_id = 1;
  const int64_t num_experts_per_rank = 2;
  const int64_t num_total_experts = 4;
  const int64_t hidden_size = 3;
  const int64_t local_intermediate = 2;
  const int64_t full_intermediate = local_intermediate * world_size;
  const int64_t local_groups = 2;
  const int64_t full_groups = local_groups * world_size;

  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors["down_proj.qweight"] = make_arange(
      {num_total_experts, hidden_size, full_intermediate}, torch::kInt8);
  tensors["down_proj.per_channel_scale"] = make_arange(
      {num_total_experts, hidden_size, full_groups}, torch::kFloat32);
  tensors["down_proj.smooth"] =
      torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32);
  StateDict state_dict(std::move(tensors));

  torch::Tensor w2 = torch::empty(
      {num_experts_per_rank, hidden_size, local_intermediate}, torch::kInt8);
  torch::Tensor w2_scale = torch::empty(
      {num_experts_per_rank, hidden_size, local_groups}, torch::kFloat32);
  torch::Tensor act_smooth =
      torch::empty({num_experts_per_rank, local_intermediate}, torch::kFloat32);
  bool w2_loaded = false;
  bool scale_loaded = false;
  bool smooth_loaded = false;

  bool loaded = moe_weight::try_load_down_sq(state_dict,
                                             rank,
                                             world_size,
                                             start_expert_id,
                                             num_experts_per_rank,
                                             w2,
                                             w2_scale,
                                             act_smooth,
                                             w2_loaded,
                                             scale_loaded,
                                             smooth_loaded);

  torch::Tensor expected_w2 = state_dict.get_tensor("down_proj.qweight")
                                  .slice(/*dim=*/2,
                                         rank * local_intermediate,
                                         (rank + 1) * local_intermediate)
                                  .slice(/*dim=*/0,
                                         start_expert_id,
                                         start_expert_id + num_experts_per_rank)
                                  .contiguous();
  torch::Tensor expected_scale =
      state_dict.get_tensor("down_proj.per_channel_scale")
          .slice(/*dim=*/2, rank * local_groups, (rank + 1) * local_groups)
          .slice(/*dim=*/0,
                 start_expert_id,
                 start_expert_id + num_experts_per_rank)
          .contiguous();
  torch::Tensor expected_smooth =
      state_dict.get_tensor("down_proj.smooth")
          .slice(/*dim=*/0,
                 rank * local_intermediate,
                 (rank + 1) * local_intermediate)
          .reshape({1, -1})
          .expand({num_experts_per_rank, local_intermediate});

  EXPECT_TRUE(loaded);
  EXPECT_TRUE(w2_loaded);
  EXPECT_TRUE(scale_loaded);
  EXPECT_TRUE(smooth_loaded);
  EXPECT_TRUE(torch::equal(w2.cpu(), expected_w2.cpu()));
  EXPECT_TRUE(torch::equal(w2_scale.cpu(), expected_scale.cpu()));
  EXPECT_TRUE(torch::equal(act_smooth.cpu(), expected_smooth.cpu()));
}

TEST(MoEWeightHelperTest, LoadsDownSmoothquantShardWith2DScale) {
  const int64_t rank = 1;
  const int64_t world_size = 2;
  const int64_t start_expert_id = 1;
  const int64_t num_experts_per_rank = 2;
  const int64_t num_total_experts = 4;
  const int64_t hidden_size = 3;
  const int64_t local_intermediate = 2;
  const int64_t full_intermediate = local_intermediate * world_size;

  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors["down_proj.qweight"] = make_arange(
      {num_total_experts, hidden_size, full_intermediate}, torch::kInt8);
  tensors["down_proj.per_channel_scale"] =
      make_arange({num_total_experts, full_intermediate}, torch::kFloat32);
  tensors["down_proj.smooth"] =
      torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32);
  StateDict state_dict(std::move(tensors));

  torch::Tensor w2 = torch::empty(
      {num_experts_per_rank, hidden_size, local_intermediate}, torch::kInt8);
  torch::Tensor w2_scale =
      torch::empty({num_experts_per_rank, local_intermediate}, torch::kFloat32);
  torch::Tensor act_smooth =
      torch::empty({num_experts_per_rank, local_intermediate}, torch::kFloat32);
  bool w2_loaded = false;
  bool scale_loaded = false;
  bool smooth_loaded = false;

  bool loaded = moe_weight::try_load_down_sq(state_dict,
                                             rank,
                                             world_size,
                                             start_expert_id,
                                             num_experts_per_rank,
                                             w2,
                                             w2_scale,
                                             act_smooth,
                                             w2_loaded,
                                             scale_loaded,
                                             smooth_loaded);

  torch::Tensor expected_w2 = state_dict.get_tensor("down_proj.qweight")
                                  .slice(/*dim=*/2,
                                         rank * local_intermediate,
                                         (rank + 1) * local_intermediate)
                                  .slice(/*dim=*/0,
                                         start_expert_id,
                                         start_expert_id + num_experts_per_rank)
                                  .contiguous();
  torch::Tensor expected_scale =
      state_dict.get_tensor("down_proj.per_channel_scale")
          .slice(/*dim=*/1,
                 rank * local_intermediate,
                 (rank + 1) * local_intermediate)
          .slice(/*dim=*/0,
                 start_expert_id,
                 start_expert_id + num_experts_per_rank)
          .contiguous();
  torch::Tensor expected_smooth =
      state_dict.get_tensor("down_proj.smooth")
          .slice(/*dim=*/0,
                 rank * local_intermediate,
                 (rank + 1) * local_intermediate)
          .reshape({1, -1})
          .expand({num_experts_per_rank, local_intermediate});

  EXPECT_TRUE(loaded);
  EXPECT_TRUE(w2_loaded);
  EXPECT_TRUE(scale_loaded);
  EXPECT_TRUE(smooth_loaded);
  EXPECT_TRUE(torch::equal(w2.cpu(), expected_w2.cpu()));
  EXPECT_TRUE(torch::equal(w2_scale.cpu(), expected_scale.cpu()));
  EXPECT_TRUE(torch::equal(act_smooth.cpu(), expected_smooth.cpu()));
}

TEST(MoEWeightHelperTest, MissingFusedKeyReturnsFalseWithoutMutation) {
  StateDict state_dict(std::unordered_map<std::string, torch::Tensor>{});
  torch::Tensor w13 = torch::full({2, 4, 3}, -7, torch::kInt8);
  torch::Tensor w13_scale = torch::full({2, 4}, -3.0f, torch::kFloat32);
  torch::Tensor input_smooth = torch::full({4, 3}, -5.0f, torch::kFloat32);
  torch::Tensor w2 = torch::full({2, 3, 2}, -9, torch::kInt8);
  torch::Tensor w2_scale = torch::full({2, 3, 2}, -2.0f, torch::kFloat32);
  torch::Tensor act_smooth = torch::full({2, 2}, -4.0f, torch::kFloat32);
  torch::Tensor expected_w13 = w13.clone();
  torch::Tensor expected_w13_scale = w13_scale.clone();
  torch::Tensor expected_input_smooth = input_smooth.clone();
  torch::Tensor expected_w2 = w2.clone();
  torch::Tensor expected_w2_scale = w2_scale.clone();
  torch::Tensor expected_act_smooth = act_smooth.clone();
  bool w13_loaded = false;
  bool w13_scale_loaded = false;
  bool input_smooth_loaded = false;
  bool w2_loaded = false;
  bool w2_scale_loaded = false;
  bool act_smooth_loaded = false;

  bool gate_up_loaded =
      moe_weight::try_load_gate_up_sq(state_dict,
                                      /*rank=*/0,
                                      /*world_size=*/2,
                                      /*start_expert_id=*/0,
                                      /*num_experts_per_rank=*/2,
                                      w13,
                                      w13_scale,
                                      input_smooth,
                                      w13_loaded,
                                      w13_scale_loaded,
                                      input_smooth_loaded);
  bool down_loaded = moe_weight::try_load_down_sq(state_dict,
                                                  /*rank=*/0,
                                                  /*world_size=*/2,
                                                  /*start_expert_id=*/0,
                                                  /*num_experts_per_rank=*/2,
                                                  w2,
                                                  w2_scale,
                                                  act_smooth,
                                                  w2_loaded,
                                                  w2_scale_loaded,
                                                  act_smooth_loaded);

  EXPECT_FALSE(gate_up_loaded);
  EXPECT_FALSE(down_loaded);
  EXPECT_FALSE(w13_loaded);
  EXPECT_FALSE(w13_scale_loaded);
  EXPECT_FALSE(input_smooth_loaded);
  EXPECT_FALSE(w2_loaded);
  EXPECT_FALSE(w2_scale_loaded);
  EXPECT_FALSE(act_smooth_loaded);
  EXPECT_TRUE(torch::equal(w13.cpu(), expected_w13.cpu()));
  EXPECT_TRUE(torch::equal(w13_scale.cpu(), expected_w13_scale.cpu()));
  EXPECT_TRUE(torch::equal(input_smooth.cpu(), expected_input_smooth.cpu()));
  EXPECT_TRUE(torch::equal(w2.cpu(), expected_w2.cpu()));
  EXPECT_TRUE(torch::equal(w2_scale.cpu(), expected_w2_scale.cpu()));
  EXPECT_TRUE(torch::equal(act_smooth.cpu(), expected_act_smooth.cpu()));
}

TEST(MoEWeightHelperTest, ShapeMismatchReturnsFalseWithoutMutation) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors["gate_up_proj.qweight"] = make_arange({4, 8, 3}, torch::kInt8);
  tensors["gate_up_proj.per_channel_scale"] =
      make_arange({4, 8}, torch::kFloat32);
  tensors["gate_up_proj.smooth"] =
      torch::tensor({0.25f, 0.5f, 0.75f}, torch::kFloat32);
  tensors["down_proj.qweight"] = make_arange({4, 3, 4}, torch::kInt8);
  tensors["down_proj.per_channel_scale"] =
      make_arange({4, 3, 4}, torch::kFloat32);
  tensors["down_proj.smooth"] =
      torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32);
  StateDict state_dict(std::move(tensors));

  torch::Tensor w13 = torch::full({2, 4, 3}, -7, torch::kInt8);
  torch::Tensor bad_w13_scale = torch::full({2, 3}, -3.0f, torch::kFloat32);
  torch::Tensor input_smooth = torch::full({4, 3}, -5.0f, torch::kFloat32);
  torch::Tensor w2 = torch::full({2, 3, 2}, -9, torch::kInt8);
  torch::Tensor bad_w2_scale = torch::full({2, 3, 1}, -2.0f, torch::kFloat32);
  torch::Tensor act_smooth = torch::full({2, 2}, -4.0f, torch::kFloat32);
  torch::Tensor expected_w13 = w13.clone();
  torch::Tensor expected_w13_scale = bad_w13_scale.clone();
  torch::Tensor expected_input_smooth = input_smooth.clone();
  torch::Tensor expected_w2 = w2.clone();
  torch::Tensor expected_w2_scale = bad_w2_scale.clone();
  torch::Tensor expected_act_smooth = act_smooth.clone();
  bool w13_loaded = false;
  bool w13_scale_loaded = false;
  bool input_smooth_loaded = false;
  bool w2_loaded = false;
  bool w2_scale_loaded = false;
  bool act_smooth_loaded = false;

  bool gate_up_loaded =
      moe_weight::try_load_gate_up_sq(state_dict,
                                      /*rank=*/1,
                                      /*world_size=*/2,
                                      /*start_expert_id=*/1,
                                      /*num_experts_per_rank=*/2,
                                      w13,
                                      bad_w13_scale,
                                      input_smooth,
                                      w13_loaded,
                                      w13_scale_loaded,
                                      input_smooth_loaded);
  bool down_loaded = moe_weight::try_load_down_sq(state_dict,
                                                  /*rank=*/1,
                                                  /*world_size=*/2,
                                                  /*start_expert_id=*/1,
                                                  /*num_experts_per_rank=*/2,
                                                  w2,
                                                  bad_w2_scale,
                                                  act_smooth,
                                                  w2_loaded,
                                                  w2_scale_loaded,
                                                  act_smooth_loaded);

  EXPECT_FALSE(gate_up_loaded);
  EXPECT_FALSE(down_loaded);
  EXPECT_FALSE(w13_loaded);
  EXPECT_FALSE(w13_scale_loaded);
  EXPECT_FALSE(input_smooth_loaded);
  EXPECT_FALSE(w2_loaded);
  EXPECT_FALSE(w2_scale_loaded);
  EXPECT_FALSE(act_smooth_loaded);
  EXPECT_TRUE(torch::equal(w13.cpu(), expected_w13.cpu()));
  EXPECT_TRUE(torch::equal(bad_w13_scale.cpu(), expected_w13_scale.cpu()));
  EXPECT_TRUE(torch::equal(input_smooth.cpu(), expected_input_smooth.cpu()));
  EXPECT_TRUE(torch::equal(w2.cpu(), expected_w2.cpu()));
  EXPECT_TRUE(torch::equal(bad_w2_scale.cpu(), expected_w2_scale.cpu()));
  EXPECT_TRUE(torch::equal(act_smooth.cpu(), expected_act_smooth.cpu()));
}

}  // namespace
}  // namespace layer
}  // namespace xllm
