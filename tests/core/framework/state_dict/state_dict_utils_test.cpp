/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace {

TEST(StateDictUtilsTest, LoadsMoeWeightsIntoPhysicalExpertPrefix) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("0.up_proj.weight",
                  torch::tensor({{1, 2}, {3, 4}}, torch::kInt32));
  tensors.emplace("1.up_proj.weight",
                  torch::tensor({{5, 6}, {7, 8}}, torch::kInt32));
  StateDict state_dict(std::move(tensors));

  torch::Tensor physical_weight = torch::full({3, 2, 2}, -1, torch::kInt32);
  std::vector<torch::Tensor> accumulated_tensors;
  bool weight_is_loaded = false;

  weight::load_moe_weight(state_dict,
                          "up_proj.",
                          "weight",
                          /*dim=*/-1,
                          /*rank=*/0,
                          /*world_size=*/1,
                          /*start_expert_id=*/0,
                          /*num_experts_per_rank=*/2,
                          accumulated_tensors,
                          physical_weight,
                          weight_is_loaded);

  EXPECT_TRUE(weight_is_loaded);
  EXPECT_TRUE(torch::equal(physical_weight[0],
                           torch::tensor({{1, 2}, {3, 4}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(physical_weight[1],
                           torch::tensor({{5, 6}, {7, 8}}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(physical_weight[2], torch::full({2, 2}, -1, torch::kInt32)));
}

TEST(StateDictUtilsTest, ReshapesMoeScaleInsidePhysicalExpertPrefix) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.emplace("0.up_proj.weight_scale",
                  torch::tensor({{1}, {2}}, torch::kFloat32));
  tensors.emplace("1.up_proj.weight_scale",
                  torch::tensor({{3}, {4}}, torch::kFloat32));
  StateDict state_dict(std::move(tensors));

  torch::Tensor physical_scale = torch::full({3, 2}, -1, torch::kFloat32);
  std::vector<torch::Tensor> accumulated_tensors;
  bool scale_is_loaded = false;

  weight::load_moe_weight(state_dict,
                          "up_proj.",
                          "weight_scale",
                          /*dim=*/-1,
                          /*rank=*/0,
                          /*world_size=*/1,
                          /*start_expert_id=*/0,
                          /*num_experts_per_rank=*/2,
                          accumulated_tensors,
                          physical_scale,
                          scale_is_loaded);

  EXPECT_TRUE(scale_is_loaded);
  EXPECT_TRUE(
      torch::equal(physical_scale,
                   torch::tensor({{1, 2}, {3, 4}, {-1, -1}}, torch::kFloat32)));
}

}  // namespace
}  // namespace xllm
