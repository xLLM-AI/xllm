/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "framework/state_dict/utils.h"

namespace xllm {
namespace weight {
namespace {

torch::Tensor make_weight(int64_t rows, int64_t cols) {
  return torch::arange(rows * cols, torch::kFloat32).reshape({rows, cols});
}

}  // namespace

TEST(StateDictUtilsTest, UsesReplicatedKvShardWhenPreviousKvTensorIsLoaded) {
  StateDict state_dict({
      {"q_proj.weight", make_weight(/*rows=*/16, /*cols=*/2)},
      {"k_proj.weight", make_weight(/*rows=*/4, /*cols=*/2)},
      {"v_proj.weight", make_weight(/*rows=*/4, /*cols=*/2) + 100.0f},
  });
  const std::vector<std::string> prefixes = {"q_proj.", "k_proj.", "v_proj."};
  std::vector<torch::Tensor> tensors = {torch::ones({4, 2}, torch::kFloat32),
                                        torch::ones({2, 2}, torch::kFloat32),
                                        torch::Tensor()};

  const bool is_loaded = load_tensor_list(state_dict,
                                          prefixes,
                                          "weight",
                                          /*dim=*/0,
                                          /*rank=*/2,
                                          /*world_size=*/4,
                                          tensors,
                                          /*num_kv_head_replicas=*/2);

  ASSERT_TRUE(is_loaded);
  const torch::Tensor expected_v =
      state_dict.get_sharded_tensor("v_proj.weight",
                                    /*dim=*/0,
                                    /*rank=*/1,
                                    /*world_size=*/2);
  EXPECT_TRUE(torch::equal(tensors[2], expected_v));
}

}  // namespace weight
}  // namespace xllm
