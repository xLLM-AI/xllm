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

#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

torch::Device mlu_device() {
  return torch::Device(torch::kPrivateUse1, /*index=*/0);
}

TEST(PackCacheBlocksTest, GathersPairsWithoutChangingSurroundingStorage) {
  const torch::Device device = mlu_device();
  const torch::DeviceGuard guard(device);
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::Tensor block_ids = torch::tensor(
      {3, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  const torch::Tensor float_source =
      torch::arange(/*end=*/24, float_options).view({4, 2, 3});
  const torch::Tensor int_source =
      torch::arange(/*end=*/16, int_options).view({4, 2, 2});
  torch::Tensor float_storage = torch::full({4, 2, 3}, -7, float_options);
  torch::Tensor int_storage = torch::full({4, 2, 2}, -9, int_options);
  torch::Tensor float_destination = float_storage.narrow(0, 1, 2);
  torch::Tensor int_destination = int_storage.narrow(0, 1, 2);

  pack_cache_blocks({float_source, int_source},
                    block_ids,
                    {float_destination, int_destination});

  EXPECT_TRUE(torch::equal(float_destination.cpu(),
                           float_source.index_select(0, block_ids).cpu()));
  EXPECT_TRUE(torch::equal(int_destination.cpu(),
                           int_source.index_select(0, block_ids).cpu()));
  EXPECT_TRUE(torch::equal(float_storage[0].cpu(),
                           torch::full({2, 3}, -7, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(float_storage[3].cpu(),
                           torch::full({2, 3}, -7, torch::kFloat32)));
  EXPECT_TRUE(torch::equal(int_storage[0].cpu(),
                           torch::full({2, 2}, -9, torch::kInt32)));
  EXPECT_TRUE(torch::equal(int_storage[3].cpu(),
                           torch::full({2, 2}, -9, torch::kInt32)));
}

TEST(PackCacheBlocksDeathTest, RejectsMismatchedPairCounts) {
  EXPECT_DEATH(pack_cache_blocks({}, torch::Tensor(), {torch::Tensor()}),
               "Check failed: sources.size\\(\\) == destinations.size\\(\\)");
}

}  // namespace
}  // namespace xllm::kernel::mlu
