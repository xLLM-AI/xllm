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

#include "core/layers/common/dflash2_grouped_conv.h"

namespace xllm::layer {

TEST(DFlash2GroupedConvTest, ResetsHistoryAtEachBlockBoundary) {
  torch::Tensor hidden = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f},
                                        {5.0f, 6.0f, 7.0f, 8.0f},
                                        {9.0f, 10.0f, 11.0f, 12.0f},
                                        {13.0f, 14.0f, 15.0f, 16.0f}});
  torch::Tensor delta = torch::zeros({4, 2, 2});
  torch::Tensor base = torch::ones({2, 4});

  torch::Tensor output = dflash2_grouped_conv(hidden,
                                              delta,
                                              base,
                                              /*block_size=*/2,
                                              /*num_groups=*/2,
                                              /*group_size=*/2,
                                              /*taps=*/2);
  torch::Tensor expected = torch::stack(
      {hidden[0], hidden[1] + hidden[0], hidden[2], hidden[3] + hidden[2]});
  EXPECT_TRUE(torch::allclose(output, expected));
}

TEST(DFlash2GroupedConvTest, BroadcastsDynamicKernelWithinEachGroup) {
  torch::Tensor hidden =
      torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}});
  torch::Tensor delta = torch::tensor({{{1.0f, 2.0f}}, {{3.0f, 4.0f}}});
  torch::Tensor base = torch::zeros({1, 4});

  torch::Tensor output = dflash2_grouped_conv(hidden,
                                              delta,
                                              base,
                                              /*block_size=*/2,
                                              /*num_groups=*/2,
                                              /*group_size=*/2,
                                              /*taps=*/1);
  torch::Tensor expected =
      torch::tensor({{1.0f, 2.0f, 6.0f, 8.0f}, {15.0f, 18.0f, 28.0f, 32.0f}});
  EXPECT_TRUE(torch::allclose(output, expected));
}

}  // namespace xllm::layer
