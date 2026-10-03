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

#include "layers/npu_torch/dcp_attention.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>

namespace xllm {
namespace {

TEST(NpuDcpAttentionMergeTest, IgnoresNonFiniteLseBeforeMax) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const torch::Tensor partial_outputs =
      torch::tensor({nan, nan, infinity, -infinity, 3.0f, 9.0f})
          .reshape({3, 1, 1, 2});
  const torch::Tensor partial_lse =
      torch::tensor({nan, infinity, 1000.0f}).reshape({3, 1, 1, 1});

  const layer::DcpAttentionResult result =
      layer::detail::merge_dcp_tnd_attention_shards(partial_outputs,
                                                    partial_lse);

  EXPECT_TRUE(torch::equal(result.output,
                           torch::tensor({3.0f, 9.0f}).reshape({1, 1, 2})));
  EXPECT_TRUE(
      torch::equal(result.lse, torch::tensor({1000.0f}).reshape({1, 1})));
}

}  // namespace
}  // namespace xllm
