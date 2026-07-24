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

#include "core/common/flash_comm1_context.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(FlashComm1ContextTest, PadsRowsAndZerosTail) {
  torch::Tensor input =
      torch::arange(6, torch::TensorOptions().dtype(torch::kFloat));

  torch::Tensor padded = pad_rows_by_copy(input, 8);

  EXPECT_EQ(padded.size(0), 8);
  EXPECT_TRUE(torch::equal(padded.slice(0, 0, 6), input));
  EXPECT_TRUE(torch::equal(padded.slice(0, 6, 8), torch::zeros({2})));
}

TEST(FlashComm1ContextTest, SelectsFusedOrUnfusedReduceScatter) {
  FlashComm1Context context;
  context.enable_mmrs_fusion = false;
  EXPECT_EQ(row_parallel_reduce_mode_for_fc1(context),
            RowParallelReduceMode::REDUCE_SCATTER);

  context.enable_mmrs_fusion = true;
  EXPECT_EQ(row_parallel_reduce_mode_for_fc1(context),
            RowParallelReduceMode::MATMUL_REDUCE_SCATTER);
}

}  // namespace
}  // namespace xllm
