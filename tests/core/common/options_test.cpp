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

#include "core/common/options.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(OptionsTest, SpeculativeTokenCountRemainsNonnegative) {
  Options options;
  EXPECT_EQ(options.num_speculative_tokens(), 0);
  options.num_speculative_tokens(/*value=*/4);
  EXPECT_EQ(options.num_speculative_tokens(), 4);
  options.num_speculative_tokens(/*value=*/-1);
  EXPECT_EQ(options.num_speculative_tokens(), 0);
}

TEST(OptionsTest, ContextParallelDefaultsToOne) {
  Options options;

  EXPECT_EQ(options.cp_size(), 1);
}

TEST(OptionsTest, ContextParallelAcceptsExplicitValues) {
  Options options;
  options.cp_size(4);

  EXPECT_EQ(options.cp_size(), 4);
}

}  // namespace
}  // namespace xllm
