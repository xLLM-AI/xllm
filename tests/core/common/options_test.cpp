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

#include "core/common/options.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "c_api/default.h"
#include "cc_api/types.h"

namespace xllm {
namespace {

TEST(OptionsTest, ContextParallelDefaultsToOneAcrossPublicApis) {
  Options options;
  const XLLM_InitLLMOptions cc_options;

  EXPECT_EQ(options.cp_size(), 1);
  EXPECT_EQ(cc_options.cp_size, 1);
  EXPECT_EQ(XLLM_INIT_LLM_OPTIONS_DEFAULT.cp_size, 1U);
  EXPECT_EQ(XLLM_C_ABI_VERSION_MAJOR, 1);
  static_assert(std::is_same_v<decltype(XLLM_InitOptions::cp_size), uint32_t>);
}

TEST(OptionsTest, ContextParallelAcceptsExplicitValuesAcrossPublicApis) {
  Options options;
  options.cp_size(4);
  XLLM_InitLLMOptions cc_options;
  cc_options.cp_size = 4;
  XLLM_InitOptions c_options = XLLM_INIT_LLM_OPTIONS_DEFAULT;
  c_options.cp_size = 4;

  EXPECT_EQ(options.cp_size(), 4);
  EXPECT_EQ(cc_options.cp_size, 4);
  EXPECT_EQ(c_options.cp_size, 4U);
}

}  // namespace
}  // namespace xllm
