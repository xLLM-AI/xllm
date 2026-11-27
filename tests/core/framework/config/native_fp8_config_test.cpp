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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "core/framework/config/kv_cache_config.h"

namespace xllm {
namespace {

TEST(NativeFP8ConfigTest, XqaPageSizesAreCheckedBeforeAllocation) {
  KVCacheConfig config;
  config.kv_cache_dtype("fp8_e4m3");
  for (const int32_t size : {16, 32, 64, 128}) {
    config.block_size(size);
    EXPECT_NO_FATAL_FAILURE(config.validate());
  }
  for (const int32_t size : {0, -1, 8, 256}) {
    config.block_size(size);
    EXPECT_DEATH(config.validate(), "FP8 XQA requires block_size");
  }
}

TEST(NativeFP8ConfigTest, PageSizeRestrictionDoesNotAffectOtherFormats) {
  KVCacheConfig config;
  config.block_size(/*value=*/256);
  for (const std::string& dtype : {"auto", "int8"}) {
    config.kv_cache_dtype(dtype);
    EXPECT_NO_FATAL_FAILURE(config.validate());
  }
}

}  // namespace
}  // namespace xllm
