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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

#include "core/framework/config/config_utils.h"
#include "core/framework/config/kv_cache_config.h"

namespace xllm {
namespace {

TEST(NativeFP8ConfigTest, XqaPageSizesAreCheckedBeforeAllocation) {
  KVCacheConfig config;
  config.kv_cache_dtype("fp8_e4m3");
  for (const std::string& backend : {"auto", "xqa"}) {
    SCOPED_TRACE(backend);
    config.kv_cache_decode_backend(backend);
    for (const int32_t size : {16, 32, 64, 128}) {
      config.block_size(size);
      EXPECT_NO_FATAL_FAILURE(config.validate());
    }
    for (const int32_t size : {0, -1, 8, 256}) {
      config.block_size(size);
      EXPECT_DEATH(config.validate(), "FP8 XQA requires block_size");
    }
  }
}

TEST(NativeFP8ConfigTest, PageSizeRestrictionDoesNotAffectOtherBackends) {
  KVCacheConfig config;
  config.kv_cache_dtype("fp8_e4m3")
      .kv_cache_decode_backend("flashinfer")
      .block_size(256);
  EXPECT_NO_FATAL_FAILURE(config.validate());
  for (const std::string& dtype : {"auto", "int8"}) {
    config.kv_cache_dtype(dtype).kv_cache_decode_backend("auto");
    EXPECT_NO_FATAL_FAILURE(config.validate());
  }
}

TEST(NativeFP8ConfigTest, ScalesAndBackendsRoundTripThroughJson) {
  google::FlagSaver flag_saver;
  KVCacheConfig original;
  original.kv_cache_dtype("fp8_e4m3")
      .kv_cache_k_scale(0.125)
      .kv_cache_v_scale(0.5)
      .kv_cache_prefill_backend("fa3")
      .kv_cache_decode_backend("xqa");
  nlohmann::ordered_json encoded = nlohmann::ordered_json::object();
  original.append_config_json(encoded);
  KVCacheConfig restored;
  restored.from_json(config::parse_json_string(encoded.dump()));
  EXPECT_NO_FATAL_FAILURE(restored.validate());
  EXPECT_EQ(restored.kv_cache_dtype(), "fp8_e4m3");
  EXPECT_DOUBLE_EQ(restored.kv_cache_k_scale(), 0.125);
  EXPECT_DOUBLE_EQ(restored.kv_cache_v_scale(), 0.5);
  EXPECT_EQ(restored.kv_cache_prefill_backend(), "fa3");
  EXPECT_EQ(restored.kv_cache_decode_backend(), "xqa");
}

}  // namespace
}  // namespace xllm
