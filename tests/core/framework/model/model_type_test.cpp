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

#include "core/framework/model/model_type.h"

#include <gtest/gtest.h>

#include <string_view>

namespace xllm::util {
namespace {

TEST(ModelTypeTest, LooksUpBorrowedSubstringWithoutTerminator) {
  constexpr std::string_view kStorage = "glm_moe_dsa_mtp_unrelated_suffix";
  const std::string_view model_type = kStorage.substr(/*pos=*/0, /*count=*/15);
  EXPECT_TRUE(is_mla_model_type(model_type));
  EXPECT_FALSE(is_mla_model_type(kStorage));
  EXPECT_FALSE(is_mla_model_type(std::string_view()));
}

TEST(ModelTypeTest, GatesPackedC8ByModelAndBackend) {
  EXPECT_TRUE(supports_mla_kv_cache_quant("glm_moe_dsa"));
  EXPECT_TRUE(supports_mla_kv_cache_quant("glm_moe_dsa_mtp"));
  EXPECT_FALSE(supports_mla_kv_cache_quant("deepseek_v3"));
  EXPECT_FALSE(
      enable_mla_packed_c8(/*enable_kv_cache_quant=*/false, "glm_moe_dsa"));
  EXPECT_FALSE(
      enable_mla_packed_c8(/*enable_kv_cache_quant=*/true, "deepseek_v3"));
#if defined(USE_NPU)
  EXPECT_TRUE(
      enable_mla_packed_c8(/*enable_kv_cache_quant=*/true, "glm_moe_dsa"));
#else
  EXPECT_FALSE(
      enable_mla_packed_c8(/*enable_kv_cache_quant=*/true, "glm_moe_dsa"));
#endif
}

}  // namespace
}  // namespace xllm::util
