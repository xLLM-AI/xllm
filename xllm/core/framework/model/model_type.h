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

#pragma once

#include <algorithm>
#include <array>
#include <string_view>
#include <unordered_set>

namespace xllm {

inline constexpr std::array<std::string_view, 3> kGlm5NextModel = {
    "glm5_3_flash",
    "glm5_next",
    "glm5_next_text",
};

inline bool is_glm5_next_model(std::string_view model_type) {
  return std::ranges::find(kGlm5NextModel, model_type) != kGlm5NextModel.end();
}

inline bool is_deepseek_v4_family(std::string_view model_type) {
  return model_type == "deepseek_v4" || model_type == "deepseek_v41" ||
         model_type == "deepseek_v4_dspark" || model_type == "deepseek_v4_mtp";
}

}  // namespace xllm

namespace xllm::util {

inline const std::unordered_set<std::string_view>& mla_model_type_set() {
  static const std::unordered_set<std::string_view> kMlaModelTypeSet = {
      "deepseek_v2",
      "deepseek_v3",
      "deepseek_v32",
      "deepseek_v3_mtp",
      "deepseek_v32_mtp",
      "kimi_k2",
      "kimi_k25",
      "glm4_moe_lite",
      "glm_moe_dsa",  // glm5 model type
      "glm_moe_dsa_mtp",
      "glm5_dspark",    // Dense MLA DSpark draft; no sparse indexer.
      "glm5_next",      // GLM-5.3-Flash (absorbed MLA, NoPE, kv_lora=512)
      "glm5_3_flash",   // GLM-5.3-Flash public alias
      "glm5_next_mtp",  // GLM-5.3-Flash MTP draft (same absorbed-MLA attention)
      "joyai_llm_flash"};
  return kMlaModelTypeSet;
}

inline bool is_mla_model_type(std::string_view model_type) {
  return mla_model_type_set().contains(model_type);
}

inline bool enable_mla_packed_c8(bool enable_kv_cache_quant,
                                 std::string_view model_type) {
#if defined(USE_NPU)
  return enable_kv_cache_quant &&
         (model_type == "glm_moe_dsa" || model_type == "glm_moe_dsa_mtp");
#else
  (void)enable_kv_cache_quant;
  (void)model_type;
  return false;
#endif
}

// Shared message for the three sites that reject non-MLU KV-cache quant. Kept
// as one string so the sites cannot drift apart.
inline constexpr const char* kNonMluKvCacheQuantRejectMsg =
    "KV cache quantization on non-MLU backends only supports GLM-5.2 "
    "MLA SFA C8 (model_type=glm_moe_dsa).";

}  // namespace xllm::util
