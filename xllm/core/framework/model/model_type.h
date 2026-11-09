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

#include <string_view>
#include <unordered_set>

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

inline bool has_mtp_model_type_marker(std::string_view model_type) {
  return model_type.find("mtp") != std::string_view::npos;
}

inline bool starts_with_model_type(std::string_view model_type,
                                   std::string_view target_model_type) {
  return model_type.size() >= target_model_type.size() &&
         model_type.compare(0, target_model_type.size(), target_model_type) ==
             0;
}

inline bool is_target_mtp_model_type(std::string_view model_type,
                                     std::string_view target_model_type) {
  return starts_with_model_type(model_type, target_model_type) &&
         has_mtp_model_type_marker(model_type);
}

inline constexpr std::string_view kDeepseekV4DSparkModelType =
    "deepseek_v4_dspark";

inline bool is_deepseek_v4_dspark_model_type(std::string_view model_type) {
  return model_type == kDeepseekV4DSparkModelType;
}

inline constexpr std::string_view kDeepseekV4ModelType = "deepseek_v4";
inline constexpr std::string_view kDeepseekV41ModelType = "deepseek_v41";

inline bool is_deepseek_v41_model_type(std::string_view model_type) {
  return model_type == kDeepseekV41ModelType;
}

inline bool is_deepseek_v4_model_type(std::string_view model_type) {
  return model_type == kDeepseekV4ModelType ||
         is_deepseek_v41_model_type(model_type) ||
         is_deepseek_v4_dspark_model_type(model_type) ||
         is_target_mtp_model_type(model_type, kDeepseekV4ModelType);
}

// SFA C8: packed [int8 nope | bf16 rope | fp32 scale] KV cache. Currently only
// GLM-5.2 (glm_moe_dsa) drives the packed 656B write/read pipeline.
inline bool supports_mla_kv_cache_quant(std::string_view model_type) {
  return model_type == "glm_moe_dsa" || model_type == "glm_moe_dsa_mtp";
}

// Combined predicate for the SFA C8 packed KV-cache layout. The C8 layout is
// only defined for the (small) set of models covered by
// supports_mla_kv_cache_quant(), and every such model_type is by construction
// in mla_model_type_set(), so a redundant enable_mla parameter is not needed
// here. Callers that also need to honor the runtime enable_mla config bit
// AND it in at the call site (estimation is the only such caller today).
//
// The packed 656B layout is implemented on NPU only. Gating the build here
// rather than at each consumer is what lets every caller read this flat: on a
// non-NPU build the predicate is unconditionally false, so a caller that
// eagerly forwards it into KVCacheCapacity / KVCacheCreateOptions can never
// push a packed shape at a backend that has no write or read kernel for it.
// If the layout ever lands on another backend, flip this single gate and every
// consumer follows.
inline bool enable_mla_packed_c8(bool enable_kv_cache_quant,
                                 std::string_view model_type) {
#if defined(USE_NPU)
  return enable_kv_cache_quant && supports_mla_kv_cache_quant(model_type);
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
