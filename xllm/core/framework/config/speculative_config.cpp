/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/speculative_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_string(speculative_algorithm,
              "MTP",
              "Speculative decoding algorithm. Supported options: MTP, Eagle3, "
              "Suffix, DFlash. Default is MTP.");

DEFINE_int32(speculative_suffix_cache_max_depth,
             64,
             "Maximum suffix tree depth for suffix speculative decoding.");

DEFINE_double(speculative_suffix_max_spec_factor,
              1.0,
              "Suffix speculation max tokens factor relative to match length.");

DEFINE_double(speculative_suffix_max_spec_offset,
              0.0,
              "Suffix speculation max tokens additive offset.");

DEFINE_double(speculative_suffix_min_token_prob,
              0.1,
              "Minimum token probability used in suffix speculation.");

DEFINE_int32(speculative_suffix_max_cached_requests,
             -1,
             "Maximum globally cached requests for suffix speculation (-1 "
             "unlimited, 0 disabled).");

DEFINE_bool(speculative_suffix_use_tree_spec,
            false,
            "Whether to use tree-based suffix speculation instead of path "
            "speculation.");

DEFINE_bool(enable_opt_validate_probs,
            false,
            "Whether validate uses selected-only draft_probs [B,S] directly. "
            "If false, selected-only cache values are restored to dense "
            "[B,S,V].");

DEFINE_bool(enable_mtp_draft_body_tp1,
            false,
            "Whether to run the MTP draft body with tensor-parallel size 1 "
            "while keeping the draft LMHead on the target TP group.");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

namespace xllm {

void SpeculativeConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(draft_model);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_opt_validate_probs);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mtp_draft_body_tp1);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_atb_spec_kernel);
}

void SpeculativeConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(draft_model);
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_opt_validate_probs);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mtp_draft_body_tp1);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_atb_spec_kernel);
}

void SpeculativeConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const SpeculativeConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, draft_model);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_speculative_tokens);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_algorithm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_cache_max_depth);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_offset);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_min_token_prob);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_cached_requests);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_use_tree_spec);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_opt_validate_probs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mtp_draft_body_tp1);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_atb_spec_kernel);
}

SpeculativeConfig& SpeculativeConfig::get_instance() {
  static SpeculativeConfig config;
  return config;
}

void SpeculativeConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
