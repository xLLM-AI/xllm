/* Copyright 2025-2026 The xLLM Authors.

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

#include <boost/algorithm/string/predicate.hpp>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <string_view>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class SpeculativeConfig final {
 public:
  SpeculativeConfig() = default;
  ~SpeculativeConfig() = default;

  static SpeculativeConfig& get_instance();

  static bool is_eagle3_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "Eagle3");
  }

  static bool is_dflash_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "DFlash");
  }

  static bool is_dflash2_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "DFlash2");
  }

  static bool is_dspark_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "DSpark");
  }

  static bool is_suffix_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "Suffix");
  }

  static bool uses_ssm_engine(std::string_view draft_model_path,
                              std::string_view algorithm,
                              int32_t num_speculative_tokens) {
    return !draft_model_path.empty() ||
           (is_suffix_algorithm(algorithm) && num_speculative_tokens > 0);
  }

  static bool is_mtp_algorithm(std::string_view algorithm) {
    return boost::iequals(algorithm, "MTP");
  }

  // Whether the PD handoff must carry the bootstrap hidden state.
  static bool requires_bootstrap_hidden(std::string_view algorithm) {
    return is_mtp_algorithm(algorithm) || is_eagle3_algorithm(algorithm);
  }

  // Drafts that generate a block of speculative tokens in one forward.
  static bool is_block_diffusion_algorithm(std::string_view algorithm) {
    return is_dflash_algorithm(algorithm) || is_dflash2_algorithm(algorithm) ||
           is_dspark_algorithm(algorithm);
  }

  // Algorithm-level admission only; callers validate runtime geometry.
  static bool supports_task_pipeline(std::string_view algorithm) {
    return is_mtp_algorithm(algorithm) ||
           is_block_diffusion_algorithm(algorithm);
  }

  static bool supports_adaptive_target_verify(std::string_view algorithm) {
    return is_dspark_algorithm(algorithm) || is_mtp_algorithm(algorithm);
  }

  static bool requires_aux_hidden_capture(std::string_view algorithm) {
    return is_eagle3_algorithm(algorithm) ||
           is_block_diffusion_algorithm(algorithm);
  }

  // Aux-capture drafts share the target's KV layout, so they are included.
  static bool supports_host_kv_cache(std::string_view algorithm) {
    return is_mtp_algorithm(algorithm) || is_eagle3_algorithm(algorithm) ||
           is_block_diffusion_algorithm(algorithm);
  }

  // DFlash2 carries the dense proposal only when probabilistic rejection
  // requires it; greedy-only DFlash and Suffix stay out.
  static bool is_probabilistic_draft_sampling_supported(
      std::string_view algorithm) {
    return is_mtp_algorithm(algorithm) || is_dspark_algorithm(algorithm) ||
           is_dflash2_algorithm(algorithm) || is_eagle3_algorithm(algorithm);
  }

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void validate() const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "SPECULATIVE OPTIONS",
        {"draft_model",
         "num_speculative_tokens",
         "speculative_algorithm",
         "enable_draft_swa_manage",
         "speculative_suffix_cache_max_depth",
         "speculative_suffix_max_spec_factor",
         "speculative_suffix_max_spec_offset",
         "speculative_suffix_min_token_prob",
         "speculative_suffix_max_cached_requests",
         "speculative_suffix_use_tree_spec",
         "enable_probabilistic_draft",
         "enable_mtp_draft_body_tp1",
         "enable_atb_spec_kernel",
         "enable_adaptive_target_verify",
         "enable_adaptive_draft_step",
         "adaptive_draft_step_config",
         "adaptive_target_verify_min_gain",
         "adaptive_target_verify_cost_table",
         "adaptive_target_verify_calibration",
         "adaptive_target_verify_cost_table_output"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, draft_model);

  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "MTP";

  PROPERTY(bool, enable_draft_swa_manage) = false;

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  PROPERTY(bool, enable_probabilistic_draft) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;

  PROPERTY(bool, enable_atb_spec_kernel) = false;

  PROPERTY(bool, enable_adaptive_target_verify) = false;
  PROPERTY(bool, enable_adaptive_draft_step) = false;
  PROPERTY(std::string, adaptive_draft_step_config);

  PROPERTY(double, adaptive_target_verify_min_gain) = 0.0;

  PROPERTY(std::string, adaptive_target_verify_cost_table);
  PROPERTY(std::string, adaptive_target_verify_calibration);
  PROPERTY(std::string, adaptive_target_verify_cost_table_output);
};

}  // namespace xllm
