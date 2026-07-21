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

#pragma once

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

  // Whether a speculative algorithm requires the target model to capture
  // intermediate-layer aux hidden states to drive the draft (Eagle3 and
  // DFlash). Centralizes the algorithm-string classification in one place. The
  // worker consults it to decide whether to populate the target's
  // layers_to_capture; the model then keys off that list alone, never the
  // algorithm string. Takes the algorithm explicitly so a spawned worker
  // process, which reads its own Options rather than this global config, can
  // classify without an initialized singleton.
  static bool requires_aux_hidden_capture(std::string_view algorithm) {
    return algorithm == "Eagle3" || algorithm == "DFlash";
  }

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "SPECULATIVE OPTIONS",
        {"draft_model",
         "num_speculative_tokens",
         "speculative_algorithm",
         "speculative_suffix_cache_max_depth",
         "speculative_suffix_max_spec_factor",
         "speculative_suffix_max_spec_offset",
         "speculative_suffix_min_token_prob",
         "speculative_suffix_max_cached_requests",
         "speculative_suffix_use_tree_spec",
         "enable_opt_validate_probs",
         "enable_mtp_draft_body_tp1",
         "enable_atb_spec_kernel"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, draft_model);

  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "MTP";

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  PROPERTY(bool, enable_opt_validate_probs) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;

  PROPERTY(bool, enable_atb_spec_kernel) = false;
};

}  // namespace xllm
