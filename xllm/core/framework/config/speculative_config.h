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
    return algorithm == "Eagle3" || algorithm == "DFlash" ||
           algorithm == "DSpark";
  }

  static bool is_mtp_algorithm(std::string_view algorithm) {
    return algorithm.size() == 3 &&
           (algorithm[0] == 'M' || algorithm[0] == 'm') &&
           (algorithm[1] == 'T' || algorithm[1] == 't') &&
           (algorithm[2] == 'P' || algorithm[2] == 'p');
  }

  // True for the block-diffusion draft algorithms (DFlash, DSpark) that record
  // validate metrics inline per-seq and drive the adaptive per-seq varlen
  // prune. Case-insensitive so it matches however the flag was cased. MTP is
  // classified separately via is_mtp_algorithm; callers that also accept MTP
  // must OR the two.
  static bool is_block_diffusion_algorithm(std::string_view algorithm) {
    return iequals(algorithm, "dflash") || iequals(algorithm, "dspark");
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
         "enable_atb_spec_kernel",
         "enable_adaptive_speculative_decode",
         "adaptive_speculative_min_gain"}};
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

  PROPERTY(bool, enable_adaptive_speculative_decode) = false;

  PROPERTY(double, adaptive_speculative_min_gain) = 0.0;

 private:
  // ASCII case-insensitive equality. Mirrors the manual case handling in
  // is_mtp_algorithm rather than pulling <algorithm>/<cctype> into this header.
  static bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      char ca = a[i];
      char cb = b[i];
      if (ca >= 'A' && ca <= 'Z') {
        ca = static_cast<char>(ca - 'A' + 'a');
      }
      if (cb >= 'A' && cb <= 'Z') {
        cb = static_cast<char>(cb - 'A' + 'a');
      }
      if (ca != cb) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace xllm
