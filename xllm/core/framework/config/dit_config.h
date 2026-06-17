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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class DiTConfig final {
 public:
  DiTConfig() = default;
  ~DiTConfig() = default;

  static DiTConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "DiT MODEL OPTIONS",
        {"max_requests_per_batch",
         "dit_cache_policy",
         "dit_cache_warmup_steps",
         "dit_cache_n_derivatives",
         "dit_cache_skip_interval_steps",
         "dit_cache_residual_diff_threshold",
         "dit_cache_start_steps",
         "dit_cache_end_steps",
         "dit_cache_start_blocks",
         "dit_cache_end_blocks",
         "dit_sp_communication_overlap",
         "dit_debug_print"
         "dit_generation_image_area_max"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, max_requests_per_batch) = 1;

  PROPERTY(std::string, dit_cache_policy) = "TaylorSeer";

  PROPERTY(int64_t, dit_cache_warmup_steps) = 0;

  PROPERTY(int64_t, dit_cache_n_derivatives) = 3;

  PROPERTY(int64_t, dit_cache_skip_interval_steps) = 3;

  PROPERTY(double, dit_cache_residual_diff_threshold) = 0.09;

  PROPERTY(int64_t, dit_cache_start_steps) = 5;

  PROPERTY(int64_t, dit_cache_end_steps) = 5;

  PROPERTY(int64_t, dit_cache_start_blocks) = 5;

  PROPERTY(int64_t, dit_cache_end_blocks) = 5;

  PROPERTY(int64_t, dit_sp_communication_overlap) = 1;

  PROPERTY(bool, dit_debug_print) = false;

  PROPERTY(int64_t, dit_generation_image_area_max) = 0;
};

}  // namespace xllm
