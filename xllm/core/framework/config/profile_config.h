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
#include <limits>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ProfileConfig final {
 public:
  ProfileConfig() = default;
  ~ProfileConfig() = default;

  static ProfileConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "PROFILE OPTIONS",
        {"enable_profile_step_time",
         "enable_profile_token_budget",
         "enable_latency_aware_schedule",
         "profile_max_prompt_length",
         "max_global_ttft_ms",
         "max_global_tpot_ms",
         "enable_profile_kv_blocks",
         "disable_ttft_profiling",
         "enable_forward_interruption",
         "enable_online_profile",
         "profile_backend",
         "profile_dir"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_profile_step_time) = false;

  PROPERTY(bool, enable_profile_token_budget) = false;

  PROPERTY(bool, enable_latency_aware_schedule) = false;

  PROPERTY(int32_t, profile_max_prompt_length) = 2048;

  PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();

  PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

  PROPERTY(bool, enable_profile_kv_blocks) = true;

  PROPERTY(bool, disable_ttft_profiling) = false;

  PROPERTY(bool, enable_forward_interruption) = false;

  // Whether to enable the online timeline profiling endpoints
  // (/start_profile and /stop_profile). CUDA only for now.
  PROPERTY(bool, enable_online_profile) = false;

  // Online profiling backend. "torch" (default) records CPU+CUDA activities
  // in-process via libtorch's Kineto profiler and writes a Chrome trace on
  // /stop_profile, no external profiler required (mirrors vLLM's default).
  // "cuda" only toggles the CUDA profiler capture range
  // (cudaProfilerStart/Stop) and requires launching the server under nsys with
  // --capture-range=cudaProfilerApi to record a trace.
  PROPERTY(std::string, profile_backend) = "torch";

  // Directory the "torch" backend writes timeline traces to. Empty means the
  // current working directory. Mirrors vLLM's torch_profiler_dir.
  PROPERTY(std::string, profile_dir) = "";
};

}  // namespace xllm
