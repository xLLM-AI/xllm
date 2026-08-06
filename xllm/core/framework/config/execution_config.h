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

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ExecutionConfig final {
 public:
  ExecutionConfig() = default;
  ~ExecutionConfig() = default;

  static ExecutionConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "EXECUTION OPTIONS",
        {"enable_graph",
         "enable_graph_double_buffer",
         "enable_graph_mode_decode_no_padding",
         "enable_prefill_piecewise_graph",
         "enable_graph_vmm_pool",
         "max_tokens_for_graph_mode",
         "acl_graph_decode_batch_size_limit",
         "enable_shm",
         "use_contiguous_input_buffer",
         "input_shm_size",
         "output_shm_size",
         "random_seed",
         "python_graph_backend"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_graph) = false;

  PROPERTY(bool, enable_graph_double_buffer) = true;

  PROPERTY(bool, enable_graph_mode_decode_no_padding) = false;

  PROPERTY(bool, enable_prefill_piecewise_graph) = false;

  PROPERTY(bool, enable_graph_vmm_pool) = true;

  PROPERTY(int32_t, max_tokens_for_graph_mode) = 2048;

  PROPERTY(int32_t, acl_graph_decode_batch_size_limit) = 16;

  PROPERTY(bool, enable_shm) = false;

  PROPERTY(bool, use_contiguous_input_buffer) = true;

  PROPERTY(uint64_t, input_shm_size) = 1024;

  PROPERTY(uint64_t, output_shm_size) = 128;

  PROPERTY(int32_t, random_seed) = -1;

  PROPERTY(std::string, python_graph_backend) = "off";
};

}  // namespace xllm
