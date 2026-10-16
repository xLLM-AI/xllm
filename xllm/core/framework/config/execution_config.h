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

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

// Resolved decode geometry shared by scheduler profiling and graph warmup.
// For MTP, num_decoding_tokens is the number of token rows produced per
// sequence in a decode step; it is normally num_speculative_tokens + 1.
struct DecodeGraphWarmupConfig {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
  int32_t max_graph_batch_size = 0;
};

class ExecutionConfig final {
 public:
  ExecutionConfig() = default;
  ~ExecutionConfig() = default;

  static ExecutionConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  DecodeGraphWarmupConfig decode_graph_warmup_config(
      int32_t num_speculative_tokens,
      bool enable_no_padding) const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "EXECUTION OPTIONS",
        {"enable_task_pipeline",
         "enable_graph",
         "enable_unified_mtp_graph",
         "disable_graph_warmup",
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
         "python_graph_backend",
         "enable_fia_decode",
         "enable_dsa_multi_stream",
         "enable_attn_dp_weight_sharding"}};
    return kOptionCategory;
  }

  // Enable the task pipeline; scheduler overlap determines the slot count.
  PROPERTY(bool, enable_task_pipeline) = false;

  PROPERTY(bool, enable_graph) = false;

  // Opt supported NPU MTP requests into the unified Python graph path.
  // Disabling this keeps the legacy ACL graph/decode route available for
  // comparison and fallback without disabling ordinary graph execution.
  PROPERTY(bool, enable_unified_mtp_graph) = false;

  PROPERTY(bool, disable_graph_warmup) = false;

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

  PROPERTY(bool, enable_fia_decode) = false;

  PROPERTY(bool, enable_dsa_multi_stream) = false;

  PROPERTY(bool, enable_attn_dp_weight_sharding) = false;
};

}  // namespace xllm
