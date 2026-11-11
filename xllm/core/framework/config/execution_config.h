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
#include <string_view>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

// Describe one independently captured draft invocation. Block drafts round
// whole requests before expanding rows; autoregressive drafts round token rows.
struct DraftGraphWarmupShape {
  int32_t tokens_per_sequence = 1;
  bool bucket_by_sequence = false;
};

// Resolved decode geometry shared by scheduler profiling and graph warmup.
// For MTP, num_decoding_tokens is the number of token rows produced per
// sequence in a decode step; it is normally num_speculative_tokens + 1.
struct DecodeGraphWarmupConfig {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
  int32_t max_graph_batch_size = 0;
  std::vector<DraftGraphWarmupShape> draft_shapes;
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
      bool enable_no_padding,
      std::string_view speculative_algorithm = "MTP") const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "EXECUTION OPTIONS",
        {"extension_impl",
         "enable_graph",
         "disable_graph_warmup",
         "block_draft_graph_warmup_max_kv_len",
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

  // Select a native extension implementation family for registered components.
  PROPERTY(std::string, extension_impl) = "";

  PROPERTY(bool, enable_graph) = false;

  PROPERTY(bool, disable_graph_warmup) = false;

  // Zero covers the full model context; -1 keeps lazy KV-capacity capture.
  PROPERTY(int32_t, block_draft_graph_warmup_max_kv_len) = 0;

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
