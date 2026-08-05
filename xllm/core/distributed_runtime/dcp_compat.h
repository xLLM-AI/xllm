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

#include <optional>
#include <string>

#include "common/options.h"
#include "common/types.h"

namespace xllm {

inline std::optional<std::string> validate_dcp_first_version_options(
    const Options& options,
    EngineType engine_type) {
  if (options.decode_context_parallel_size() <= 1) {
    return std::nullopt;
  }
  if (options.enable_chunked_prefill()) {
    return "decode_context_parallel_size first version does not yet support "
           "chunked prefill; set --enable_chunked_prefill=false or set "
           "--decode_context_parallel_size=1";
  }
  if (options.enable_prefix_cache()) {
    return "decode_context_parallel_size first version does not yet support "
           "prefix cache; set --enable_prefix_cache=false or set "
           "--decode_context_parallel_size=1";
  }
  if (options.enable_schedule_overlap()) {
    return "decode_context_parallel_size first version does not yet support "
           "schedule overlap; set --enable_schedule_overlap=false or set "
           "--decode_context_parallel_size=1";
  }
  if (options.enable_disagg_pd() ||
      options.instance_role() != InstanceRole::DEFAULT) {
    return "decode_context_parallel_size first version does not yet support "
           "disaggregated prefill-decode; set --enable_disagg_pd=false, "
           "--instance_role=DEFAULT, or set --decode_context_parallel_size=1";
  }
  if (engine_type == EngineType::SSM ||
      !options.draft_model_path().value_or("").empty() ||
      options.num_speculative_tokens() > 0) {
    return "decode_context_parallel_size first version does not yet support "
           "speculative decoding; unset --draft_model, set "
           "--num_speculative_tokens=0, or set "
           "--decode_context_parallel_size=1";
  }
  return std::nullopt;
}

inline std::optional<std::string> validate_dcp_first_version_model_type(
    const std::string& model_type) {
  if (model_type == "qwen3_5_moe_text") {
    return "decode_context_parallel_size first version does not yet support "
           "Qwen3.5 MoE; use dense Qwen3.5 or set "
           "--decode_context_parallel_size=1";
  }
  return std::nullopt;
}

}  // namespace xllm
