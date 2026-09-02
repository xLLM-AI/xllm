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
#include <string_view>

#include "core/framework/model/model_args.h"
#include "core/util/utils.h"

namespace xllm {

inline int64_t mtp_hidden_state_width(const ModelArgs& model_args) {
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    return model_args.hc_mult() * model_args.hidden_size();
  }
  return model_args.hidden_size();
}

inline bool requires_glm_mtp_cache_ownership_fence(
    bool schedule_overlap_enabled,
    int32_t num_speculative_tokens,
    int32_t dp_size,
    std::string_view model_type,
    bool graph_enabled) {
  return schedule_overlap_enabled && num_speculative_tokens > 0 &&
         dp_size > 1 && model_type == "glm_moe_dsa" && !graph_enabled;
}

}  // namespace xllm
