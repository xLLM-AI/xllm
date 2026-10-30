/* Copyright 2026 The xLLM Authors.

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
#include <string_view>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/util/utils.h"

namespace xllm {

// NPU MTP targets whose num_speculative_tokens flows from engine Options into
// ModelArgs; other targets keep the checkpoint default.
inline bool uses_mtp_target_spec_width(std::string_view model_type) {
  return is_glm5_next_target_model_type(model_type) ||
         util::is_deepseek_v4_model_type(model_type);
}

inline int64_t mtp_hidden_state_width(const ModelArgs& model_args) {
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    return model_args.hc_mult() * model_args.hidden_size();
  }
  return model_args.hidden_size();
}

inline bool uses_embedded_eagle3_draft(std::string_view algorithm,
                                       const ModelArgs& target_model_args) {
  return SpeculativeConfig::is_eagle3_algorithm(algorithm) &&
         target_model_args.enable_embedded_eagle3_draft();
}

}  // namespace xllm
