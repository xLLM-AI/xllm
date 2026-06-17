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

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class BeamSearchConfig final {
 public:
  BeamSearchConfig() = default;
  ~BeamSearchConfig() = default;

  static BeamSearchConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {"BEAM SEARCH OPTIONS",
                                                   {"enable_beam_search_kernel",
                                                    "beam_width",
                                                    "enable_block_copy_kernel",
                                                    "enable_topk_sorted"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_beam_search_kernel) = false;

  PROPERTY(int32_t, beam_width) = 1;

#if defined(USE_NPU) || defined(USE_CUDA)
  PROPERTY(bool, enable_block_copy_kernel) = true;
#else
  PROPERTY(bool, enable_block_copy_kernel) = false;
#endif

  PROPERTY(bool, enable_topk_sorted) = true;
};

}  // namespace xllm
