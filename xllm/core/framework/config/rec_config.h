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

class RecConfig final {
 public:
  RecConfig() = default;
  ~RecConfig() = default;

  static RecConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "REC OPTIONS",
        {"enable_rec_fast_sampler",
         "enable_rec_prefill_only",
         "enable_xattention_one_stage",
         "max_decode_rounds",
         "enable_constrained_decoding",
         "output_rec_logprobs",
         "enable_convert_tokens_to_item",
         "enable_output_sku_logprobs",
         "enable_extended_item_info",
         "each_conversion_threshold",
         "total_conversion_threshold",
         "request_queue_size",
         "rec_worker_max_concurrency"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_rec_fast_sampler) = true;

  PROPERTY(bool, enable_rec_prefill_only) = false;

  PROPERTY(bool, enable_xattention_one_stage) = false;

  PROPERTY(int32_t, max_decode_rounds) = 0;

  PROPERTY(bool, enable_constrained_decoding) = false;

  PROPERTY(bool, output_rec_logprobs) = false;

  PROPERTY(bool, enable_convert_tokens_to_item) = false;

  PROPERTY(bool, enable_output_sku_logprobs) = false;

  PROPERTY(bool, enable_extended_item_info) = false;

  PROPERTY(int32_t, each_conversion_threshold) = 50;

  PROPERTY(int32_t, total_conversion_threshold) = 1000;

  PROPERTY(int32_t, request_queue_size) = 100000;

  PROPERTY(uint32_t, rec_worker_max_concurrency) = 1;
};

}  // namespace xllm
