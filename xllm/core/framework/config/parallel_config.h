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

class ParallelConfig final {
 public:
  ParallelConfig() = default;
  ~ParallelConfig() = default;

  static ParallelConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "PARALLEL OPTIONS",
        {"dp_size",
         "ep_size",
         "cp_size",
         "tp_size",
         "sp_size",
         "cfg_size",
         "vae_size",
         "communication_backend",
         "enable_mm_encoder_dp",
         "enable_multi_stream_parallel",
         "micro_batch_num",
         "enable_dp_balance"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, dp_size) = 1;

  PROPERTY(int32_t, ep_size) = 1;

  PROPERTY(int32_t, cp_size) = 1;

  // 0 means follow cp_size (legacy KV-split width).
  PROPERTY(int32_t, kv_split_size) = 1;

  PROPERTY(int64_t, tp_size) = 1;

  PROPERTY(int64_t, sp_size) = 1;

  PROPERTY(int64_t, cfg_size) = 1;

  PROPERTY(int64_t, vae_size) = 1;

  PROPERTY(std::string, communication_backend) = "hccl";

  PROPERTY(bool, enable_mm_encoder_dp) = false;

  PROPERTY(bool, enable_multi_stream_parallel) = false;

  PROPERTY(int32_t, micro_batch_num) = 1;

  PROPERTY(bool, enable_dp_balance) = false;

  [[nodiscard]] int32_t kv_split_size_effective() const noexcept {
    return kv_split_size_ > 0 ? kv_split_size_ : cp_size_;
  }
};

}  // namespace xllm
