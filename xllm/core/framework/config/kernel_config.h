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

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class KernelConfig final {
 public:
  KernelConfig() = default;
  ~KernelConfig() = default;

  static KernelConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "KERNEL OPTIONS",
        {"enable_customize_mla_kernel",
         "npu_kernel_backend",
         "enable_intralayer_addnorm",
         "enable_fused_mc2",
         "enable_interlayer_addnorm",
         "enable_split_rmsnorm_rope",
         "enable_flash_comm"}};
    return kOptionCategory;
  }

#if defined(USE_NPU)
  PROPERTY(bool, enable_customize_mla_kernel) = false;

  PROPERTY(std::string, npu_kernel_backend) = "AUTO";

  PROPERTY(bool, enable_intralayer_addnorm) = false;

  PROPERTY(int32_t, enable_fused_mc2) = 0;

  PROPERTY(bool, enable_interlayer_addnorm) = false;

  PROPERTY(bool, enable_split_rmsnorm_rope) = false;

  PROPERTY(bool, enable_flash_comm) = false;
#endif
};

}  // namespace xllm
