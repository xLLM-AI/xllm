/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "util/env_var.h"

namespace xllm {

inline bool use_replicated_qwen35_mtp_draft(bool enabled_by_config) {
#if defined(USE_NPU)
  return enabled_by_config ||
         util::get_bool_env("XLLM_NPU_REPLICATE_QWEN35_MTP_DRAFT", false);
#else
  (void)enabled_by_config;
  return false;
#endif
}

}  // namespace xllm
