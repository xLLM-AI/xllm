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
#include <string>

namespace xllm {

// Stable startup policy supplied by the engine assembly boundary. Managers
// and policies retain this value instead of reading process configuration.
class EplbOptions final {
 public:
  int32_t redundant_experts_num = 1;
  int64_t eplb_update_interval = 1000;
  double eplb_min_peak_load_improvement = 0.05;
  int32_t eplb_prepare_timeout_seconds = 30;
  std::string eplb_policy_kind = "balanced";

  void validate() const;
};

struct EplbControllerConfig {
  bool enabled = false;
  EplbOptions policy;
};

}  // namespace xllm
