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
#include <string>

namespace xllm {

class EPLBConfig;

// Value-type snapshot of the EPLB tunables consumed by the manager / policy
// layer. Constructed once by EplbManager at start-up from the global
// EPLBConfig singleton and passed through the factory and every policy
// constructor, so nothing under `framework/eplb/` reaches back into the
// singleton at runtime. Snapshotting also makes the settings stable over the
// lifetime of one manager: rebalance rounds see a consistent view rather
// than picking up mid-round config mutations.
class EplbOptions final {
 public:
  int32_t redundant_experts_num = 1;
  int64_t eplb_update_interval = 1000;
  double eplb_min_peak_load_improvement = 0.05;
  int32_t eplb_prepare_timeout_seconds = 30;
  std::string eplb_policy_kind = "balanced";

  void validate() const;

  // Build an EplbOptions from the process-wide EPLBConfig singleton. Defined
  // in eplb_options.cpp so the header does not have to depend on the concrete
  // EPLBConfig definition.
  static EplbOptions from_global_config();
};

}  // namespace xllm
