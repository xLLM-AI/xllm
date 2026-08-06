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

#include "core/framework/eplb/eplb_options.h"

#include <glog/logging.h>

#include "core/framework/config/eplb_config.h"

namespace xllm {

void EplbOptions::validate() const {
  CHECK_GE(redundant_experts_num, 0)
      << "eplb redundant_experts_num must be non-negative.";
  CHECK_GE(eplb_update_interval, 0)
      << "eplb_update_interval must be non-negative.";
  CHECK(eplb_min_peak_load_improvement >= 0.0 &&
        eplb_min_peak_load_improvement <= 1.0)
      << "eplb_min_peak_load_improvement must be in [0, 1].";
  CHECK_GT(eplb_prepare_timeout_seconds, 0)
      << "eplb_prepare_timeout_seconds must be positive.";
}

EplbOptions EplbOptions::from_global_config() {
  const EPLBConfig& cfg = EPLBConfig::get_instance();
  EplbOptions o;
  o.redundant_experts_num = cfg.redundant_experts_num();
  o.eplb_update_interval = cfg.eplb_update_interval();
  o.eplb_min_peak_load_improvement = cfg.eplb_min_peak_load_improvement();
  o.eplb_prepare_timeout_seconds = cfg.eplb_prepare_timeout_seconds();
  o.eplb_policy_kind = cfg.eplb_policy_kind();
  o.validate();
  return o;
}

}  // namespace xllm
