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
#include <string>

#include "common/types.h"

namespace xllm {

struct PdTopo {
  int32_t dp_size = 0;
  int32_t tp_size = 0;
};

enum class PdTopoStatus : int8_t {
  ALLOW_HOMO = 0,
  ALLOW_HETERO = 1,
  DENY_HETERO = 2,
  INVALID_LOCAL = 3,
  INVALID_REMOTE = 4,
};

struct PdTopoResult {
  PdTopoStatus status = PdTopoStatus::DENY_HETERO;
  std::string reason = "";
};

bool try_get_pd_topo(const InstanceInfo& info,
                     PdTopo* topo,
                     std::string* reason);

PdTopo get_pd_topo(const InstanceInfo& info);

PdTopoResult check_pd_topo(const InstanceInfo& local,
                           const InstanceInfo& remote,
                           const std::string& kv_mode,
                           bool kv_cache_is_tp_invariant,
                           bool enable_heterogeneous_pd = false);

}  // namespace xllm
