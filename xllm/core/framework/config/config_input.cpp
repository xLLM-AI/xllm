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

#include "core/framework/config/config_input.h"

#include <gflags/gflags.h>

#include <utility>
#include <vector>

namespace xllm::config {

FlagInputSnapshot FlagInputSnapshot::capture() {
  std::vector<google::CommandLineFlagInfo> flags;
  google::GetAllFlags(&flags);

  FlagInputSnapshot snapshot;
  snapshot.specified_flags_.reserve(flags.size());
  for (google::CommandLineFlagInfo& flag : flags) {
    if (flag.is_default) {
      continue;
    }
    snapshot.specified_flags_.emplace(std::move(flag.name));
  }
  return snapshot;
}

bool FlagInputSnapshot::is_specified(const std::string& flag_name) const {
  return specified_flags_.contains(flag_name);
}

}  // namespace xllm::config
