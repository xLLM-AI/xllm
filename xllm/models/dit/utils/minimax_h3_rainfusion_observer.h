/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "models/dit/utils/minimax_h3_qualified_execution_context.h"

namespace xllm {

inline H3QualifiedLaunchSnapshot h3_qualified_launch_snapshot(
    const H3QualifiedExecutionContext& execution,
    H3QualifiedLaunchSnapshot snapshot = {}) {
  snapshot.rf_blocks = execution.rainfusion.borrowed_kernel_context.launches;
#if defined(USE_NPU)
  snapshot.selector_launches =
      execution.rainfusion.kernel_launches.selector_launches;
  snapshot.attention_launches =
      execution.rainfusion.kernel_launches.attention_launches;
#endif
  return snapshot;
}

}  // namespace xllm
