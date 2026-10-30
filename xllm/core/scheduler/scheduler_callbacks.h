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

#include <functional>

#include "core/framework/batch/batch_group.h"

namespace xllm {

// Flat forward-step callbacks; the scheduler and ProfileManager share them so
// neither component sees an engine type.
using StepCallback = std::function<ForwardOutput(BatchGroup&)>;
using UpdateLastStepResultCallback = std::function<void(BatchGroup&)>;

// The step/update callback pair bound to one engine.
struct SchedulerCallbacks {
  StepCallback step;
  UpdateLastStepResultCallback update_last_step_result;
};

// Binds an engine (by pointer) into the flat scheduler callbacks. The result
// callback collects the engine's step outputs and applies them in the
// commit-pending action, matching the engine's collect_step_outputs contract.
template <typename EngineT>
SchedulerCallbacks make_scheduler_callbacks(EngineT* engine) {
  return {[engine](BatchGroup& batch) { return engine->step(batch); },
          [engine](BatchGroup& batch) {
            batch.apply_outputs(engine->collect_step_outputs(batch),
                                BatchOutputAction::COMMIT_PENDING);
          }};
}

}  // namespace xllm
