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

using StepCallback = std::function<ForwardOutput(BatchGroup&)>;

struct SchedulerCallbacks {
  StepCallback step;
  std::function<void(BatchGroup&)> update_last_step_result;
};

// The callbacks capture `engine` as a raw pointer: the caller (each master)
// must destroy the scheduler that owns these callbacks before the engine it
// captures, e.g. by declaring the scheduler after the engine.
template <typename EngineT>
SchedulerCallbacks make_scheduler_callbacks(EngineT* engine) {
  return {[engine](BatchGroup& batch) { return engine->step(batch); },
          [engine](BatchGroup& batch) {
            batch.apply_outputs(engine->collect_step_outputs(batch),
                                BatchOutputAction::COMMIT_PENDING);
          }};
}

}  // namespace xllm
