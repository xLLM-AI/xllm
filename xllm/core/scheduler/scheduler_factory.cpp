/* Copyright 2025-2026 The xLLM Authors.

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

#include "scheduler/scheduler_factory.h"

#include "core/common/global_flags.h"
#include "core/disaggregation/disagg_pd_scheduler.h"
#include "core/distributed_runtime/rec_engine.h"
#include "core/framework/config/parallel_config.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"

namespace xllm {

SchedulerKind select_scheduler_kind(const SchedulerOptions& options) {
  if (options.enable_disagg_pd()) {
    return SchedulerKind::DISAGG_PD;
  }

  return SchedulerKind::CONTINUOUS;
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    RecEngine* engine,
    SchedulerOptions options) {
  return std::make_unique<FixedStepsScheduler>(engine, options);
}

}  // namespace xllm
