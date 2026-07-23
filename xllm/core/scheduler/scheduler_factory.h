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
#include <memory>

#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"

namespace xllm {

enum class SchedulerKind : int8_t {
  CONTINUOUS = 0,
  ZERO_EVICTION = 4,
  DISAGG_PD = 5,
  DISAGG_PD_CHUNKED_PREFILL = 6,
  PD_OOC = 7
};

SchedulerKind select_scheduler_kind(
    const ContinuousScheduler::Options& options);

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    Engine* engine,
    DiTScheduler::Options options);

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

}  // namespace xllm
