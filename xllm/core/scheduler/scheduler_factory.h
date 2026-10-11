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

#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <utility>

#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

namespace xllm {

class RecEngine;

enum class SchedulerKind : int8_t {
  CONTINUOUS = 0,
  ZERO_EVICTION = 4,
  DISAGG_PD = 5
};

SchedulerKind select_scheduler_kind(const SchedulerOptions& options);

template <typename TargetEngine>
  requires requires(TargetEngine* engine, BatchGroup& batch) {
    static_cast<Engine*>(engine);
    { engine->step(batch) } -> std::same_as<ForwardOutput>;
    { engine->update_last_step_result(batch) } -> std::same_as<void>;
  }
std::unique_ptr<Scheduler> create_continuous_scheduler(
    TargetEngine* engine,
    SchedulerOptions options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager =
        nullptr,
    std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator =
        nullptr,
    std::shared_ptr<KVCacheTransferCoordinator> kv_transfer_coordinator =
        nullptr) {
  switch (select_scheduler_kind(options)) {
    case SchedulerKind::DISAGG_PD:
      return std::make_unique<DisaggPDScheduler>(
          engine,
          options,
          std::move(distributed_worker_manager),
          std::move(model_residency_coordinator),
          std::move(kv_transfer_coordinator));
    case SchedulerKind::ZERO_EVICTION:
      return std::make_unique<ZeroEvictionScheduler>(
          engine,
          options,
          std::move(model_residency_coordinator),
          std::move(distributed_worker_manager));
    case SchedulerKind::CONTINUOUS:
      return std::make_unique<ContinuousScheduler<TargetEngine>>(
          engine,
          options,
          std::move(model_residency_coordinator),
          std::move(distributed_worker_manager));
  }
  return std::make_unique<ContinuousScheduler<TargetEngine>>(
      engine,
      options,
      std::move(model_residency_coordinator),
      std::move(distributed_worker_manager));
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options);

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    RecEngine* engine,
    SchedulerOptions options);

}  // namespace xllm
