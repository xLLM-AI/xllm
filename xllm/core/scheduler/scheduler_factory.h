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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

// Creates the scheduler kind selected by the options, binding the flat
// scheduler callbacks. All scheduler constructors take the flat dependency
// set (pool, model args, tokenizer, callbacks, ...).
inline std::unique_ptr<Scheduler> create_continuous_scheduler(
    BlockManagerPool* block_manager_pool,
    const ModelArgs* model_args,
    const Tokenizer* tokenizer,
    SchedulerCallbacks callbacks,
    SchedulerOptions options,
    std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator =
        nullptr,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager =
        nullptr,
    std::shared_ptr<KVCacheTransferCoordinator> kv_transfer_coordinator =
        nullptr,
    DisaggPDServiceOptions response_options = {},
    DraftStepController* draft_step_controller = nullptr) {
  switch (select_scheduler_kind(options)) {
    case SchedulerKind::DISAGG_PD:
      return std::make_unique<DisaggPDScheduler>(
          block_manager_pool,
          model_args,
          tokenizer,
          std::move(callbacks),
          options,
          std::move(distributed_worker_manager),
          std::move(model_residency_coordinator),
          std::move(kv_transfer_coordinator),
          std::move(response_options),
          draft_step_controller);
    case SchedulerKind::ZERO_EVICTION:
      return std::make_unique<ZeroEvictionScheduler>(
          block_manager_pool,
          model_args,
          tokenizer,
          std::move(callbacks),
          options,
          std::move(model_residency_coordinator),
          std::move(distributed_worker_manager),
          draft_step_controller);
    case SchedulerKind::CONTINUOUS:
      break;
  }
  return std::make_unique<ContinuousScheduler>(
      block_manager_pool,
      model_args,
      tokenizer,
      std::move(callbacks),
      options,
      std::move(model_residency_coordinator),
      std::move(distributed_worker_manager),
      draft_step_controller);
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options);

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    RecEngine* engine,
    SchedulerOptions options);

}  // namespace xllm
