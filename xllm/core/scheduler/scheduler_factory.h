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

#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"

namespace xllm {

class DiTEngine;
class KVTransferCoordinator;
struct PDServiceOptions;

enum class SchedulerKind : int8_t { CONTINUOUS = 0, DISAGG_PD = 5 };

SchedulerKind select_scheduler_kind(const SchedulerOptions& options);

std::unique_ptr<Scheduler> create_continuous_scheduler(
    BlockManagerPool* block_manager_pool,
    const ModelArgs* model_args,
    const Tokenizer* tokenizer,
    SchedulerCallbacks callbacks,
    SchedulerOptions options,
    std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager,
    std::shared_ptr<KVTransferCoordinator> kv_transfer_coordinator,
    PDServiceOptions response_options);

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options);

}  // namespace xllm
