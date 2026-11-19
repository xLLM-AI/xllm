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
#include "core/disaggregation/pd_scheduler.h"
#include "core/framework/config/parallel_config.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"

namespace xllm {

SchedulerKind select_scheduler_kind(const SchedulerOptions& options) {
  if (options.enable_disagg_pd()) {
    return SchedulerKind::DISAGG_PD;
  }

  return SchedulerKind::CONTINUOUS;
}

std::unique_ptr<Scheduler> create_continuous_scheduler(
    BlockManagerPool* block_manager_pool,
    const ModelArgs* model_args,
    const Tokenizer* tokenizer,
    SchedulerCallbacks callbacks,
    SchedulerOptions options,
    std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager,
    std::shared_ptr<KVTransferCoordinator> kv_transfer_coordinator,
    PDServiceOptions response_options) {
  if (select_scheduler_kind(options) == SchedulerKind::DISAGG_PD) {
    auto scheduler =
        std::make_unique<PDScheduler>(block_manager_pool,
                                      model_args,
                                      tokenizer,
                                      std::move(callbacks),
                                      options,
                                      std::move(distributed_worker_manager),
                                      std::move(model_residency_coordinator),
                                      std::move(kv_transfer_coordinator));
    scheduler->start_runtime(std::move(response_options));
    return scheduler;
  }
  return std::make_unique<ContinuousScheduler>(
      block_manager_pool,
      model_args,
      tokenizer,
      std::move(callbacks),
      options,
      std::move(model_residency_coordinator),
      std::move(distributed_worker_manager));
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

}  // namespace xllm
