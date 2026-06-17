/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "scheduler/scheduler_factory.h"

#include "core/common/global_flags.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_chunked_prefill_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"
#include "scheduler/mix_scheduler.h"
#include "scheduler/pd_ooc_scheduler.h"
#include "scheduler/prefill_only_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

namespace xllm {

SchedulerKind select_scheduler_kind(
    const ContinuousScheduler::Options& options) {
  if (::xllm::SchedulerConfig::get_instance().use_mix_scheduler()) {
    return SchedulerKind::MIX;
  }

  if (options.enable_disagg_pd()) {
    if (options.enable_pd_ooc()) {
      return SchedulerKind::PD_OOC;
    }
    if (options.enable_chunked_prefill()) {
      return SchedulerKind::DISAGG_PD_CHUNKED_PREFILL;
    }
    return SchedulerKind::DISAGG_PD;
  }

  if (options.enable_chunked_prefill()) {
    if (::xllm::ParallelConfig::get_instance().enable_prefill_sp() ||
        options.num_speculative_tokens() > 0) {
      return SchedulerKind::PREFILL_ONLY;
    }
    return SchedulerKind::CHUNKED_PREFILL;
  }

  if (::xllm::SchedulerConfig::get_instance().use_zero_evict()) {
    return SchedulerKind::ZERO_EVICTION;
  }

  return SchedulerKind::CONTINUOUS;
}

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  switch (select_scheduler_kind(options)) {
    case SchedulerKind::MIX:
      CHECK(options.enable_chunked_prefill())
          << "mix scheduler requires enabling chunked prefill";
      return std::make_unique<MixScheduler>(engine, options);
    case SchedulerKind::PD_OOC:
      return std::make_unique<PDOOCScheduler>(engine, options);
    case SchedulerKind::DISAGG_PD_CHUNKED_PREFILL:
      return std::make_unique<DisaggPDChunkedPrefillScheduler>(engine, options);
    case SchedulerKind::DISAGG_PD:
      return std::make_unique<DisaggPDScheduler>(engine, options);
    case SchedulerKind::PREFILL_ONLY:
      return std::make_unique<PrefillOnlyScheduler>(engine, options);
    case SchedulerKind::CHUNKED_PREFILL:
      return std::make_unique<ChunkedPrefillScheduler>(engine, options);
    case SchedulerKind::ZERO_EVICTION:
      return std::make_unique<ZeroEvictionScheduler>(engine, options);
    case SchedulerKind::CONTINUOUS:
      return std::make_unique<ContinuousScheduler>(engine, options);
  }

  return std::make_unique<ContinuousScheduler>(engine, options);
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    Engine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

std::unique_ptr<FixedStepsScheduler> create_fixed_steps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  return std::make_unique<FixedStepsScheduler>(engine, options);
}

}  // namespace xllm
