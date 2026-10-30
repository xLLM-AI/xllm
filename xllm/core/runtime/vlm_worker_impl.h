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
#include <optional>

#include "core/runtime/vlm_forward_params.h"
#include "runtime/worker_impl.h"

namespace xllm {

class VLMWorkerImpl final : public WorkerImpl {
 public:
  enum class ForwardSyncPolicy : int8_t {
    LEGACY = 0,
    NO_SYNC,
  };

  VLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~VLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  bool uses_worker_task_pipeline() const override { return true; }
  ::xllm::Status create_task_pipeline(
      std::unique_ptr<TaskExecutionPipeline>& output) override;

  std::optional<ForwardOutput> step(const VlmForwardInput& input) override;

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      const VlmForwardInput& input) override;

 private:
  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const VlmForwardInput& input,
      Stream& compute_stream,
      ForwardCompletion completion) override;

  std::optional<ForwardOutput> step_internal(
      const VlmForwardInput& input,
      ForwardSyncPolicy sync_policy = ForwardSyncPolicy::LEGACY,
      ForwardCompletion completion = ForwardCompletion::CALLER_MANAGED);
};

}  // namespace xllm
