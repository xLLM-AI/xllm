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

#include <torch/types.h>

#include <functional>

#include "forward_params.h"
#include "framework/eplb/eplb_executor.h"
#include "framework/model/causal_lm.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class LLMWorkerImpl : public WorkerImpl {
 public:
  enum class ForwardSyncPolicy : int8_t {
    LEGACY = 0,
    NO_SYNC,
  };

  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                bool is_input_release_deferred = false);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const LlmForwardInput& input) override;

  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const LlmForwardInput& input,
      Stream& compute_stream,
      ForwardCompletion completion,
      const std::function<void()>& prepare_sampling = {}) override;

  std::optional<ForwardOutput> step_internal(
      const LlmForwardInput& input,
      ForwardSyncPolicy sync_policy = ForwardSyncPolicy::LEGACY,
      ForwardCompletion completion = ForwardCompletion::CALLER_MANAGED,
      const std::function<void()>& prepare_sampling = {});

 protected:
  void prepare_expert_input(ExpertInput& input) const override;

  std::optional<ForwardOutput> step_for_schedule_overlap(
      const LlmForwardInput& input,
      const std::function<void()>& prepare_sampling = {}) override;

 public:
#if defined(USE_NPU)
  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const Stream& signal_stream);
#endif

 protected:
  torch::Tensor expert_load_data_;
  std::unique_ptr<EplbExecutor> eplb_executor_;
};

}  // namespace xllm
