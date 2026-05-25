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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_vlm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class VLMWorkerImpl : public WorkerImpl {
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

  std::optional<ForwardOutput> step(const ForwardInput& input) override;
  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& input,
      Stream& compute_stream) override;

  std::optional<ForwardOutput> step_internal(
      const ForwardInput& input,
      ForwardSyncPolicy sync_policy = ForwardSyncPolicy::LEGACY);
};

}  // namespace xllm
