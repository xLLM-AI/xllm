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

#include <folly/futures/Future.h>
#include <torch/types.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_params_view.h"
#include "framework/quantization/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class EmbedWorkerImpl : public WorkerImpl {
 public:
  EmbedWorkerImpl(const ParallelArgs& parallel_args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~EmbedWorkerImpl() override = default;

  std::optional<ForwardOutput> step(const LlmForwardInput& input) override;
};

}  // namespace xllm
