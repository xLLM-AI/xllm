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

#include "core/common/global_flags.h"
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

class EmbedVLMWorkerImpl : public WorkerImpl {
 public:
  EmbedVLMWorkerImpl(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const runtime::Options& options);

  ~EmbedVLMWorkerImpl() override = default;

  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;
};

}  // namespace xllm