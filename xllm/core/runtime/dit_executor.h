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

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "core/framework/dit_model_loader.h"
#include "forward_params.h"
#include "framework/batch/dit_batch.h"
#include "framework/model/dit_model.h"
#include "framework/model/model_input_params.h"
#include "framework/request/dit_request_state.h"
#include "runtime/options.h"

namespace xllm {

class DiTExecutor {
 public:
  DiTExecutor(DiTModel* model, const runtime::Options& options);

  ~DiTExecutor() = default;

  DiTForwardInput prepare_inputs(DiTBatch& batch);

  DiTForwardOutput forward(const DiTForwardInput& input);

 private:
  // not own
  DiTModel* model_;
  runtime::Options options_;
};

}  // namespace xllm
