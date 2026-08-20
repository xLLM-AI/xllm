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

#include "dit_executor.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm {

DiTExecutor::DiTExecutor(DiTModel* model, const runtime::Options& options)
    : model_(model), options_(options) {}

DiTForwardInput DiTExecutor::prepare_inputs(DiTBatch& batch) {
  return batch.prepare_forward_input();
}

DiTForwardOutput DiTExecutor::forward(const DiTForwardInput& input) {
  return model_->forward(input);
}

}  // namespace xllm
