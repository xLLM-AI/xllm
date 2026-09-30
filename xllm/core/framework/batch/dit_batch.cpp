/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/dit_batch.h"

#include <glog/logging.h>

#include <cstdint>
#include <utility>

#include "core/framework/batch/dit_forward_input_builder.h"

namespace xllm {

void DiTBatch::add(std::shared_ptr<DiTRequest> request) {
  CHECK(request != nullptr);
  request_vec_.emplace_back(std::move(request));
}

DiTForwardInput DiTBatch::prepare_forward_input() {
  return DiTForwardInputBuilder(request_vec_).build_forward_input();
}

void DiTBatch::process_forward_output(const DiTForwardOutput& output) {
  // Text diffusion models produce text output directly.
  if (!output.text_output.empty()) {
    CHECK(request_vec_.size() == output.text_output.size());
    for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
         ++idx) {
      auto& request = request_vec_[idx];
      request->handle_forward_text_output(output.text_output[idx]);
    }
    return;
  }
  CHECK(request_vec_.size() == output.tensors.size());
  for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
       ++idx) {
    auto& request = request_vec_[idx];
    request->handle_forward_output(output.tensors[idx]);
  }
}

}  // namespace xllm
