/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/batch/batch_group.h"

#include <glog/logging.h>

namespace xllm {

BatchGroup::BatchGroup(size_t dp_size) : batches_() {
  CHECK_GT(dp_size, 0);
  batches_.reserve(dp_size);
  for (size_t rank = 0; rank < dp_size; ++rank) {
    batches_.emplace_back();
  }
}

void BatchGroup::apply_outputs(const std::vector<RawForwardOutput>& outputs,
                               BatchOutputAction action) {
  CHECK_EQ(outputs.size(), size())
      << "Each DP rank must have exactly one output.";
  for (size_t dp_rank = 0; dp_rank < size(); ++dp_rank) {
    Batch& batch = batches_[dp_rank];
    const RawForwardOutput& output = outputs[dp_rank];
    batch.process_sample_output(output, action);
  }
}

}  // namespace xllm
