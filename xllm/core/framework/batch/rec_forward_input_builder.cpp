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

#include "core/framework/batch/rec_forward_input_builder.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "core/framework/batch/onerec_forward_input_builder.h"
#include "core/framework/batch/onerec_xattention_forward_input_builder.h"
#include "core/framework/batch/rec_multi_round_forward_input_builder.h"

namespace xllm {

std::unique_ptr<RecForwardInputBuilder> RecForwardInputBuilder::create(
    BatchInputType input_type,
    const BatchInputData& data,
    const ModelArgs* args,
    MPMCThreadPool* thread_pool) {
  switch (input_type) {
    case BatchInputType::ONEREC:
      return std::make_unique<OneRecForwardInputBuilder>(
          data, args, thread_pool);
    case BatchInputType::ONEREC_XATTENTION:
      return std::make_unique<OneRecXAttentionForwardInputBuilder>(
          data, args, thread_pool);
    case BatchInputType::REC_MULTI_ROUND:
      return std::make_unique<RecMultiRoundForwardInputBuilder>(
          data, args, thread_pool);
    case BatchInputType::SEQUENCE:
      break;
  }
  LOG(FATAL) << "Unsupported Rec batch input type: "
             << static_cast<int32_t>(input_type);
  return nullptr;
}

}  // namespace xllm
