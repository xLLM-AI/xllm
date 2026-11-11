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

#pragma once

#include <vector>

#include "core/runtime/forward_params.h"
#include "worker.pb.h"

namespace xllm {

// Encode only completed, contiguous CPU outputs. A failed conversion leaves
// the destination unchanged.
bool encode_llm_output(const ForwardOutput& output,
                       const std::vector<SpeculativeTokenStats>& stats,
                       proto::LLMForwardResponse* response);
bool decode_llm_output(const proto::LLMForwardResponse& response,
                       RawForwardOutput& output);

bool encode_vlm_output(const ForwardOutput& output,
                       const std::vector<SpeculativeTokenStats>& stats,
                       proto::VLMForwardResponse* response);
bool decode_vlm_output(const proto::VLMForwardResponse& response,
                       RawForwardOutput& output);

bool encode_dit_output(const ForwardOutput& output,
                       proto::DITForwardResponse* response);
bool decode_dit_output(const proto::DITForwardResponse& response,
                       RawForwardOutput& output);

}  // namespace xllm
