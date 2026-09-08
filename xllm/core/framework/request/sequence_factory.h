/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/request/sequence.h"

namespace xllm {

// Creates the Sequence type matching `seq_params.rec_type`: a plain Sequence
// for LLM / VLM requests, LlmRecSequence or OneRecSequence for recommendation
// requests.
std::unique_ptr<Sequence> create_sequence(
    size_t index,
    const std::vector<int32_t>& prompt_token_ids,
    torch::Tensor input_embedding,
    const MMData& mm_data,
    const IncrementalDecoder& incremental_decoder,
    const SequenceParams& seq_params);

}  // namespace xllm
