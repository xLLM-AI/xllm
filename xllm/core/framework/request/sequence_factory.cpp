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

#include "core/framework/request/sequence_factory.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

#include "core/framework/request/onerec_sequence.h"
#include "core/framework/request/rec_sequence.h"
#include "core/framework/request/rec_type.h"

namespace xllm {

std::unique_ptr<Sequence> create_sequence(
    size_t index,
    const std::vector<int32_t>& prompt_token_ids,
    torch::Tensor input_embedding,
    const MMData& mm_data,
    const IncrementalDecoder& incremental_decoder,
    const SequenceParams& seq_params) {
  switch (seq_params.rec_type) {
    case RecType::kNone:
      return std::make_unique<Sequence>(index,
                                        prompt_token_ids,
                                        std::move(input_embedding),
                                        mm_data,
                                        incremental_decoder,
                                        seq_params);
    case RecType::kLlmRec:
      return std::make_unique<LlmRecSequence>(index,
                                              prompt_token_ids,
                                              std::move(input_embedding),
                                              mm_data,
                                              incremental_decoder,
                                              seq_params);
    case RecType::kOneRec:
      return std::make_unique<OneRecSequence>(index,
                                              prompt_token_ids,
                                              std::move(input_embedding),
                                              mm_data,
                                              incremental_decoder,
                                              seq_params);
  }
  LOG(FATAL) << "unknown rec type " << static_cast<int>(seq_params.rec_type);
  return nullptr;
}

}  // namespace xllm
