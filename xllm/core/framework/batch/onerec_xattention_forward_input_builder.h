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

#include "core/framework/batch/onerec_forward_input_builder.h"

namespace xllm {

// Adds xattention decode metadata to the common OneRec encoder/decoder input.
class OneRecXAttentionForwardInputBuilder final
    : public OneRecForwardInputBuilder {
 public:
  OneRecXAttentionForwardInputBuilder(const BatchInputData& data,
                                      const ModelArgs* args,
                                      MPMCThreadPool* thread_pool = nullptr)
      : OneRecForwardInputBuilder(data, args, thread_pool),
        sequence_groups_(data.sequence_groups),
        args_(args) {}

  RecForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  const std::vector<SequencesGroup*>& sequence_groups_;
  const ModelArgs* args_ = nullptr;
};

}  // namespace xllm
