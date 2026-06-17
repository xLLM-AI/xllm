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

#include "onerec_batch_input_builder.h"

namespace xllm {

// Isolated builder type for the future OneRec xattention pipeline.
// It currently reuses the legacy OneRec builder behavior, but keeps a separate
// type boundary so new step-meta / multi-round input organization can be added
// without polluting OneRecBatchInputBuilder.
class OneRecXAttentionBatchInputBuilder final : public OneRecBatchInputBuilder {
 public:
  explicit OneRecXAttentionBatchInputBuilder(
      const std::vector<SequencesGroup*>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      const uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      MPMCThreadPool* thread_pool = nullptr)
      : OneRecBatchInputBuilder(sequence_groups,
                                allowed_max_tokens,
                                input_embeddings_vec,
                                mm_data_vec,
                                swap_block_transfer_infos,
                                batch_id,
                                args,
                                batch_forward_type,
                                thread_pool),
        sequence_groups_(sequence_groups),
        allowed_max_tokens_(allowed_max_tokens),
        args_(args) {}

  ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  const std::vector<SequencesGroup*>& sequence_groups_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const ModelArgs* args_ = nullptr;
};

}  // namespace xllm
