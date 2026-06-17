/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "rec_batch_input_builder.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "core/util/rec_model_utils.h"
#include "onerec_batch_input_builder.h"
#include "onerec_xattention_batch_input_builder.h"
#include "rec_multi_round_batch_input_builder.h"

namespace xllm {

std::unique_ptr<RecBatchInputBuilder> RecBatchInputBuilder::create(
    RecType rec_type,
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    MPMCThreadPool* thread_pool) {
  switch (rec_type) {
    case RecType::kOneRec:
      if (is_onerec_xattention_mode()) {
        return std::make_unique<OneRecXAttentionBatchInputBuilder>(
            sequence_groups,
            allowed_max_tokens,
            input_embeddings_vec,
            mm_data_vec,
            swap_block_transfer_infos,
            batch_id,
            args,
            batch_forward_type,
            thread_pool);
      }
      return std::make_unique<OneRecBatchInputBuilder>(
          sequence_groups,
          allowed_max_tokens,
          input_embeddings_vec,
          mm_data_vec,
          swap_block_transfer_infos,
          batch_id,
          args,
          batch_forward_type,
          thread_pool);
    case RecType::kLlmRec:
      // Check if Rec multi-round mode is enabled
      if (is_rec_multi_round_mode()) {
        return std::make_unique<RecMultiRoundBatchInputBuilder>(
            sequence_groups,
            allowed_max_tokens,
            input_embeddings_vec,
            mm_data_vec,
            swap_block_transfer_infos,
            batch_id,
            args,
            batch_forward_type,
            thread_pool);
      }
      // Fall through for non-multi-round LlmRec (not yet implemented)
      break;
    case RecType::kNone:
      break;
  }

  LOG(FATAL) << "Unsupported RecType for RecBatchInputBuilder: "
             << static_cast<int32_t>(rec_type);
  return nullptr;
}

}  // namespace xllm
