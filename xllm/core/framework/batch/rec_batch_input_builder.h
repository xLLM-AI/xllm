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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "framework/model/model_args.h"
#include "framework/request/rec_type.h"
#include "framework/request/sequences_group.h"
#include "runtime/forward_params.h"
#include "util/threadpool.h"

namespace xllm {

class RecBatchInputBuilder {
 public:
  virtual ~RecBatchInputBuilder() = default;

  virtual ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) = 0;

  static std::unique_ptr<RecBatchInputBuilder> create(
      RecType rec_type,
      const std::vector<SequencesGroup*>& sequence_groups,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      MPMCThreadPool* thread_pool = nullptr);
};

}  // namespace xllm
