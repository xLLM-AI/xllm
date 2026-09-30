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

#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/sequences_group.h"

namespace xllm {

// Selected by the scheduler's factory, independent of mutable sequence views.
enum class BatchInputType : int8_t {
  SEQUENCE,
  ONEREC,
  ONEREC_XATTENTION,
  REC_MULTI_ROUND,
};

// Non-owning per-forward view. Its sequence plan, owning batch and requests
// must outlive the builder. Builders may advance sequence execution state.
struct BatchInputData {
  const std::vector<Sequence*>& sequences;
  const std::vector<SequencesGroup*>& sequence_groups;
  const std::vector<uint32_t>& allowed_max_tokens;
  const std::vector<torch::Tensor>& input_embeddings;
  const std::vector<MMData>& mm_data;
  std::vector<BlockTransferInfo>* swap_block_transfer_infos;
  uint64_t batch_id;
  BatchForwardType forward_type;
};

}  // namespace xllm
