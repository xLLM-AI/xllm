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

#include "core/framework/batch/rec_batch_factory.h"

namespace xllm {

RecBatchFactory::RecBatchFactory(int32_t dp_size, BatchInputType input_type)
    : builder_(dp_size, input_type) {}

RecBatchGroup RecBatchFactory::create_batches(
    const std::vector<std::shared_ptr<Request>>& running_requests,
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos)
    const {
  return builder_.build(running_requests,
                        running_sequences,
                        running_sequences_budgets,
                        swap_block_transfer_infos);
}

}  // namespace xllm
