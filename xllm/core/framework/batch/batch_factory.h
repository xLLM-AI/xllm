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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/batch/batch_builder.h"
#include "core/framework/batch/batch_group.h"
#include "core/framework/request/request.h"

namespace xllm {

// LLM and VLM share sequence scheduling, KV state and token budgets. Their
// multimodal differences belong to input building, not DP batch assembly.
class BatchFactory final {
 public:
  explicit BatchFactory(int32_t dp_size);

  BatchGroup create_batches(
      const std::vector<std::shared_ptr<Request>>& running_requests,
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos =
          nullptr) const;

 private:
  BatchBuilder builder_;
};

}  // namespace xllm
