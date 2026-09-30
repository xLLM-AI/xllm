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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/request/request.h"

namespace xllm {

// Builds LLM/VLM batches grouped by DP rank.
class BatchBuilder final {
 public:
  explicit BatchBuilder(int32_t dp_size);

  BatchGroup build(
      const std::vector<std::shared_ptr<Request>>& requests,
      const std::vector<Sequence*>& sequences,
      const std::vector<size_t>& budgets,
      std::vector<std::vector<BlockTransferInfo>>* swap_infos) const;

 private:
  int32_t dp_size_;
};

}  // namespace xllm
