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

#include "common/metrics.h"
#include "framework/batch/batch.h"

namespace xllm {

class BatchFactory {
 public:
  static BatchFactory* get_instance(int32_t dp_size, int32_t cp_size = 1) {
    static BatchFactory instance(dp_size, cp_size);
    return &instance;
  }

  std::vector<Batch> create_batches(
      const std::vector<std::shared_ptr<Request>>& running_requests,
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      // for beam-search
      std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos =
          nullptr);

  std::vector<Batch> create_rec_batches(
      const std::vector<std::shared_ptr<Request>>& running_requests,
      const std::vector<Sequence*>& running_sequences,
      const std::vector<size_t>& running_sequences_budgets,
      std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos =
          nullptr);

 private:
  BatchFactory(int32_t dp_size, int32_t cp_size)
      : dp_size_(dp_size), cp_size_(cp_size) {}
  ~BatchFactory() = default;

  DISALLOW_COPY_AND_ASSIGN(BatchFactory);

 private:
  int32_t dp_size_;
  int32_t cp_size_;
};
}  // namespace xllm
