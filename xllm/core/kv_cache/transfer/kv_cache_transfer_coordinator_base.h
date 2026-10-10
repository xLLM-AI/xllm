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

#include <folly/futures/Future.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/model/model_input_types.h"
#include "core/kv_cache/transfer/prefetch_result.h"

namespace xllm {

// Coordinates transfers prepared by the hierarchy cache pool. The pool owns
// allocation, cache admission, and publication.
class KVCacheTransferCoordinatorBase {
 public:
  virtual ~KVCacheTransferCoordinatorBase() = default;

  virtual std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) = 0;

  virtual void transfer_kv_blocks(
      uint32_t dp_rank,
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) = 0;

  virtual void prefetch_from_storage(
      uint32_t dp_rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) = 0;
};

}  // namespace xllm
