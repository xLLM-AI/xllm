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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/kv_cache/transfer/kv_cache_transfer_coordinator_base.h"

namespace xllm {

class DistributedWorkerManager;

// Coordinates KV transfer participation and completion across distributed
// workers. The master composes this alongside the model engine.
class KVCacheTransferCoordinator final : public KVCacheTransferCoordinatorBase {
 public:
  struct Options {
    int32_t dp_size = 1;
    uint32_t prefetch_timeout_ms = 0;
  };

  KVCacheTransferCoordinator(
      Options options,
      std::shared_ptr<DistributedWorkerManager> distributed_worker_manager);

  bool pull_kv_blocks(int32_t src_dp_size,
                      int32_t src_dp_rank,
                      const std::vector<uint64_t>& src_cluster_ids,
                      const std::vector<std::string>& src_addrs,
                      int32_t dst_dp_rank,
                      const std::vector<KVTransferMapping>& mappings);

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  void transfer_kv_blocks(
      uint32_t dp_rank,
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  void prefetch_from_storage(
      uint32_t dp_rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) override;

 private:
  const Options options_;
  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_;
};

}  // namespace xllm
