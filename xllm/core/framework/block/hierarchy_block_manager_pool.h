/* Copyright 2025-2026 The xLLM Authors.

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

#include "block_manager_pool.h"
#include "distributed_runtime/engine.h"
#include "util/blockingconcurrentqueue.h"

namespace xllm {

class Engine;

struct OffloadBlockPair {
  OffloadBlockPair(Block& s, Block& d) : src(s), dst(d) {}

  OffloadBlockPair(Block&& s, Block&& d)
      : src(std::move(s)), dst(std::move(d)) {}

  OffloadBlockPair(Block& s) : src(s) {}

  OffloadBlockPair(Block&& s) : src(std::move(s)) {}

  Block src;
  Block dst;
};

class HierarchyBlockManagerPool : public BlockManagerPool {
 public:
  using OffloadBlockPairQueue =
      moodycamel::BlockingConcurrentQueue<std::shared_ptr<OffloadBlockPair>>;

  explicit HierarchyBlockManagerPool(const BlockManagerPool::Options& options,
                                     Engine* engine,
                                     int32_t dp_size = 1);
  ~HierarchyBlockManagerPool() = default;

  bool allocate(Sequence* sequence, size_t num_tokens) override;

  // control the copy in blocks num
  bool allocate(Sequence* sequence,
                size_t num_tokens,
                size_t max_copy_in_blocks_num) override;

  void allocate_shared(Sequence* sequence) override;

  void deallocate(Sequence* sequence) override;

  void transfer_blocks(std::vector<Batch>& batches) override;
  void transfer_blocks() override;

  void prefetch_from_storage(std::shared_ptr<Request>& request) override;

  bool update_prefetch_result(std::shared_ptr<Request>& request,
                              const uint32_t timeout) override;

 private:
  void allocate_host_shared(Sequence* sequence);
  // Move offload-eligible device/host KV block pairs out of the sequence into
  // offload_block_pair_queues_[dp_rank] for the next D2H transfer.
  void collect_offload_pairs(Sequence* sequence, int32_t dp_rank);

 private:
  Engine* engine_;
  std::vector<std::unique_ptr<BlockManager>> host_block_managers_;

  // BlockTransferInfo per step
  std::vector<std::vector<BlockTransferInfo>> load_block_transfer_infos_;
  std::vector<OffloadBlockPairQueue> offload_block_pair_queues_;
};

}  // namespace xllm
