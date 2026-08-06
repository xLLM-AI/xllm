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

#include <vector>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/model/model_input_params.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

struct HostCacheRestorePoint {
  size_t restore_target_tokens = 0;
  size_t copy_units = 0;
};

class KVCacheManager {
 public:
  virtual ~KVCacheManager() = default;

  virtual bool allocate(Sequence* sequence) = 0;
  virtual bool allocate(std::vector<Sequence*>& sequences) = 0;
  virtual bool allocate(Sequence* sequence, size_t num_tokens) = 0;
  virtual bool try_allocate(Sequence* sequence) = 0;

  virtual void transfer_blocks(std::vector<Batch>& batches) {};
  virtual void transfer_blocks() {};

  virtual void prefetch_from_storage(std::shared_ptr<Request>& request) {};

  virtual bool update_prefetch_result(std::shared_ptr<Request>& request,
                                      const uint32_t timeout) {
    return true;
  };

  virtual std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank) = 0;
  virtual void deallocate(Request* request) = 0;
  virtual void deallocate(std::vector<Sequence*>& sequences) = 0;
  virtual void deallocate(Sequence* sequence) = 0;

  virtual void allocate_shared(Sequence* sequence) = 0;
  virtual bool supports_host_cache_restore() const { return false; }
  virtual HostCacheRestorePoint select_host_cache_restore(
      Sequence* sequence,
      size_t /*max_copy_units*/) {
    return HostCacheRestorePoint{
        /*restore_target_tokens=*/sequence->kv_state().kv_cache_tokens_num(),
        /*copy_units=*/0};
  }
  virtual void trim_host_cache(Sequence* sequence,
                               const HostCacheRestorePoint& selected_restore) {
    sequence->set_host_cache_restore(selected_restore.restore_target_tokens,
                                     selected_restore.copy_units);
  }
  virtual void cache(Sequence* sequence) = 0;
  // Cache only the full blocks covered by the first `num_tokens` tokens. Used
  // by in-batch prefix cache to publish admitted prefill blocks before they are
  // deallocated, so later requests in the same batch can share them.
  virtual void cache(Sequence* sequence, size_t num_tokens) = 0;

  virtual std::vector<std::vector<BlockTransferInfo>>*
  get_swap_block_transfer_infos() = 0;

  virtual uint32_t num_blocks() const = 0;
  virtual int32_t block_size() const = 0;
  // Drop all prefix-cache entries (RL sleep/wakeup: discarded KV would make
  // cached prefixes point to garbage). Default no-op for managers without a
  // prefix cache.
  virtual void reset_prefix_cache() {}
  virtual std::vector<size_t> num_blocks_in_prefix_cache() const = 0;
  virtual std::vector<size_t> num_free_blocks() const = 0;
  virtual std::vector<size_t> num_used_blocks() const = 0;
  virtual double kv_cache_utilization() const = 0;

  // Reserve XTensor padding blocks after KV tensors are created.
  virtual void reserve_xtensor_padding_blocks() {}

 protected:
  KVCacheManager() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(KVCacheManager);
};

}  // namespace xllm
