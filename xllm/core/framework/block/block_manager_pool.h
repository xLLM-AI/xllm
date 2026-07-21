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

#include <queue>
#include <vector>

#include "block_manager.h"
#include "framework/block/kv_cache_manager.h"
#include "framework/block/single_block_manager.h"

namespace xllm {

class BlockManagerPool : public KVCacheManager {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(uint32_t, host_num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(bool, enable_linear_state) = false;
    // Total physical linear-state slots [0, N) for the unified slot pool
    // (= num_linear_state_blocks). Only used when enable_linear_state is true.
    PROPERTY(int32_t, linear_state_num_slots) = 0;
    // Linear-state checkpoint stride in tokens (one prefill chunk), forwarded
    // to the LINEAR leaf so its checkpoint index probes the correct hash
    // domain. Only used when enable_linear_state is true; -1 disables the
    // probe.
    PROPERTY(int32_t, linear_chunk_stride) = -1;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
    // Host prefix-cache offload (host_blocks_factor > 1). Wraps composite
    // leaves in ConcurrentBlockManagerImpl so the async D2H offload callback
    // can free blocks off-thread safely.
    PROPERTY(bool, enable_host_offload) = false;
    PROPERTY(bool, enable_xtensor) = false;
    PROPERTY(int64_t, num_layers) = 0;  // Required when enable_xtensor is true
    PROPERTY(int64_t, slot_size) = 0;   // Memory size per slot (for xtensor)
    PROPERTY(std::string, model_id);    // Model ID for multi-model support
    // Token-level sliding window size for CompositeBlockManager.
    PROPERTY(uint32_t, sliding_window_size) = 0;
    // Base SWA/cache-state block rows retained per sequence.
    PROPERTY(uint32_t, swa_blocks_per_seq) = 0;
    // Scheduler token budget used to size the shared SWA burst pool.
    PROPERTY(uint32_t, max_tokens_per_batch) = 0;
    // For CompositeBlockManager.
    PROPERTY(std::vector<uint32_t>, manager_types) = {};
    PROPERTY(std::vector<uint32_t>, compress_ratios) = {};
    PROPERTY(uint32_t, max_seqs_per_batch) = 0;
    // Hasher type bound to the engine (TEXT for LLM, MM for VLM).
    PROPERTY(BlockHasherType, hasher_type) = BlockHasherType::TEXT;
    PROPERTY(uint32_t, num_single_blocks) = 0;
    PROPERTY(uint32_t, num_speculative_tokens) = 0;
  };

  explicit BlockManagerPool(const Options& options, int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  virtual bool allocate(Sequence* sequence) override;
  virtual bool allocate(std::vector<Sequence*>& sequences) override;
  virtual bool allocate(Sequence* sequence, size_t num_tokens) override;
  virtual bool allocate(Sequence* sequence,
                        size_t num_tokens,
                        size_t needed_copy_in_blocks_num) override;

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  virtual std::vector<Block> allocate(size_t num_tokens,
                                      int32_t& dp_rank) override;

  virtual bool try_allocate(Sequence* sequence) override;

  virtual void deallocate(Request* request) override;
  virtual void deallocate(std::vector<Sequence*>& sequences) override;
  virtual void deallocate(Sequence* sequence) override;

  void deallocate_without_cache(Sequence* sequence);

  virtual void allocate_shared(Sequence* sequence) override;
  virtual void cache(Sequence* sequence) override;
  virtual void cache(Sequence* sequence, size_t num_tokens) override;

  virtual std::vector<std::vector<BlockTransferInfo>>*
  get_swap_block_transfer_infos() override;

  virtual float get_gpu_cache_usage_perc() const;

  virtual uint32_t num_blocks() const override;
  virtual int32_t block_size() const override;
  void reset_prefix_cache() override;
  virtual std::vector<size_t> num_blocks_in_prefix_cache() const override;
  virtual std::vector<size_t> num_free_blocks() const override;
  virtual std::vector<size_t> num_used_blocks() const override;
  virtual double kv_cache_utilization() const override;

  // get the options for the block manager
  const Options& options() const { return options_; }

  // Reserve XTensor padding blocks for each DP manager.
  // Should be called after KV tensors are created.
  void reserve_xtensor_padding_blocks() override;

 protected:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  bool process_beam_search(Sequence* sequence, bool need_swap = false);

 private:
  friend class BlockManagerPoolTestPeer;

  std::vector<std::vector<BlockTransferInfo>> swap_block_transfer_infos_;

 protected:
  // the options for the block manager
  Options options_;
  std::vector<std::unique_ptr<BlockManager>> block_managers_;
};

}  // namespace xllm
