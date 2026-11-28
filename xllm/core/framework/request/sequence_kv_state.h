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
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/kv_cache/block/block.h"
#include "core/kv_cache/block/cache_partition_binding.h"
#include "core/util/slice.h"

namespace xllm {

class KVCacheState {
 public:
  // get the number of tokens in the kvcache
  size_t kv_cache_tokens_num() const { return kv_cache_tokens_num_; }
  void set_kv_cache_tokens_num(size_t num);
  void incr_kv_cache_tokens_num(size_t num);
  // Advance kv_cache_tokens_num_ to `new_target` if it grows the counter. No-op
  // when `new_target <= kv_cache_tokens_num_`. CHECK-fails when `new_target`
  // exceeds current_max_tokens_capacity() -- callers must clamp before calling.
  // CHECK-fails when the counter is already past capacity (drift detection).
  // Used by the host-cache H2D restore path in HierarchyBlockManagerPool where
  // the "up to" semantics naturally arise from mismatched host/device counters.
  void incr_kv_cache_tokens_num_up_to(size_t new_target);

  // Register layout identity before mounting resources. A legacy binding is
  // optional and exists only for worker/export callers still using BlockType.
  void bind_partition(const CachePartitionId& id,
                      CacheRetention retention,
                      std::optional<BlockType> legacy_type = std::nullopt);
  SequenceCacheView view(const CachePartitionId& id) const;
  Slice<Block> blocks(const CachePartitionId& id) const;
  size_t num_blocks(const CachePartitionId& id) const;
  std::vector<int32_t> cache_slots(const CachePartitionId& id,
                                   int32_t pos_start,
                                   int32_t pos_end) const;
  void add_blocks(const CachePartitionId& id, const std::vector<Block>& blocks);
  void mount_composite_shared(const CachePartitionId& id,
                              std::vector<Block>&& blocks);
  void replace_composite_blocks(const CachePartitionId& id,
                                std::vector<Block>&& blocks,
                                size_t shared,
                                size_t published);
  void set_shared_blocks_num(const CachePartitionId& id, size_t num);
  void erase_blocks(const CachePartitionId& id);
  std::vector<Block> take_blocks(const CachePartitionId& id);
  size_t shared_blocks_num(const CachePartitionId& id) const;
  size_t num_cached_blocks(const CachePartitionId& id) const;
  void set_num_cached_blocks(const CachePartitionId& id, size_t n);
  size_t next_group_transfer_block_idx(const CachePartitionId& id) const;
  void advance_group_transfer_block_idx(const CachePartitionId& id, size_t idx);

  // Drop a logical suffix and clamp shared reach and paged publication. The
  // recurrent publication cursor counts checkpoints, not compact live slots.
  void truncate_blocks(const CachePartitionId& id, size_t logical_end);
  // Release a physical prefix while retaining its logical positions (SWA).
  void release_blocks_before(const CachePartitionId& id, size_t logical_end);

  // Transitional borrow for prefix insertion and recurrent live-slot updates.
  // Prefer mount/truncate/take operations for ownership changes.
  std::vector<Block>* mutable_blocks(const CachePartitionId& id);

  // Legacy worker-slot compatibility. New allocation paths use partition IDs.
  // Blocks held under `type`; empty slice when the type is absent.
  Slice<Block> blocks(BlockType type) const;
  // Mutable block list for `type`, created on demand.
  std::vector<Block>* mutable_blocks(BlockType type);
  // Number of blocks held under `type`.
  size_t num_blocks(BlockType type) const;
  // True if the sequence holds any cache-bearing blocks (KV / SWA / C4 / C128).
  // Excludes EMBEDDING, which is a per-sequence resource, not token cache. Used
  // to decide whether an allocation that started from an empty sequence should
  // be fully rolled back on failure (vs. a grow on an already-populated seq).
  bool has_any_blocks() const;

  // True after this cache tier has completed its prefix-cache probe.  This is
  // deliberately independent from has_any_blocks(): a completed probe may
  // have matched zero blocks, and a Mooncake prefetch may have populated Host
  // blocks before HBM has been probed.
  bool prefix_cache_matched() const { return prefix_cache_matched_; }
  void set_prefix_cache_matched(bool matched = true) {
    prefix_cache_matched_ = matched;
  }
  // token <-> physical slot mapping for `type` (paged attention). CHECKs the
  // type is present.
  std::vector<int32_t> cache_slots(BlockType type,
                                   int32_t pos_start,
                                   int32_t pos_end);

  void add_blocks(BlockType type, const std::vector<Block>& new_blocks);
  void add_shared_blocks(BlockType type,
                         std::vector<Block>&& blocks,
                         size_t current_total_num_tokens);
  // Composite mount for DSV4 admission: install the (possibly gap-containing)
  // shared block vector for `type` at logical positions [0, blocks.size()).
  // The vector came directly from this state's/type's prefix-cache probe, so
  // its logical length is also the actual per-type cache hit cursor. Does NOT
  // touch kv_cache_tokens_num_; the composite advances that once after all
  // leaves have mounted, so all leaves observe a consistent usable-token
  // count while retaining their independent cache hit cursors.
  void mount_composite_shared(BlockType type,
                              std::vector<Block>&& shared_blocks);
  // Replace the full block vector and its shared/cache publication metadata.
  void replace_composite_blocks(BlockType type,
                                std::vector<Block>&& blocks,
                                size_t num_shared_blocks,
                                size_t cache_publish_cursor);
  void set_shared_blocks_num(BlockType type, size_t num);
  // Drop all blocks held under `type` (releases their Block refs and removes
  // the map entry).
  void erase_blocks(BlockType type);

  // Move the blocks for `type` out without dropping their references.  The
  // caller becomes responsible for the returned aliases and may mount them
  // again after combining probes from another cache tier.
  std::vector<Block> take_blocks(BlockType type);

  // Number of shared (prefix-cache-hit) blocks held under `type`.
  size_t shared_blocks_num(BlockType type) const;
  // Number of shared tokens for this sequence. Sequence-level: the value is the
  // same across block types, so it takes no BlockType.
  size_t shared_tokens_num() const;

  // Prefix-cache cursor in units of this BlockType's blocks. It starts at the
  // actual logical reach returned by this state/type's probe (not the common
  // sequence restore length), then advances as newly forwarded blocks are
  // inserted. For sparse SWA this is a logical position and may span invalid
  // placeholders. Grows monotonically:
  //   - Admission mount: set to that type's retained probe-vector length.
  //   - Pre-grow hook: after processing positions through `end`, advance the
  //     cursor to `end`. Sparse SWA may intentionally insert only the trailing
  //     window of each cache unit while skipping earlier positions.
  //   - reset(): cleared alongside the rest of the sequence's cache state.
  size_t num_cached_blocks(BlockType type) const;
  // Per-type cursor table. Callers that need a stable snapshot must copy it.
  std::map<BlockType, size_t> num_cached_blocks() const;
  void set_num_cached_blocks(BlockType type, size_t n);

  size_t current_max_tokens_capacity() const;

  // Groups exported to the worker's multi_block_tables, in the fixed
  // kMultiBlockExportOrder. Each pair is (type, &blocks); only types currently
  // present in the map are returned. Empty for the flat KV path.
  std::vector<std::pair<BlockType, const std::vector<Block>*>>
  multi_block_export_view() const;
  // True when this sequence holds any multi_block_tables-exported group
  // (SWA / C4 / C128).
  bool has_multi_block_export() const;

  // Per-sequence embedding-row block (BlockType::EMBEDDING): returns the
  // EMBEDDING block id, or -1 when absent. Used for embedding_ids export
  // (spec-decode EmbeddingCache). Other block types read via blocks(type).
  int32_t get_embedding_block_id() const;

  // Linear-state live slot id (BlockType::LINEAR), or -1 when absent.
  int32_t get_linear_block_id() const;

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  size_t next_transfer_block_idx() const;
  void advance_transfer_block_idx(size_t idx);

  size_t next_group_transfer_block_idx(BlockType type) const;
  void advance_group_transfer_block_idx(BlockType type, size_t idx);

  void reset();

 private:
  const CachePartitionId& partition_id(BlockType type) const;
  const CachePartitionId& writable_partition_id(BlockType type);
  void remember_restore_point(const CachePartitionId& id,
                              const std::vector<Block>& blocks,
                              size_t shared);
  void remember_block_size(const CachePartitionId& id,
                           const std::vector<Block>& blocks);

  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  bool prefix_cache_matched_ = false;

  // Compatibility bindings do not own resources and are copied on fork.
  std::map<BlockType, CachePartitionId> legacy_bindings_;
  std::map<CachePartitionId, CacheRetention> retentions_;

  // Sorted identities keep allocation/release traversal deterministic. Worker
  // export order comes from the compatibility bindings and export order table.
  std::map<CachePartitionId, std::vector<Block>> composite_blocks_;

  // Logical block size is layout metadata, not Block ownership state. Host
  // offload moves physical Block handles into an asynchronous queue and leaves
  // invalid placeholders behind, so capacity calculations cannot read size()
  // from the first current handle.
  std::map<CachePartitionId, size_t> block_sizes_;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // next logical prompt block index that needs PD PUSH transfer.
  size_t next_transfer_block_idx_ = 0;

  // Cache groups can have different block sizes, so each group advances its
  // transfer cursor independently.
  std::map<CachePartitionId, size_t> next_group_transfer_block_idxes_;

  // Mounted recurrent restore boundary survives live-vector compaction.
  std::map<CachePartitionId, size_t> restore_tokens_;

  // Shared logical reach per partition.
  std::map<CachePartitionId, uint32_t> num_owned_shared_blocks_;

  // Pre-grow cache insert cursor per partition. See num_cached_blocks() above.
  std::map<CachePartitionId, size_t> num_cached_blocks_;
};

}  // namespace xllm
