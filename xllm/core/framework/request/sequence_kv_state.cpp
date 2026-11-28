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

#include "core/framework/request/sequence_kv_state.h"

#include <algorithm>
#include <iterator>

#include "core/kv_cache/block/block.h"
#include "core/kv_cache/block/block_manager.h"

namespace xllm {

namespace {
// Empty block list returned by read views when a BlockType is absent from the
// map. Static so the returned Slice / reference stays valid.
const std::vector<Block>& empty_blocks() {
  static const std::vector<Block> kEmpty;
  return kEmpty;
}

void try_replace_unique_blocks(std::vector<Block>&& matched_shared_blocks,
                               uint32_t* num_owned_shared_blocks,
                               std::vector<Block>* owned_blocks) {
  uint32_t num_matched_shared_blocks = matched_shared_blocks.size();
  if (*num_owned_shared_blocks < num_matched_shared_blocks) {
    CHECK_GE(owned_blocks->size(), num_matched_shared_blocks);
    std::move(matched_shared_blocks.begin(),
              matched_shared_blocks.begin() + num_matched_shared_blocks,
              owned_blocks->begin());
    *num_owned_shared_blocks = num_matched_shared_blocks;
  }
}
}  // namespace

const CachePartitionId& KVCacheState::partition_id(BlockType type) const {
  const auto it = legacy_bindings_.find(type);
  return it == legacy_bindings_.end() ? legacy_cache_partition_id(type)
                                      : it->second;
}

const CachePartitionId& KVCacheState::writable_partition_id(BlockType type) {
  const CachePartitionId& id = partition_id(type);
  retentions_.try_emplace(id, legacy_cache_retention(type));
  legacy_bindings_.try_emplace(type, id);
  return partition_id(type);
}

void KVCacheState::bind_partition(const CachePartitionId& id,
                                  CacheRetention retention,
                                  std::optional<BlockType> legacy_type) {
  if (legacy_type.has_value()) {
    const CachePartitionId previous = partition_id(*legacy_type);
    if (!(previous == id)) {
      const auto binding = legacy_bindings_.find(*legacy_type);
      CHECK(binding == legacy_bindings_.end() ||
            binding->second == legacy_cache_partition_id(*legacy_type))
          << "Cannot rebind an active model's worker cache slot";
      // Legacy callers can populate a sequence before its manager sees it.
      // Move those handles and cursors; never duplicate ownership.
      auto rekey = [&](auto& table) {
        auto node = table.extract(previous);
        if (!node.empty()) {
          CHECK(!table.contains(id));
          node.key() = id;
          table.insert(std::move(node));
        }
      };
      rekey(composite_blocks_);
      rekey(block_sizes_);
      rekey(num_owned_shared_blocks_);
      rekey(num_cached_blocks_);
      rekey(next_group_transfer_block_idxes_);
      rekey(restore_tokens_);
      retentions_.erase(previous);
    }
    legacy_bindings_.insert_or_assign(*legacy_type, id);
  }
  const auto [it, inserted] = retentions_.try_emplace(id, retention);
  CHECK(inserted || it->second == retention);
}

SequenceCacheView KVCacheState::view(const CachePartitionId& id) const {
  SequenceCacheView result;
  const Slice<Block> bs = blocks(id);
  result.logical_blocks = bs.size();
  result.resident_blocks = std::ranges::count_if(
      bs, [](const Block& block) { return block.is_valid(); });
  const auto stride = block_sizes_.find(id);
  result.token_stride = stride == block_sizes_.end() ? 0 : stride->second;
  result.shared_logical_blocks = shared_blocks_num(id);
  result.published_logical_blocks = num_cached_blocks(id);
  const auto restore = restore_tokens_.find(id);
  if (restore != restore_tokens_.end() && result.shared_logical_blocks > 0 &&
      result.shared_logical_blocks <= bs.size() &&
      bs[result.shared_logical_blocks - 1].is_valid()) {
    result.restore_tokens = restore->second;
  }
  return result;
}

void KVCacheState::truncate_blocks(const CachePartitionId& id,
                                   size_t logical_end) {
  auto it = composite_blocks_.find(id);
  CHECK(it != composite_blocks_.end());
  CHECK_LE(logical_end, it->second.size());
  if (logical_end == it->second.size()) {
    return;
  }
  const bool removed_restore = logical_end < shared_blocks_num(id);
  std::vector<Block> suffix;
  suffix.reserve(it->second.size() - logical_end);
  std::move(it->second.begin() + logical_end,
            it->second.end(),
            std::back_inserter(suffix));
  it->second.resize(logical_end);
  Block::release(std::move(suffix));
  num_owned_shared_blocks_[id] = std::min(shared_blocks_num(id), logical_end);
  const auto retention = retentions_.find(id);
  if (retention == retentions_.end() ||
      retention->second != CacheRetention::RECURRENT) {
    num_cached_blocks_[id] = std::min(num_cached_blocks(id), logical_end);
  }
  if (removed_restore) {
    remember_restore_point(id, it->second, shared_blocks_num(id));
  }
}

void KVCacheState::release_blocks_before(const CachePartitionId& id,
                                         size_t logical_end) {
  auto it = composite_blocks_.find(id);
  CHECK(it != composite_blocks_.end());
  CHECK_LE(logical_end, it->second.size());
  std::vector<Block> released;
  released.reserve(logical_end);
  std::move(it->second.begin(),
            it->second.begin() + logical_end,
            std::back_inserter(released));
  Block::release(std::move(released));
}

void KVCacheState::set_kv_cache_tokens_num(size_t num) {
  kv_cache_tokens_num_ = num;
}

void KVCacheState::incr_kv_cache_tokens_num(size_t num) {
  CHECK(kv_cache_tokens_num_ + num <= current_max_tokens_capacity());
  kv_cache_tokens_num_ += num;
}

void KVCacheState::incr_kv_cache_tokens_num_up_to(size_t new_target) {
  const size_t capacity = current_max_tokens_capacity();
  // Drift check: the counter was already advanced past capacity by an earlier
  // call. Failing here loudly names the producer instead of silently letting
  // the drift propagate to forward_input_builder's CHECK.
  CHECK_LE(kv_cache_tokens_num_, capacity)
      << "kv_cache_tokens_num_ drifted past capacity: " << kv_cache_tokens_num_
      << " > " << capacity;
  CHECK_LE(new_target, capacity)
      << "incr_kv_cache_tokens_num_up_to target above capacity: " << new_target
      << " > " << capacity;
  if (new_target > kv_cache_tokens_num_) {
    kv_cache_tokens_num_ = new_target;
  }
}

Slice<Block> KVCacheState::blocks(const CachePartitionId& id) const {
  const auto it = composite_blocks_.find(id);
  return it == composite_blocks_.end() ? Slice<Block>(empty_blocks())
                                       : Slice<Block>(it->second);
}

std::vector<Block>* KVCacheState::mutable_blocks(const CachePartitionId& id) {
  return &composite_blocks_[id];
}

size_t KVCacheState::num_blocks(const CachePartitionId& id) const {
  const auto it = composite_blocks_.find(id);
  return it == composite_blocks_.end() ? 0 : it->second.size();
}

bool KVCacheState::has_any_blocks() const {
  return std::ranges::any_of(composite_blocks_, [this](const auto& entry) {
    const auto retention = retentions_.find(entry.first);
    const CacheRetention kind = retention == retentions_.end()
                                    ? CacheRetention::HISTORY
                                    : retention->second;
    return !entry.second.empty() &&
           (kind == CacheRetention::HISTORY || kind == CacheRetention::WINDOW);
  });
}

std::vector<int32_t> KVCacheState::cache_slots(const CachePartitionId& id,
                                               int32_t pos_start,
                                               int32_t pos_end) const {
  const auto it = composite_blocks_.find(id);
  CHECK(it != composite_blocks_.end() && !it->second.empty())
      << "no cache blocks available";
  const std::vector<Block>& bs = it->second;

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = block_sizes_.at(id);
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = bs[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

void KVCacheState::add_blocks(const CachePartitionId& id,
                              const std::vector<Block>& new_blocks) {
  remember_block_size(id, new_blocks);
  std::vector<Block>& bs = composite_blocks_[id];
  bs.insert(bs.end(), new_blocks.begin(), new_blocks.end());
}

void KVCacheState::remember_block_size(const CachePartitionId& id,
                                       const std::vector<Block>& blocks) {
  for (const Block& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    const auto [it, inserted] = block_sizes_.emplace(id, block.size());
    CHECK(inserted || it->second == block.size())
        << "block size changed for cache type " << id.model_id << ":"
        << id.ordinal;
    return;
  }
}

void KVCacheState::remember_restore_point(const CachePartitionId& id,
                                          const std::vector<Block>& bs,
                                          size_t shared) {
  restore_tokens_.erase(id);
  const auto retention = retentions_.find(id);
  if (retention == retentions_.end() ||
      retention->second != CacheRetention::RECURRENT) {
    return;
  }
  for (size_t index = shared; index > 0; --index) {
    if (bs[index - 1].is_valid()) {
      restore_tokens_[id] = index * bs[index - 1].size();
      return;
    }
  }
}

void KVCacheState::set_shared_blocks_num(const CachePartitionId& id,
                                         size_t num) {
  CHECK_LE(num, num_blocks(id));
  num_owned_shared_blocks_[id] = static_cast<uint32_t>(num);
  if (num == 0) {
    restore_tokens_.erase(id);
  }
}

void KVCacheState::erase_blocks(const CachePartitionId& id) {
  restore_tokens_.erase(id);
  composite_blocks_.erase(id);
  block_sizes_.erase(id);
  num_owned_shared_blocks_.erase(id);
  num_cached_blocks_.erase(id);
}

std::vector<Block> KVCacheState::take_blocks(const CachePartitionId& id) {
  restore_tokens_.erase(id);
  auto it = composite_blocks_.find(id);
  if (it == composite_blocks_.end()) {
    num_owned_shared_blocks_.erase(id);
    num_cached_blocks_.erase(id);
    block_sizes_.erase(id);
    return {};
  }
  std::vector<Block> blocks = std::move(it->second);
  composite_blocks_.erase(it);
  block_sizes_.erase(id);
  num_owned_shared_blocks_.erase(id);
  num_cached_blocks_.erase(id);
  return blocks;
}

size_t KVCacheState::shared_blocks_num(const CachePartitionId& id) const {
  const auto it = num_owned_shared_blocks_.find(id);
  return it == num_owned_shared_blocks_.end() ? 0 : it->second;
}

size_t KVCacheState::shared_tokens_num() const {
  // Shared token count is a sequence-level value: shared_blocks * block_size
  // is the same across cache-bearing block types by construction (composite
  // trim keeps every leaf aligned to a common safe_hit). Read it off any type
  // that carries shared blocks. Use the first VALID block in the vector for
  // block_size -- vec[0] may be an invalid placeholder in the gap-tolerant
  // SWA case (last_hit at position K, vec_len = K+1, positions [0..K-1] may
  // include invalid placeholders), and reading .size() off an invalid Block
  // returns 0.
  for (const auto& [type, shared] : num_owned_shared_blocks_) {
    if (shared == 0) {
      continue;
    }
    const Slice<Block> bs = blocks(type);
    for (const auto& b : bs) {
      if (b.is_valid()) {
        return static_cast<size_t>(shared) * b.size();
      }
    }
  }
  return 0;
}

void KVCacheState::add_shared_blocks(BlockType type,
                                     std::vector<Block>&& shared_blocks,
                                     size_t current_total_num_tokens) {
  const CachePartitionId& id = writable_partition_id(type);
  if (shared_blocks.empty()) {
    return;
  }
  remember_block_size(id, shared_blocks);
  const size_t matched_blocks = shared_blocks.size();
  std::vector<Block>& bs = composite_blocks_[id];
  uint32_t& shared = num_owned_shared_blocks_[id];
  // The number of matched blocks may be fewer than the number of blocks held by
  // the sequence itself. In this case, try to replace the blocks computed by
  // the sequence with blocks from the prefix_cache and release the computed
  // blocks to save kv_cache as much as possible.
  if (shared_blocks.size() <= bs.size()) {
    try_replace_unique_blocks(std::move(shared_blocks), &shared, &bs);
    num_cached_blocks_[id] = std::max(num_cached_blocks(type), matched_blocks);
    return;
  }

  bs.clear();
  shared = shared_blocks.size();
  bs = std::move(shared_blocks);

  // update the kv cache position
  size_t num_shared_tokens = bs.size() * bs[0].size();
  // It is possible that num_shared_tokens == current_total_num_tokens,
  // indicating that the exact same prompt has been received again. In this
  // case, it becomes necessary to adjust the kv cache position to the
  // previous token, allowing the model proceed. While the shared blocks
  // should be immutable ideally, but it remains safe to regenerate the kv
  // cache in this context, given the utiliztion of the exact same token.
  if (num_shared_tokens == current_total_num_tokens) {
    size_t block_size = bs[0].size();
    CHECK_GT(block_size, 0);
    num_shared_tokens =
        ((current_total_num_tokens - 1) / block_size) * block_size;
    if (shared > 0) {
      shared--;
      Block& released = bs.back();
      released.manager()->release(std::span<Block>(&released, /*size=*/1));
      bs.pop_back();
    }
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
  // `shared` may be one block shorter than the raw match after the
  // exact-repeat pop above. Only the retained prefix is mounted on this
  // sequence, so it is also the cache insertion cursor.
  num_cached_blocks_[id] = shared;
  // update the kv cache position
  kv_cache_tokens_num_ = num_shared_tokens;
}

void KVCacheState::mount_composite_shared(const CachePartitionId& id,
                                          std::vector<Block>&& shared_blocks) {
  if (shared_blocks.empty()) {
    return;
  }
  remember_block_size(id, shared_blocks);
  std::vector<Block>& bs = composite_blocks_[id];
  CHECK(bs.empty()) << "mount_composite_shared: existing blocks under type "
                    << id.model_id << ":" << id.ordinal;
  const size_t count = shared_blocks.size();
  remember_restore_point(id, shared_blocks, count);
  bs = std::move(shared_blocks);
  num_owned_shared_blocks_[id] = static_cast<uint32_t>(count);
  // The probe vector length is the actual matched logical reach for this
  // state and BlockType. For SWA it may include invalid placeholders before
  // the deepest hit; insert() uses this as a position cursor, not as a count
  // of valid physical blocks.
  num_cached_blocks_[id] = count;
}

void KVCacheState::replace_composite_blocks(const CachePartitionId& id,
                                            std::vector<Block>&& blocks,
                                            size_t num_shared_blocks,
                                            size_t cache_publish_cursor) {
  CHECK_LE(num_shared_blocks, blocks.size());
  CHECK_LE(cache_publish_cursor, blocks.size());
  remember_block_size(id, blocks);
  remember_restore_point(id, blocks, num_shared_blocks);
  composite_blocks_[id] = std::move(blocks);
  num_owned_shared_blocks_[id] = static_cast<uint32_t>(num_shared_blocks);
  num_cached_blocks_[id] = cache_publish_cursor;
}

size_t KVCacheState::num_cached_blocks(const CachePartitionId& id) const {
  const auto it = num_cached_blocks_.find(id);
  return it == num_cached_blocks_.end() ? 0 : it->second;
}

std::map<BlockType, size_t> KVCacheState::num_cached_blocks() const {
  std::map<BlockType, size_t> cursors;
  for (const auto& [type, id] : legacy_bindings_) {
    const auto cursor = num_cached_blocks_.find(id);
    if (cursor != num_cached_blocks_.end()) {
      cursors.emplace(type, cursor->second);
    }
  }
  return cursors;
}

void KVCacheState::set_num_cached_blocks(const CachePartitionId& id, size_t n) {
  num_cached_blocks_[id] = n;
}

size_t KVCacheState::current_max_tokens_capacity() const {
  size_t capacity = 0;
  for (const auto& [id, blocks] : composite_blocks_) {
    const auto retention = retentions_.find(id);
    if (blocks.empty() || (retention != retentions_.end() &&
                           retention->second != CacheRetention::HISTORY)) {
      continue;
    }
    const size_t partition_capacity = blocks.size() * block_sizes_.at(id);
    capacity = capacity == 0 ? partition_capacity
                             : std::min(capacity, partition_capacity);
  }
  return capacity;
}

std::vector<std::pair<BlockType, const std::vector<Block>*>>
KVCacheState::multi_block_export_view() const {
  std::vector<std::pair<BlockType, const std::vector<Block>*>> result;
  result.reserve(kMultiBlockExportOrder.size());
  for (const BlockType type : kMultiBlockExportOrder) {
    const auto it = composite_blocks_.find(partition_id(type));
    if (it != composite_blocks_.end() && !it->second.empty()) {
      result.emplace_back(type, &it->second);
    }
  }
  return result;
}

bool KVCacheState::has_multi_block_export() const {
  return std::ranges::any_of(kMultiBlockExportOrder, [this](BlockType type) {
    return num_blocks(type) != 0;
  });
}

int32_t KVCacheState::get_embedding_block_id() const {
  const Slice<Block> bs = blocks(BlockType::EMBEDDING);
  return bs.empty() || !bs[0].is_valid() ? -1 : bs[0].id();
}

int32_t KVCacheState::get_linear_block_id() const {
  const Slice<Block> bs = blocks(BlockType::LINEAR);
  return bs.empty() || !bs.back().is_valid() ? -1 : bs.back().id();
}

void KVCacheState::set_transfer_kv_info(TransferKVInfo&& info) {
  transfer_kv_info_ = std::move(info);
}

std::optional<TransferKVInfo>& KVCacheState::transfer_kv_info() {
  return transfer_kv_info_;
}

size_t KVCacheState::next_transfer_block_idx() const {
  return next_transfer_block_idx_;
}

void KVCacheState::advance_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = std::max(next_transfer_block_idx_, idx);
}

size_t KVCacheState::next_group_transfer_block_idx(
    const CachePartitionId& id) const {
  const auto it = next_group_transfer_block_idxes_.find(id);
  return it == next_group_transfer_block_idxes_.end() ? 0 : it->second;
}

void KVCacheState::advance_group_transfer_block_idx(const CachePartitionId& id,
                                                    size_t idx) {
  size_t& cursor = next_group_transfer_block_idxes_[id];
  cursor = std::max(cursor, idx);
}

void KVCacheState::reset() {
  kv_cache_tokens_num_ = 0;
  prefix_cache_matched_ = false;
  restore_tokens_.clear();
  num_owned_shared_blocks_.clear();
  num_cached_blocks_.clear();
  pushed_local_block_count_ = 0;
  for (auto& [type, blocks] : composite_blocks_) {
    Block::release(std::move(blocks));
  }
  composite_blocks_.clear();
  block_sizes_.clear();
  transfer_kv_info_.reset();
  next_transfer_block_idx_ = 0;
  next_group_transfer_block_idxes_.clear();
}

Slice<Block> KVCacheState::blocks(BlockType type) const {
  return blocks(partition_id(type));
}
std::vector<Block>* KVCacheState::mutable_blocks(BlockType type) {
  return mutable_blocks(writable_partition_id(type));
}
size_t KVCacheState::num_blocks(BlockType type) const {
  return num_blocks(partition_id(type));
}
std::vector<int32_t> KVCacheState::cache_slots(BlockType type,
                                               int32_t pos_start,
                                               int32_t pos_end) {
  return cache_slots(partition_id(type), pos_start, pos_end);
}
void KVCacheState::add_blocks(BlockType type,
                              const std::vector<Block>& blocks) {
  add_blocks(writable_partition_id(type), blocks);
}
void KVCacheState::mount_composite_shared(BlockType type,
                                          std::vector<Block>&& blocks) {
  mount_composite_shared(writable_partition_id(type), std::move(blocks));
}
void KVCacheState::replace_composite_blocks(BlockType type,
                                            std::vector<Block>&& blocks,
                                            size_t shared,
                                            size_t published) {
  replace_composite_blocks(
      writable_partition_id(type), std::move(blocks), shared, published);
}
void KVCacheState::set_shared_blocks_num(BlockType type, size_t num) {
  set_shared_blocks_num(writable_partition_id(type), num);
}
void KVCacheState::erase_blocks(BlockType type) {
  erase_blocks(partition_id(type));
}
std::vector<Block> KVCacheState::take_blocks(BlockType type) {
  return take_blocks(partition_id(type));
}
size_t KVCacheState::shared_blocks_num(BlockType type) const {
  return shared_blocks_num(partition_id(type));
}
size_t KVCacheState::num_cached_blocks(BlockType type) const {
  return num_cached_blocks(partition_id(type));
}
void KVCacheState::set_num_cached_blocks(BlockType type, size_t n) {
  set_num_cached_blocks(writable_partition_id(type), n);
}
size_t KVCacheState::next_group_transfer_block_idx(BlockType type) const {
  return next_group_transfer_block_idx(partition_id(type));
}
void KVCacheState::advance_group_transfer_block_idx(BlockType type,
                                                    size_t idx) {
  advance_group_transfer_block_idx(writable_partition_id(type), idx);
}

}  // namespace xllm
