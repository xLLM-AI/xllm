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

#include "sequence_kv_state.h"

#include <algorithm>

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

size_t KVCacheState::kv_cache_tokens_num() const {
  return kv_cache_tokens_num_;
}

void KVCacheState::set_kv_cache_tokens_num(size_t num) {
  kv_cache_tokens_num_ = num;
}

void KVCacheState::incr_kv_cache_tokens_num(size_t num) {
  CHECK(kv_cache_tokens_num_ + num <= current_max_tokens_capacity());
  kv_cache_tokens_num_ += num;
  slice_window_pos_ += num;
}

void KVCacheState::incr_kv_cache_tokens_num_up_to(size_t new_target) {
  const size_t capacity = current_max_tokens_capacity();
  // Drift check: the counter was already advanced past capacity by an earlier
  // call. Failing here loudly names the producer instead of silently letting
  // the drift propagate to batch_input_builder's CHECK.
  CHECK_LE(kv_cache_tokens_num_, capacity)
      << "kv_cache_tokens_num_ drifted past capacity: " << kv_cache_tokens_num_
      << " > " << capacity;
  CHECK_LE(new_target, capacity)
      << "incr_kv_cache_tokens_num_up_to target above capacity: " << new_target
      << " > " << capacity;
  if (new_target > kv_cache_tokens_num_) {
    const size_t delta = new_target - kv_cache_tokens_num_;
    kv_cache_tokens_num_ = new_target;
    slice_window_pos_ += delta;
  }
}

Slice<Block> KVCacheState::blocks(BlockType type) const {
  const auto it = composite_blocks_.find(type);
  return it == composite_blocks_.end() ? Slice<Block>(empty_blocks())
                                       : Slice<Block>(it->second);
}

std::vector<Block>* KVCacheState::mutable_blocks(BlockType type) {
  return &composite_blocks_[type];
}

size_t KVCacheState::num_blocks(BlockType type) const {
  const auto it = composite_blocks_.find(type);
  return it == composite_blocks_.end() ? 0 : it->second.size();
}

bool KVCacheState::has_any_blocks() const {
  // Cache-bearing types only; SINGLE is a per-sequence resource block, not
  // token cache, and must not count toward "the sequence already holds cache".
  for (const BlockType type :
       {BlockType::KV, BlockType::SWA, BlockType::C4, BlockType::C128}) {
    const auto it = composite_blocks_.find(type);
    if (it != composite_blocks_.end() && !it->second.empty()) {
      return true;
    }
  }
  return false;
}

std::vector<int32_t> KVCacheState::cache_slots(BlockType type,
                                               int32_t pos_start,
                                               int32_t pos_end) {
  const auto it = composite_blocks_.find(type);
  CHECK(it != composite_blocks_.end() && !it->second.empty())
      << "no cache blocks available";
  const std::vector<Block>& bs = it->second;

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = bs[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = bs[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

void KVCacheState::add_blocks(BlockType type,
                              const std::vector<Block>& new_blocks) {
  std::vector<Block>& bs = composite_blocks_[type];
  bs.insert(bs.end(), new_blocks.begin(), new_blocks.end());
}

void KVCacheState::incr_shared_blocks_num(BlockType type, size_t num) {
  uint32_t& shared = num_owned_shared_blocks_[type];
  CHECK(shared + num <= num_blocks(type));
  shared += num;
}

void KVCacheState::erase_blocks(BlockType type) {
  composite_blocks_.erase(type);
  num_owned_shared_blocks_.erase(type);
  if (type == BlockType::LINEAR) {
    pending_linear_save_hash_.reset();
    linear_restore_src_block_.reset();
  }
}

Block KVCacheState::copy_block(BlockType type) const {
  DCHECK(type == BlockType::SINGLE || type == BlockType::LINEAR)
      << "copy_block is for singleton block types only";
  auto it = composite_blocks_.find(type);
  if (it == composite_blocks_.end() || it->second.empty()) {
    return Block();
  }
  return it->second[0];
}

size_t KVCacheState::shared_blocks_num(BlockType type) const {
  const auto it = num_owned_shared_blocks_.find(type);
  return it == num_owned_shared_blocks_.end() ? 0 : it->second;
}

size_t KVCacheState::shared_tokens_num() const {
  // Shared token count is a sequence-level value: shared_blocks * block_size is
  // the same across block types. Read it off whichever type carries shared
  // blocks (KV is the canonical source).
  for (const auto& [type, shared] : num_owned_shared_blocks_) {
    if (shared == 0) {
      continue;
    }
    const Slice<Block> bs = blocks(type);
    if (!bs.empty()) {
      return shared * bs[0].size();
    }
  }
  return 0;
}

void KVCacheState::add_shared_blocks(BlockType type,
                                     std::vector<Block>&& shared_blocks,
                                     size_t current_total_num_tokens) {
  if (shared_blocks.empty()) {
    return;
  }
  std::vector<Block>& bs = composite_blocks_[type];
  uint32_t& shared = num_owned_shared_blocks_[type];
  // The number of matched blocks may be fewer than the number of blocks held by
  // the sequence itself. In this case, try to replace the blocks computed by
  // the sequence with blocks from the prefix_cache and release the computed
  // blocks to save kv_cache as much as possible.
  if (shared_blocks.size() <= bs.size()) {
    try_replace_unique_blocks(std::move(shared_blocks), &shared, &bs);
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
      bs.pop_back();
    }
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
  // update the kv cache position
  kv_cache_tokens_num_ = num_shared_tokens;
}

void KVCacheState::set_slice_window_size(uint32_t size) {
  CHECK(size > 0);
  CHECK(!blocks(BlockType::SWA).empty());
  slice_window_size_ = size;
  slice_window_pos_ = 0;
  slice_window_buffer_ = size;
}

void KVCacheState::update_slice_window_pos() {
  if (slice_window_size_ > 0) {
    if (slice_window_pos_ >= slice_window_buffer_) {
      // Preserve the legacy cursor contract for non-composite block managers.
      // CompositeBlockManager exposes DSA SWA tables as absolute logical
      // columns instead.
      slice_window_pos_ = slice_window_pos_ % slice_window_size_;
    }
  }
}

size_t KVCacheState::current_max_tokens_capacity() const {
  const Slice<Block> kv = blocks(BlockType::KV);
  if (!kv.empty()) {
    // all blocks have the same size
    const size_t block_size = kv[0].size();
    return kv.size() * block_size;
  }
  // DSV4: only the compressed incremental groups (C4 / C128) have a linear
  // token capacity. The SWA ring is excluded on purpose -- its committed tokens
  // keep advancing past ring_capacity * block_size, so counting it here would
  // make incr_kv_cache_tokens_num's CHECK fail.
  size_t capacity = 0;
  for (const BlockType type : {BlockType::C4, BlockType::C128}) {
    const Slice<Block> bs = blocks(type);
    if (bs.empty()) {
      continue;
    }
    const size_t group_capacity = bs.size() * bs[0].size();
    capacity =
        capacity == 0 ? group_capacity : std::min(capacity, group_capacity);
  }
  return capacity;
}

std::vector<std::pair<BlockType, const std::vector<Block>*>>
KVCacheState::multi_block_export_view() const {
  std::vector<std::pair<BlockType, const std::vector<Block>*>> view;
  view.reserve(kMultiBlockExportOrder.size());
  for (const BlockType type : kMultiBlockExportOrder) {
    const auto it = composite_blocks_.find(type);
    if (it != composite_blocks_.end() && !it->second.empty()) {
      view.emplace_back(type, &it->second);
    }
  }
  return view;
}

bool KVCacheState::has_multi_block_export() const {
  for (const BlockType type : kMultiBlockExportOrder) {
    const auto it = composite_blocks_.find(type);
    if (it != composite_blocks_.end() && !it->second.empty()) {
      return true;
    }
  }
  return false;
}

int32_t KVCacheState::get_single_block_id() const {
  const auto it = composite_blocks_.find(BlockType::SINGLE);
  if (it == composite_blocks_.end() || it->second.empty() ||
      !it->second[0].is_valid()) {
    return -1;
  }
  return it->second[0].id();
}

int32_t KVCacheState::get_linear_block_id() const {
  const auto it = composite_blocks_.find(BlockType::LINEAR);
  if (it == composite_blocks_.end() || it->second.empty() ||
      !it->second[0].is_valid()) {
    return -1;
  }
  return it->second[0].id();
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

void KVCacheState::set_next_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = idx;
}

void KVCacheState::advance_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = std::max(next_transfer_block_idx_, idx);
}

size_t KVCacheState::next_group_transfer_block_idx(BlockType type) const {
  const auto it = next_group_transfer_block_idxes_.find(type);
  return it == next_group_transfer_block_idxes_.end() ? 0 : it->second;
}

void KVCacheState::advance_group_transfer_block_idx(BlockType type,
                                                    size_t idx) {
  size_t& cursor = next_group_transfer_block_idxes_[type];
  cursor = std::max(cursor, idx);
}

void KVCacheState::reset() {
  kv_cache_tokens_num_ = 0;
  num_owned_shared_blocks_.clear();
  pushed_local_block_count_ = 0;
  composite_blocks_.clear();
  src_blocks_.clear();
  need_swap_ = false;
  transfer_kv_info_.reset();
  next_transfer_block_idx_ = 0;
  pending_linear_save_hash_.reset();
  linear_restore_src_block_.reset();
  next_group_transfer_block_idxes_.clear();
}

void KVCacheState::process_beam_search(std::optional<Block> new_block) {
  // Beam search only operates on the flat attention KV group.
  std::vector<Block>& kv = composite_blocks_[BlockType::KV];
  kv.clear();
  kv = std::move(src_blocks_);

  if (new_block.has_value()) {
    kv.pop_back();
    kv.emplace_back(new_block.value());
  }
}

}  // namespace xllm
