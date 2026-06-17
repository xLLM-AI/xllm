/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "composite_block_manager.h"

#include <algorithm>
#include <utility>

#include "block_manager_impl.h"
#include "framework/block/block_utils.h"
#include "framework/kv_cache/kv_cache_event.h"
#include "sliding_window_block_manager.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) {
  CHECK_GT(denominator, 0u);
  return (numerator + denominator - 1) / denominator;
}

}  // namespace

CompositeBlockManager::CompositeBlockManager(
    const BlockManager::Options& options)
    : BlockManager(options) {
  const size_t n = options_.manager_types().size();
  CHECK_EQ(n, options_.compress_ratios().size())
      << "manager_types and compress_ratios must have the same size";
  CHECK_GT(n, 0u) << "CompositeBlockManager requires at least one sub-manager";

  sub_managers_.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const uint32_t type = options_.manager_types()[i];
    const uint32_t compress_ratio = options_.compress_ratios()[i];
    BlockManager::Options opts = options_;

    if (type == kManagerTypeBlockManagerImpl) {
      opts.block_size(static_cast<uint32_t>(options_.block_size()) *
                      compress_ratio);
      opts.num_blocks(static_cast<uint32_t>(options_.num_blocks()) /
                      compress_ratio);
      sub_managers_.push_back(std::make_unique<BlockManagerImpl>(opts));
    } else if (type == kManagerTypeSlidingWindowBlockManager) {
      const uint32_t swa_blocks_per_seq = options_.swa_blocks_per_seq();
      CHECK_GT(swa_blocks_per_seq, 0u) << "swa_blocks_per_seq must be positive";
      CHECK_GT(options_.block_size(), 0) << "block_size must be positive";
      const uint32_t sliding_window_size =
          std::max(options_.sliding_window_size(), 1u);
      const uint32_t max_seqs = std::max(options_.max_seqs_per_batch(), 1u);
      const uint32_t burst_blocks =
          ceil_div(std::max(options_.max_tokens_per_batch(), 1u),
                   static_cast<uint32_t>(options_.block_size()));
      const uint32_t swa_total_blocks =
          swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;
      opts.num_blocks(swa_total_blocks)
          .swa_blocks_per_seq(swa_blocks_per_seq)
          .sliding_window_size(sliding_window_size);
      LOG(INFO)
          << "CompositeBlockManager uses dynamic sliding-window allocation: "
             "swa_blocks_per_seq="
          << swa_blocks_per_seq
          << ", sliding_window_size=" << sliding_window_size
          << ", block_size=" << options_.block_size()
          << ", burst_blocks=" << burst_blocks
          << ", total_blocks=" << swa_total_blocks << ", max_seqs=" << max_seqs
          << ". SWA blocks outside the sliding window are returned to the "
             "physical SW cache pool.";
      sub_managers_.push_back(
          std::make_unique<SlidingWindowBlockManager>(opts));
    } else {
      LOG(FATAL) << "Unknown manager_type " << type;
    }
  }
}

bool CompositeBlockManager::allocate_for_sequence(Sequence* seq,
                                                  size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  std::vector<std::vector<Block>>* composite =
      seq->kv_state().mutable_composite_blocks();
  const size_t original_manager_count = composite->size();
  composite->resize(sub_managers_.size());
  std::vector<size_t> original_sizes(composite->size(), 0);
  for (size_t i = 0; i < composite->size(); ++i) {
    original_sizes[i] = composite->at(i).size();
  }

  auto rollback_new_blocks = [&]() {
    for (size_t i = 0; i < composite->size(); ++i) {
      auto& manager_blocks = composite->at(i);
      if (manager_blocks.size() <= original_sizes[i]) {
        continue;
      }
      sub_managers_[i]->deallocate(
          Slice<Block>(manager_blocks.data() + original_sizes[i],
                       manager_blocks.size() - original_sizes[i]));
      manager_blocks.resize(original_sizes[i]);
    }
    composite->resize(original_manager_count);
  };

  auto& swa_blocks = composite->at(0);
  const size_t block_size = sub_managers_[0]->block_size();
  const size_t cached_tokens = seq->kv_state().kv_cache_tokens_num();
  const size_t sliding_window_tokens =
      std::max<size_t>(sub_managers_[0]->options().sliding_window_size(), 1);
  size_t release_blocks = 0;
  if (cached_tokens >= sliding_window_tokens) {
    const size_t skipped_tokens = cached_tokens - sliding_window_tokens + 1;
    const size_t skipped_blocks = skipped_tokens / block_size;
    release_blocks = std::min(skipped_blocks, swa_blocks.size());
  }

  const size_t swa_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (swa_blocks.size() < swa_blocks_needed) {
    const size_t old_size = swa_blocks.size();
    swa_blocks.resize(swa_blocks_needed);
    std::vector<Block> blocks =
        sub_managers_[0]->allocate(swa_blocks_needed - old_size);
    if (blocks.size() != swa_blocks_needed - old_size) {
      swa_blocks.resize(old_size);
      composite->resize(original_manager_count);
      return false;
    }
    std::move(blocks.begin(), blocks.end(), swa_blocks.begin() + old_size);
  }

  for (size_t i = 1; i < sub_managers_.size(); ++i) {
    const size_t num_blocks = composite->at(i).size();
    const size_t block_size = sub_managers_[i]->block_size();
    const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
    if (num_blocks_needed <= num_blocks) {
      continue;
    }

    const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

    const auto blocks = sub_managers_[i]->allocate(num_additional_blocks);
    if (blocks.size() != num_additional_blocks) {
      rollback_new_blocks();
      return false;
    }

    composite->at(i).insert(
        composite->at(i).end(), blocks.begin(), blocks.end());
  }
  if (release_blocks > 0) {
    std::vector<Block> blocks_to_release;
    blocks_to_release.reserve(release_blocks);
    for (size_t j = 0; j < release_blocks; ++j) {
      if (swa_blocks[j].is_valid()) {
        blocks_to_release.emplace_back(std::move(swa_blocks[j]));
      }
    }
    if (!blocks_to_release.empty()) {
      sub_managers_[0]->deallocate(blocks_to_release);
    }
  }
  return true;
}

void CompositeBlockManager::deallocate_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  const std::vector<std::vector<Block>>& composite =
      seq->kv_state().composite_blocks();
  const size_t n = std::min(composite.size(), sub_managers_.size());
  for (size_t i = 0; i < n; ++i) {
    if (!composite[i].empty()) {
      sub_managers_[i]->deallocate(composite[i]);
    }
  }
}

void CompositeBlockManager::deallocate(const Slice<Block>& blocks) {
  if (blocks.empty()) {
    return;
  }

  size_t run_start = 0;
  BlockManager* run_manager = nullptr;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    if (!block.is_valid()) {
      if (run_manager != nullptr) {
        run_manager->deallocate(blocks.slice(run_start, i));
        run_manager = nullptr;
      }
      run_start = i + 1;
      continue;
    }
    BlockManager* manager = block.manager();
    CHECK(manager != nullptr)
        << "CompositeBlockManager got a valid block without owner manager";
    const auto it =
        std::find_if(sub_managers_.begin(),
                     sub_managers_.end(),
                     [manager](const std::unique_ptr<BlockManager>& sub_mgr) {
                       return sub_mgr.get() == manager;
                     });
    CHECK(it != sub_managers_.end())
        << "CompositeBlockManager cannot deallocate block " << block.id()
        << " from a manager outside this composite manager";

    if (run_manager == nullptr) {
      run_manager = manager;
      run_start = i;
    } else if (run_manager != manager) {
      run_manager->deallocate(blocks.slice(run_start, i));
      run_manager = manager;
      run_start = i;
    }
  }
  if (run_manager != nullptr) {
    run_manager->deallocate(blocks.slice(run_start, blocks.size()));
  }
}

std::vector<Block> CompositeBlockManager::allocate(size_t /*num_blocks*/) {
  NOT_IMPLEMENTED();
  return {};
}

std::vector<Block> CompositeBlockManager::allocate_shared(
    const Slice<int32_t>& /*tokens_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/) {
  NOT_IMPLEMENTED();
  return {};
}

void CompositeBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                  std::vector<Block>& /*blocks*/,
                                  size_t /*existed_shared_blocks_num*/,
                                  const MMData& /*mm_data*/) {
  // Prefix cache is not supported for CompositeBlockManager yet.
  // Keep cache() as a no-op so scheduler-side prefill cache hooks
  // do not crash when composite manager is enabled (e.g. DeepSeek V4).
}

void CompositeBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  // Prefix cache is not supported for CompositeBlockManager yet.
}

void CompositeBlockManager::get_merged_kvcache_event(
    KvCacheEvent* event) const {
  for (const auto& mgr : sub_managers_) {
    mgr->get_merged_kvcache_event(event);
  }
}

size_t CompositeBlockManager::num_blocks_in_prefix_cache() const {
  LOG(FATAL) << "CompositeBlockManager not support prefix cache yet";
  return 0;
}

size_t CompositeBlockManager::num_free_blocks() const {
  return sub_managers_[1]->num_free_blocks();
}

size_t CompositeBlockManager::num_used_blocks() const {
  return sub_managers_[1]->num_used_blocks();
}

double CompositeBlockManager::kv_cache_utilization() const {
  return sub_managers_[1]->kv_cache_utilization();
}

void CompositeBlockManager::free(int32_t block_id) {
  LOG(FATAL) << "CompositeBlockManager::free should not be called";
}

Block CompositeBlockManager::allocate() {
  LOG(FATAL) << "CompositeBlockManager::allocate should not be called";
  return Block();
}

size_t CompositeBlockManager::num_total_blocks() const {
  return sub_managers_[1]->num_total_blocks();
}

}  // namespace xllm
