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

#include "hierarchy_block_manager_pool.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <tuple>
#include <unordered_map>

#include "block_manager_impl.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"
#include "sliding_window_block_manager.h"

namespace xllm {

namespace {

using ProbeResult = CompositeBlockManager::ProbeResult;

// Wrap a leaf in the concurrency adapter when the D2H offload callback frees
// blocks off-thread. Host-offload leaves always need this wrap.
std::unique_ptr<BlockManager> wrap_for_offload(std::unique_ptr<BlockManager> l,
                                               const BlockManager::Options& o) {
  if (o.enable_disagg_pd() || o.enable_kvcache_store() ||
      o.enable_host_offload()) {
    return std::make_unique<ConcurrentBlockManagerImpl>(std::move(l));
  }
  return l;
}

ProbeResult* find_probe(std::vector<ProbeResult>* probes, BlockType type) {
  if (probes == nullptr) {
    return nullptr;
  }
  for (ProbeResult& probe : *probes) {
    if (probe.type == type) {
      return &probe;
    }
  }
  return nullptr;
}

void trim_blocks_from_back(BlockManager* leaf,
                           std::vector<Block>* blocks,
                           size_t keep) {
  CHECK(leaf != nullptr);
  CHECK(blocks != nullptr);
  if (blocks->size() <= keep) {
    return;
  }
  std::vector<Block> dropped;
  dropped.reserve(blocks->size() - keep);
  for (size_t i = keep; i < blocks->size(); ++i) {
    dropped.emplace_back(std::move((*blocks)[i]));
  }
  blocks->resize(keep);
  leaf->deallocate(dropped);
}

bool has_valid_swa_window(size_t tokens,
                          size_t block_size,
                          size_t blocks_per_window,
                          const Slice<Block>& hbm_blocks,
                          const Slice<Block>& host_blocks = {}) {
  size_t index = tokens / block_size;
  if (index < blocks_per_window) {
    return false;
  }
  for (size_t i = 0; i < blocks_per_window; ++i) {
    --index;
    const bool hbm_valid =
        index < hbm_blocks.size() && hbm_blocks[index].is_valid();
    const bool host_valid =
        index < host_blocks.size() && host_blocks[index].is_valid();
    if (!hbm_valid && !host_valid) {
      return false;
    }
  }
  return true;
}

struct PrefixMatch {
  size_t restore_tokens = 0;
  size_t hbm_tokens = 0;
  size_t host_tokens = 0;
  size_t copy_units = 0;
};

size_t probe_reach(const ProbeResult& probe, bool sparse) {
  if (sparse) {
    size_t reach = 0;
    for (size_t i = 0; i < probe.blocks.size(); ++i) {
      if (probe.blocks[i].is_valid()) {
        reach = i + 1;
      }
    }
    return reach;
  }
  for (size_t i = 0; i < probe.blocks.size(); ++i) {
    if (!probe.blocks[i].is_valid()) {
      return i;
    }
  }
  return probe.blocks.size();
}

PrefixMatch trim_shared_probes(
    CompositeBlockManager::LeafCombination combination,
    std::vector<ProbeResult>* hbm_probes,
    std::vector<ProbeResult>* host_probes,
    size_t prompt_tokens,
    size_t target_tokens) {
  PrefixMatch match;
  if (hbm_probes == nullptr || hbm_probes->empty() || host_probes == nullptr ||
      host_probes->empty()) {
    return match;
  }

  const size_t max_prefix_tokens = prompt_tokens == 0 ? 0 : prompt_tokens - 1;
  auto cap_and_align = [&](size_t tokens, size_t alignment) {
    CHECK_GT(alignment, 0u);
    const size_t capped = std::min({tokens, target_tokens, max_prefix_tokens});
    return (capped / alignment) * alignment;
  };

  if (combination == CompositeBlockManager::LeafCombination::FLAT_KV) {
    ProbeResult* hbm_kv = find_probe(hbm_probes, BlockType::KV);
    ProbeResult* host_kv = find_probe(host_probes, BlockType::KV);
    CHECK(hbm_kv != nullptr && host_kv != nullptr);
    const size_t block_size = hbm_kv->block_size;
    CHECK_EQ(block_size, host_kv->block_size);
    match.restore_tokens = cap_and_align(
        std::max(probe_reach(*hbm_kv, false), probe_reach(*host_kv, false)) *
            block_size,
        block_size);
    const size_t restore_blocks = match.restore_tokens / block_size;
    match.hbm_tokens =
        std::min(probe_reach(*hbm_kv, false), restore_blocks) * block_size;
    match.host_tokens =
        std::min(probe_reach(*host_kv, false), restore_blocks) * block_size;
    match.copy_units = restore_blocks > probe_reach(*hbm_kv, false)
                           ? restore_blocks - probe_reach(*hbm_kv, false)
                           : 0;
    trim_blocks_from_back(hbm_kv->leaf, &hbm_kv->blocks, restore_blocks);
    return match;
  }

  CHECK(combination == CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
  ProbeResult* hbm_swa = find_probe(hbm_probes, BlockType::SWA);
  ProbeResult* hbm_c4 = find_probe(hbm_probes, BlockType::C4);
  ProbeResult* hbm_c128 = find_probe(hbm_probes, BlockType::C128);
  ProbeResult* host_swa = find_probe(host_probes, BlockType::SWA);
  ProbeResult* host_c4 = find_probe(host_probes, BlockType::C4);
  ProbeResult* host_c128 = find_probe(host_probes, BlockType::C128);
  CHECK(hbm_swa != nullptr && hbm_c4 != nullptr && hbm_c128 != nullptr);
  CHECK(host_swa != nullptr && host_c4 != nullptr && host_c128 != nullptr);

  const size_t swa_block_size = hbm_swa->block_size;
  const size_t c128_block_size = hbm_c128->block_size;

  const size_t blocks_per_window =
      static_cast<size_t>(hbm_swa->leaf->options().swa_blocks_per_seq());

  size_t max_matched_tokens = cap_and_align(
      std::min(
          {std::max(probe_reach(*hbm_swa, true), probe_reach(*host_swa, true)) *
               swa_block_size,
           std::max(probe_reach(*hbm_c4, false), probe_reach(*host_c4, false)) *
               hbm_c4->block_size,
           std::max(probe_reach(*hbm_c128, false),
                    probe_reach(*host_c128, false)) *
               c128_block_size}),
      c128_block_size);
  while (max_matched_tokens > 0 && !has_valid_swa_window(max_matched_tokens,
                                                         swa_block_size,
                                                         blocks_per_window,
                                                         hbm_swa->blocks,
                                                         host_swa->blocks)) {
    max_matched_tokens -= c128_block_size;
  }

  auto tier_matched_tokens = [&](const ProbeResult& swa,
                                 const ProbeResult& c4,
                                 const ProbeResult& c128) {
    size_t tokens =
        cap_and_align(std::min({probe_reach(swa, true) * swa_block_size,
                                probe_reach(c4, false) * c4.block_size,
                                probe_reach(c128, false) * c128_block_size}),
                      c128_block_size);
    tokens = std::min(tokens, max_matched_tokens);
    while (tokens > 0 &&
           !has_valid_swa_window(
               tokens, swa_block_size, blocks_per_window, swa.blocks)) {
      tokens -= c128_block_size;
    }
    return tokens;
  };

  const size_t max_hbm_matched_tokens =
      tier_matched_tokens(*hbm_swa, *hbm_c4, *hbm_c128);
  const size_t max_host_matched_tokens =
      tier_matched_tokens(*host_swa, *host_c4, *host_c128);

  const size_t restore_units = max_matched_tokens / c128_block_size;
  match.restore_tokens = max_matched_tokens;
  match.hbm_tokens = max_hbm_matched_tokens;
  match.host_tokens = max_host_matched_tokens;
  match.copy_units = restore_units > probe_reach(*hbm_c128, false)
                         ? restore_units - probe_reach(*hbm_c128, false)
                         : 0;

  trim_blocks_from_back(
      hbm_swa->leaf, &hbm_swa->blocks, max_matched_tokens / swa_block_size);
  trim_blocks_from_back(
      hbm_c4->leaf, &hbm_c4->blocks, max_matched_tokens / hbm_c4->block_size);
  trim_blocks_from_back(hbm_c128->leaf, &hbm_c128->blocks, restore_units);
  return match;
}

}  // namespace

HierarchyBlockManagerPool::HierarchyBlockManagerPool(
    const BlockManagerPool::Options& options,
    Engine* engine,
    int32_t dp_size)
    : engine_(engine), BlockManagerPool(options, dp_size) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  host_block_managers_.reserve(dp_size);

  for (int32_t i = 0; i < dp_size; ++i) {
    CompositeBlockManager::LeafMap per_type;
    auto* composite =
        static_cast<CompositeBlockManager*>(block_managers_[i].get());
    const CompositeBlockManager::LeafCombination combination =
        composite->leaf_combination();
    switch (combination) {
      case CompositeBlockManager::LeafCombination::FLAT_KV:
      case CompositeBlockManager::LeafCombination::SWA_COMPRESSED:
        break;
      case CompositeBlockManager::LeafCombination::FLAT_KV_LINEAR:
      case CompositeBlockManager::LeafCombination::UNSUPPORTED:
      default:
        LOG(FATAL) << "HierarchyBlockManagerPool supports only FLAT_KV and "
                      "SWA_COMPRESSED cache layouts; got "
                   << static_cast<int32_t>(combination);
    }
    const CompositeBlockManager::LeafMap& device_leaves =
        composite->leaf_entries();
    std::map<BlockType, uint32_t> host_capacities =
        options_.host_num_blocks_by_type();
    if (host_capacities.empty() && options_.host_num_blocks() > 0) {
      host_capacities.emplace(BlockType::KV, options_.host_num_blocks());
    }
    for (const auto& [type, num_blocks] : host_capacities) {
      const auto device_leaf_it = device_leaves.find(type);
      if (device_leaf_it == device_leaves.end() || num_blocks == 0) {
        continue;
      }
      BlockManager::Options host_options =
          device_leaf_it->second.leaf->options();
      host_options.num_blocks(num_blocks)
          .enable_disagg_pd(options_.enable_disagg_pd())
          .enable_kvcache_store(options_.enable_kvcache_store())
          .enable_host_offload(options_.enable_host_offload());

      std::unique_ptr<BlockManager> leaf;
      if (type == BlockType::SWA) {
        leaf = std::make_unique<SlidingWindowBlockManager>(host_options);
      } else {
        leaf = std::make_unique<BlockManagerImpl>(host_options);
      }
      leaf = wrap_for_offload(std::move(leaf), host_options);
      per_type.emplace(
          type,
          CompositeBlockManager::LeafEntry{
              std::move(leaf),
              /*participates_in_admission=*/false,
              /*supports_prefix_cache=*/host_options.enable_prefix_cache()});
    }
    host_block_managers_.emplace_back(std::move(per_type));
  }

  load_block_transfer_infos_.resize(host_block_managers_.size());
  offload_block_pair_queues_.resize(host_block_managers_.size());
}

void HierarchyBlockManagerPool::release_host_match(Sequence* sequence,
                                                   int32_t dp_rank) {
  CHECK(sequence != nullptr);
  KVCacheState& host_state = sequence->host_kv_state();
  for (const auto& [type, entry] : host_block_managers_[dp_rank]) {
    const Slice<Block> blocks = host_state.blocks(type);
    if (!blocks.empty()) {
      entry.leaf->deallocate(blocks);
    }
  }
  host_state.reset();
  sequence->clear_host_cache_match();
}

void HierarchyBlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->cache_full_blocks_for_sequence(sequence);

  collect_offload_pairs(
      sequence, dp_rank, sequence->kv_state().kv_cache_tokens_num());

  // Release the host blocks still held by the sequence. Blocks moved into the
  // offload queue are now invalid in this vector and are skipped by
  // deallocate; their host ids stay reserved (held by the queue) until the
  // D2H copy completes and the offload callback caches + frees them.
  for (const auto& [type, entry] : host_block_managers_[dp_rank]) {
    const Slice<Block> host_blocks = sequence->host_kv_state().blocks(type);
    if (!host_blocks.empty()) {
      entry.leaf->deallocate(host_blocks);
    }
  }

  // Release device blocks via the composite (includes prefix cache flush).
  composite->deallocate_for_sequence(sequence);
  sequence->reset();
}

void HierarchyBlockManagerPool::collect_offload_pairs(Sequence* sequence,
                                                      int32_t dp_rank,
                                                      size_t completed_tokens) {
  if (!options_.enable_prefix_cache()) {
    return;
  }

  KVCacheState& hbm_state = sequence->kv_state();
  KVCacheState& host_state = sequence->host_kv_state();
  for (const auto& [type, entry] : host_block_managers_[dp_rank]) {
    std::vector<Block>* hbm_blocks = hbm_state.mutable_blocks(type);
    std::vector<Block>* host_blocks = host_state.mutable_blocks(type);
    const size_t block_size = entry.leaf->block_size();
    CHECK_GT(block_size, 0u);
    const size_t completed_blocks = completed_tokens / block_size;
    const size_t comparable_blocks =
        std::min({hbm_blocks->size(), host_blocks->size(), completed_blocks});
    for (size_t i = 0; i < comparable_blocks; ++i) {
      Block& hbm_block = (*hbm_blocks)[i];
      Block& host_block = (*host_blocks)[i];
      // Prefix-capable HBM leaves are held by both the sequence and the device
      // prefix cache. Decode SWA deliberately skips prefix insertion, so its
      // completed block is sequence-only until this offload pair retains it.
      // Host is the sequence's unfilled destination block in both cases.
      const uint32_t expected_hbm_refs = entry.supports_prefix_cache ? 2 : 1;
      if (hbm_block.ref_count() != expected_hbm_refs ||
          host_block.ref_count() != 1) {
        continue;
      }
      host_block.set_hash_value(hbm_block.get_immutable_hash_value());
      auto pair = std::make_shared<OffloadBlockPair>(
          OffloadBlockPair{/*src=*/hbm_block,
                           /*dst=*/std::move(host_block),
                           /*block_type=*/type});
      offload_block_pair_queues_[dp_rank].enqueue(std::move(pair));
    }
  }
}

void HierarchyBlockManagerPool::trim_host_cache(
    Sequence* sequence,
    const HostCacheRestorePoint& selected_restore) {
  CHECK(sequence != nullptr);
  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const size_t restore_tokens = selected_restore.restore_target_tokens;
  KVCacheState& hbm_state = sequence->kv_state();
  KVCacheState& host_state = sequence->host_kv_state();
  const size_t matched_tokens = sequence->kv_cache_tokens_num();
  CHECK_LE(hbm_state.kv_cache_tokens_num(), restore_tokens);
  CHECK_LE(restore_tokens, matched_tokens);

  const auto* composite =
      static_cast<const CompositeBlockManager*>(block_managers_[dp_rank].get());
  for (const auto& [type, entry] : composite->leaf_entries()) {
    if (type == BlockType::EMBEDDING || type == BlockType::LINEAR) {
      continue;
    }
    const size_t block_size = entry.leaf->block_size();
    CHECK_GT(block_size, 0u);
    const size_t keep = restore_tokens / block_size;
    host_state.set_num_cached_blocks(
        type, std::min(host_state.num_cached_blocks(type), keep));
    const Slice<Block> current = hbm_state.blocks(type);
    if (current.size() <= keep) {
      continue;
    }

    const size_t shared = std::min(hbm_state.shared_blocks_num(type), keep);
    const size_t cached = std::min(hbm_state.num_cached_blocks(type), keep);
    std::vector<Block> blocks = std::move(*hbm_state.mutable_blocks(type));
    trim_blocks_from_back(entry.leaf.get(), &blocks, keep);
    if (blocks.empty()) {
      hbm_state.erase_blocks(type);
    } else {
      hbm_state.replace_composite_blocks(
          type, std::move(blocks), shared, cached);
    }
  }

  host_state.set_kv_cache_tokens_num(
      std::min(host_state.kv_cache_tokens_num(), restore_tokens));
  sequence->set_host_cache_restore(restore_tokens, selected_restore.copy_units);
}

bool HierarchyBlockManagerPool::allocate(Sequence* sequence,
                                         size_t num_tokens) {
  CHECK(sequence != nullptr);
  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  if (should_probe_prefix_cache(sequence)) {
    allocate_shared(sequence);
  }

  const size_t restore_tokens = sequence->kv_cache_tokens_num();
  KVCacheState& hbm_state = sequence->kv_state();
  CHECK_LE(hbm_state.kv_cache_tokens_num(), restore_tokens);
  CHECK_LE(restore_tokens, num_tokens);
  std::map<BlockType, size_t> previous_hbm_cached_blocks =
      hbm_state.num_cached_blocks();

  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  if (!composite->allocate_sequence(sequence, num_tokens)) {
    release_host_match(sequence, dp_rank);
    return false;
  }

  KVCacheState& host_state = sequence->host_kv_state();

  // Host growth is best-effort and must not reject an HBM allocation that has
  // already succeeded. Stage by BlockType so partial Host growth can be
  // released without touching either cache state.
  std::map<BlockType, std::vector<Block>> staged;

  auto release_staged = [&]() {
    for (auto& [type, blocks] : staged) {
      host_block_managers_[dp_rank].at(type).leaf->deallocate(blocks);
    }
    staged.clear();
  };

  for (auto& [type, entry] : host_block_managers_[dp_rank]) {
    std::optional<std::vector<Block>> blocks =
        entry.leaf->allocate_for_sequence(sequence, host_state, num_tokens);
    if (!blocks.has_value()) {
      release_staged();
      break;
    }
    if (!blocks->empty()) {
      staged.emplace(type, std::move(*blocks));
    }
    const size_t leaf_block_size = entry.leaf->block_size();
    const size_t needed = (num_tokens + leaf_block_size - 1) / leaf_block_size;
    const auto staged_it = staged.find(type);
    const size_t staged_for_type =
        staged_it == staged.end() ? 0 : staged_it->second.size();
    const size_t total = host_state.num_blocks(type) + staged_for_type;
    if (total < needed) {
      release_staged();
      break;
    }
  }

  for (auto& [type, blocks] : staged) {
    host_state.add_blocks(type, blocks);
  }
  staged.clear();

  for (auto& [type, entry] : host_block_managers_[dp_rank]) {
    entry.leaf->release_out_of_window(sequence, host_state);
  }

  std::vector<BlockTransferInfo>& load_infos =
      load_block_transfer_infos_[dp_rank];
  for (const auto& [type, entry] : host_block_managers_[dp_rank]) {
    std::vector<Block>* host_blocks = host_state.mutable_blocks(type);
    std::vector<Block>* hbm_blocks = hbm_state.mutable_blocks(type);
    const auto hbm_matched_it = previous_hbm_cached_blocks.find(type);
    const size_t hbm_matched_blocks =
        hbm_matched_it == previous_hbm_cached_blocks.end()
            ? 0
            : hbm_matched_it->second;
    const size_t host_matched_blocks = host_state.num_cached_blocks(type);
    const size_t comparable_blocks =
        std::min(host_blocks->size(), hbm_blocks->size());
    for (size_t i = 0; i < comparable_blocks; ++i) {
      Block& host_block = (*host_blocks)[i];
      Block& hbm_block = (*hbm_blocks)[i];
      if (i < hbm_matched_blocks || i >= host_matched_blocks ||
          hbm_block.ref_count() != 2 || host_block.ref_count() != 2) {
        continue;
      }
      host_block.set_hash_value(hbm_block.get_immutable_hash_value());
      load_infos.emplace_back(host_block.id(),
                              hbm_block.id(),
                              hbm_block.get_immutable_hash_value(),
                              TransferType::H2D,
                              type);
    }
  }

  collect_offload_pairs(sequence, dp_rank, restore_tokens);
  CHECK_GE(hbm_state.current_max_tokens_capacity(), restore_tokens);
  hbm_state.set_kv_cache_tokens_num(restore_tokens);
  sequence->clear_host_cache_match();
  return true;
}

void HierarchyBlockManagerPool::allocate_shared(Sequence* sequence) {
  CHECK(sequence != nullptr);
  if (!should_probe_prefix_cache(sequence)) {
    return;
  }

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());

  // Decode participates in the device prefix cache only. In particular, the
  // DSV4 decode layout probes C4/C128 (SWA is disabled by the composite role
  // predicate), but must never mount Host aliases or schedule H2D restores.
  if (sequence->stage() == SequenceStage::DECODE) {
    composite->allocate_shared_for_sequence(sequence);
    return;
  }

  KVCacheState& hbm_state = sequence->kv_state();
  KVCacheState& host_state = sequence->host_kv_state();
  std::vector<ProbeResult> hbm_probes;
  std::vector<ProbeResult> host_probes;

  if (hbm_state.prefix_cache_matched()) {
    for (const auto& [type, entry] : composite->leaf_entries()) {
      if (!entry.supports_prefix_cache || entry.leaf == nullptr) {
        continue;
      }
      hbm_probes.emplace_back(ProbeResult{type,
                                          entry.leaf.get(),
                                          hbm_state.take_blocks(type),
                                          entry.leaf->block_size()});
    }
  } else {
    hbm_probes = CompositeBlockManager::probe_prefix_cache(
        sequence, composite->leaf_entries(), hbm_state);
  }

  if (host_state.prefix_cache_matched()) {
    for (const auto& [type, entry] : host_block_managers_[dp_rank]) {
      if (!entry.supports_prefix_cache || entry.leaf == nullptr) {
        continue;
      }
      host_probes.emplace_back(ProbeResult{type,
                                           entry.leaf.get(),
                                           host_state.take_blocks(type),
                                           entry.leaf->block_size()});
    }
  } else {
    host_probes = CompositeBlockManager::probe_prefix_cache(
        sequence, host_block_managers_[dp_rank], host_state);
  }

  PrefixMatch match = trim_shared_probes(composite->leaf_combination(),
                                         &hbm_probes,
                                         &host_probes,
                                         sequence->tokens().size(),
                                         sequence->num_tokens());
  VLOG(1) << "[HostCache][PrefixMatch] sequence_id=" << sequence->seq_id()
          << " hbm_tokens=" << match.hbm_tokens
          << " host_tokens=" << match.host_tokens;

  sequence->clear_host_cache_match();
  for (ProbeResult& probe : host_probes) {
    host_state.mount_composite_shared(probe.type, std::move(probe.blocks));
  }
  CHECK_GE(host_state.current_max_tokens_capacity(), match.host_tokens);
  host_state.incr_kv_cache_tokens_num_up_to(match.host_tokens);

  for (ProbeResult& probe : hbm_probes) {
    hbm_state.mount_composite_shared(probe.type, std::move(probe.blocks));
  }
  CHECK_LE(hbm_state.kv_cache_tokens_num(), match.hbm_tokens);
  CHECK_GE(hbm_state.current_max_tokens_capacity(), match.hbm_tokens);
  hbm_state.set_kv_cache_tokens_num(match.hbm_tokens);
  CompositeBlockManager::release_probes(&hbm_probes);
  CompositeBlockManager::release_probes(&host_probes);
  hbm_state.set_prefix_cache_matched();
  host_state.set_prefix_cache_matched();

  if (match.restore_tokens > match.hbm_tokens) {
    sequence->set_host_cache_match(match.restore_tokens, match.copy_units);
  }
}

HostCacheRestorePoint HierarchyBlockManagerPool::select_host_cache_restore(
    Sequence* sequence,
    size_t max_copy_units) {
  CHECK(sequence != nullptr);
  const size_t restore_tokens = sequence->kv_cache_tokens_num();

  const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const auto* composite =
      static_cast<const CompositeBlockManager*>(block_managers_[dp_rank].get());
  const bool is_swa_compressed =
      composite->leaf_combination() ==
      CompositeBlockManager::LeafCombination::SWA_COMPRESSED;
  const BlockType unit_type =
      is_swa_compressed ? BlockType::C128 : BlockType::KV;
  const auto unit_leaf = composite->leaf_entries().find(unit_type);
  CHECK(unit_leaf != composite->leaf_entries().end());
  const size_t copy_unit_tokens = unit_leaf->second.leaf->block_size();
  CHECK_GT(copy_unit_tokens, 0u);
  const size_t restore_units = restore_tokens / copy_unit_tokens;
  const size_t hbm_cached_units = std::min(
      sequence->kv_state().num_cached_blocks(unit_type), restore_units);
  const size_t full_copy_units = restore_units - hbm_cached_units;
  if (full_copy_units == 0) {
    return HostCacheRestorePoint{/*restore_target_tokens=*/restore_tokens,
                                 /*copy_units=*/0};
  }

  size_t selected_copy_units = std::min(max_copy_units, full_copy_units);
  size_t selected_tokens =
      (hbm_cached_units + selected_copy_units) * copy_unit_tokens;

  if (!is_swa_compressed) {
    return HostCacheRestorePoint{/*restore_target_tokens=*/selected_tokens,
                                 /*copy_units=*/selected_copy_units};
  }

  const auto swa_leaf = composite->leaf_entries().find(BlockType::SWA);
  CHECK(swa_leaf != composite->leaf_entries().end());
  const size_t swa_block_size = swa_leaf->second.leaf->block_size();
  const size_t blocks_per_window = static_cast<size_t>(
      swa_leaf->second.leaf->options().swa_blocks_per_seq());
  CHECK_GT(swa_block_size, 0u);
  CHECK_GT(blocks_per_window, 0u);

  const Slice<Block> hbm_swa = sequence->kv_state().blocks(BlockType::SWA);
  const Slice<Block> host_swa =
      sequence->host_kv_state().blocks(BlockType::SWA);
  while (selected_copy_units > 0 && !has_valid_swa_window(selected_tokens,
                                                          swa_block_size,
                                                          blocks_per_window,
                                                          hbm_swa,
                                                          host_swa)) {
    --selected_copy_units;
    selected_tokens -= copy_unit_tokens;
  }

  return HostCacheRestorePoint{/*restore_target_tokens=*/selected_tokens,
                               /*copy_units=*/selected_copy_units};
}

bool HierarchyBlockManagerPool::should_probe_prefix_cache(
    Sequence* sequence) const {
  if (!options_.enable_prefix_cache() || sequence == nullptr) {
    return false;
  }
  // Decode only has a device cache.  Prefill may enter with Host already
  // matched by Mooncake, so the tier completion flags, rather than block
  // presence, define whether another probe is needed.
  KVCacheState& hbm_state = sequence->kv_state();
  if (sequence->stage() == SequenceStage::DECODE) {
    return !hbm_state.prefix_cache_matched();
  }
  if (!hbm_state.prefix_cache_matched()) {
    return true;
  }
  // A Host match released after an HBM allocation failure may be re-probed
  // while HBM still contains only its immutable prefix aliases. Once forward
  // has advanced beyond that prefix, re-probing would detach computed DSV4
  // SWA/C4/C128 blocks and replace the cursor with a shorter cache match.
  if (hbm_state.kv_cache_tokens_num() > hbm_state.shared_tokens_num()) {
    return false;
  }
  return !sequence->host_kv_state().prefix_cache_matched();
}

BlockManager* HierarchyBlockManagerPool::leaf_of(BlockType type,
                                                 int32_t dp_rank) const {
  auto it = host_block_managers_[dp_rank].find(type);
  return it == host_block_managers_[dp_rank].end() ? nullptr
                                                   : it->second.leaf.get();
}

void HierarchyBlockManagerPool::prefetch_from_storage(
    std::shared_ptr<Request>& request) {
  if (!options_.enable_kvcache_store()) {
    return;
  }

  for (auto& prefill_sequence : request->sequences()) {
    Sequence* sequence = prefill_sequence.get();
    CHECK(sequence != nullptr);
    CHECK(!sequence->has_any_blocks())
        << "Mooncake prefetch admission requires an empty Sequence state.";

    const int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
    auto plan = std::make_shared<PrefetchPlan>();
    plan->sequence = sequence;

    plan->host_probes = CompositeBlockManager::probe_prefix_cache(
        sequence, host_block_managers_[dp_rank]);
    for (size_t probe_index = 0; probe_index < plan->host_probes.size();
         ++probe_index) {
      ProbeResult& probe = plan->host_probes[probe_index];
      CHECK(probe.leaf != nullptr);
      const size_t full_blocks = sequence->tokens().size() / probe.block_size;
      if (probe.blocks.size() > full_blocks) {
        trim_blocks_from_back(probe.leaf, &probe.blocks, full_blocks);
      }
      probe.blocks.resize(full_blocks);

      std::vector<size_t> missing;
      for (size_t block_index = 0; block_index < full_blocks; ++block_index) {
        if (!probe.blocks[block_index].is_valid()) {
          missing.emplace_back(block_index);
        }
      }
      if (missing.empty()) {
        continue;
      }

      const size_t requested_blocks = missing.size();
      size_t allocatable_blocks =
          std::min(requested_blocks,
                   probe.leaf->num_free_blocks() +
                       probe.leaf->num_blocks_in_prefix_cache());
      std::vector<Block> allocated = probe.leaf->allocate(allocatable_blocks);
      if (allocated.empty() && allocatable_blocks > 0) {
        allocatable_blocks =
            std::min(requested_blocks, probe.leaf->num_free_blocks());
        allocated = probe.leaf->allocate(allocatable_blocks);
      }
      if (allocated.empty()) {
        LOG(WARNING) << "[Mooncake][Prefetch] insufficient Host blocks: seq="
                     << sequence->seq_id()
                     << ", type=" << static_cast<int32_t>(probe.type)
                     << ", requested=" << requested_blocks << ", allocated=0";
        continue;
      }
      CHECK_EQ(allocated.size(), allocatable_blocks);
      if (allocated.size() < requested_blocks) {
        LOG(WARNING) << "[Mooncake][Prefetch] partial Host allocation: seq="
                     << sequence->seq_id()
                     << ", type=" << static_cast<int32_t>(probe.type)
                     << ", requested=" << requested_blocks
                     << ", allocated=" << allocated.size();
      }
      sequence->update_block_hashes(probe.block_size,
                                    probe.leaf->options().hasher_type());
      const Slice<XXH3Key> hashes = sequence->block_hashes();
      CHECK_GE(hashes.size(), full_blocks);
      for (size_t i = 0; i < allocated.size(); ++i) {
        const size_t block_index = missing[i];
        allocated[i].set_hash_value(hashes[block_index].data);
        probe.blocks[block_index] = std::move(allocated[i]);
        plan->queries.emplace_back(PrefetchQuery{
            probe_index, block_index, 0, block_index * probe.block_size});
      }
    }

    std::sort(
        plan->queries.begin(),
        plan->queries.end(),
        [](const PrefetchQuery& lhs, const PrefetchQuery& rhs) {
          return std::tie(lhs.token_start, lhs.probe_index, lhs.block_index) <
                 std::tie(rhs.token_start, rhs.probe_index, rhs.block_index);
        });

    std::vector<BlockTransferInfo> transfer_infos;
    transfer_infos.reserve(plan->queries.size());
    for (size_t result_index = 0; result_index < plan->queries.size();
         ++result_index) {
      PrefetchQuery& query = plan->queries[result_index];
      ProbeResult& probe = plan->host_probes[query.probe_index];
      Block& block = probe.blocks[query.block_index];
      query.result_index = result_index;
      transfer_infos.emplace_back(/*src_id=*/-1,
                                  /*dst_id=*/block.id(),
                                  block.get_immutable_hash_value(),
                                  TransferType::G2H,
                                  probe.type);
    }

    if (transfer_infos.empty()) {
      release_prefetch_plan(plan.get(), /*publish_store_hits=*/true);
      continue;
    }

    CHECK(engine_ != nullptr) << "Mooncake prefetch requires an Engine.";
    plan->result = engine_->prefetch_from_storage(dp_rank, transfer_infos);
    CHECK(plan->result != nullptr);
    {
      std::lock_guard<std::mutex> lock(prefetch_plans_mutex_);
      const bool inserted = prefetch_plans_.emplace(sequence, plan).second;
      CHECK(inserted) << "Duplicate Mooncake prefetch plan for sequence "
                      << sequence->seq_id();
    }
  }
}

bool HierarchyBlockManagerPool::update_prefetch_result(
    std::shared_ptr<Request>& request,
    const uint32_t timeout) {
  if (!options_.enable_kvcache_store()) {
    return true;
  }

  bool all_completed = true;
  for (auto& prefill_sequence : request->sequences()) {
    Sequence* sequence = prefill_sequence.get();
    std::shared_ptr<PrefetchPlan> plan;
    {
      std::lock_guard<std::mutex> lock(prefetch_plans_mutex_);
      const auto it = prefetch_plans_.find(sequence);
      if (it != prefetch_plans_.end()) {
        plan = it->second;
      }
    }
    if (plan == nullptr) {
      continue;
    }
    const bool discard_result = request->finished() || request->cancelled();
    if (!plan->result->completed()) {
      if (discard_result && plan->result->request_stop()) {
        VLOG(1) << "[Mooncake][Prefetch] cancelled sequence "
                << sequence->seq_id()
                << "; stop requested, waiting for every TP worker's "
                   "in-flight batch to finish.";
      } else if (timeout > 0 && plan->timer.elapsed_milliseconds() >= timeout &&
                 plan->result->request_stop()) {
        LOG(WARNING) << "[Mooncake][Prefetch] timeout after " << timeout
                     << " ms for sequence " << sequence->seq_id()
                     << "; stop requested, waiting for every TP worker's "
                        "in-flight batch to finish.";
      }
      all_completed = false;
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(prefetch_plans_mutex_);
      const size_t erased = prefetch_plans_.erase(sequence);
      CHECK_EQ(erased, 1u);
    }
    release_prefetch_plan(plan.get(),
                          /*publish_store_hits=*/!discard_result);
  }

  return all_completed;
}

void HierarchyBlockManagerPool::release_prefetch_plan(PrefetchPlan* plan,
                                                      bool publish_store_hits) {
  CHECK(plan != nullptr);
  CHECK(plan->sequence != nullptr);
  if (!publish_store_hits) {
    for (ProbeResult& probe : plan->host_probes) {
      CHECK(probe.leaf != nullptr);
      probe.leaf->deallocate(probe.blocks);
      probe.blocks.clear();
    }
    return;
  }

  std::vector<uint8_t> merged_hits;
  if (plan->result != nullptr) {
    merged_hits = plan->result->merged_hits();
  }

  for (const PrefetchQuery& query : plan->queries) {
    ProbeResult& probe = plan->host_probes[query.probe_index];
    CHECK_LT(query.block_index, probe.blocks.size());
    const bool hit = query.result_index < merged_hits.size() &&
                     merged_hits[query.result_index] != 0;
    if (!hit) {
      std::vector<Block> miss;
      miss.emplace_back(std::move(probe.blocks[query.block_index]));
      probe.leaf->deallocate(miss);
    }
  }

  KVCacheState& host_state = plan->sequence->host_kv_state();
  for (ProbeResult& probe : plan->host_probes) {
    CHECK(probe.leaf != nullptr);
    if (!probe.blocks.empty()) {
      // PrefixCache::insert skips invalid SWA placeholders. Valid Store hits
      // are inserted before the vector is shortened for the usable prefix.
      probe.leaf->cache(probe.blocks);
    }

    const bool sparse = probe.type == BlockType::SWA;
    const size_t reach = probe_reach(probe, sparse);
    if (probe.blocks.size() > reach) {
      trim_blocks_from_back(probe.leaf, &probe.blocks, reach);
    }
    if (!probe.blocks.empty()) {
      host_state.mount_composite_shared(probe.type, std::move(probe.blocks));
    }

    VLOG(1) << "[Mooncake][PrefetchComplete] type="
            << static_cast<int32_t>(probe.type) << ", reach=" << reach;
  }
  host_state.set_prefix_cache_matched();
}

void HierarchyBlockManagerPool::transfer_blocks(std::vector<Batch>& batches) {
  for (size_t i = 0; i < load_block_transfer_infos_.size(); ++i) {
    if (load_block_transfer_infos_[i].empty()) {
      continue;
    }
    CHECK_LT(i, batches.size())
        << "Missing batch for pending H2D transfer at dp_rank=" << i;
    batches[i].set_batch_id();
    engine_->transfer_kv_blocks(
        i, batches[i].batch_id(), std::move(load_block_transfer_infos_[i]));
    load_block_transfer_infos_[i].clear();
  }

  transfer_offload_blocks();
}

void HierarchyBlockManagerPool::transfer_blocks() { transfer_offload_blocks(); }

bool HierarchyBlockManagerPool::has_pending_async_block_release() const {
  if (pending_offload_transfers_.load(std::memory_order_relaxed) > 0) {
    return true;
  }
  for (const OffloadBlockPairQueue& queue : offload_block_pair_queues_) {
    if (queue.size_approx() > 0) {
      return true;
    }
  }
  return false;
}

void HierarchyBlockManagerPool::transfer_offload_blocks() {
  for (size_t i = 0; i < offload_block_pair_queues_.size(); i++) {
    std::vector<BlockTransferInfo> transfer_infos;
    std::vector<Block> src_blocks;
    std::vector<Block> dst_blocks;
    std::vector<BlockType> block_types;

    std::shared_ptr<OffloadBlockPair> block_pair;
    while (offload_block_pair_queues_[i].try_dequeue(block_pair)) {
      src_blocks.emplace_back(std::move(block_pair->src));
      dst_blocks.emplace_back(std::move(block_pair->dst));
      transfer_infos.emplace_back(
          BlockTransferInfo(src_blocks.back().id(),
                            dst_blocks.back().id(),
                            dst_blocks.back().get_immutable_hash_value(),
                            TransferType::D2H2G));
      // Preserve the BlockType so the completion callback publishes to the
      // right host leaf. The engine transfer path stamps the outbound info's
      // block_type from device layer coverage; this side just needs it to
      // route publish/free.
      transfer_infos.back().block_type = block_pair->block_type;
      block_types.emplace_back(block_pair->block_type);
      block_pair.reset();
    }

    if (!transfer_infos.empty()) {
      // Capture per-leaf host manager pointers so the completion callback can
      // route each block to the correct host leaf on publish.
      std::unordered_map<BlockType, BlockManager*> host_leaves_snapshot;
      for (const auto& [type, entry] : host_block_managers_[i]) {
        host_leaves_snapshot.emplace(type, entry.leaf.get());
      }
      pending_offload_transfers_.fetch_add(1, std::memory_order_relaxed);
      std::atomic<size_t>* pending_offload_transfers =
          &pending_offload_transfers_;
      folly::collectAll(
          std::move(engine_->transfer_kv_blocks(i, std::move(transfer_infos))))
          .via(folly::getGlobalCPUExecutor())
          .thenValue([device_blocks = std::move(src_blocks),
                      host_blocks = std::move(dst_blocks),
                      block_types_vec = std::move(block_types),
                      device_block_mgr_ptr = block_managers_[i].get(),
                      host_leaves = std::move(host_leaves_snapshot)](
                         std::vector<folly::Try<uint32_t>>&& results) mutable {
            bool copy_ok = true;
            for (auto&& result : results) {
              if (result.hasException()) {
                LOG(ERROR) << "Offload RPC failed: "
                           << result.exception().what();
                copy_ok = false;
                continue;
              }
              if (result.value() != host_blocks.size()) {
                LOG(ERROR) << "Offload copy fail, expected "
                           << host_blocks.size() << ", got " << result.value();
                copy_ok = false;
              }
            }

            // Always release the reserved ids so the block pools do not leak.
            device_block_mgr_ptr->deallocate(device_blocks);
            device_blocks.clear();

            if (copy_ok) {
              // Group host blocks by type and cache/free them on the matching
              // host leaf. Publishing per-type keeps a partial success from
              // corrupting sibling leaves.
              std::unordered_map<BlockType, std::vector<Block>> by_type;
              for (size_t k = 0; k < host_blocks.size(); ++k) {
                by_type[block_types_vec[k]].emplace_back(
                    std::move(host_blocks[k]));
              }
              for (auto& [type, blocks] : by_type) {
                auto it = host_leaves.find(type);
                if (it == host_leaves.end()) {
                  LOG(ERROR) << "Missing Host block manager for block type "
                             << static_cast<int32_t>(type);
                  blocks.clear();
                  continue;
                }
                it->second->cache(blocks);
                it->second->deallocate(blocks);
                blocks.clear();
              }
              host_blocks.clear();
            } else {
              // Publish nothing on failure. Return all host ids to their
              // owning leaf so no host slot leaks.
              std::unordered_map<BlockType, std::vector<Block>> by_type;
              for (size_t k = 0; k < host_blocks.size(); ++k) {
                by_type[block_types_vec[k]].emplace_back(
                    std::move(host_blocks[k]));
              }
              for (auto& [type, blocks] : by_type) {
                auto it = host_leaves.find(type);
                if (it == host_leaves.end()) {
                  LOG(ERROR) << "Missing Host block manager for block type "
                             << static_cast<int32_t>(type);
                  blocks.clear();
                  continue;
                }
                it->second->deallocate(blocks);
                blocks.clear();
              }
              host_blocks.clear();
            }

            return 0;
          })
          .ensure([pending_offload_transfers]() {
            pending_offload_transfers->fetch_sub(1, std::memory_order_relaxed);
          });
    }
  }
}

}  // namespace xllm
