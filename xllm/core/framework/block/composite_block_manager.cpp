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

#include "core/framework/block/composite_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

#include "core/framework/block/embedding_block_manager.h"
#include "core/framework/block/vmm_kv_block_manager.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/request/sequence.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/block/concurrent_block_manager_impl.h"
#include "core/kv_cache/block/linear_state_block_manager.h"
#include "core/kv_cache/block/sliding_window_block_manager.h"
#include "core/kv_cache/layout/static_kv_cache_layout.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

// KV completion bounds the block contents; confirmed tokens independently
// provide the hasher's lookahead. The lookahead token needs no completed KV.
size_t publishable_tokens(const Sequence& seq, const BlockManager& leaf) {
  const BlockHasherType type = leaf.options().hasher_type();
  const size_t token_count = seq.hash_tokens(type).size();
  const size_t lookahead = block_hash_lookahead(type);
  const size_t ready_tokens =
      token_count > lookahead ? token_count - lookahead : 0;
  return std::min(seq.kv_cache_tokens_num(), ready_tokens);
}

// Publish a dependency-complete range and advance its cursor together. Callers
// select the range; the leaf's identity policy owns the extra hash tokens.
void publish_blocks(Sequence* seq,
                    BlockManager& leaf,
                    BlockType type,
                    size_t begin,
                    size_t end) {
  if (end <= begin) {
    return;
  }
  const BlockHasherType hasher_type = leaf.options().hasher_type();
  const size_t block_size = leaf.block_size();
  const size_t token_end = end * block_size + block_hash_lookahead(hasher_type);
  KVCacheState& kv = seq->kv_state();
  std::vector<Block>* blocks = kv.mutable_blocks(type);
  CHECK(blocks != nullptr);
  CHECK_LE(end, blocks->size());
  seq->update_block_hashes(static_cast<uint32_t>(block_size), hasher_type);
  leaf.cache(seq->hash_tokens(hasher_type).slice(0, token_end),
             *blocks,
             begin,
             seq->mm_data(),
             seq->block_hashes());
  kv.set_num_cached_blocks(type, end);
}

inline BlockType block_type_from_compress_ratio(uint32_t ratio) {
  if (ratio == 1) {
    return BlockType::C1;
  }
  if (ratio == 2) {
    return BlockType::C2;
  }
  if (ratio == 4) {
    return BlockType::C4;
  }
  return BlockType::C128;
}

// Wrap the leaf in a concurrency adapter when sequence-level calls may run
// off the scheduler thread (disagg PD / kvcache store / host-offload).
std::unique_ptr<BlockManager> maybe_concurrent(
    std::unique_ptr<BlockManager> leaf,
    const BlockManager::Options& options) {
  if (options.enable_disagg_pd() || options.enable_kvcache_store() ||
      options.enable_host_offload()) {
    return std::make_unique<ConcurrentBlockManagerImpl>(std::move(leaf));
  }
  return leaf;
}

// Choose the paged virtual memory manager or the flat free-list manager.
std::unique_ptr<BlockManager> make_kv_leaf(const BlockManager::Options& kv_opts,
                                           int32_t dp_rank,
                                           VmmKVPageAllocator* page_allocator) {
  if (!kv_opts.enable_vmm_kvcache()) {
    return std::make_unique<BlockManagerImpl>(kv_opts);
  }
  CHECK_GT(kv_opts.num_layers(), 0)
      << "num_layers must be set when enable_vmm_kvcache is true";
  CHECK_GT(kv_opts.slot_size(), 0)
      << "slot_size must be set when enable_vmm_kvcache is true";
  CHECK(page_allocator != nullptr)
      << "Virtual memory KV cache requires a page allocator";
  const size_t page_size = page_allocator->page_size();
  CHECK_GT(page_size, 0u);
  const size_t block_mem_size = kv_opts.virtual_block_bytes();
  CHECK_GT(block_mem_size, 0);
  CHECK_EQ(page_size % block_mem_size, 0);
  return std::make_unique<VmmKVBlockManager>(kv_opts,
                                             *page_allocator,
                                             kv_opts.num_layers(),
                                             block_mem_size,
                                             page_size,
                                             dp_rank,
                                             kv_opts.model_id());
}

using ProbeResult = CompositeBlockManager::ProbeResult;

// Trim outcome. `to_mount` feeds the shape's mount step; `to_drop` releases
// the aliases that are outside the usable prefix.
struct TrimOutcome {
  std::vector<ProbeResult> to_mount;
  std::vector<ProbeResult> to_drop;
  size_t safe_hit_tokens = 0;
};

std::optional<ProbeResult> take_probe(std::vector<ProbeResult>& probes,
                                      BlockType type) {
  for (auto it = probes.begin(); it != probes.end(); ++it) {
    if (it->type == type) {
      ProbeResult r = std::move(*it);
      probes.erase(it);
      return r;
    }
  }
  return std::nullopt;
}

// FLAT_KV: no trim; Sequence::add_shared_blocks owns replace + exact-repeat.
TrimOutcome trim_flat_kv(std::vector<ProbeResult> probes) {
  CHECK_EQ(probes.size(), 1u) << "FLAT_KV expects a single KV probe";
  TrimOutcome out;
  out.to_mount = std::move(probes);
  return out;
}

// FLAT_KV_LINEAR: retain only a KV prefix whose token boundary has an exact
// LINEAR checkpoint.
TrimOutcome trim_flat_kv_linear(std::vector<ProbeResult> probes,
                                size_t prompt_tokens) {
  auto linear = take_probe(probes, BlockType::LINEAR);
  CHECK_EQ(probes.size(), 1u)
      << "FLAT_KV_LINEAR expects one KV probe (LINEAR removed above)";
  TrimOutcome out;

  ProbeResult& kv = probes.front();
  const size_t kv_block_size = kv.block_size;
  size_t safe_count = 0;
  size_t source_index = 0;
  if (linear.has_value() && kv_block_size > 0 && linear->block_size > 0) {
    const size_t max_hit_tokens =
        std::min(kv.blocks.size() * kv_block_size,
                 prompt_tokens == 0 ? 0 : prompt_tokens - 1);
    const size_t checkpoint_count =
        std::min(linear->blocks.size(), max_hit_tokens / linear->block_size);
    for (size_t index = checkpoint_count; index > 0; --index) {
      const size_t checkpoint_tokens = index * linear->block_size;
      if (checkpoint_tokens % kv_block_size == 0 &&
          linear->blocks[index - 1].is_valid()) {
        safe_count = checkpoint_tokens / kv_block_size;
        source_index = index - 1;
        break;
      }
    }
  }

  if (safe_count < kv.blocks.size()) {
    ProbeResult drop = {kv.type,
                        kv.leaf,
                        /*blocks=*/{},
                        /*block_size=*/kv.block_size};
    drop.blocks.reserve(kv.blocks.size() - safe_count);
    for (size_t i = safe_count; i < kv.blocks.size(); ++i) {
      drop.blocks.emplace_back(std::move(kv.blocks[i]));
    }
    kv.blocks.resize(safe_count);
    out.to_drop.emplace_back(std::move(drop));
  }
  out.safe_hit_tokens = safe_count * kv_block_size;
  out.to_mount = std::move(probes);
  if (linear.has_value()) {
    if (out.safe_hit_tokens > 0) {
      // Only the checkpoint itself survives; the leading slots stay empty so
      // the LINEAR leaf is mounted with the same shape as a full hit.
      std::vector<Block> source_blocks(source_index + 1);
      source_blocks.back() = std::move(linear->blocks[source_index]);
      out.to_mount.emplace_back(ProbeResult{linear->type,
                                            linear->leaf,
                                            std::move(source_blocks),
                                            linear->block_size});
    }
    out.to_drop.emplace_back(std::move(*linear));
  }
  return out;
}

// SWA_COMPRESSED: cross-leaf min -> C128-stride clamp -> SWA tail-continuity
// (fallback in C128 steps) -> per-leaf trim.
TrimOutcome trim_swa_compressed(std::vector<ProbeResult> probes,
                                size_t prompt_tokens) {
  TrimOutcome out;
  size_t c128_block_size = 0;
  for (const auto& p : probes) {
    if (p.type == BlockType::C128) {
      c128_block_size = p.block_size;
    }
  }
  CHECK_GT(c128_block_size, 0u)
      << "SWA_COMPRESSED trim requires a C128 leaf with non-zero block_size";

  size_t safe_hit_tokens = std::numeric_limits<size_t>::max();
  for (const auto& p : probes) {
    safe_hit_tokens = std::min(safe_hit_tokens, p.blocks.size() * p.block_size);
  }
  safe_hit_tokens = (safe_hit_tokens / c128_block_size) * c128_block_size;

  // A forward pass must retain at least one C128 unit; clamp an exact prompt
  // hit before validating SWA, whose preceding window may already be evicted.
  const size_t max_prefix_tokens = prompt_tokens == 0 ? 0 : prompt_tokens - 1;
  safe_hit_tokens =
      (std::min(safe_hit_tokens, max_prefix_tokens) / c128_block_size) *
      c128_block_size;

  // SWA attention reads the last `swa_blocks_per_seq` base blocks; a middle
  // invalid placeholder in that tail means garbage KV. Fall back in C128
  // steps until the tail is fully valid.
  auto swa_it =
      std::find_if(probes.begin(), probes.end(), [](const ProbeResult& p) {
        return p.type == BlockType::SWA;
      });
  if (swa_it != probes.end() && safe_hit_tokens > 0) {
    const size_t swa_block_size = swa_it->block_size;
    const size_t tail_required =
        static_cast<size_t>(swa_it->leaf->options().swa_blocks_per_seq());
    const std::vector<Block>& swa_vector = swa_it->blocks;
    if (swa_block_size > 0 && tail_required > 0) {
      while (safe_hit_tokens >= c128_block_size) {
        const size_t trimmed_len = safe_hit_tokens / swa_block_size;
        bool tail_ok = trimmed_len >= tail_required;
        if (tail_ok) {
          for (size_t i = trimmed_len - tail_required; i < trimmed_len; ++i) {
            if (i >= swa_vector.size() || !swa_vector[i].is_valid()) {
              tail_ok = false;
              break;
            }
          }
        }
        if (tail_ok) {
          break;
        }
        safe_hit_tokens -= c128_block_size;
      }
      if (safe_hit_tokens < c128_block_size) {
        safe_hit_tokens = 0;
      }
    }
  }

  if (safe_hit_tokens == 0) {
    out.to_drop = std::move(probes);
    return out;
  }

  out.to_mount.reserve(probes.size());
  for (auto& p : probes) {
    const size_t target_len = safe_hit_tokens / p.block_size;
    if (p.blocks.size() > target_len) {
      ProbeResult drop = {p.type,
                          p.leaf,
                          /*blocks=*/{},
                          /*block_size=*/p.block_size};
      drop.blocks.reserve(p.blocks.size() - target_len);
      for (size_t i = target_len; i < p.blocks.size(); ++i) {
        drop.blocks.emplace_back(std::move(p.blocks[i]));
      }
      p.blocks.resize(target_len);
      out.to_drop.emplace_back(std::move(drop));
    }
    out.to_mount.emplace_back(std::move(p));
  }
  out.safe_hit_tokens = safe_hit_tokens;
  return out;
}

}  // namespace

CompositeBlockManager::LeafMap build_composite_leaves(
    const BlockManager::Options& options,
    int32_t dp_rank,
    VmmKVPageAllocator* page_allocator) {
  CompositeBlockManager::LeafMap leaves;
  BlockManager::Options cache_options = options;
  if (options.enable_shared_kv_cache()) {
    CHECK(!options.enable_vmm_kvcache());
    CHECK(options.enable_linear_state());
    CHECK(options.manager_types().empty());
    CHECK_EQ(options.num_blocks(),
             static_cast<uint32_t>(options.linear_state_num_slots()));
    cache_options.shared_block_pool(
        options.shared_layout()
            ? std::make_shared<StaticKVCacheBlockPool>(options.num_blocks(),
                                                       *options.shared_layout())
            : std::make_shared<StaticKVCacheBlockPool>(
                  options.num_blocks(), options.shared_linear_state_slots()));
  }

  const size_t n = options.manager_types().size();
  if (n != 0) {
    CHECK_EQ(n, options.compress_ratios().size())
        << "manager_types and compress_ratios must have the same size";
  }

  bool prefix_cache_on = options.enable_prefix_cache();
  const bool is_decode = options.instance_is_decode();
  // C1/C2 cannot restore prefixes; disable the whole composite rather than
  // cache SWA alone.
  bool grouped_anchor_participates = false;
  bool has_token_group = false;
  for (size_t i = 0; i < n; ++i) {
    if (options.manager_types()[i] != kManagerTypeBlockManagerImpl) {
      continue;
    }
    const uint32_t ratio = options.compress_ratios()[i];
    const BlockType key = block_type_from_compress_ratio(ratio);
    has_token_group |= key == BlockType::C1 || key == BlockType::C2;
    if (key == BlockType::C4 || key == BlockType::C128) {
      grouped_anchor_participates = true;
      break;
    }
  }
  if (prefix_cache_on && has_token_group && !grouped_anchor_participates) {
    LOG(WARNING) << "grouped cache has no prefix-cache-capable group; "
                    "disabling prefix cache for this instance";
    prefix_cache_on = false;
  }

  const bool prefill_prefix_cache = prefix_cache_on && !is_decode;

  // LINEAR holds recurrent state in addition to KV and must not determine
  // token capacity. The concurrent wrapper also covers off-scheduler callers.
  if (options.enable_linear_state()) {
    CHECK_GT(options.linear_state_num_slots(), 0)
        << "linear_state_num_slots must be set when linear state is enabled";
    int32_t chunk_stride =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    // With prefix caching, the slot stride is the checkpoint boundary.
    // Decode can disable chunked prefill, leaving its unused stride unset.
    if (prefill_prefix_cache) {
      CHECK_GT(chunk_stride, 0) << "max_tokens_per_chunk_for_prefill must be "
                                   "positive for linear state prefix cache";
    } else if (chunk_stride <= 0) {
      chunk_stride = 1;
    }
    BlockManager::Options linear_opts = cache_options;
    linear_opts
        .num_blocks(static_cast<uint32_t>(options.linear_state_num_slots()))
        .block_size(chunk_stride)
        .enable_prefix_cache(prefill_prefix_cache)
        .instance_is_decode(is_decode);
    leaves.emplace(
        BlockType::LINEAR,
        CompositeBlockManager::LeafEntry{
            maybe_concurrent(std::make_unique<LinearStateBlockManager>(
                                 std::move(linear_opts)),
                             options),
            /*participates_in_admission=*/false,
            /*supports_prefix_cache=*/prefill_prefix_cache});
  }

  if (options.manager_types().empty()) {
    // Normal / Qwen / virtual memory: a single KV leaf.
    BlockManager::Options kv_opts = cache_options;
    kv_opts.block_type(BlockType::KV);
    kv_opts.enable_prefix_cache(prefix_cache_on);
    leaves.emplace(
        BlockType::KV,
        CompositeBlockManager::LeafEntry{
            maybe_concurrent(make_kv_leaf(kv_opts, dp_rank, page_allocator),
                             options),
            /*participates_in_admission=*/true,
            /*supports_prefix_cache=*/prefix_cache_on});
    return leaves;
  }

  for (size_t i = 0; i < n; ++i) {
    const uint32_t type = options.manager_types()[i];
    const uint32_t compress_ratio = options.compress_ratios()[i];
    BlockManager::Options opts = options;

    if (type == kManagerTypeBlockManagerImpl) {
      CHECK(compress_ratio == 1 || compress_ratio == 2 || compress_ratio == 4 ||
            compress_ratio == 128)
          << "unexpected compress_ratio " << compress_ratio
          << " for composite BlockManagerImpl sub-manager";
      opts.block_size(static_cast<uint32_t>(options.block_size()) *
                      compress_ratio);
      opts.num_blocks(static_cast<uint32_t>(options.num_blocks()) /
                      compress_ratio);
      const BlockType key = block_type_from_compress_ratio(compress_ratio);
      const bool compressed_prefix_cache =
          prefix_cache_on && (key == BlockType::C4 || key == BlockType::C128);
      opts.block_type(key)
          .enable_prefix_cache(compressed_prefix_cache)
          .hasher_type(options.hasher_type());
      // C4/C128: full-history attention (no window), so hits must form a
      // solid left-to-right prefix. The default BlockManagerImpl probe is
      // exactly right. Cross-leaf min happens in the composite.
      leaves.emplace(key,
                     CompositeBlockManager::LeafEntry{
                         maybe_concurrent(
                             std::make_unique<BlockManagerImpl>(opts), options),
                         /*participates_in_admission=*/true,
                         /*supports_prefix_cache=*/compressed_prefix_cache});
    } else if (type == kManagerTypeSlidingWindowBlockManager) {
      const uint32_t swa_blocks_per_seq = options.swa_blocks_per_seq();
      CHECK_GT(swa_blocks_per_seq, 0u) << "swa_blocks_per_seq must be positive";
      CHECK_GT(options.block_size(), 0) << "block_size must be positive";
      const uint32_t sliding_window_size =
          std::max(options.sliding_window_size(), 1u);
      const uint32_t swa_total_blocks = options.swa_num_blocks();
      CHECK_GT(swa_total_blocks, 0u)
          << "swa_num_blocks must be provided by the KV cache estimator";
      opts.num_blocks(swa_total_blocks)
          .swa_blocks_per_seq(swa_blocks_per_seq)
          .sliding_window_size(sliding_window_size)
          .block_type(BlockType::SWA)
          .enable_prefix_cache(prefill_prefix_cache)
          .hasher_type(options.hasher_type());
      // SWA is not an admission leaf (pool sized by ring, not token budget)
      // but serves gap-tolerant prefix cache.
      leaves.emplace(
          BlockType::SWA,
          CompositeBlockManager::LeafEntry{
              maybe_concurrent(
                  std::make_unique<SlidingWindowBlockManager>(opts), options),
              /*participates_in_admission=*/false,
              /*supports_prefix_cache=*/prefill_prefix_cache});
    } else {
      LOG(FATAL) << "Unknown manager_type " << type;
    }
  }
  return leaves;
}

CompositeBlockManager::CompositeBlockManager(
    LeafMap leaves,
    const BlockManager::Options& options,
    CacheTier cache_tier)
    : BlockManager(options),
      leaves_(std::move(leaves)),
      combination_(classify_leaf_combination(leaves_)),
      cache_tier_(cache_tier) {
  CHECK(!leaves_.empty()) << "CompositeBlockManager requires at least one leaf";
  if (options.enable_shared_kv_cache()) {
    CHECK(cache_tier == CacheTier::DEVICE);
    shared_block_pool_ = leaf_of(BlockType::KV)->options().shared_block_pool();
    CHECK(shared_block_pool_ != nullptr);
    CHECK_EQ(shared_block_pool_,
             leaf_of(BlockType::LINEAR)->options().shared_block_pool());
  }
}

CompositeBlockManager::LeafCombination
CompositeBlockManager::classify_leaf_combination(const LeafMap& leaves) {
  auto has_prefix = [&](BlockType t) {
    const auto it = leaves.find(t);
    return it != leaves.end() && it->second.supports_prefix_cache;
  };
  const bool has_kv = has_prefix(BlockType::KV);
  const bool has_swa = has_prefix(BlockType::SWA);
  const bool has_c4 = has_prefix(BlockType::C4);
  const bool has_c128 = has_prefix(BlockType::C128);
  const bool has_linear = has_prefix(BlockType::LINEAR);

  // SWA_COMPRESSED requires C4 + C128 to make sense (a compressed shape with
  // only one granularity is not a shape we ship). SWA is optional here on
  // purpose: the DECODE role turns off SWA prefix cache via the role
  // predicate, so has_swa is false while C4/C128 stay on -- trim /
  // classify still route through this branch, but probes will be missing
  // the SWA entry and trim_swa_compressed guards accordingly.
  if (has_c4 || has_c128) {
    CHECK(has_c4 && has_c128)
        << "SWA_COMPRESSED requires both C4 and C128 with prefix cache on; "
        << "got C4=" << has_c4 << " C128=" << has_c128;
    CHECK(!has_kv) << "SWA_COMPRESSED must not carry a KV leaf";
    CHECK(!has_linear) << "SWA_COMPRESSED must not carry a LINEAR leaf";
    return LeafCombination::SWA_COMPRESSED;
  }
  if (has_swa) {
    // A prefix-cache-on SWA leaf paired with no compressed leaves is not a
    // shape we build; the classifier surfaces the misconfiguration loudly
    // instead of silently mounting garbage.
    LOG(FATAL) << "SWA prefix cache is on but neither C4 nor C128 is present";
  }
  if (has_kv) {
    return has_linear ? LeafCombination::FLAT_KV_LINEAR
                      : LeafCombination::FLAT_KV;
  }
  return LeafCombination::UNSUPPORTED;
}

BlockManager* CompositeBlockManager::leaf_of(BlockType type) const {
  auto it = leaves_.find(type);
  return it == leaves_.end() ? nullptr : it->second.leaf.get();
}

void CompositeBlockManager::publish_prefix(Sequence* seq) {
  publish_completed_prefix(seq, std::numeric_limits<size_t>::max());
}

void CompositeBlockManager::publish_restored_prefix(Sequence* seq,
                                                    size_t restored_tokens) {
  publish_completed_prefix(seq, restored_tokens);
}

void CompositeBlockManager::publish_completed_prefix(Sequence* seq,
                                                     size_t token_limit) {
  if (seq == nullptr || seq->is_graph_warmup() ||
      cache_tier_ == CacheTier::HOST) {
    return;
  }
  KVCacheState& kv = seq->kv_state();

  size_t cacheable_tokens = 0;
  size_t cache_unit_size = 0;
  if (combination_ == LeafCombination::SWA_COMPRESSED) {
    BlockManager* c128_leaf = leaf_of(BlockType::C128);
    CHECK(c128_leaf != nullptr);
    cacheable_tokens =
        std::min(token_limit, publishable_tokens(*seq, *c128_leaf));
    cache_unit_size = c128_leaf->block_size();
    CHECK_GT(cache_unit_size, 0u);

    // A DSV4 prefix is restorable only at a complete C128 boundary. Cap the
    // boundary by every participating leaf's allocated logical capacity so no
    // partially allocated composite unit can become visible.
    for (const auto& [type, entry] : leaves_) {
      if (!entry.supports_prefix_cache || type == BlockType::EMBEDDING ||
          type == BlockType::LINEAR) {
        continue;
      }
      const size_t block_size = entry.leaf->block_size();
      CHECK_GT(block_size, 0u);
      CHECK_EQ(cache_unit_size % block_size, 0u)
          << "DSV4 cache leaf block size must divide the C128 cache unit";
      cacheable_tokens =
          std::min(cacheable_tokens, kv.num_blocks(type) * block_size);
    }
    cacheable_tokens = (cacheable_tokens / cache_unit_size) * cache_unit_size;
  }

  for (auto& [type, entry] : leaves_) {
    // EMBEDDING / LINEAR hold no token cache. KV participates here after the
    // hierarchy pool confirms a Host restore; later calls flush its tail.
    if (type == BlockType::EMBEDDING || type == BlockType::LINEAR) {
      continue;
    }
    if (!entry.supports_prefix_cache) {
      continue;
    }
    BlockManager& leaf = *entry.leaf;
    const size_t block_size = leaf.block_size();
    if (block_size == 0) {
      continue;
    }
    std::vector<Block>* blocks = kv.mutable_blocks(type);
    if (blocks == nullptr || blocks->empty()) {
      continue;
    }
    const size_t num_full =
        combination_ == LeafCombination::SWA_COMPRESSED
            ? cacheable_tokens / block_size
            : std::min(token_limit, publishable_tokens(*seq, leaf)) /
                  block_size;
    size_t cached = kv.num_cached_blocks(type);
    const size_t end = std::min(num_full, blocks->size());
    if (end <= cached) {
      continue;
    }

    if (combination_ == LeafCombination::SWA_COMPRESSED &&
        type == BlockType::SWA) {
      const size_t blocks_per_unit = cache_unit_size / block_size;
      const size_t blocks_per_window =
          static_cast<size_t>(leaf.options().swa_blocks_per_seq());
      CHECK_GT(blocks_per_unit, 0u);
      CHECK_GT(blocks_per_window, 0u);

      // SWA only needs the window immediately preceding each restorable C128
      // checkpoint. Advance the cursor over earlier positions without
      // inserting them; LinearStatePrefixCache keeps those positions sparse.
      const size_t completed_units = cacheable_tokens / cache_unit_size;
      for (size_t unit = cached / blocks_per_unit + 1; unit <= completed_units;
           ++unit) {
        const size_t unit_end = unit * blocks_per_unit;
        const size_t window_begin =
            unit_end > blocks_per_window ? unit_end - blocks_per_window : 0;
        const size_t publish_begin = std::max(cached, window_begin);
        publish_blocks(seq, leaf, type, publish_begin, unit_end);
        cached = unit_end;
      }
      continue;
    }

    publish_blocks(seq, leaf, type, cached, end);
  }
}

KVCacheState& CompositeBlockManager::kv_state_for(Sequence& seq) const {
  if (cache_tier_ == CacheTier::HOST) {
    return seq.host_kv_state();
  }
  return seq.kv_state();
}

bool CompositeBlockManager::allocate_sequence(Sequence* seq,
                                              size_t num_tokens,
                                              bool cache_prefix) {
  if (seq == nullptr) {
    return false;
  }
  KVCacheState& kv_state = kv_state_for(*seq);
  const size_t previous_checkpoint =
      kv_state.num_cached_blocks(BlockType::LINEAR);

  // Publish blocks completed by the previous forward before growing again.
  // SlidingWindowBlockManager can then release a cached, slid-out block and
  // retry when its first allocation attempt exhausts the SWA pool.
  if (cache_prefix) {
    publish_prefix(seq);
  }

  // Fan out growth. Each leaf returns its newly allocated blocks (or nullopt
  // on failure). Stage keyed by BlockType; commit only after every leaf
  // succeeds so a failure rolls back cleanly.
  std::map<BlockType, std::vector<Block>> staged;

  auto release_staged = [&]() {
    for (auto& [type, blocks] : staged) {
      Block::release(std::move(blocks));
    }
    staged.clear();
  };

  for (auto& [type, entry] : leaves_) {
    auto blocks = entry.leaf->allocate_for_sequence(seq, kv_state, num_tokens);
    if (!blocks.has_value() && shared_block_pool_ &&
        entry.leaf->options().shared_block_pool() == shared_block_pool_) {
      // Bound retries to the cache snapshot: another thread may publish new
      // cache entries while each leaf lock is released between attempts.
      size_t remaining_evictions = num_blocks_in_prefix_cache();
      while (!blocks.has_value() && remaining_evictions > 0 &&
             reclaim_shared_cache()) {
        --remaining_evictions;
        blocks = entry.leaf->allocate_for_sequence(seq, kv_state, num_tokens);
      }
    }
    if (!blocks.has_value()) {
      release_staged();
      return false;
    }
    if (!blocks->empty()) {
      staged.emplace(type, std::move(*blocks));
    }
  }

  const size_t checkpoint = kv_state.num_cached_blocks(BlockType::LINEAR);
  if (shared_block_pool_ && shared_block_pool_->uses_slabs() &&
      checkpoint > previous_checkpoint) {
    record_shared_checkpoint(*seq, checkpoint);
  }

  // Grow-or-fail: every cache-bearing leaf must cover num_tokens now.
  // Without this a leaf that mistakenly returned an empty vector under
  // pressure would let the pool report admission success while the device
  // is under-provisioned -- forward_input_builder's CHECK then fires
  // downstream. Rolling back lets the scheduler defer to the next tick.
  // EMBEDDING / LINEAR are per-sequence resource slots, not token cache.
  for (const auto& [type, entry] : leaves_) {
    if (type == BlockType::EMBEDDING || type == BlockType::LINEAR) {
      continue;
    }
    const size_t leaf_block_size = entry.leaf->block_size();
    if (leaf_block_size == 0) {
      continue;
    }
    const size_t needed = (num_tokens + leaf_block_size - 1) / leaf_block_size;
    const auto staged_it = staged.find(type);
    const size_t staged_for_type =
        staged_it == staged.end() ? 0 : staged_it->second.size();
    const size_t total = kv_state.num_blocks(type) + staged_for_type;
    if (total < needed) {
      release_staged();
      return false;
    }
  }

  // Commit staged blocks before publishing the completed prefix. The token
  // cursor bounds publication, so newly allocated compute blocks are not
  // cached until their tokens have been forwarded.
  for (auto& [type, blocks] : staged) {
    kv_state.add_blocks(type, blocks);
  }
  staged.clear();
  if (cache_prefix) {
    publish_prefix(seq);
  }

  // Release slid-out SWA blocks only after they have had a chance to enter the
  // prefix cache. Other leaves are no-ops here.
  release_out_of_window_for_sequence(seq);
  return true;
}

void CompositeBlockManager::release_out_of_window_for_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  KVCacheState& kv_state = kv_state_for(*seq);
  for (auto& [type, entry] : leaves_) {
    entry.leaf->release_out_of_window(seq, kv_state);
  }
}

void CompositeBlockManager::deallocate_for_sequence(Sequence* seq,
                                                    bool cache_prefix) {
  if (seq == nullptr || cache_tier_ == CacheTier::HOST) {
    return;
  }
  // Prefix publication needs the live cache state. Reset consumes its block
  // ownership as well as the sequence metadata.
  if (cache_prefix) {
    publish_prefix(seq);
  }
  seq->reset();
}

std::vector<CompositeBlockManager::ProbeResult>
CompositeBlockManager::probe_prefix_cache(Sequence* seq,
                                          const LeafMap& leaves) {
  if (seq == nullptr) {
    return {};
  }
  return probe_prefix_cache(seq, leaves, seq->kv_state());
}

std::vector<CompositeBlockManager::ProbeResult>
CompositeBlockManager::probe_prefix_cache(Sequence* seq,
                                          const LeafMap& leaves,
                                          const KVCacheState& existing_state) {
  std::vector<ProbeResult> probes;
  if (seq == nullptr) {
    return probes;
  }
  probes.reserve(leaves.size());
  for (const auto& [type, entry] : leaves) {
    if (!entry.supports_prefix_cache || entry.leaf == nullptr) {
      continue;
    }
    BlockManager& leaf = *entry.leaf;
    Slice<XXH3Key> hashes;
    Slice<Block> existed;
    if (type == BlockType::LINEAR) {
      seq->update_linear_state_hashes(static_cast<uint32_t>(leaf.block_size()));
      hashes = seq->linear_state_hashes();
    } else {
      seq->update_block_hashes(static_cast<uint32_t>(leaf.block_size()),
                               leaf.options().hasher_type());
      hashes = seq->block_hashes();
      existed = existing_state.blocks(type).slice(
          0, existing_state.shared_blocks_num(type));
    }
    probes.emplace_back(ProbeResult{
        /*type=*/type,
        /*leaf=*/&leaf,
        /*blocks=*/
        leaf.allocate_shared(seq->hash_tokens(leaf.options().hasher_type()),
                             existed,
                             seq->mm_data(),
                             hashes),
        /*block_size=*/leaf.block_size()});
  }
  return probes;
}

void CompositeBlockManager::release_probes(std::vector<ProbeResult>* probes) {
  if (probes == nullptr) {
    return;
  }
  for (ProbeResult& probe : *probes) {
    if (probe.blocks.empty()) {
      continue;
    }
    Block::release(std::move(probe.blocks));
  }
  probes->clear();
}

void CompositeBlockManager::allocate_shared_for_sequence(Sequence* seq) {
  // Shared flow: probe -> trim -> mount. Trim strategy is per-shape (see
  // trim_flat_kv / trim_flat_kv_linear / trim_swa_compressed).
  if (seq == nullptr || combination_ == LeafCombination::UNSUPPORTED) {
    return;
  }
  KVCacheState& kv_state = kv_state_for(*seq);
  if (combination_ == LeafCombination::FLAT_KV_LINEAR &&
      kv_state.prefix_cache_matched()) {
    return;
  }
  if (shared_block_pool_ && shared_block_pool_->uses_slabs() &&
      combination_ == LeafCombination::FLAT_KV_LINEAR) {
    const size_t block_size = leaf_of(BlockType::KV)->block_size();
    const size_t full_blocks = seq->tokens().size() / block_size +
                               (seq->tokens().size() % block_size != 0);
    // Restore keeps its cached source until a private Linear output exists.
    // Decline that transient two-state footprint before pinning the prefix;
    // cold execution needs one state and can reclaim the inactive cache pair.
    if (!options_.shared_layout()->can_fit(full_blocks, 2)) {
      kv_state.set_prefix_cache_matched();
      return;
    }
  }
  std::vector<ProbeResult> probes = probe_prefix_cache(seq, leaves_, kv_state);
  if (probes.empty()) {
    kv_state.set_prefix_cache_matched();
    return;
  }

  TrimOutcome trimmed;
  switch (combination_) {
    case LeafCombination::FLAT_KV:
      trimmed = trim_flat_kv(std::move(probes));
      break;
    case LeafCombination::FLAT_KV_LINEAR:
      trimmed = trim_flat_kv_linear(std::move(probes), seq->tokens().size());
      break;
    case LeafCombination::SWA_COMPRESSED:
      trimmed = trim_swa_compressed(std::move(probes), seq->tokens().size());
      break;
    case LeafCombination::UNSUPPORTED:
      return;
  }
  if (shared_block_pool_ && shared_block_pool_->uses_slabs() &&
      combination_ == LeafCombination::FLAT_KV_LINEAR &&
      trimmed.safe_hit_tokens > 0) {
    record_shared_checkpoint(
        *seq,
        trimmed.safe_hit_tokens / leaf_of(BlockType::LINEAR)->block_size());
  }

  release_probes(&trimmed.to_drop);

  switch (combination_) {
    case LeafCombination::FLAT_KV:
    case LeafCombination::FLAT_KV_LINEAR: {
      for (ProbeResult& probe : trimmed.to_mount) {
        if (probe.type == BlockType::LINEAR) {
          kv_state.mount_composite_shared(probe.type, std::move(probe.blocks));
        } else {
          kv_state.add_shared_blocks(
              probe.type, std::move(probe.blocks), seq->num_tokens());
        }
      }
      break;
    }
    case LeafCombination::SWA_COMPRESSED: {
      for (ProbeResult& probe : trimmed.to_mount) {
        kv_state.mount_composite_shared(probe.type, std::move(probe.blocks));
      }
      kv_state.set_kv_cache_tokens_num(trimmed.safe_hit_tokens);
      break;
    }
    case LeafCombination::UNSUPPORTED:
      break;
  }
  kv_state.set_prefix_cache_matched();
}

void CompositeBlockManager::record_shared_checkpoint(Sequence& seq,
                                                     size_t checkpoint_index) {
  BlockManager* full = leaf_of(BlockType::KV);
  BlockManager* linear = leaf_of(BlockType::LINEAR);
  CHECK(full != nullptr && linear != nullptr);
  const size_t tokens = checkpoint_index * linear->block_size();
  if (tokens == 0 || tokens % full->block_size() != 0) {
    return;
  }
  seq.update_block_hashes(static_cast<uint32_t>(full->block_size()),
                          full->options().hasher_type());
  seq.update_linear_state_hashes(static_cast<uint32_t>(linear->block_size()));
  const size_t full_count = tokens / full->block_size();
  CHECK_LE(full_count, seq.block_hashes().size());
  CHECK_LE(checkpoint_index, seq.linear_state_hashes().size());
  // MTP Full keys include the next token, while the recurrent checkpoint key
  // describes only the prefix. Multiple Full endpoints may share one state.
  full->set_prefix_dependency(seq.block_hashes()[full_count - 1],
                              seq.linear_state_hashes()[checkpoint_index - 1]);
}

void CompositeBlockManager::publish_prefix(Sequence* seq, size_t num_tokens) {
  // Chunked-prefill mid-step: only KV needs a token-clamped flush now;
  // SWA / C4 / C128 get their flush from the pre-grow hook at the top of
  // the next allocate_sequence.
  if (seq == nullptr || seq->is_graph_warmup() ||
      combination_ == LeafCombination::UNSUPPORTED ||
      cache_tier_ == CacheTier::HOST) {
    return;
  }
  switch (combination_) {
    case LeafCombination::FLAT_KV:
    case LeafCombination::FLAT_KV_LINEAR: {
      BlockManager& kv_leaf = *leaf_of(BlockType::KV);
      KVCacheState& kv_state = seq->kv_state();
      const size_t block_size = kv_leaf.block_size();
      const BlockHasherType hasher_type = kv_leaf.options().hasher_type();
      const Slice<int32_t> tokens = seq->hash_tokens(hasher_type);
      const size_t completed_tokens = block_hash_lookahead(hasher_type) == 0
                                          ? num_tokens
                                          : kv_state.kv_cache_tokens_num();
      const size_t available_tokens_num =
          std::min({num_tokens,
                    completed_tokens,
                    kv_state.num_blocks(BlockType::KV) * block_size});
      const size_t publish_end =
          std::min(available_tokens_num / block_size,
                   num_hash_blocks(hasher_type, tokens.size(), block_size));
      const size_t publish_begin = kv_state.num_cached_blocks(BlockType::KV);
      publish_blocks(seq, kv_leaf, BlockType::KV, publish_begin, publish_end);
      break;
    }
    case LeafCombination::SWA_COMPRESSED:
    case LeafCombination::UNSUPPORTED:
      break;
  }
}

void CompositeBlockManager::trim_prefetch_blocks(Sequence& seq,
                                                 size_t max_hit_tokens) {
  for (auto& [type, entry] : leaves_) {
    if (type == BlockType::EMBEDDING) {
      continue;
    }
    entry.leaf->trim_prefetch_blocks(seq, max_hit_tokens);
  }
}
std::vector<Block> CompositeBlockManager::allocate_blocks(BlockType type,
                                                          size_t num_blocks) {
  BlockManager* leaf = leaf_of(type);
  CHECK(leaf != nullptr) << "CompositeBlockManager has no leaf for block type "
                         << static_cast<int>(type);
  std::vector<Block> blocks = leaf->allocate(num_blocks);
  if (blocks.size() != num_blocks && shared_block_pool_ &&
      leaf->options().shared_block_pool() == shared_block_pool_) {
    size_t remaining_evictions = num_blocks_in_prefix_cache();
    while (blocks.size() != num_blocks && remaining_evictions > 0 &&
           reclaim_shared_cache()) {
      --remaining_evictions;
      blocks = leaf->allocate(num_blocks);
    }
  }
  return blocks;
}

void CompositeBlockManager::deallocate(const Slice<Block>& blocks) {
  // Route each contiguous run to its owning leaf by Block::manager().
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

std::optional<std::vector<Block>> CompositeBlockManager::allocate_for_sequence(
    Sequence* /*seq*/,
    KVCacheState& /*kv_state*/,
    size_t /*num_tokens*/) {
  NOT_IMPLEMENTED();
  return std::nullopt;
}

std::vector<Block> CompositeBlockManager::allocate_shared(
    const Slice<int32_t>& /*tokens_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
  return {};
}

void CompositeBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                  std::vector<Block>& /*blocks*/,
                                  size_t /*existed_shared_blocks_num*/,
                                  const MMData& /*mm_data*/,
                                  const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

void CompositeBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  NOT_IMPLEMENTED();
}

size_t CompositeBlockManager::num_blocks_in_prefix_cache() const {
  size_t total = 0;
  for (const auto& [type, entry] : leaves_) {
    total += entry.leaf->num_blocks_in_prefix_cache();
  }
  return total;
}

const CompositeBlockManager::LeafEntry* CompositeBlockManager::capacity_leaf()
    const {
  // Smallest block_size = finest granularity = closest to the base block the
  // scheduler assumes. KV for normal models; C4 for DSV4.
  const LeafEntry* chosen = nullptr;
  for (const auto& [type, entry] : leaves_) {
    if (!entry.participates_in_admission) {
      continue;
    }
    if (chosen == nullptr ||
        entry.leaf->block_size() < chosen->leaf->block_size()) {
      chosen = &entry;
    }
  }
  return chosen;
}

bool CompositeBlockManager::reclaim_shared_cache() {
  CHECK(shared_block_pool_ != nullptr);
  if (shared_block_pool_->uses_slabs()) {
    return reclaim_shared_prefix();
  }
  for (auto& [type, entry] : leaves_) {
    if (entry.leaf->options().shared_block_pool() != shared_block_pool_) {
      continue;
    }
    // Each call takes only this leaf lock. A cached Block destructor may
    // re-enter its own free(), but never another leaf allocator.
    if (entry.leaf->evict_prefix_cache(1) != 0) {
      return true;
    }
  }
  return false;
}

bool CompositeBlockManager::reclaim_shared_prefix() {
  BlockManager& full_leaf = *leaf_of(BlockType::KV);
  BlockManager& linear_leaf = *leaf_of(BlockType::LINEAR);
  const std::vector<PrefixCacheEntry> full = full_leaf.prefix_cache_entries();
  const std::vector<PrefixCacheEntry> linear =
      linear_leaf.prefix_cache_entries();
  using HashIndex = std::
      unordered_map<XXH3Key, size_t, FixedStringKeyHash, FixedStringKeyEqual>;
  HashIndex full_indices;
  full_indices.reserve(full.size());
  for (size_t index = 0; index < full.size(); ++index) {
    full_indices.emplace(full[index].hash, index);
  }
  std::vector<size_t> order(full.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&full](size_t left, size_t right) {
    return full[left].block_index < full[right].block_index;
  });
  std::vector<std::optional<size_t>> parents(full.size());
  std::vector<bool> complete(full.size(), false);
  for (const size_t index : order) {
    const auto parent = full[index].parent
                            ? full_indices.find(*full[index].parent)
                            : full_indices.end();
    if (parent != full_indices.end()) {
      parents[index] = parent->second;
    }
    complete[index] = full[index].chain_known &&
                      (!full[index].parent || (parents[index].has_value() &&
                                               complete[*parents[index]]));
  }
  HashIndex linear_indices;
  linear_indices.reserve(linear.size());
  for (size_t index = 0; index < linear.size(); ++index) {
    linear_indices.emplace(linear[index].hash, index);
  }
  constexpr size_t kNoCheckpoint = std::numeric_limits<size_t>::max();
  constexpr size_t kSharedCheckpoint = kNoCheckpoint - 1;
  std::vector<size_t> owners(full.size(), kNoCheckpoint);
  std::vector<std::optional<size_t>> checkpoints(full.size());
  std::vector<bool> restorable(linear.size(), false);
  for (size_t index = 0; index < full.size(); ++index) {
    const auto checkpoint = full[index].dependency
                                ? linear_indices.find(*full[index].dependency)
                                : linear_indices.end();
    if (complete[index] && checkpoint != linear_indices.end()) {
      checkpoints[index] = checkpoint->second;
      owners[index] = checkpoint->second;
      restorable[checkpoint->second] = true;
    }
  }
  // Merge checkpoint identity, rather than counting endpoints: MTP branches
  // belonging to one state still have exclusive ownership of their ancestors.
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    if (parents[*it].has_value()) {
      size_t& parent_owner = owners[*parents[*it]];
      if (parent_owner == kNoCheckpoint) {
        parent_owner = owners[*it];
      } else if (owners[*it] != kNoCheckpoint && parent_owner != owners[*it]) {
        parent_owner = kSharedCheckpoint;
      }
    }
  }
  const auto slab_benefit =
      [this](BlockType type,
             const std::vector<const PrefixCacheEntry*>& entries) {
        std::vector<int32_t> ids;
        ids.reserve(entries.size());
        for (const PrefixCacheEntry* entry : entries) {
          ids.emplace_back(entry->block_id);
        }
        return shared_block_pool_->reclaimable_slab_count(type, ids);
      };
  std::vector<const PrefixCacheEntry*> full_orphans;
  std::vector<const PrefixCacheEntry*> linear_orphans;
  full_orphans.reserve(full.size());
  linear_orphans.reserve(linear.size());
  for (size_t index = 0; index < full.size(); ++index) {
    if (owners[index] == kNoCheckpoint && full[index].evictable) {
      full_orphans.emplace_back(&full[index]);
    }
  }
  for (size_t index = 0; index < linear.size(); ++index) {
    if (!restorable[index] && linear[index].evictable) {
      linear_orphans.emplace_back(&linear[index]);
    }
  }
  const size_t full_benefit = slab_benefit(BlockType::KV, full_orphans);
  const size_t linear_benefit = slab_benefit(BlockType::LINEAR, linear_orphans);
  const auto evict_orphans =
      [](BlockManager& leaf,
         const std::vector<const PrefixCacheEntry*>& entries,
         size_t benefit) {
        bool reclaimed = false;
        for (const PrefixCacheEntry* entry : entries) {
          reclaimed = leaf.evict_cached_block(entry->hash) || reclaimed;
          if (reclaimed && benefit == 0) {
            break;
          }
        }
        return reclaimed;
      };
  const bool prefer_full = linear_orphans.empty() ||
                           (!full_orphans.empty() &&
                            (full_benefit > linear_benefit ||
                             (full_benefit == linear_benefit &&
                              full_orphans.front()->last_access_time <=
                                  linear_orphans.front()->last_access_time)));
  if (prefer_full) {
    if (evict_orphans(full_leaf, full_orphans, full_benefit) ||
        evict_orphans(linear_leaf, linear_orphans, linear_benefit)) {
      return true;
    }
  } else if (evict_orphans(linear_leaf, linear_orphans, linear_benefit) ||
             evict_orphans(full_leaf, full_orphans, full_benefit)) {
    return true;
  }

  // A checkpoint and the Full suffix used only by it form one eviction unit.
  // Shared ancestors stay cached for other recoverable checkpoints.
  std::vector<size_t> exclusive_counts(linear.size(), 0);
  for (size_t index = 0; index < full.size(); ++index) {
    if (owners[index] < linear.size() && full[index].evictable) {
      ++exclusive_counts[owners[index]];
    }
  }
  std::vector<std::vector<const PrefixCacheEntry*>> exclusive(linear.size());
  for (size_t index = 0; index < linear.size(); ++index) {
    exclusive[index].reserve(exclusive_counts[index]);
  }
  for (size_t index = 0; index < full.size(); ++index) {
    if (owners[index] < linear.size() && full[index].evictable) {
      exclusive[owners[index]].emplace_back(&full[index]);
    }
  }
  std::optional<size_t> victim;
  size_t best_benefit = 0;
  for (size_t index = 0; index < linear.size(); ++index) {
    if (!linear[index].evictable || !restorable[index]) {
      continue;
    }
    const size_t benefit = slab_benefit(BlockType::KV, exclusive[index]) +
                           shared_block_pool_->reclaimable_slab_count(
                               BlockType::LINEAR, {linear[index].block_id});
    if (!victim.has_value() || benefit > best_benefit) {
      victim = index;
      best_benefit = benefit;
    }
  }
  if (victim.has_value() &&
      linear_leaf.evict_cached_block(linear[*victim].hash)) {
    for (const PrefixCacheEntry* entry : exclusive[*victim]) {
      full_leaf.evict_cached_block(entry->hash);
    }
    return true;
  }

  // If checkpoints are pinned, an unused Full block may still be reclaimed.
  // Drop its dependent cache records where possible; live aliases remain valid.
  for (size_t index = 0; index < full.size(); ++index) {
    if (!full[index].evictable ||
        !full_leaf.evict_cached_block(full[index].hash)) {
      continue;
    }
    std::vector<bool> survives(linear.size(), false);
    for (size_t endpoint = 0; endpoint < full.size(); ++endpoint) {
      if (!checkpoints[endpoint].has_value()) {
        continue;
      }
      std::optional<size_t> current = endpoint;
      while (current.has_value() && *current != index) {
        current = parents[*current];
      }
      if (!current.has_value()) {
        survives[*checkpoints[endpoint]] = true;
      }
    }
    for (size_t state = 0; state < linear.size(); ++state) {
      if (restorable[state] && !survives[state]) {
        linear_leaf.evict_cached_block(linear[state].hash);
      }
    }
    return true;
  }
  return false;
}

size_t CompositeBlockManager::num_free_blocks() const {
  if (shared_block_pool_) {
    return shared_block_pool_->num_free_slots();
  }
  // Reports one admission leaf's raw block count. Mixing leaves of different
  // block_size would make num_free * block_size() meaningless.
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_free_blocks();
}

size_t CompositeBlockManager::num_used_blocks() const {
  if (shared_block_pool_) {
    return shared_block_pool_->num_used_slots();
  }
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_used_blocks();
}

double CompositeBlockManager::kv_cache_utilization() const {
  if (shared_block_pool_) {
    return static_cast<double>(shared_block_pool_->num_used_slots()) /
           shared_block_pool_->num_total_slots();
  }
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0.0 : leaf->leaf->kv_cache_utilization();
}

void CompositeBlockManager::free(int32_t /*block_id*/) {
  LOG(FATAL) << "CompositeBlockManager::free should not be called";
}

Block CompositeBlockManager::allocate() {
  LOG(FATAL) << "CompositeBlockManager::allocate should not be called";
  return Block();
}

size_t CompositeBlockManager::admission_capacity() const {
  if (shared_block_pool_) {
    return shared_block_pool_->num_total_slots();
  }
  const auto* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->admission_capacity();
}

size_t CompositeBlockManager::num_total_blocks() const {
  if (shared_block_pool_) {
    return shared_block_pool_->num_total_slots();
  }
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_total_blocks();
}

void CompositeBlockManager::reserve_virtual_memory_padding_blocks() {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->reserve_virtual_memory_padding_blocks();
  }
}

}  // namespace xllm
