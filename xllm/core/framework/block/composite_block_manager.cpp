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

#include "composite_block_manager.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "block_manager_impl.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "embedding_block_manager.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"
#include "linear_state_block_manager.h"
#include "sliding_window_block_manager.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) {
  CHECK_GT(denominator, 0u);
  return (numerator + denominator - 1) / denominator;
}

// Whether a leaf of the given BlockType participates in prefix cache under
// the current role. On the PREFILL side (instance_is_decode == false) every
// cache-bearing leaf participates. On the DECODE side we skip SWA and
// LINEAR because the D forward never reads their shared prefix before P
// overwrites it -- SWA hits carry gap-invalid placeholders that would break
// the grouped-response CHECK, and LINEAR restore is a net waste (D does no
// prefill). EMBEDDING has never had a prefix cache and stays out on both sides.
//
// Host offload is intentionally broader than this predicate: decode keeps an
// offload-only SWA Host leaf, while this predicate still prevents SWA from
// being probed or restored as a prefix.
bool leaf_participates_in_prefix_cache(BlockType type,
                                       bool instance_is_decode) {
  if (!instance_is_decode) {
    return true;
  }
  switch (type) {
    case BlockType::KV:
    case BlockType::C4:
    case BlockType::C128:
      return true;
    case BlockType::SWA:
    case BlockType::LINEAR:
    case BlockType::EMBEDDING:
      return false;
  }
  // Fail loudly on unhandled BlockType. Falling back to false would silently
  // disable prefix cache for the new leaf across both P and D, and every
  // request would end up re-prefilling its prompt with no visible error.
  LOG(FATAL) << "leaf_participates_in_prefix_cache: unhandled BlockType="
             << static_cast<int32_t>(type);
  return false;
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

// Xtensor VMM manager or flat free-list BlockManagerImpl. Xtensor has no
// prefix cache.
std::unique_ptr<BlockManager> make_kv_leaf(const BlockManager::Options& kv_opts,
                                           int32_t dp_rank) {
  if (!kv_opts.enable_xtensor()) {
    return std::make_unique<BlockManagerImpl>(kv_opts);
  }
  CHECK_GT(kv_opts.num_layers(), 0)
      << "num_layers must be set when enable_xtensor is true";
  CHECK_GT(kv_opts.slot_size(), 0)
      << "slot_size must be set when enable_xtensor is true";
  const size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  // K and V share block size; divide by 2.
  const size_t block_mem_size =
      static_cast<size_t>(kv_opts.block_size()) * kv_opts.slot_size() / 2;
  return std::make_unique<XTensorBlockManagerImpl>(kv_opts,
                                                   kv_opts.num_layers(),
                                                   block_mem_size,
                                                   page_size,
                                                   dp_rank,
                                                   kv_opts.model_id());
}

}  // namespace

CompositeBlockManager::LeafMap build_composite_leaves(
    const BlockManager::Options& options,
    int32_t dp_rank) {
  CompositeBlockManager::LeafMap leaves;

  const bool prefix_cache_on = options.enable_prefix_cache();
  const bool is_decode = options.instance_is_decode();
  const bool linear_participates =
      leaf_participates_in_prefix_cache(BlockType::LINEAR, is_decode);
  const bool swa_participates =
      leaf_participates_in_prefix_cache(BlockType::SWA, is_decode);
  const bool c4_participates =
      leaf_participates_in_prefix_cache(BlockType::C4, is_decode);
  const bool c128_participates =
      leaf_participates_in_prefix_cache(BlockType::C128, is_decode);
  const bool kv_participates =
      leaf_participates_in_prefix_cache(BlockType::KV, is_decode);

  // LINEAR resource leaf (Qwen3.5-Next GDN). Additive on top of the KV
  // family: a GDN model holds both KV and LINEAR. Not an admission leaf
  // (block_size==1 would misreport pool capacity). Scheduler-thread only, so
  // no ConcurrentBlockManagerImpl wrap. supports_prefix_cache follows the
  // role predicate; on DECODE it is off so probe_prefix_cache skips the
  // leaf and the composite classifies FLAT_KV_LINEAR down to FLAT_KV.
  if (options.enable_linear_state()) {
    CHECK_GT(options.linear_state_num_slots(), 0)
        << "linear_state_num_slots must be set when linear state is enabled";
    const bool linear_prefix_cache = prefix_cache_on && linear_participates;
    int32_t chunk_stride =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    // The chunk stride is the checkpoint boundary for the LINEAR prefix cache.
    // It only matters when that cache is on (PREFILL role); the LINEAR leaf
    // holds no token cache otherwise (see set_kv_cache_info's LINEAR skip) and
    // each sequence takes exactly one slot regardless of block_size. When the
    // cache is off (DECODE role) chunked prefill is legitimately disabled and
    // the stride is left at its -1 default, so require a positive stride only
    // when the LINEAR prefix cache is actually enabled, and fall back to a
    // valid positive block_size otherwise so the pool sizing stays well-formed.
    if (linear_prefix_cache) {
      CHECK_GT(chunk_stride, 0) << "max_tokens_per_chunk_for_prefill must be "
                                   "positive for linear state prefix cache";
    } else if (chunk_stride <= 0) {
      chunk_stride = 1;
    }
    leaves.emplace(
        BlockType::LINEAR,
        CompositeBlockManager::LeafEntry{
            std::make_unique<LinearStateBlockManager>(
                static_cast<uint32_t>(options.linear_state_num_slots()),
                chunk_stride,
                linear_prefix_cache),
            /*participates_in_admission=*/false,
            /*supports_prefix_cache=*/linear_prefix_cache});
  }

  if (options.manager_types().empty()) {
    // Normal / Qwen / xtensor: a single KV leaf. Prefix cache is on only for
    // the flat (non-xtensor) KV leaf.
    BlockManager::Options kv_opts = options;
    kv_opts.block_type(BlockType::KV);
    const bool kv_prefix_cache =
        prefix_cache_on && !options.enable_xtensor() && kv_participates;
    kv_opts.enable_prefix_cache(kv_prefix_cache);
    leaves.emplace(
        BlockType::KV,
        CompositeBlockManager::LeafEntry{
            maybe_concurrent(make_kv_leaf(kv_opts, dp_rank), options),
            /*participates_in_admission=*/true,
            /*supports_prefix_cache=*/kv_prefix_cache});
    return leaves;
  }

  // DSV4: SWA + compressed (C4 / C128) leaves.
  const size_t n = options.manager_types().size();
  CHECK_EQ(n, options.compress_ratios().size())
      << "manager_types and compress_ratios must have the same size";
  CHECK_GT(n, 0u) << "Composite requires at least one sub-manager";

  for (size_t i = 0; i < n; ++i) {
    const uint32_t type = options.manager_types()[i];
    const uint32_t compress_ratio = options.compress_ratios()[i];
    BlockManager::Options opts = options;

    if (type == kManagerTypeBlockManagerImpl) {
      opts.block_size(static_cast<uint32_t>(options.block_size()) *
                      compress_ratio);
      opts.num_blocks(static_cast<uint32_t>(options.num_blocks()) /
                      compress_ratio);
      CHECK(compress_ratio == 4 || compress_ratio == 128)
          << "unexpected compress_ratio " << compress_ratio
          << " for composite BlockManagerImpl sub-manager";
      const BlockType key =
          compress_ratio == 4 ? BlockType::C4 : BlockType::C128;
      const bool compressed_participates =
          key == BlockType::C4 ? c4_participates : c128_participates;
      const bool compressed_prefix_cache =
          prefix_cache_on && compressed_participates;
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
      const uint32_t max_seqs = std::max(options.max_seqs_per_batch(), 1u);
      const uint32_t burst_blocks =
          ceil_div(std::max(options.max_tokens_per_batch(), 1u),
                   static_cast<uint32_t>(options.block_size()));
      // Slack fits the peak "old blocks not yet released + new tail".
      const uint32_t swa_total_blocks =
          swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;
      const bool swa_prefix_cache = prefix_cache_on && swa_participates;
      opts.num_blocks(swa_total_blocks)
          .swa_blocks_per_seq(swa_blocks_per_seq)
          .sliding_window_size(sliding_window_size)
          .block_type(BlockType::SWA)
          .enable_prefix_cache(swa_prefix_cache)
          .hasher_type(options.hasher_type());
      // SWA is not an admission leaf (pool sized by ring, not token budget)
      // but serves gap-tolerant prefix cache.
      leaves.emplace(
          BlockType::SWA,
          CompositeBlockManager::LeafEntry{
              maybe_concurrent(
                  std::make_unique<SlidingWindowBlockManager>(opts), options),
              /*participates_in_admission=*/false,
              /*supports_prefix_cache=*/swa_prefix_cache});
    } else {
      LOG(FATAL) << "Unknown manager_type " << type;
    }
  }
  return leaves;
}

CompositeBlockManager::CompositeBlockManager(LeafMap leaves)
    : BlockManager(BlockManager::Options()),
      leaves_(std::move(leaves)),
      combination_(classify_leaf_combination(leaves_)) {
  CHECK(!leaves_.empty()) << "CompositeBlockManager requires at least one leaf";
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

void CompositeBlockManager::cache_full_blocks_for_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  KVCacheState& kv = seq->kv_state();
  for (auto& [type, entry] : leaves_) {
    // EMBEDDING / LINEAR hold no token cache. KV also participates here so a
    // Host-restored prefix is published immediately after its HBM destination
    // blocks are allocated; cache_for_sequence later flushes only the tail.
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
    const size_t num_full = seq->kv_cache_tokens_num() / block_size;
    const size_t cached = kv.num_cached_blocks(type);
    const size_t end = std::min(num_full, blocks->size());
    if (end <= cached) {
      continue;
    }
    // Clamp tokens to `end * block_size`. The leaf's cache() re-derives its own
    // n_blocks bound from `tokens.size() / block_size_`; without this clamp,
    // chunked prefill (where seq->tokens() spans the whole prompt but only
    // `num_full` blocks are actually forwarded) would let the leaf stamp a
    // partial tail block with a full-block hash key and admit garbage KV into
    // the prefix cache. Also cap at seq->tokens().size(): unit tests build
    // sequences whose token vector is shorter than kv_cache_tokens_num.
    const size_t token_end = std::min(end * block_size, seq->tokens().size());
    if (token_end / block_size <= cached) {
      continue;
    }
    seq->update_block_hashes(static_cast<uint32_t>(block_size),
                             leaf.options().hasher_type());
    leaf.cache(seq->tokens().slice(0, token_end),
               *blocks,
               cached,
               seq->mm_data(),
               seq->block_hashes());
    kv.set_num_cached_blocks(type, token_end / block_size);
  }
}

bool CompositeBlockManager::allocate_sequence(Sequence* seq,
                                              size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  KVCacheState& kv_state = seq->kv_state();

  // Fan out growth. Each leaf returns its newly allocated blocks (or nullopt
  // on failure). Stage keyed by BlockType; commit only after every leaf
  // succeeds so a failure rolls back cleanly.
  std::map<BlockType, std::vector<Block>> staged;

  auto release_staged = [&]() {
    for (auto& [type, blocks] : staged) {
      leaf_of(type)->deallocate(blocks);
    }
    staged.clear();
  };

  for (auto& [type, entry] : leaves_) {
    std::optional<std::vector<Block>> blocks =
        entry.leaf->allocate_for_sequence(seq, num_tokens);
    if (!blocks.has_value()) {
      release_staged();
      return false;
    }
    if (!blocks->empty()) {
      staged.emplace(type, std::move(*blocks));
    }
  }

  // Grow-or-fail: every cache-bearing leaf must cover num_tokens now.
  // Without this a leaf that mistakenly returned an empty vector under
  // pressure would let the pool report admission success while the device
  // is under-provisioned -- batch_input_builder's CHECK then fires
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
    const size_t total = seq->kv_state().num_blocks(type) + staged_for_type;
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
  cache_full_blocks_for_sequence(seq);

  // Release slid-out SWA blocks only after they have had a chance to enter the
  // prefix cache. Other leaves are no-ops here.
  release_out_of_window_for_sequence(seq);
  return true;
}

void CompositeBlockManager::release_out_of_window_for_sequence(Sequence* seq) {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->release_out_of_window(seq);
  }
}

void CompositeBlockManager::deallocate_for_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  // Publish prefix cache first, then release blocks. seq->reset() belongs to
  // the pool caller.
  cache_for_sequence(seq);
  for (auto& [type, entry] : leaves_) {
    entry.leaf->deallocate(seq->kv_state().blocks(type));
  }
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
        leaf.allocate_shared(seq->tokens(), existed, seq->mm_data(), hashes),
        /*block_size=*/leaf.block_size()});
  }
  return probes;
}

void CompositeBlockManager::release_probes(std::vector<ProbeResult>* probes) {
  if (probes == nullptr) {
    return;
  }
  // Clear aliases immediately after returning them. Concurrent deallocate
  // accounting depends on observing only the cache and current caller refs.
  for (ProbeResult& probe : *probes) {
    if (probe.blocks.empty()) {
      continue;
    }
    probe.leaf->deallocate(probe.blocks);
    probe.blocks.clear();
  }
  probes->clear();
}

namespace {

using ProbeResult = CompositeBlockManager::ProbeResult;

// Trim outcome. `to_mount` feeds the shape's mount step; `to_drop` goes to
// leaf->deallocate() (cache release, not physical free).
// `linear_restore_src`, when set, is the LINEAR checkpoint that the caller
// stashes via Sequence::set_linear_restore_src_block.
struct TrimOutcome {
  std::vector<ProbeResult> to_mount;
  std::vector<ProbeResult> to_drop;
  size_t safe_hit_tokens = 0;
  std::optional<Block> linear_restore_src;
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

// FLAT_KV_LINEAR: extract LINEAR's deepest checkpoint as a restore source
// (its vector is `[inv, ..., deepest_valid]` at chunk stride, so reach in
// tokens is `blocks.size() * block_size`), then clamp KV shared blocks to
// that recoverable budget.
TrimOutcome trim_flat_kv_linear(std::vector<ProbeResult> probes) {
  auto linear = take_probe(probes, BlockType::LINEAR);
  CHECK_EQ(probes.size(), 1u)
      << "FLAT_KV_LINEAR expects one KV probe (LINEAR removed above)";
  TrimOutcome out;

  size_t linear_recoverable_tokens = 0;
  if (linear.has_value()) {
    linear_recoverable_tokens = linear->blocks.size() * linear->block_size;
    for (auto it = linear->blocks.rbegin(); it != linear->blocks.rend(); ++it) {
      if (it->is_valid()) {
        out.linear_restore_src = std::move(*it);
        break;
      }
    }
  }

  ProbeResult& kv = probes.front();
  const size_t kv_block_size = kv.block_size;
  const size_t recoverable_blocks =
      kv_block_size == 0 ? 0 : linear_recoverable_tokens / kv_block_size;
  const size_t safe_count = std::min(kv.blocks.size(), recoverable_blocks);
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
  return out;
}

// SWA_COMPRESSED: cross-leaf min -> C128-stride clamp -> SWA tail-continuity
// (fallback in C128 steps) -> exact-repeat pop -> per-leaf trim.
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

  // Exact-repeat pop: forward needs at least one C128 block to compute.
  if (safe_hit_tokens == prompt_tokens && safe_hit_tokens >= c128_block_size) {
    safe_hit_tokens -= c128_block_size;
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

void CompositeBlockManager::allocate_shared_for_sequence(Sequence* seq) {
  // Shared flow: probe -> trim -> mount. Trim strategy is per-shape (see
  // trim_flat_kv / trim_flat_kv_linear / trim_swa_compressed).
  if (seq == nullptr || combination_ == LeafCombination::UNSUPPORTED) {
    return;
  }
  std::vector<ProbeResult> probes = probe_prefix_cache(seq, leaves_);
  if (probes.empty()) {
    return;
  }

  TrimOutcome trimmed;
  switch (combination_) {
    case LeafCombination::FLAT_KV:
      trimmed = trim_flat_kv(std::move(probes));
      break;
    case LeafCombination::FLAT_KV_LINEAR:
      trimmed = trim_flat_kv_linear(std::move(probes));
      break;
    case LeafCombination::SWA_COMPRESSED:
      trimmed = trim_swa_compressed(std::move(probes), seq->tokens().size());
      break;
    case LeafCombination::UNSUPPORTED:
      return;
  }

  release_probes(&trimmed.to_drop);
  if (trimmed.linear_restore_src.has_value()) {
    seq->set_linear_restore_src_block(std::move(*trimmed.linear_restore_src));
  }

  // FLAT_KV{,_LINEAR} defer to Sequence::add_shared_blocks, which owns replace
  // and exact-repeat. Composite layouts mount every leaf before advancing the
  // sequence-level token count once.
  switch (combination_) {
    case LeafCombination::FLAT_KV:
    case LeafCombination::FLAT_KV_LINEAR: {
      if (!trimmed.to_mount.empty()) {
        ProbeResult& kv = trimmed.to_mount.front();
        seq->add_shared_blocks(kv.type, std::move(kv.blocks));
      }
      break;
    }
    case LeafCombination::SWA_COMPRESSED: {
      for (ProbeResult& probe : trimmed.to_mount) {
        seq->kv_state().mount_composite_shared(probe.type,
                                               std::move(probe.blocks));
      }
      seq->kv_state().set_kv_cache_tokens_num(trimmed.safe_hit_tokens);
      break;
    }
    case LeafCombination::UNSUPPORTED:
      break;
  }
}

void CompositeBlockManager::cache_for_sequence(Sequence* seq) {
  // Final flush at deallocate. KV shapes flush the tail via leaf->cache();
  // SWA_COMPRESSED re-runs the pre-grow hook (cursor-guarded, idempotent) so
  // the window-tail block that never triggered another allocate_sequence
  // still makes it into the cache.
  if (seq == nullptr || combination_ == LeafCombination::UNSUPPORTED) {
    return;
  }
  switch (combination_) {
    case LeafCombination::FLAT_KV:
    case LeafCombination::FLAT_KV_LINEAR: {
      BlockManager& kv_leaf = *leaf_of(BlockType::KV);
      seq->update_block_hashes(static_cast<uint32_t>(kv_leaf.block_size()),
                               kv_leaf.options().hasher_type());
      KVCacheState& kv_state = seq->kv_state();
      std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::KV);
      const size_t block_size = kv_leaf.block_size();
      const Slice<int32_t> cached_tokens = seq->cached_tokens();
      const size_t publish_end =
          std::min(cached_tokens.size() / block_size, blocks->size());
      const size_t publish_begin = kv_state.num_cached_blocks(BlockType::KV);
      kv_leaf.cache(cached_tokens,
                    *blocks,
                    publish_begin,
                    seq->mm_data(),
                    seq->block_hashes());
      kv_state.set_num_cached_blocks(BlockType::KV,
                                     std::max(publish_begin, publish_end));
      break;
    }
    case LeafCombination::SWA_COMPRESSED: {
      cache_full_blocks_for_sequence(seq);
      break;
    }
    case LeafCombination::UNSUPPORTED:
      break;
  }
}

void CompositeBlockManager::cache_for_sequence(Sequence* seq,
                                               size_t num_tokens) {
  // Chunked-prefill mid-step: only KV needs a token-clamped flush now;
  // SWA / C4 / C128 get their flush from the pre-grow hook at the top of
  // the next allocate_sequence.
  if (seq == nullptr || combination_ == LeafCombination::UNSUPPORTED) {
    return;
  }
  switch (combination_) {
    case LeafCombination::FLAT_KV:
    case LeafCombination::FLAT_KV_LINEAR: {
      BlockManager& kv_leaf = *leaf_of(BlockType::KV);
      KVCacheState& kv_state = seq->kv_state();
      const size_t block_size = kv_leaf.block_size();
      const size_t available_tokens_num =
          std::min({num_tokens,
                    kv_state.num_blocks(BlockType::KV) * block_size,
                    seq->tokens().size()});
      const size_t publish_begin = kv_state.num_cached_blocks(BlockType::KV);
      if (available_tokens_num > publish_begin * block_size) {
        seq->update_block_hashes(static_cast<uint32_t>(block_size),
                                 kv_leaf.options().hasher_type());
        std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::KV);
        CHECK_GE(blocks->size(), publish_begin);
        kv_leaf.cache(seq->tokens().slice(0, available_tokens_num),
                      *blocks,
                      publish_begin,
                      seq->mm_data(),
                      seq->block_hashes());
        kv_state.set_num_cached_blocks(BlockType::KV,
                                       available_tokens_num / block_size);
      }
      break;
    }
    case LeafCombination::SWA_COMPRESSED:
    case LeafCombination::UNSUPPORTED:
      break;
  }
}

std::vector<Block> CompositeBlockManager::allocate_blocks(BlockType type,
                                                          size_t num_blocks) {
  BlockManager* leaf = leaf_of(type);
  CHECK(leaf != nullptr) << "CompositeBlockManager has no leaf for block type "
                         << static_cast<int>(type);
  return leaf->allocate(num_blocks);
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
    size_t /*num_tokens*/) {
  // The pool drives the composite via allocate_sequence(); the leaf-level
  // entry point is meaningless on the composite itself.
  NOT_IMPLEMENTED();
  return std::nullopt;
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

void CompositeBlockManager::reset_prefix_cache() {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->reset_prefix_cache();
  }
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

size_t CompositeBlockManager::num_free_blocks() const {
  // Reports one admission leaf's raw block count. Mixing leaves of different
  // block_size would make num_free * block_size() meaningless.
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_free_blocks();
}

size_t CompositeBlockManager::num_used_blocks() const {
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_used_blocks();
}

double CompositeBlockManager::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks()) / static_cast<double>(total);
}

void CompositeBlockManager::free(int32_t /*block_id*/) {
  LOG(FATAL) << "CompositeBlockManager::free should not be called";
}

Block CompositeBlockManager::allocate() {
  LOG(FATAL) << "CompositeBlockManager::allocate should not be called";
  return Block();
}

size_t CompositeBlockManager::num_total_blocks() const {
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_total_blocks();
}

void CompositeBlockManager::reserve_xtensor_padding_blocks() {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->reserve_xtensor_padding_blocks();
  }
}

}  // namespace xllm
