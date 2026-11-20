/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>

#include "acl/acl.h"

namespace xllm::kernel::npu::tilelang {

// Frozen MiniMax-H3 RainFusion device constants. These mirror the TileLang DSL
// modules h3_ref2va_rf_v2_selector.py / h3_rainfusion_attention.py and must not
// drift from them: the AOT specializations are compiled against exactly these
// values and the wrappers reject anything else before launch.
inline constexpr int64_t kH3RainfusionBlock = 128;
inline constexpr int64_t kH3RainfusionHeads = 4;
inline constexpr int64_t kH3RainfusionHeadDim = 128;
inline constexpr int64_t kH3RainfusionNumCores = 24;
inline constexpr int64_t kH3RainfusionVecNum = 2;
inline constexpr int64_t kH3RainfusionWorkers =
    kH3RainfusionNumCores * kH3RainfusionVecNum;
inline constexpr int64_t kH3RainfusionChunkRows = 64;
inline constexpr int64_t kH3RainfusionMTile = 16;
inline constexpr int64_t kH3RainfusionPlanAlign = 8;
inline constexpr int64_t kH3RainfusionStackBlocks = 4;
inline constexpr int64_t kH3RainfusionTaskSlots = 3;
inline constexpr int64_t kH3RainfusionRingSlots = 2;
inline constexpr int64_t kH3RainfusionKvPackRows = 256;

// rf_v2 record flags (MindIE-SD sparse_flash_attn_rf_v2 semantics, Mulan PSL
// v2).
inline constexpr int32_t kH3RainfusionFlagFirstFrame = 1;
inline constexpr int32_t kH3RainfusionFlagDense = 2;
inline constexpr int32_t kH3RainfusionFlagTail = 4;

// The qualified packed geometry of one local-U8 rank. Every field is a
// compile-time axis of the two AOT specializations, so a request whose derived
// geometry differs in any field must fall back before launching.
// The CAPACITY tuple is the single source of truth for every physical buffer,
// stride and GM extent, and it is what the AOT registry dispatches on. The
// logical geometry stays a runtime argument pair-for-pair with it: one admitted
// (selector, attention) capacity object serves every logical geometry that
// fits.
//
// Storage note: total_rows is the TND STORAGE extent and must be explicitly
// 256-aligned (kH3RainfusionKvPackRows), while valid_rows counts real tokens
// only. For 1440x736 valid_rows is 60891 and storage is round_up(60891,256) =
// 60928; the two must never be conflated.
struct H3RainfusionCapacity {
  int64_t total_rows = 0;
  int64_t num_blocks = 0;
  int64_t record_count = 0;
  int64_t records_padded = 0;
  int64_t topk_r = 0;
  int64_t plan_width = 0;
  int64_t q_pool_pad = 0;
  int64_t k_pool_pad = 0;
};

// The one qualified capacity both kernels are compiled against. Admitting the
// pair means both objects exist for exactly this tuple; a logical geometry is
// dispatchable only when it fits, and the fit is checked before any launch.
inline constexpr H3RainfusionCapacity kH3RainfusionQualifiedCapacity{
    /*total_rows=*/61952,
    /*num_blocks=*/483,
    /*record_count=*/5568,
    /*records_padded=*/5568,
    /*topk_r=*/97,
    /*plan_width=*/488,
    /*q_pool_pad=*/480,
    /*k_pool_pad=*/496};

// The selector unrolls cap_n_tiles = ceildiv(k_pool_pad, N_TILE) tiles and
// guards every tile at or beyond ceil(rt_total_blocks/N_TILE), so an inactive
// tile performs no GM copy, no event set/wait, no MMA and no FIX store, and L0A
// is released on the last ACTIVE tile with no MMA after it. The logical tile
// count therefore only has to be
// <= cap_n_tiles; it no longer has to equal it.
inline constexpr int64_t kH3RainfusionNTile = 128;
inline constexpr int64_t kH3RainfusionCapNTiles = 4;  // ceildiv(496, 128)

struct H3RainfusionGeometry {
  int64_t total_rows = 0;
  int64_t valid_rows = 0;
  int64_t video_start = 0;
  int64_t grid_t = 0;
  int64_t grid_h = 0;
  int64_t grid_w = 0;
  int64_t topk_r = 0;
};

// Quantities derived from the geometry. Mirrors _derive() in the frozen
// selector DSL module; the pool/ring extents below are the exact shapes the
// AOT-compiled selector expects.
struct H3RainfusionDerived {
  int64_t video_rows = 0;
  int64_t video_end = 0;
  int64_t total_blocks = 0;
  int64_t tail_rows = 0;
  int64_t ff_rows = 0;
  int64_t ff_blocks = 0;
  int64_t video_full_blocks = 0;
  int64_t num_sq = 0;
  int64_t q_pool_pad = 0;
  int64_t k_pool_pad = 0;
  int64_t plan_width = 0;
};

inline constexpr int64_t h3_ceildiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

inline constexpr int64_t h3_align_up(int64_t a, int64_t b) {
  return h3_ceildiv(a, b) * b;
}

// Pure host arithmetic; throws std::invalid_argument on an unqualified
// geometry so dispatch can fall back instead of launching.
H3RainfusionDerived h3_rainfusion_derive(const H3RainfusionGeometry& geometry);

// Active record count (the frozen delivery value is 5557 for the qualified
// geometry). This is the extent the attention inverse epilogue iterates over;
// it is NOT the physical table extent.
int64_t h3_rainfusion_record_count(const H3RainfusionGeometry& geometry);

// Physical [N,4] table extent, always a multiple of the 48 AIV workers.
int64_t h3_rainfusion_records_padded(const H3RainfusionGeometry& geometry);

// Build the constant worker-major rf_v2 reorder record table on `device`.
// Pure function of the geometry: build once per rank and reuse across every
// layer, forward and request.
torch::Tensor build_h3_rainfusion_reorder_records(
    const H3RainfusionGeometry& geometry,
    const torch::Device& device);

// One combined workspace per rank, reused sequentially by every executed
// block. The pipeline runs the 50 blocks serially on one stream, so per-layer
// workspaces are forbidden and concurrent use of one workspace is unsafe.
struct H3RainfusionWorkspace {
  // Record extents, resolved once at allocation. The active count drives the
  // attention inverse epilogue's per-worker iteration bound and is a dispatch
  // axis of both kernels, so it must never be recomputed on the hot path nor
  // confused with the physical table extent.
  int64_t record_count = 0;
  int64_t records_padded = 0;
  // Constant input, built once.
  torch::Tensor reorder_records;  // int32 [records_padded, 4]
  // Selector outputs, consumed directly by the attention launch.
  torch::Tensor plan_ids;     // int32 [total_blocks, heads, plan_width]
  torch::Tensor plan_count;   // int32 [total_blocks, 8]
  torch::Tensor reordered_q;  // bf16 [total_rows, heads*dim]
  torch::Tensor reordered_k;
  torch::Tensor reordered_v;
  // Selector persistent scratch.
  torch::Tensor q_pool;      // bf16 [heads, q_pool_pad, dim]
  torch::Tensor k_pool;      // bf16 [heads, k_pool_pad, dim]
  torch::Tensor score_ring;  // fp32 [cores, ring_slots, m_tile, k_pool_pad]
  // Attention scratch.
  torch::Tensor workspace_s;  // fp32 [slots, cores, block, stack*block]
  torch::Tensor workspace_p;  // bf16 [slots, cores, block, stack*block]
  torch::Tensor workspace_o;  // fp32 [slots, cores, block, dim]
  torch::Tensor packed_k;     // bf16 [heads, total_rows, dim]
  torch::Tensor packed_v;
  torch::Tensor reordered_output;  // bf16 [total_rows, heads*dim]
};

// True only when both AOT specializations for this exact geometry are present
// in the generated registries. Must be checked on every rank before the first
// block so the backend choice reaches consensus. Resolves the record extents
// once, so it belongs in request setup and not in the per-block path.
bool has_h3_rainfusion_specialization(const H3RainfusionGeometry& geometry);

// Classifies WHY the capacity pair cannot serve this geometry, so a legal-but-
// unsupported request reports a specific dense-fallback reason instead of one
// opaque string. Returns "" when the geometry is servable, i.e. exactly when
// has_h3_rainfusion_specialization() is true. Depends only on the geometry,
// which is identical on every rank, so the reason is rank-symmetric and safe to
// log per rank. Never throws.
const char* h3_rainfusion_unavailability_reason(
    const H3RainfusionGeometry& geometry);

H3RainfusionWorkspace allocate_h3_rainfusion_workspace(
    const H3RainfusionGeometry& geometry,
    const torch::Device& device);

// Counts successful submissions to the stream, not device completion.
// A successful request still requires an explicit final stream synchronize.
struct H3RainfusionLaunchCounters {
  int64_t selector_launches = 0;
  int64_t attention_launches = 0;
};

// The complete hot path: selector then final attention, both on the current
// NPU stream, with no host synchronization and no ATen operation between the
// two launches. `output` is written in the original packed row order; rows
// [valid_rows, total_rows) are never touched and stay caller-owned.
void h3_rainfusion_attention_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const H3RainfusionGeometry& geometry,
    H3RainfusionWorkspace& workspace,
    torch::Tensor& output,
    H3RainfusionLaunchCounters* counters = nullptr);

// Qualified RainFusion geometry profiles. The frozen production profile is
// preserved byte-for-byte; the snow Ref2VA profile is ADDITIVE and was derived
// from the real one-card official Qwen3-VL layer-50 encoder observation
// (11110 condition tokens => valid 59892, aligned 59904, video_start 22596),
// never from CLI literals. Grid and topk_r are shared production constants:
// topk_r is the fixed production Top-R budget, so keeping it at 94 keeps the
// block-selection strategy unchanged (94 <= total_blocks 468 holds).
// Keep ceil(20% of the KV blocks) before applying the dense protection set.
inline constexpr int64_t kH3RainfusionSparsityPercent = 20;  // keep 1 - 0.8

inline int64_t h3_rainfusion_derive_topk_r(int64_t total_blocks) {
  return (total_blocks * kH3RainfusionSparsityPercent + 99) / 100;
}

inline bool is_qualified_h3_rainfusion_profile(
    const H3RainfusionGeometry& geometry) {
  if (geometry.total_rows <= 0 || geometry.valid_rows <= 0 ||
      geometry.valid_rows > geometry.total_rows) {
    return false;
  }
  // Storage must be explicitly block- and KvPack-aligned, and must cover the
  // full tail block read. valid_rows counts real tokens only; total_rows is TND
  // storage.
  if (geometry.total_rows % kH3RainfusionBlock != 0 ||
      geometry.total_rows % kH3RainfusionKvPackRows != 0) {
    return false;
  }
  const int64_t total_blocks =
      (geometry.valid_rows + kH3RainfusionBlock - 1) / kH3RainfusionBlock;
  if (total_blocks * kH3RainfusionBlock > geometry.total_rows) {
    return false;
  }
  if (geometry.grid_t <= 0 || geometry.grid_h <= 0 || geometry.grid_w <= 0) {
    return false;
  }
  const int64_t video_rows =
      geometry.grid_t * geometry.grid_h * geometry.grid_w;
  if (geometry.video_start < 0 ||
      geometry.video_start + video_rows > geometry.valid_rows) {
    return false;
  }
  // The first-frame / video-block relation is NOT a legality question. For any
  // legal packed slice, floor(grid_t*grid_h*grid_w / 128) <= ceil(grid_h*grid_w
  // / 128) simply means RainFusion has no sparse video query to select, so it
  // is NOT APPLICABLE and must reach dense through the existing all-rank
  // availability aggregation exactly like every other unsupported geometry.
  // Rejecting it here made the caller throw BEFORE the collective, so a legal
  // small geometry (e.g. 32x32/124: grid 37x1x1, prefix 22596, valid 22633, 177
  // blocks, ff_blocks 1, video_full_blocks 0) was reported as a bad request
  // instead of as dense, and one rank could throw while the others entered the
  // world collective. Legality is established by the extent, alignment and span
  // checks above. The sparse derive() stays STRICT and still throws for this
  // case, which has_h3_rainfusion_specialization() converts into "unavailable"
  // and h3_rainfusion_unavailability_reason() names
  // no_sparse_video_query_blocks. The fixed-K strategy: K is derived, then
  // bounded by the block count.
  if (geometry.topk_r != h3_rainfusion_derive_topk_r(total_blocks) ||
      geometry.topk_r < 1 || geometry.topk_r > total_blocks) {
    return false;
  }
  return true;
}

namespace detail {

// Per-family predicates and launchers. Each wrapper translation unit owns its
// own generated registry include, so the facade reaches the registries only
// through these entry points and never includes a registry itself. The
// record extents are always passed in by the caller: deriving them rebuilds
// the full span table, which must not happen per block.
// Capacity-keyed admission: the registry key is the capacity tuple, so these
// resolve the pair independently of the logical geometry.
bool has_selector_capacity_specialization();
bool has_attention_capacity_specialization();
// The variant keys the runtime actually selected, so a per-rank receipt can
// record what was dispatched rather than what a registry happens to list. Empty
// when the capacity object is absent. Declared in detail because that is where
// the wrappers define them; declaring them at the enclosing scope produced an
// undefined reference at link time.
const char* h3_rainfusion_selected_selector_key();
const char* h3_rainfusion_selected_attention_key();
// Fits the logical geometry inside the qualified capacity, including the
// explicit cap_n_tiles scope limit. Returns false (never throws) so the caller
// can fall back to dense and report it as dense.
bool h3_rainfusion_logical_fits_capacity(const H3RainfusionGeometry& geometry,
                                         const H3RainfusionDerived& derived,
                                         int64_t record_count);
void validate_selector(const torch::Tensor& query,
                       const torch::Tensor& key,
                       const torch::Tensor& value,
                       const H3RainfusionGeometry& geometry,
                       const H3RainfusionDerived& derived,
                       const H3RainfusionWorkspace& workspace);

void launch_selector(const torch::Tensor& query,
                     const torch::Tensor& key,
                     const torch::Tensor& value,
                     const H3RainfusionGeometry& geometry,
                     const H3RainfusionDerived& derived,
                     const H3RainfusionWorkspace& workspace,
                     aclrtStream stream);

void validate_attention(const H3RainfusionGeometry& geometry,
                        const H3RainfusionDerived& derived,
                        const H3RainfusionWorkspace& workspace,
                        torch::Tensor& output);

void launch_attention(const H3RainfusionGeometry& geometry,
                      const H3RainfusionDerived& derived,
                      const H3RainfusionWorkspace& workspace,
                      torch::Tensor& output,
                      aclrtStream stream);

}  // namespace detail

}  // namespace xllm::kernel::npu::tilelang
