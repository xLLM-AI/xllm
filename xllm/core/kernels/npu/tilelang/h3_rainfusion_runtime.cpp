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

#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kH3RainfusionPoolAlign = 16;

void check_geometry(const H3RainfusionGeometry& geometry) {
  // Bound arithmetic before multiplying dimensions or narrowing dispatch axes.
  constexpr int64_t kMaxAxis = std::numeric_limits<int32_t>::max();
  if (geometry.total_rows <= 0 || geometry.total_rows > kMaxAxis ||
      geometry.grid_t <= 0 || geometry.grid_h <= 0 || geometry.grid_w <= 0 ||
      geometry.grid_t > kMaxAxis / geometry.grid_h ||
      geometry.grid_t * geometry.grid_h > kMaxAxis / geometry.grid_w ||
      geometry.video_start < 0 || geometry.video_start > kMaxAxis ||
      geometry.topk_r <= 0) {
    throw std::invalid_argument(
        "TileLang H3 RainFusion: invalid dispatch axes");
  }
  if (geometry.total_rows % kH3RainfusionBlock != 0 ||
      geometry.valid_rows <= 0 || geometry.valid_rows > geometry.total_rows) {
    throw std::invalid_argument(
        "TileLang H3 RainFusion: total_rows must be block aligned and cover "
        "valid_rows");
  }
  if (geometry.video_start < 0 || geometry.grid_t <= 0 ||
      geometry.grid_h <= 0 || geometry.grid_w <= 0) {
    throw std::invalid_argument(
        "TileLang H3 RainFusion: video start and grid dimensions must be "
        "positive");
  }
  if (geometry.total_rows % kH3RainfusionKvPackRows != 0) {
    throw std::invalid_argument(
        "TileLang H3 RainFusion: total_rows must be divisible by the 256-row "
        "K/V pack tile");
  }
}

}  // namespace

H3RainfusionDerived h3_rainfusion_derive(const H3RainfusionGeometry& geometry) {
  check_geometry(geometry);
  H3RainfusionDerived derived;
  derived.video_rows = geometry.grid_t * geometry.grid_h * geometry.grid_w;
  derived.total_blocks = h3_ceildiv(geometry.valid_rows, kH3RainfusionBlock);
  derived.tail_rows =
      geometry.valid_rows - (derived.total_blocks - 1) * kH3RainfusionBlock;
  derived.ff_rows = geometry.grid_h * geometry.grid_w;
  derived.ff_blocks = h3_ceildiv(derived.ff_rows, kH3RainfusionBlock);
  derived.video_full_blocks = derived.video_rows / kH3RainfusionBlock;
  derived.num_sq = derived.video_full_blocks - derived.ff_blocks;
  derived.q_pool_pad = h3_align_up(derived.num_sq, kH3RainfusionMTile);
  derived.k_pool_pad =
      h3_align_up(derived.total_blocks, kH3RainfusionPoolAlign);
  derived.plan_width =
      h3_align_up(derived.total_blocks, kH3RainfusionPlanAlign);
  derived.video_end = geometry.video_start + derived.video_rows;

  if (derived.video_end > geometry.valid_rows ||
      derived.ff_blocks > derived.video_full_blocks || derived.num_sq < 1 ||
      geometry.topk_r > derived.total_blocks) {
    throw std::invalid_argument(
        "TileLang H3 RainFusion: invalid sparse-video geometry");
  }
  return derived;
}

namespace {

// Worker-major rf_v2 [src, dst, rows, flags] span table, in semantic order.
// Ported verbatim from build_reorder_records() in the frozen selector DSL
// module: the first frame stays untouched when the latent grid has an H or W
// remainder, every later frame is emitted as its tiled main region followed by
// that frame's H remainder and W remainder, and the dense context rows are
// appended after the video span.
std::vector<std::array<int32_t, 4>> build_semantic_records(
    const H3RainfusionGeometry& geometry,
    const H3RainfusionDerived& derived) {
  std::vector<std::array<int32_t, 4>> records;
  const auto add_span =
      [&](int64_t src, int64_t dst, int64_t rows, int32_t flags) {
        if (rows <= 0) {
          return;
        }
        int64_t offset = 0;
        while (offset < rows) {
          const int64_t count = std::min(kH3RainfusionChunkRows, rows - offset);
          int32_t record_flags = flags;
          if (offset + count == rows && rows % kH3RainfusionChunkRows != 0) {
            record_flags |= kH3RainfusionFlagTail;
          }
          records.push_back({static_cast<int32_t>(src + offset),
                             static_cast<int32_t>(dst + offset),
                             static_cast<int32_t>(count),
                             record_flags});
          offset += count;
        }
      };

  const int64_t remainder_h = geometry.grid_h % 8;
  const int64_t remainder_w = geometry.grid_w % 8;
  const int64_t h_main = geometry.grid_h - remainder_h;
  const int64_t w_main = geometry.grid_w - remainder_w;
  int64_t dst = 0;
  int64_t first_rearranged_frame = 0;
  if (remainder_h != 0 || remainder_w != 0) {
    add_span(geometry.video_start,
             dst,
             derived.ff_rows,
             kH3RainfusionFlagFirstFrame);
    dst += derived.ff_rows;
    first_rearranged_frame = 1;
  }

  for (int64_t frame = first_rearranged_frame; frame < geometry.grid_t;
       ++frame) {
    const int64_t base = geometry.video_start + frame * derived.ff_rows;
    for (int64_t hn = 0; hn < geometry.grid_h / 8; ++hn) {
      for (int64_t wn = 0; wn < w_main / 8; ++wn) {
        for (int64_t hb = 0; hb < 8; ++hb) {
          const int64_t src = base + (hn * 8 + hb) * geometry.grid_w + wn * 8;
          add_span(src, dst, 8, 0);
          dst += 8;
        }
      }
    }
    if (remainder_h != 0) {
      add_span(base + h_main * geometry.grid_w,
               dst,
               remainder_h * geometry.grid_w,
               0);
      dst += remainder_h * geometry.grid_w;
    }
    if (remainder_w != 0) {
      for (int64_t h = 0; h < h_main; ++h) {
        add_span(base + h * geometry.grid_w + w_main, dst, remainder_w, 0);
        dst += remainder_w;
      }
    }
  }

  add_span(0, dst, geometry.video_start, kH3RainfusionFlagDense);
  dst += geometry.video_start;
  if (geometry.valid_rows > derived.video_end) {
    add_span(derived.video_end,
             dst,
             geometry.valid_rows - derived.video_end,
             kH3RainfusionFlagDense);
    dst += geometry.valid_rows - derived.video_end;
  }

  if (dst != geometry.valid_rows) {
    throw std::invalid_argument("TileLang H3 RainFusion: record table covers " +
                                std::to_string(dst) + " rows, expected " +
                                std::to_string(geometry.valid_rows));
  }
  return records;
}

}  // namespace

int64_t h3_rainfusion_record_count(const H3RainfusionGeometry& geometry) {
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);
  return static_cast<int64_t>(build_semantic_records(geometry, derived).size());
}

bool h3_rainfusion_logical_fits_capacity(const H3RainfusionGeometry& geometry,
                                         const H3RainfusionDerived& derived,
                                         int64_t record_count) {
  const H3RainfusionCapacity& cap = kH3RainfusionQualifiedCapacity;
  if (geometry.total_rows > cap.total_rows ||
      geometry.valid_rows > geometry.total_rows) {
    return false;
  }
  if (derived.total_blocks > cap.num_blocks ||
      record_count > cap.record_count) {
    return false;
  }
  const int64_t records_padded =
      kH3RainfusionWorkers * h3_ceildiv(record_count, kH3RainfusionWorkers);
  if (records_padded > cap.records_padded) {
    return false;
  }
  if (geometry.topk_r > cap.topk_r || geometry.topk_r > derived.total_blocks) {
    return false;
  }
  if (derived.num_sq > cap.q_pool_pad ||
      derived.total_blocks > cap.k_pool_pad) {
    return false;
  }
  if (h3_align_up(derived.total_blocks, kH3RainfusionPlanAlign) >
      cap.plan_width) {
    return false;
  }
  // The selector Phase C loop is unrolled at the CAPACITY tile count but now
  // guards every tile at or beyond ceil(rt_total_blocks/N_TILE): an inactive
  // tile performs no GM copy, no event set/wait, no MMA and no FIX store, and
  // the l0a release stays on the last ACTIVE tile so no MMA can follow it. The
  // logical tile count therefore only has to FIT the unrolled capacity rather
  // than equal it, which is what lets one capacity object serve geometries with
  // fewer tiles.
  if (h3_ceildiv(derived.total_blocks, kH3RainfusionNTile) >
      kH3RainfusionCapNTiles) {
    return false;
  }
  // Storage must be explicitly 256-aligned and must cover the full tail block
  // read.
  if (geometry.total_rows % kH3RainfusionKvPackRows != 0) {
    return false;
  }
  if (h3_ceildiv(geometry.valid_rows, kH3RainfusionBlock) * kH3RainfusionBlock >
      geometry.total_rows) {
    return false;
  }
  return true;
}

int64_t h3_rainfusion_records_padded(const H3RainfusionGeometry& geometry) {
  const int64_t count = h3_rainfusion_record_count(geometry);
  return kH3RainfusionWorkers * h3_ceildiv(count, kH3RainfusionWorkers);
}

torch::Tensor build_h3_rainfusion_reorder_records(
    const H3RainfusionGeometry& geometry,
    const torch::Device& device) {
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);
  const std::vector<std::array<int32_t, 4>> semantic =
      build_semantic_records(geometry, derived);
  const int64_t count = static_cast<int64_t>(semantic.size());
  const int64_t records_per_worker = h3_ceildiv(count, kH3RainfusionWorkers);
  const int64_t padded = kH3RainfusionWorkers * records_per_worker;

  const auto cpu_options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  torch::Tensor table = torch::zeros({padded, 4}, cpu_options);
  int32_t* table_ptr = table.data_ptr<int32_t>();
  // Worker-major scatter: worker w owns the contiguous slot range
  // [w*records_per_worker, (w+1)*records_per_worker), so each AIV worker
  // preloads its own records with a single DMA. Unused slots stay zero, which
  // is why the active extent is `count` and never `padded`.
  for (int64_t index = 0; index < count; ++index) {
    const int64_t slot = (index % kH3RainfusionWorkers) * records_per_worker +
                         index / kH3RainfusionWorkers;
    for (int64_t field = 0; field < 4; ++field) {
      table_ptr[slot * 4 + field] =
          semantic[static_cast<size_t>(index)][static_cast<size_t>(field)];
    }
  }
  return table.to(torch::TensorOptions().device(device).dtype(torch::kInt32));
}

bool has_h3_rainfusion_specialization(const H3RainfusionGeometry& geometry) {
#ifndef XLLM_ENABLE_H3_RAINFUSION
  return false;
#else
  // Only this production geometry was qualified. Reject other dispatch axes
  // before building records or narrowing integers for a generated registry.
  if (!is_qualified_h3_rainfusion_profile(geometry)) {
    return false;
  }
  try {
    // derive() validates the geometry and throws when it is not qualified.
    (void)h3_rainfusion_derive(geometry);
    const int64_t record_count = h3_rainfusion_record_count(geometry);
    const int64_t records_padded =
        kH3RainfusionWorkers * h3_ceildiv(record_count, kH3RainfusionWorkers);
    // Admit the CAPACITY PAIR, then require the logical geometry to fit it. The
    // registry key is the capacity tuple, so both objects must exist for
    // exactly that tuple; a logical geometry that does not fit falls back to
    // dense and is reported as dense, never as sparse.
    if (!detail::has_selector_capacity_specialization() ||
        !detail::has_attention_capacity_specialization()) {
      return false;
    }
    const H3RainfusionDerived cap_derived = h3_rainfusion_derive(geometry);
    return h3_rainfusion_logical_fits_capacity(
        geometry, cap_derived, record_count);
  } catch (const std::invalid_argument&) {
    return false;
  }
#endif
}

const char* h3_rainfusion_unavailability_reason(
    const H3RainfusionGeometry& geometry) {
#ifndef XLLM_ENABLE_H3_RAINFUSION
  (void)geometry;
  return "rainfusion_disabled_at_build_time";
#else
  if (!is_qualified_h3_rainfusion_profile(geometry)) {
    return "unqualified_geometry";
  }
  // Computed directly rather than via derive(), because derive() THROWS on
  // num_sq < 1 and the whole point of this classifier is to name that case
  // instead of swallowing it. A geometry with no sparse video query block is
  // legal: the first-frame context already covers every video block, so there
  // is nothing for RainFusion to select and the request must run dense.
  const int64_t video_rows =
      geometry.grid_t * geometry.grid_h * geometry.grid_w;
  const int64_t ff_blocks =
      h3_ceildiv(geometry.grid_h * geometry.grid_w, kH3RainfusionBlock);
  if (video_rows / kH3RainfusionBlock - ff_blocks < 1) {
    return "no_sparse_video_query_blocks";
  }
  if (!detail::has_selector_capacity_specialization() ||
      !detail::has_attention_capacity_specialization()) {
    return "capacity_object_absent";
  }
  try {
    const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);
    const int64_t record_count = h3_rainfusion_record_count(geometry);
    if (!h3_rainfusion_logical_fits_capacity(geometry, derived, record_count)) {
      return "logical_geometry_exceeds_capacity";
    }
    return "";
  } catch (const std::invalid_argument&) {
    return "invalid_sparse_video_geometry";
  }
#endif
}

H3RainfusionWorkspace allocate_h3_rainfusion_workspace(
    const H3RainfusionGeometry& geometry,
    const torch::Device& device) {
  if (!has_h3_rainfusion_specialization(geometry)) {
    throw std::invalid_argument("H3 RainFusion specialization is unavailable");
  }
  CHECK(device.type() == c10::DeviceType::PrivateUse1)
      << "TileLang H3 RainFusion: workspace device must be NPU";
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);

  const auto bf16 =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto fp32 =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const auto int32 = torch::TensorOptions().device(device).dtype(torch::kInt32);

  const int64_t rows = geometry.total_rows;
  const int64_t row_width = kH3RainfusionHeads * kH3RainfusionHeadDim;
  const int64_t stack_width = kH3RainfusionStackBlocks * kH3RainfusionBlock;

  H3RainfusionWorkspace workspace;
  workspace.reorder_records =
      build_h3_rainfusion_reorder_records(geometry, device);
  workspace.records_padded = workspace.reorder_records.size(0);
  workspace.record_count = h3_rainfusion_record_count(geometry);
  CHECK_EQ(workspace.records_padded % kH3RainfusionWorkers, 0)
      << "TileLang H3 RainFusion: record table must be divisible by the "
      << kH3RainfusionWorkers << " AIV workers";
  CHECK_LE(workspace.record_count, workspace.records_padded)
      << "TileLang H3 RainFusion: active records exceed the table extent";
  // PHYSICAL extents come from the qualified capacity, never from the logical
  // geometry: the compiled objects index GM at capacity strides, so an
  // undersized tensor is an out-of-bounds access. The logical extents stay in
  // `derived` and are passed to the kernels as runtime scalars.
  const H3RainfusionCapacity& cap = kH3RainfusionQualifiedCapacity;
  CHECK_GE(cap.total_rows, rows)
      << "TileLang H3 RainFusion: storage exceeds the qualified capacity";
  workspace.plan_ids =
      torch::empty({cap.num_blocks, kH3RainfusionHeads, cap.plan_width}, int32);
  workspace.plan_count =
      torch::empty({cap.num_blocks, kH3RainfusionPlanAlign}, int32);
  // INPUTS AND OUTPUTS KEEP THE LOGICAL STORAGE EXTENT. These are row-major
  // [rows, heads*dim], so the element stride is heads*dim regardless of the row
  // count and the kernels address only rows below the logical storage extent.
  // Only internal scratch, whose per-head / per-block strides are baked from
  // the capacity at compile time, is allocated at capacity.
  workspace.reordered_q = torch::empty({rows, row_width}, bf16);
  workspace.reordered_k = torch::empty({rows, row_width}, bf16);
  workspace.reordered_v = torch::empty({rows, row_width}, bf16);
  workspace.q_pool = torch::empty(
      {kH3RainfusionHeads, cap.q_pool_pad, kH3RainfusionHeadDim}, bf16);
  workspace.k_pool = torch::empty(
      {kH3RainfusionHeads, cap.k_pool_pad, kH3RainfusionHeadDim}, bf16);
  workspace.score_ring = torch::empty({kH3RainfusionNumCores,
                                       kH3RainfusionRingSlots,
                                       kH3RainfusionMTile,
                                       cap.k_pool_pad},
                                      fp32);
  workspace.workspace_s = torch::empty({kH3RainfusionTaskSlots,
                                        kH3RainfusionNumCores,
                                        kH3RainfusionBlock,
                                        stack_width},
                                       fp32);
  workspace.workspace_p = torch::empty({kH3RainfusionTaskSlots,
                                        kH3RainfusionNumCores,
                                        kH3RainfusionBlock,
                                        stack_width},
                                       bf16);
  workspace.workspace_o = torch::empty({kH3RainfusionTaskSlots,
                                        kH3RainfusionNumCores,
                                        kH3RainfusionBlock,
                                        kH3RainfusionHeadDim},
                                       fp32);
  // packed_k/packed_v are INTERNAL SCRATCH whose per-head stride is baked from
  // the capacity total_rows at compile time, so they must be allocated at
  // capacity even though the caller-visible inputs and final outputs keep the
  // logical storage extent.
  workspace.packed_k = torch::empty({kH3RainfusionHeads,
                                     kH3RainfusionQualifiedCapacity.total_rows,
                                     kH3RainfusionHeadDim},
                                    bf16);
  workspace.packed_v = torch::empty({kH3RainfusionHeads,
                                     kH3RainfusionQualifiedCapacity.total_rows,
                                     kH3RainfusionHeadDim},
                                    bf16);
  workspace.reordered_output = torch::empty({rows, row_width}, bf16);
  return workspace;
}

void h3_rainfusion_attention_out(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
                                 const H3RainfusionGeometry& geometry,
                                 H3RainfusionWorkspace& workspace,
                                 torch::Tensor& output,
                                 H3RainfusionLaunchCounters* counters) {
#ifndef XLLM_ENABLE_H3_RAINFUSION
  throw std::runtime_error("H3 RainFusion was disabled at build time");
#else
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);

  // Both capacity kernels must be present before launching either one.
  CHECK(detail::has_selector_capacity_specialization())
      << "TileLang H3 RainFusion: no compiled selector for the qualified "
         "capacity";
  CHECK(detail::has_attention_capacity_specialization())
      << "TileLang H3 RainFusion: no compiled attention for the qualified "
         "capacity";
  CHECK(h3_rainfusion_logical_fits_capacity(
      geometry, derived, workspace.record_count))
      << "TileLang H3 RainFusion: logical geometry does not fit the qualified "
         "capacity, or is outside the explicitly supported scope";

  detail::validate_selector(query, key, value, geometry, derived, workspace);
  detail::validate_attention(geometry, derived, workspace, output);
  CHECK_EQ(query.device(), output.device());
  // Check byte intervals, including offset views. Q/K/V may share read-only
  // storage; every workspace/output allocation must be disjoint from all
  // other buffers. This is host metadata only and precedes both launches.
  const std::array<const torch::Tensor*, 4> inputs = {
      &query, &key, &value, &workspace.reorder_records};
  const std::array<const torch::Tensor*, 15> writes = {
      &workspace.plan_ids,
      &workspace.plan_count,
      &workspace.reordered_q,
      &workspace.reordered_k,
      &workspace.reordered_v,
      &workspace.q_pool,
      &workspace.k_pool,
      &workspace.score_ring,
      &workspace.workspace_s,
      &workspace.workspace_p,
      &workspace.workspace_o,
      &workspace.packed_k,
      &workspace.packed_v,
      &workspace.reordered_output,
      &output};
  const auto disjoint = [](const torch::Tensor& lhs, const torch::Tensor& rhs) {
    const uintptr_t left = reinterpret_cast<uintptr_t>(lhs.data_ptr());
    const uintptr_t right = reinterpret_cast<uintptr_t>(rhs.data_ptr());
    return left + lhs.nbytes() <= right || right + rhs.nbytes() <= left;
  };
  for (size_t index = 0; index < writes.size(); ++index) {
    for (const torch::Tensor* input : inputs) {
      CHECK(disjoint(*writes[index], *input))
          << "H3 RainFusion writable buffer aliases input, index=" << index;
    }
    for (size_t previous = 0; previous < index; ++previous) {
      CHECK(disjoint(*writes[index], *writes[previous]))
          << "H3 RainFusion workspace buffers alias, indices=" << index << ","
          << previous;
    }
  }

  const int32_t device_id = query.device().index();
  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  // Two launches on one stream. Nothing between them: no host synchronization,
  // no ATen operation, no plan readback. Fallback is only legal before the
  // first launch, because an asynchronous failure after launch leaves the
  // stream unusable for a different backend.
  detail::launch_selector(
      query, key, value, geometry, derived, workspace, stream);
  if (counters != nullptr) {
    ++counters->selector_launches;
  }
  detail::launch_attention(geometry, derived, workspace, output, stream);
  if (counters != nullptr) {
    ++counters->attention_launches;
  }
#endif
}

}  // namespace xllm::kernel::npu::tilelang
