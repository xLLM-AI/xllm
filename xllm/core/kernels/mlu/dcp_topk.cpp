/* Copyright 2026 The xLLM Authors.

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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <utility>

#include "compiler/triton/include/jit_kernel.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

using triton_jit::JITKernel;
using triton_jit::LaunchCfg;

constexpr int64_t kDcpIndexerCandidateBlockWidth = 2048;
constexpr int64_t kDcpIndexerCandidateMaxSingleRow = 160;
constexpr int64_t kDcpIndexerCandidatePrefillRowsPerProgram = 2;

constexpr int64_t kScoreSmallRowLimit = 128;
constexpr int64_t kScoreDefaultBlockWidth = 512;
constexpr int64_t kScoreStridedBlockWidth = 32;
constexpr int64_t kScoreNarrowBlockWidth = 128;
constexpr int64_t kScoreRowShardBlockWidth = 256;
constexpr int64_t kScorePreloadRowLimit = 16384;
constexpr int64_t kScoreProgramProofWidthLimit = 4096;
constexpr int32_t kScoreReducedStages = 3;
constexpr int32_t kScoreShardStages = 5;
constexpr int32_t kScoreDefaultStages = 99;
constexpr int64_t kLocalizeSmallRowLimit = 128;
constexpr int64_t kLocalizeSmallRowBlock = 1024;
constexpr int64_t kLocalizeLargeRowBlock = 512;
constexpr int64_t kLocalizeStridedBlock = 256;
constexpr int64_t kLocalizeLargeRowLimit = 1024;
constexpr int32_t kLocalizeDefaultStages = 3;
constexpr int32_t kLocalizeLargeRowStages = 4;

void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), dims);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.device() == device);
  for (int64_t axis = 0; axis < dims; ++axis) {
    CHECK(tensor.size(axis) <= 1 || tensor.stride(axis) > 0)
        << "Expanded and negative strides are unsupported";
  }
}

}  // namespace

torch::Tensor dcp_globalize_candidates(const torch::Tensor& local_slots,
                                       const torch::Tensor& context_lens,
                                       int64_t dcp_size,
                                       int64_t dcp_rank,
                                       int64_t physical_block_size) {
  const int64_t rows = local_slots.size(0);
  const int64_t width = local_slots.size(1);
  torch::Tensor global_slots = torch::empty_like(local_slots);
  if (rows == 0 || width == 0) {
    return global_slots;
  }

  const torch::DeviceGuard guard(local_slots.device());
  const int64_t rows_per_program =
      rows > kDcpIndexerCandidateMaxSingleRow
          ? kDcpIndexerCandidatePrefillRowsPerProgram
          : 1;
  const uint32_t grid_rows =
      static_cast<uint32_t>((rows + rows_per_program - 1) / rows_per_program);
  const uint32_t grid_columns =
      static_cast<uint32_t>((width + kDcpIndexerCandidateBlockWidth - 1) /
                            kDcpIndexerCandidateBlockWidth);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel& kernel = JITKernel::get(
      /*py_path=*/"xllm.core.kernels.mlu.triton_kernel.dcp_indexer_candidates",
      /*fn_name=*/rows_per_program == 1
          ? "tmo_dcp_globalize_indexer_candidates_single_row_kernel"
          : "tmo_dcp_globalize_indexer_candidates_multi_row_kernel");
  kernel.launch(static_cast<void*>(queue),
                /*grid=*/{grid_rows, grid_columns, 1},
                /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
                local_slots,
                context_lens,
                global_slots,
                rows,
                width,
                static_cast<int32_t>(dcp_size),
                static_cast<int32_t>(dcp_rank),
                static_cast<int32_t>(physical_block_size),
                local_slots.stride(0),
                local_slots.stride(1),
                global_slots.stride(0),
                global_slots.stride(1),
                kDcpIndexerCandidateBlockWidth,
                rows_per_program);
  return global_slots;
}

std::tuple<torch::Tensor, torch::Tensor> dcp_score_candidates(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& slots,
    const torch::Tensor& counts,
    int64_t dcp_size,
    int64_t dcp_rank,
    int64_t physical_block_size) {
  const int64_t rows = slots.size(0);
  const int64_t width = slots.size(1);
  const int64_t cache_slots = cache.size(0);
  torch::Tensor scores =
      torch::empty({rows, width}, query.options().dtype(torch::kFloat32));
  torch::Tensor global_slots =
      torch::empty({rows, width}, query.options().dtype(torch::kInt32));
  if (rows == 0 || width == 0) {
    return {std::move(scores), std::move(global_slots)};
  }
  const torch::DeviceGuard guard(query.device());
  const auto* properties =
      torch_mlu::getDeviceProperties(query.device().index());
  const int64_t cores =
      properties->cluster_count * properties->core_num_per_cluster;
  const bool row_owned = rows >= cores;
  const bool small_rows = rows <= kScoreSmallRowLimit;
  int64_t block_n = kScoreDefaultBlockWidth;
  int32_t num_stages = kScoreDefaultStages;
  if (query.stride(-1) != 1 || weights.stride(-1) != 1 ||
      cache.stride(-1) != 1 || slots.stride(-1) != 1 ||
      counts.stride(-1) != 1) {
    block_n = kScoreStridedBlockWidth;
    num_stages = kScoreReducedStages;
  } else if (width < kScoreRowShardBlockWidth) {
    block_n = kScoreNarrowBlockWidth;
    num_stages = kScoreReducedStages;
  } else if (!row_owned || small_rows) {
    block_n = kScoreRowShardBlockWidth;
    num_stages = !row_owned || width <= kScoreProgramProofWidthLimit
                     ? kScoreShardStages
                     : kScoreDefaultStages;
  }
  const int64_t slot_cap =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(width)));
  const int64_t page_shift =
      std::bit_width(static_cast<uint64_t>(physical_block_size)) - 1;
  const uint32_t programs = static_cast<uint32_t>(
      std::min(cores, rows * ((width + block_n - 1) / block_n)));
  const int64_t row_cap =
      rows <= kScorePreloadRowLimit
          ? static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(rows)))
          : 0;
  const int64_t bound = row_cap == 0 ? rows : row_cap;
  // Widen before multiplication: valid int64 strides can overflow int64 byte
  // bounds even when no corresponding large allocation is made (zero strides).
  constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
  const bool narrow =
      cache_slots <= kLimit &&
      (static_cast<__int128>(bound - 1) * query.stride(0) +
       static_cast<__int128>(kDcpScoreHeads - 1) * query.stride(1) +
       static_cast<__int128>(kDcpScoreHeadDim - 1) * query.stride(2)) *
              4 <
          kLimit &&
      (static_cast<__int128>(bound - 1) * weights.stride(0) +
       static_cast<__int128>(kDcpScoreHeads - 1) * weights.stride(1)) *
              4 <
          kLimit &&
      (static_cast<__int128>(cache_slots - 1) * cache.stride(0) +
       static_cast<__int128>(kDcpScoreHeadDim - 1) * cache.stride(1)) *
              4 <
          kLimit &&
      (static_cast<__int128>(bound - 1) * slots.stride(0) +
       (static_cast<__int128>(slot_cap) + block_n - 1) * slots.stride(1)) *
              4 <
          kLimit &&
      static_cast<__int128>(bound - 1) * counts.stride(0) * 4 < kLimit &&
      (static_cast<__int128>(bound) * width + block_n) * 4 < kLimit;
  const bool preload_counts =
      row_owned && rows > kScoreSmallRowLimit && rows <= kScorePreloadRowLimit;
  const bool prefetch_rows = row_owned && rows > cores && small_rows;
  const bool program_proof =
      rows >= 2 * cores && small_rows && width <= kScoreProgramProofWidthLimit;
  LaunchCfg cfg;
  cfg.num_stages = num_stages;
  cfg.enable_fp_fusion = false;
  cfg.enable_soft_i64 = true;
  cfg.force_use_shared_memory = true;
  JITKernel& kernel = JITKernel::get(
      "xllm.core.kernels.mlu.triton_kernel.dcp.dcp_candidate_score",
      "tmo_dcp_score_candidates_kernel");

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  kernel.launch(static_cast<void*>(queue),
                {programs, 1, 1},
                cfg,
                query,
                weights,
                cache,
                slots,
                counts,
                scores,
                global_slots,
                rows,
                row_cap,
                row_owned,
                small_rows,
                preload_counts,
                prefetch_rows,
                program_proof,
                narrow,
                width,
                cache_slots,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                weights.stride(0),
                weights.stride(1),
                cache.stride(0),
                cache.stride(1),
                slots.stride(0),
                slots.stride(1),
                counts.stride(0),
                dcp_size,
                dcp_rank,
                physical_block_size,
                page_shift,
                block_n,
                cores,
                slot_cap);
  return {std::move(scores), std::move(global_slots)};
}

void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  const torch::Device device = scores.device();
  check_tensor(scores, torch::kFloat32, 3, device);
  check_tensor(slots, torch::kInt32, 3, device);
  check_tensor(mapping, torch::kInt32, 1, device);
  CHECK(scores.sizes() == slots.sizes());
  const int64_t d = scores.size(0);
  const int64_t q = scores.size(1);
  const int64_t c = scores.size(2);
  CHECK_GT(d, 0);
  CHECK_GT(c, 0);
  CHECK_LE(d, std::numeric_limits<int32_t>::max() / c);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, d * c);
  CHECK_EQ(mapping.size(0), q);
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({q, topk}));
  CHECK_EQ(lengths.size(0), q);
  CHECK_LE(q, std::numeric_limits<uint32_t>::max());
  if (q != 0) {
    const int64_t small = output.stride(0) <= output.stride(1) ? 0 : 1;
    const int64_t large = 1 - small;
    CHECK(output.size(large) <= 1 ||
          output.stride(large) >
              (output.size(small) - 1) * output.stride(small));
    CHECK(!output.is_alias_of(lengths));
    for (const torch::Tensor& input : {scores, slots, mapping}) {
      CHECK(!output.is_alias_of(input));
      CHECK(!lengths.is_alias_of(input));
    }
  }
  if (q == 0) {
    return;
  }

  const torch::DeviceGuard guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const int64_t width = d * c;
  // Ascending negated scores put canonical NaN padding after valid -inf.
  torch::Tensor sortable_scores = torch::empty({q, width}, scores.options());
  torch::Tensor prepared_slots = torch::empty({q, width}, slots.options());
  const int64_t candidate_block =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(c)));
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_indexer_merge",
                 "tmo_dcp_prepare_indexer_merge_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(d), static_cast<uint32_t>(q), 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              scores,
              slots,
              sortable_scores,
              prepared_slots,
              scores.stride(0),
              scores.stride(1),
              scores.stride(2),
              slots.stride(0),
              slots.stride(1),
              slots.stride(2),
              width,
              c,
              candidate_block);

  torch::Tensor score_order = std::get<1>(torch::topk(
      sortable_scores, topk, /*dim=*/1, /*largest=*/false, /*sorted=*/true));
  const int64_t topk_block =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(topk)));
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_indexer_merge",
                 "tmo_dcp_finalize_indexer_merge_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(q), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              prepared_slots,
              score_order,
              mapping,
              output,
              lengths,
              width,
              score_order.stride(0),
              mapping.stride(0),
              output.stride(0),
              output.stride(1),
              lengths.stride(0),
              topk,
              topk_block);
}

void dcp_localize_topk(const torch::Tensor& slots,
                       const torch::Tensor& lengths,
                       int32_t rank,
                       int32_t size,
                       int32_t physical_block_size,
                       torch::Tensor& output,
                       torch::Tensor& output_lengths) {
  CHECK(slots.defined());
  CHECK(slots.device().is_privateuseone());
  const torch::Device device = slots.device();
  check_tensor(slots, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  const int64_t q = slots.size(0);
  const int64_t width = slots.size(1);
  CHECK_GT(width, 0);
  CHECK_LE(width, std::numeric_limits<int32_t>::max());
  CHECK_EQ(lengths.size(0), q);
  CHECK(lengths.is_contiguous());
  CHECK_GT(size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, size);
  CHECK_GT(physical_block_size, 0);
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(output_lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({q, width}));
  CHECK_EQ(output_lengths.size(0), q);
  CHECK_LE(q, std::numeric_limits<uint32_t>::max());
  if (q != 0) {
    const int64_t small = output.stride(0) <= output.stride(1) ? 0 : 1;
    const int64_t large = 1 - small;
    CHECK(output.size(large) <= 1 ||
          output.stride(large) >
              (output.size(small) - 1) * output.stride(small));
    CHECK(!output.is_alias_of(output_lengths));
    for (const torch::Tensor& input : {slots, lengths}) {
      CHECK(!output.is_alias_of(input));
      CHECK(!output_lengths.is_alias_of(input));
    }
  }
  CHECK(output_lengths.is_contiguous());
  if (q == 0) {
    return;
  }

  int64_t block = q <= kLocalizeSmallRowLimit ? kLocalizeSmallRowBlock
                                              : kLocalizeLargeRowBlock;
  int32_t stages = q < kLocalizeLargeRowLimit ? kLocalizeDefaultStages
                                              : kLocalizeLargeRowStages;
  if (width != 2048 || physical_block_size != 16 || (size != 4 && size != 8)) {
    stages = kLocalizeDefaultStages;
  }
  if (slots.stride(1) != 1 || output.stride(1) != 1) {
    block = kLocalizeStridedBlock;
    stages = kLocalizeDefaultStages;
  }
  LaunchCfg cfg;
  cfg.num_stages = stages;
  cfg.force_use_shared_memory = true;
  const torch::DeviceGuard guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_localize_topk",
                 "tmo_dcp_localize_topk_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(q), 1, 1},
              cfg,
              slots,
              lengths,
              output,
              output_lengths,
              rank,
              size,
              physical_block_size,
              static_cast<int32_t>(width),
              slots.stride(0),
              slots.stride(1),
              output.stride(0),
              output.stride(1),
              block);
}

}  // namespace xllm::kernel::mlu
