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

#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

#include <glog/logging.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/constants.h"
#include "framework/config/kv_cache_config.h"
#include "framework/core/MLUStream.h"
#include "triton_jit/include/jit_kernel.h"
#include "util/linalg.h"
#include "util/tensor_helper.h"

namespace xllm::layer {
namespace {

constexpr char kKPoolExpandKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool_expand";
constexpr char kKPoolKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool";
constexpr char kKPoolSelectKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool_select";
constexpr int64_t kWorkspaceBytes = 64 * 1024 * 1024;

int32_t max_sequence_length(const std::vector<int32_t>& lengths) {
  return lengths.empty() ? 0
                         : *std::max_element(lengths.begin(), lengths.end());
}

std::vector<int64_t> make_query_offsets(const std::vector<int32_t>& lengths) {
  std::vector<int64_t> offsets;
  offsets.reserve(lengths.size() + 1);
  offsets.emplace_back(0);
  for (const int32_t length : lengths) {
    offsets.emplace_back(offsets.back() + length);
  }
  return offsets;
}

std::vector<int32_t> resolve_seq_lens(const std::vector<int32_t>& host_lens,
                                      const torch::Tensor& device_lens,
                                      const char* name) {
  if (!host_lens.empty()) {
    if (device_lens.defined() && host_lens.front() == 0 &&
        host_lens.size() == static_cast<size_t>(device_lens.numel() + 1)) {
      std::vector<int32_t> per_sequence_lens;
      per_sequence_lens.reserve(host_lens.size() - 1);
      for (size_t i = 1; i < host_lens.size(); ++i) {
        per_sequence_lens.emplace_back(host_lens[i] - host_lens[i - 1]);
      }
      return per_sequence_lens;
    }
    return host_lens;
  }
  CHECK(device_lens.defined()) << name << " must be defined.";
  return tensor_to_vector<int32_t>(device_lens);
}

torch::Tensor make_row_batch(const std::vector<int32_t>& q_seq_lens,
                             const torch::Device& device) {
  std::vector<int64_t> rows;
  const int64_t row_count = std::accumulate(
      q_seq_lens.begin(), q_seq_lens.end(), static_cast<int64_t>(0));
  rows.reserve(static_cast<size_t>(row_count));
  for (size_t batch_id = 0; batch_id < q_seq_lens.size(); ++batch_id) {
    rows.insert(rows.end(),
                static_cast<size_t>(q_seq_lens[batch_id]),
                static_cast<int64_t>(batch_id));
  }
  return make_cpu_tensor(rows).to(device);
}

int64_t next_power_of_two(int64_t value) {
  return static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(value)));
}

void launch_kpool_logits(const torch::Tensor& query,
                         const torch::Tensor& weights,
                         const torch::Tensor& keys,
                         const torch::Tensor& block_table,
                         const torch::Tensor& positions,
                         const torch::Tensor& row_batch,
                         torch::Tensor& scores,
                         int64_t index_kpool,
                         int64_t pool_block_size,
                         double softmax_scale,
                         bool paged) {
  const int64_t num_rows = query.size(0);
  const int64_t num_pools = scores.size(1);
  if (num_rows == 0 || num_pools == 0) {
    return;
  }
  // Score-kernel launch tiling: BLOCK_N (pool tile) and BLOCK_H (head tile)
  // are tuned for a head count of 32. Paged scoring widens BLOCK_N once enough
  // rows amortize the per-page gather, and splits BLOCK_H (two passes over the
  // 32 heads) for large batches to halve the NRAM footprint of each pass.
  constexpr int64_t kScoreHeads = 32;
  constexpr int64_t kWideScoreMinRows = 32;
  constexpr int64_t kSplitHeadMinRows = 64;
  constexpr int64_t kUnpagedScoreTile = 1024;
  constexpr int64_t kPagedScoreTile = 256;
  constexpr int64_t kWidePagedScoreTile = 512;
  constexpr int64_t kSplitHeadTile = 16;
  constexpr int64_t kMinHeadTile = 16;
  constexpr int64_t kMaxHeadTile = 64;
  const int64_t score_heads = query.size(1);
  const bool wide_paged_score =
      paged && num_rows >= kWideScoreMinRows && score_heads == kScoreHeads;
  const bool split_head_score =
      paged && num_rows >= kSplitHeadMinRows && score_heads == kScoreHeads;
  const int64_t tile = wide_paged_score ? kWidePagedScoreTile
                                        : (!paged && score_heads <= kScoreHeads
                                               ? kUnpagedScoreTile
                                               : kPagedScoreTile);
  const int64_t head_tile =
      split_head_score
          ? kSplitHeadTile
          : std::clamp(
                next_power_of_two(score_heads), kMinHeadTile, kMaxHeadTile);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  triton_jit::JITKernel::get(kKPoolSelectKernelPath, "score")
      .launch(static_cast<void*>(queue),
              /*grid=*/
              {static_cast<uint32_t>(num_rows),
               static_cast<uint32_t>((num_pools + tile - 1) / tile),
               1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
              query,
              weights,
              keys,
              block_table,
              positions,
              row_batch,
              scores,
              query.stride(0),
              query.stride(1),
              weights.stride(0),
              block_table.stride(0),
              scores.stride(0),
              num_pools,
              static_cast<float>(softmax_scale),
              /*H=*/static_cast<int32_t>(query.size(1)),
              /*D=*/static_cast<int32_t>(query.size(2)),
              /*P=*/static_cast<int32_t>(index_kpool),
              /*POOL_BLOCK=*/static_cast<int32_t>(pool_block_size),
              /*BLOCK_H=*/static_cast<int32_t>(head_tile),
              /*BLOCK_N=*/static_cast<int32_t>(tile),
              /*PAGED=*/paged ? 1 : 0);
}

torch::Tensor make_kpool_rows(const torch::Tensor& starts, int64_t tokens) {
  torch::Tensor rows = torch::empty({tokens}, starts.options());
  if (tokens == 0) {
    return rows;
  }
  const int64_t requests = starts.numel() - 1;
  CHECK_GT(requests, 0);
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_rows")
      .launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
              {static_cast<uint32_t>((tokens + 63) / 64), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              starts,
              rows,
              tokens,
              /*N=*/static_cast<int32_t>(requests),
              /*BN=*/static_cast<int32_t>(next_power_of_two(requests)),
              /*BT=*/64);
  return rows;
}

std::shared_ptr<KPoolBatchMetadata> make_kpool_batch_metadata(
    const AttentionMetadata& metadata,
    const torch::Device& device,
    const torch::Tensor& graph_rows) {
  auto batch = std::make_shared<KPoolBatchMetadata>();
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const bool graph = metadata.enable_cuda_graph;
  batch->q_seq_lens = metadata.kpool_query_lens;
  if (batch->q_seq_lens.empty()) {
    if (graph && metadata.q_seq_lens_vec.empty()) {
      batch->q_seq_lens.assign(graph_rows.numel(), 1);
    } else {
      batch->q_seq_lens = resolve_seq_lens(
          metadata.q_seq_lens_vec, metadata.q_seq_lens, "KPool q lengths");
    }
  }
  const int64_t requests = static_cast<int64_t>(batch->q_seq_lens.size());
  const std::vector<int64_t> offsets = make_query_offsets(batch->q_seq_lens);
  const int64_t tokens = offsets.back();
  if (graph) {
    // The ordinary graph path has live cumulative query offsets. Explicit
    // speculative grouping is prepared before capture, or has uniform spans.
    if (metadata.kpool_query_lens.empty() && metadata.q_cu_seq_lens.defined() &&
        metadata.q_cu_seq_lens.numel() == requests + 1) {
      batch->query_starts = metadata.q_cu_seq_lens.to(torch::kInt64);
    } else {
      const int32_t width = batch->q_seq_lens.front();
      CHECK(std::all_of(batch->q_seq_lens.begin(),
                        batch->q_seq_lens.end(),
                        [width](int32_t len) { return len == width; }))
          << "Prepare ragged KPool metadata before graph capture.";
      batch->query_starts = torch::arange(requests + 1, options) * width;
    }
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
  } else {
    batch->query_starts = make_cpu_tensor(offsets).to(device);
    batch->row_batch = make_row_batch(batch->q_seq_lens, device);
  }
  batch->block_table = metadata.block_table;
  batch->tail_indices = metadata.linear_state_indices;
  const bool expanded = metadata.block_table.defined() &&
                        metadata.block_table.size(0) != requests;
  if (expanded) {
    CHECK_GE(metadata.block_table.size(0), tokens);
    batch->block_table = metadata.block_table.index_select(
        0, batch->query_starts.narrow(0, 0, requests));
  }
  if (batch->tail_indices.defined()) {
    CHECK_GE(batch->tail_indices.numel(), requests);
    batch->tail_indices = batch->tail_indices.narrow(0, 0, requests);
  }
  if (!graph) {
    batch->kv_seq_lens = resolve_seq_lens(
        metadata.kv_seq_lens_vec, metadata.kv_seq_lens, "KPool KV lengths");
    if (expanded) {
      std::vector<int32_t> logical_lens;
      logical_lens.reserve(requests);
      for (int64_t row = 0; row < requests; ++row) {
        logical_lens.emplace_back(batch->kv_seq_lens.at(offsets[row + 1] - 1));
      }
      batch->kv_seq_lens = std::move(logical_lens);
    }
    batch->max_kv_len = max_sequence_length(batch->kv_seq_lens);
  }
  return batch;
}

}  // namespace

void prepare_glm5_next_kpool_metadata(AttentionMetadata& metadata,
                                      const torch::Device& device) {
  metadata.kpool_batch_metadata =
      metadata.is_dummy
          ? nullptr
          : make_kpool_batch_metadata(metadata, device, metadata.kv_seq_lens);
}

torch::Tensor glm5_next_kpool_select(const torch::Tensor& query,
                                     const torch::Tensor& head_weights,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& row_batch,
                                     const torch::Tensor& index_cache,
                                     const torch::Tensor& block_table,
                                     int64_t max_kv_seq_len,
                                     int64_t block_size,
                                     int64_t index_kpool,
                                     int64_t index_topk,
                                     double softmax_scale,
                                     int64_t workspace_bytes,
                                     const KPoolBatchMetadata* batch) {
  CHECK_LE((max_kv_seq_len + block_size - 1) / block_size, block_table.size(1));
  const int64_t select_k = index_topk / index_kpool;
  const int64_t max_pools = (max_kv_seq_len + index_kpool - 1) / index_kpool;
  torch::Tensor result = torch::full(
      {query.size(0), select_k}, -1, query.options().dtype(torch::kInt64));
  if (max_pools == 0 || query.size(0) == 0 || select_k == 0) {
    return result;
  }
  CHECK_GE(workspace_bytes, max_pools * static_cast<int64_t>(sizeof(float)));
  const int64_t max_query_rows =
      batch ? max_sequence_length(batch->q_seq_lens) : query.size(0);
  const bool dense = batch && max_query_rows > 32;
  const int64_t chunk_size =
      std::min<int64_t>(dense ? max_query_rows : query.size(0),
                        workspace_bytes / (max_pools * sizeof(float)));
  torch::Tensor workspace = torch::empty(
      {chunk_size, max_pools}, query.options().dtype(torch::kFloat32));
  const torch::Tensor q = query.contiguous();
  const torch::Tensor weights = head_weights.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor rows = row_batch.contiguous();
  const torch::Tensor table = block_table.contiguous();
  const int64_t pool_block_size = block_size / index_kpool;
  // Long prefill amortizes one dense gather over many queries of each request.
  const int64_t requests = dense ? batch->q_seq_lens.size() : 1;
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  int64_t query_offset = 0;
  for (int64_t request = 0; request < requests; ++request) {
    const int64_t num_rows = dense ? batch->q_seq_lens[request] : query.size(0);
    const int64_t num_pools =
        dense ? (batch->kv_seq_lens[request] + index_kpool - 1) / index_kpool
              : max_pools;
    const int64_t request_start = query_offset;
    query_offset += num_rows;
    if (num_rows == 0 || num_pools == 0) {
      continue;
    }
    torch::Tensor keys = index_cache;
    if (dense) {
      keys = torch::empty({num_pools, query.size(2)}, query.options());
      triton_jit::JITKernel::get(kKPoolSelectKernelPath, "gather_cache")
          .launch(static_cast<void*>(queue),
                  {static_cast<uint32_t>((num_pools + 63) / 64), 1, 1},
                  {/*num_warps=*/1, /*num_stages=*/1},
                  index_cache,
                  table,
                  keys,
                  request,
                  num_pools,
                  table.stride(0),
                  /*D=*/static_cast<int32_t>(query.size(2)),
                  /*POOL_BLOCK=*/static_cast<int32_t>(pool_block_size),
                  /*BN=*/64);
    }
    // Bound top-k scratch for large histories or budgets.
    const bool streaming = num_pools > 16384 || select_k > 2048;
    const int32_t topk_tile =
        static_cast<int32_t>(streaming ? 2048 : next_power_of_two(num_pools));
    triton_jit::JITKernel& select_kernel = triton_jit::JITKernel::get(
        kKPoolSelectKernelPath,
        streaming ? "select_topk_streaming" : "select_topk");
    for (int64_t begin = 0; begin < num_rows; begin += chunk_size) {
      const int64_t count = std::min(chunk_size, num_rows - begin);
      const int64_t offset = request_start + begin;
      torch::Tensor scores =
          workspace.narrow(0, 0, count).narrow(1, 0, num_pools);
      launch_kpool_logits(q.narrow(0, offset, count),
                          weights.narrow(0, offset, count),
                          keys,
                          table,
                          pos.narrow(0, offset, count),
                          rows.narrow(0, offset, count),
                          scores,
                          index_kpool,
                          pool_block_size,
                          softmax_scale,
                          !dense);
      select_kernel.launch(static_cast<void*>(queue),
                           {static_cast<uint32_t>(count), 1, 1},
                           {/*num_warps=*/1, /*num_stages=*/streaming ? 3 : 1},
                           scores,
                           result.narrow(0, offset, count),
                           num_pools,
                           scores.stride(0),
                           /*K=*/static_cast<int32_t>(select_k),
                           /*BN=*/topk_tile);
    }
  }
  return result;
}

torch::Tensor glm5_next_kpool_normalize_key(torch::Tensor key,
                                            RMSNorm& key_norm) {
  const torch::ScalarType projection_dtype = key.scalar_type();
  key = key.to(torch::kFloat32);
  return std::get<0>(key_norm->forward(key)).to(projection_dtype);
}

void launch_kpool_update(const torch::Tensor& k,
                         const torch::Tensor& gate,
                         const torch::Tensor& ape,
                         const torch::Tensor& hadamard,
                         torch::Tensor& index_cache,
                         torch::Tensor& tail_cache,
                         const torch::Tensor& tail_block_ids,
                         const torch::Tensor& block_table,
                         const torch::Tensor& positions,
                         const torch::Tensor& row_batch,
                         const torch::Tensor& starts,
                         int64_t block_size,
                         int64_t index_kpool,
                         bool single_token) {
  if (k.size(0) == 0) {
    return;
  }
  // The Triton kernels assume stride-1 innermost lanes (k/gate) and contiguous
  // tail/ape/hadamard rows, so drop any non-row-major strides before launch.
  const torch::Tensor k_contig = k.contiguous();
  const torch::Tensor gate_contig = gate.contiguous();
  const torch::Tensor ape_contig =
      ape.to(k_contig.device(), torch::kFloat32).contiguous();
  const torch::Tensor hadamard_contig = hadamard.contiguous();
  const torch::Tensor tail_block_ids_contig = tail_block_ids.contiguous();
  const torch::Tensor block_table_contig = block_table.contiguous();
  const torch::Tensor positions_contig = positions.contiguous();
  const torch::Tensor row_batch_contig = row_batch.contiguous();
  const torch::Tensor starts_contig = starts.contiguous();
  CHECK_EQ(tail_cache.dim(), 4);
  CHECK_EQ(tail_cache.size(1), 2);
  CHECK_GE(tail_cache.size(2), index_kpool);
  CHECK_EQ(tail_cache.size(3), k.size(1));
  CHECK_EQ(starts.numel(), tail_block_ids.numel() + 1);
  CHECK(tail_cache.is_contiguous());
  const int64_t tail_len = tail_cache.size(2);
  const int64_t dim = k_contig.size(1);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_complete")
      .launch(static_cast<void*>(queue),
              /*grid=*/{static_cast<uint32_t>(k_contig.size(0)), 1, 1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
              k_contig,
              gate_contig,
              ape_contig,
              hadamard_contig,
              index_cache,
              tail_cache,
              tail_block_ids_contig,
              block_table_contig,
              positions_contig,
              row_batch_contig,
              starts_contig,
              k_contig.stride(0),
              gate_contig.stride(0),
              block_table_contig.stride(0),
              /*P=*/static_cast<int32_t>(index_kpool),
              /*T=*/static_cast<int32_t>(tail_len),
              /*D=*/static_cast<int32_t>(dim),
              /*BLOCK_P=*/static_cast<int32_t>(next_power_of_two(index_kpool)),
              /*BLOCK_D=*/static_cast<int32_t>(next_power_of_two(dim)),
              /*POOL_BLOCK=*/static_cast<int32_t>(block_size / index_kpool));
  // Separate launches order all old-tail reads before any tail write.
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_stash")
      .launch(
          static_cast<void*>(queue),
          /*grid=*/{static_cast<uint32_t>(tail_block_ids_contig.numel()), 1, 1},
          /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
          k_contig,
          gate_contig,
          tail_cache,
          tail_block_ids_contig,
          positions_contig,
          starts_contig,
          k_contig.stride(0),
          gate_contig.stride(0),
          /*T=*/static_cast<int32_t>(tail_len),
          /*D=*/static_cast<int32_t>(dim),
          /*BLOCK_P=*/static_cast<int32_t>(next_power_of_two(tail_len)),
          /*BLOCK_D=*/static_cast<int32_t>(next_power_of_two(dim)),
          /*SINGLE_TOKEN=*/single_token ? 1 : 0);
}

Glm5NextKPoolSelection glm5_next_kpool_expand_to_physical_slots(
    const torch::Tensor& selected_pool_ids,
    const torch::Tensor& query_positions,
    const torch::Tensor& row_batch,
    const torch::Tensor& block_table,
    int64_t block_size,
    int64_t index_topk,
    int64_t index_kpool,
    bool always_select_tail) {
  CHECK(selected_pool_ids.device().is_privateuseone())
      << "KPool expansion requires MLU.";
  CHECK_LE(selected_pool_ids.size(1), index_topk / index_kpool)
      << "Selected pools exceed the token output budget.";
  const int64_t num_queries = selected_pool_ids.size(0);
  const int64_t selected_pools = selected_pool_ids.size(1);
  const int64_t output_width = index_topk + index_kpool - 1;
  const auto options = selected_pool_ids.options().dtype(torch::kInt32);
  torch::Tensor physical_slots =
      torch::empty({num_queries, output_width}, options);
  torch::Tensor context_lens = torch::empty({num_queries}, options);
  if (num_queries == 0) {
    return {.physical_slots = std::move(physical_slots),
            .context_lens = std::move(context_lens)};
  }
  int64_t pool_tile =
      index_kpool > 16
          ? 16
          : std::min<int64_t>(
                selected_pools >= 1024 ? 1024 : 512,
                next_power_of_two(std::max<int64_t>(selected_pools, 1)));
  // Cross-block pools retain int64 intermediates for every member.
  if (block_size % index_kpool != 0) {
    pool_tile = std::min<int64_t>(pool_tile, 256);
  }
  const int64_t member_tile = std::min<int64_t>(index_kpool > 16 ? 128 : 16,
                                                next_power_of_two(index_kpool));
  // Larger tables stay paged; only a bounded table tile is cached in NRAM.
  const int64_t table_tile =
      block_table.size(1) <= 2048
          ? next_power_of_two(std::max<int64_t>(block_table.size(1), 1))
          : 0;
  const bool local_i32 =
      block_size % index_kpool == 0 &&
      block_table.size(1) <= std::numeric_limits<int32_t>::max() / block_size;
  const int32_t tail_tile = local_i32 ? 1024 : 256;
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  triton_jit::JITKernel::get(kKPoolExpandKernelPath, "kpool_expand")
      .launch(
          static_cast<void*>(queue),
          {static_cast<uint32_t>(std::min<int64_t>(num_queries, 65535)), 1, 1},
          {/*num_warps=*/1, /*num_stages=*/local_i32 ? 3 : 1},
          selected_pool_ids,
          query_positions,
          row_batch,
          block_table,
          physical_slots,
          context_lens,
          selected_pool_ids.stride(0),
          selected_pool_ids.stride(1),
          query_positions.stride(0),
          row_batch.stride(0),
          block_table.stride(0),
          block_table.stride(1),
          block_table.size(1),
          num_queries,
          /*K=*/selected_pools,
          /*P=*/index_kpool,
          /*B=*/block_size,
          /*W=*/output_width,
          /*BP=*/static_cast<int32_t>(pool_tile),
          /*BM=*/static_cast<int32_t>(member_tile),
          /*BT=*/static_cast<int32_t>(table_tile),
          /*TAIL=*/always_select_tail ? 1 : 0,
          /*BTAIL=*/tail_tile,
          /*LOCAL_I32=*/local_i32 ? 1 : 0);
  return {.physical_slots = std::move(physical_slots),
          .context_lens = std::move(context_lens)};
}

Glm5NextKPoolIndexerImpl::Glm5NextKPoolIndexerImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    std::shared_ptr<RotaryEmbeddingBase> rotary_emb,
    const torch::TensorOptions& options)
    : n_heads_(args.index_n_heads()),
      head_dim_(args.index_head_dim()),
      rope_head_dim_(args.qk_rope_head_dim()),
      index_topk_(args.index_topk()),
      index_kpool_(args.index_kpool()),
      block_size_(KVCacheConfig::get_instance().block_size()),
      softmax_scale_(std::pow(static_cast<double>(head_dim_), -0.5) *
                     std::pow(static_cast<double>(n_heads_), -0.5)),
      always_select_tail_(args.index_kpool_always_select_tail()),
      rotary_emb_(std::move(rotary_emb)) {
  CHECK(options.device().is_privateuseone()) << "KPool requires MLU.";
  CHECK(options.dtype() == torch::kBFloat16) << "KPool requires BF16.";
  CHECK_GE(head_dim_, 16) << "Paged KPool scoring requires D >= 16.";
  CHECK_LE(head_dim_, 128);
  CHECK_EQ(head_dim_ & (head_dim_ - 1), 0);
  CHECK_GT(index_kpool_, 0);
  CHECK_GE(index_topk_, 0);
  CHECK_LE(index_topk_,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()) -
               index_kpool_ + 1)
      << "KPool context lengths must fit int32.";
  CHECK_EQ(head_dim_ % index_kpool_, 0)
      << "KPool cache accounting requires D divisible by P.";
  CHECK_LE(index_kpool_ * head_dim_, 8192)
      << "KPool stash exceeds NRAM capacity.";
  CHECK_EQ(block_size_ % index_kpool_, 0);
  CHECK_EQ(parallel_args.kv_split_size_effective(), 1)
      << "KPool does not support DCP cache sharding.";
  CHECK_GT(block_size_, 0);
  const int64_t pool_block_size = block_size_ / index_kpool_;
  CHECK_EQ(pool_block_size & (pool_block_size - 1), 0)
      << "KPool addressing requires a power-of-two pools-per-block.";

  CHECK(rope_head_dim_ == 0 || rotary_emb_ != nullptr)
      << "GLM5-Next KPool RoPE requires a rotary embedding.";

  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(args.q_lora_rank(),
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  wk_ = register_module("wk",
                        ReplicatedLinear(args.hidden_size(),
                                         head_dim_,
                                         /*bias=*/false,
                                         QuantArgs(),
                                         options));
  weights_proj_ =
      register_module("weights_proj",
                      ReplicatedLinear(args.hidden_size(),
                                       n_heads_,
                                       /*bias=*/false,
                                       QuantArgs(),
                                       options.dtype(torch::kFloat32)));
  k_norm_ = register_module(
      "k_norm", RMSNorm(head_dim_, 1e-6, options.dtype(torch::kFloat32)));
  k_norm_->set_layernorm_mode();

  index_kpool_compress_gate_ =
      register_parameter("index_kpool_compress_gate",
                         torch::empty({head_dim_, args.hidden_size()},
                                      options.dtype(torch::kBFloat16)),
                         /*requires_grad=*/false);
  index_kpool_compress_ape_ = register_parameter(
      "index_kpool_compress_ape",
      torch::empty({index_kpool_, head_dim_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hadamard_matrix_ = util::create_hadamard_matrix(
      head_dim_, torch::kFloat32, options.device(), /*normalize=*/true);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_query(
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor q =
      wq_b_->forward(q_norm).view({q_norm.size(0), n_heads_, head_dim_});
  if (rope_head_dim_ > 0) {
    torch::Tensor q_pe = q.slice(/*dim=*/-1, 0, rope_head_dim_);
    rotary_emb_->forward(q_pe,
                         torch::clamp_min(positions, 0),
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  q = q.to(torch::kBFloat16).to(torch::kFloat32);
  return util::hadamard_transform(q, hadamard_matrix_).to(torch::kBFloat16);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_raw_k(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor k = wk_->forward(hidden_states);
  k = glm5_next_kpool_normalize_key(std::move(k), k_norm_);
  if (rope_head_dim_ > 0) {
    torch::Tensor k_pe =
        k.slice(/*dim=*/-1, 0, rope_head_dim_).unsqueeze(/*dim=*/1);
    rotary_emb_->forward(k_pe,
                         positions,
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  return k.to(torch::kBFloat16);
}

struct Glm5NextKPoolIndexerImpl::Execution {
  std::shared_ptr<const KPoolBatchMetadata> batch;
  bool graph_decode = false;
  int64_t score_capacity = 0;
};

Glm5NextKPoolIndexerImpl::Execution Glm5NextKPoolIndexerImpl::prepare_execution(
    const AttentionMetadata& metadata,
    const torch::Tensor& positions) const {
  Execution execution;
  execution.graph_decode = metadata.enable_cuda_graph;
  execution.batch = metadata.kpool_batch_metadata;
  if (!execution.batch) {
    execution.batch =
        make_kpool_batch_metadata(metadata, positions.device(), positions);
  }
  if (execution.graph_decode && metadata.kpool_batch_metadata) {
    auto batch = std::make_shared<KPoolBatchMetadata>(*execution.batch);
    const int64_t tokens = positions.numel();
    const int64_t requests = static_cast<int64_t>(batch->q_seq_lens.size());
    if (metadata.kpool_query_lens.empty() && metadata.q_cu_seq_lens.defined() &&
        metadata.q_cu_seq_lens.numel() == requests + 1) {
      batch->query_starts = metadata.q_cu_seq_lens.to(torch::kInt64);
    }
    batch->block_table = metadata.block_table;
    if (batch->block_table.defined() &&
        batch->block_table.size(0) != requests) {
      batch->block_table = batch->block_table.index_select(
          0, batch->query_starts.narrow(0, 0, requests));
    }
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
    execution.batch = std::move(batch);
  }
  execution.score_capacity =
      execution.graph_decode
          ? execution.batch->block_table.size(1) * block_size_
          : execution.batch->max_kv_len;
  return execution;
}

void Glm5NextKPoolIndexerImpl::update_cache(const torch::Tensor& raw_k,
                                            const torch::Tensor& gate_bf16,
                                            const torch::Tensor& positions,
                                            torch::Tensor& index_cache,
                                            torch::Tensor& tail_cache,
                                            const Execution& execution) {
  const auto& batch = *execution.batch;
  const bool single_token =
      !execution.graph_decode &&
      std::all_of(batch.q_seq_lens.begin(),
                  batch.q_seq_lens.end(),
                  [](int32_t length) { return length == 1; });
  launch_kpool_update(raw_k,
                      gate_bf16,
                      index_kpool_compress_ape_,
                      hadamard_matrix_,
                      index_cache,
                      tail_cache,
                      batch.tail_indices,
                      batch.block_table,
                      positions,
                      batch.row_batch,
                      batch.query_starts,
                      block_size_,
                      index_kpool_,
                      single_token);
}

torch::Tensor Glm5NextKPoolIndexerImpl::select_pools(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const torch::Tensor& index_cache,
    const AttentionMetadata& metadata,
    const Execution& execution) {
  const torch::Tensor query = project_query(q_norm, positions, metadata);
  const torch::Tensor weights =
      weights_proj_->forward(hidden_states.to(torch::kFloat32));
  return glm5_next_kpool_select(
      query,
      weights,
      positions,
      execution.batch->row_batch,
      index_cache,
      execution.batch->block_table,
      execution.score_capacity,
      block_size_,
      index_kpool_,
      index_topk_,
      softmax_scale_,
      /*workspace_bytes=*/kWorkspaceBytes,
      execution.graph_decode ? nullptr : execution.batch.get());
}

std::tuple<torch::Tensor, torch::Tensor> Glm5NextKPoolIndexerImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(tail_cache.defined()) << "KPool requires framework-owned tail storage.";
  if (hidden_states.size(0) == 0) {
    const auto options = hidden_states.options().dtype(torch::kInt32);
    return {torch::empty({0, output_width()}, options),
            torch::empty({0}, options)};
  }
  torch::Tensor raw_k = project_raw_k(hidden_states, positions, attn_metadata);
  torch::Tensor gate_score =
      torch::nn::functional::linear(hidden_states, index_kpool_compress_gate_);
  const torch::Tensor gate_bf16 = gate_score.to(torch::kBFloat16);
  const Execution execution = prepare_execution(attn_metadata, positions);
  CHECK_EQ(execution.batch->row_batch.numel(), positions.numel());
  CHECK(execution.batch->tail_indices.defined());
  if (attn_metadata.is_spec_verify) {
    CHECK_LE(max_sequence_length(execution.batch->q_seq_lens),
             tail_cache.size(2) - index_kpool_)
        << "KPool tail capacity does not cover the speculative write window.";
  }
  torch::Tensor active =
      execution.batch->tail_indices.index_select(
          0, execution.batch->row_batch) > kPaddingLinearStateId;
  if (attn_metadata.slot_mapping.defined()) {
    active = active & (attn_metadata.slot_mapping.reshape({-1}) > 0);
  }
  const torch::Tensor cache_positions = torch::where(active, positions, -1);
  update_cache(
      raw_k, gate_bf16, cache_positions, index_cache, tail_cache, execution);
  const torch::Tensor pools = select_pools(hidden_states,
                                           q_norm,
                                           cache_positions,
                                           index_cache,
                                           attn_metadata,
                                           execution);
  Glm5NextKPoolSelection selection =
      glm5_next_kpool_expand_to_physical_slots(pools,
                                               cache_positions,
                                               execution.batch->row_batch,
                                               execution.batch->block_table,
                                               block_size_,
                                               index_topk_,
                                               index_kpool_,
                                               always_select_tail_);
  return {std::move(selection.physical_slots),
          std::move(selection.context_lens)};
}

void Glm5NextKPoolIndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  wk_->load_state_dict(state_dict.get_dict_with_prefix("wk."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  LOAD_WEIGHT(index_kpool_compress_gate);
  LOAD_WEIGHT(index_kpool_compress_ape);
}

}  // namespace xllm::layer
