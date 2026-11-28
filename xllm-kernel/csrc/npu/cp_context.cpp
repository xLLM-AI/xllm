// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace xllm_kernel {
namespace {

torch::Tensor copy_indices(const std::vector<int64_t>& values,
                           torch::Device device) {
  if (device.is_cpu()) {
    return torch::tensor(values, torch::dtype(torch::kInt64));
  }
  auto host = torch::empty(
      {static_cast<int64_t>(values.size())},
      torch::dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), host.data_ptr<int64_t>());
  }
  return host.to(device, /*non_blocking=*/true);
}

// Pair the head and tail chunks on each rank. Emit the packed query and KV
// prefix ranges in local order, then build the inverse rank-major mapping.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::vector<int64_t>,
           std::vector<int64_t>,
           torch::Tensor,
           std::vector<int64_t>,
           int64_t>
build_cp_context(const std::vector<int64_t>& q_seq_lens,
                 const std::vector<int64_t>& kv_seq_lens,
                 int64_t cp_size,
                 int64_t cp_rank,
                 torch::Device device) {
  CHECK(cp_size > 1);
  CHECK(cp_rank >= 0);
  CHECK(cp_rank < cp_size);
  CHECK(q_seq_lens.size() == kv_seq_lens.size());

  const int64_t num_chunks = cp_size * 2;
  // The two chunk ids this rank owns: an early one and its mirror.
  const int64_t first_chunk = cp_rank;
  const int64_t second_chunk = num_chunks - 1 - cp_rank;

  std::vector<int64_t> shard_index;
  std::vector<int64_t> query_index;
  std::vector<int64_t> q_cu_seqlens;
  std::vector<int64_t> kv_gather_index;
  std::vector<int64_t> kv_cu_seqlens;
  std::vector<int64_t> segment_seq_indices;
  std::vector<int64_t> segment_kv_seq_lens;
  // restore_index needs the per-seq local segment offset (same on every rank)
  // and the ownership map, so accumulate it in a second pass below.
  std::vector<int64_t> chunk_lens;
  std::vector<int64_t> local_seg_offsets;
  chunk_lens.reserve(q_seq_lens.size());
  local_seg_offsets.reserve(q_seq_lens.size());
  segment_seq_indices.reserve(q_seq_lens.size() * 2);
  segment_kv_seq_lens.reserve(q_seq_lens.size() * 2);

  int64_t global_offset = 0;
  int64_t local_offset = 0;
  int64_t q_run = 0;
  int64_t kv_run = 0;
  for (size_t seq_id = 0; seq_id < q_seq_lens.size(); ++seq_id) {
    const int64_t length = q_seq_lens[seq_id];
    const int64_t kv_length = kv_seq_lens[seq_id];
    CHECK(length >= 0);
    CHECK(kv_length >= length);
    const int64_t prefix_cache_length = kv_length - length;
    const int64_t padded =
        ((length + num_chunks - 1) / num_chunks) * num_chunks;
    const int64_t chunk_len = padded / num_chunks;
    chunk_lens.push_back(chunk_len);
    local_seg_offsets.push_back(local_offset);

    // Emit the two owned segments in local order: first half then second half.
    // For each, the real rows sit at the front (small j) because real position
    // grows with j, so query_index stays front-packed per segment.
    const int64_t halves[2][2] = {{0, first_chunk}, {1, second_chunk}};
    for (const auto& half_chunk : halves) {
      const int64_t half = half_chunk[0];
      const int64_t chunk_id = half_chunk[1];
      const int64_t seg_local_base = local_offset + half * chunk_len;
      const int64_t seg_start = chunk_id * chunk_len;  // first real position
      int64_t real_count = 0;
      for (int64_t j = 0; j < chunk_len; ++j) {
        const int64_t pos_in_seq = seg_start + j;
        if (pos_in_seq < length) {
          shard_index.push_back(global_offset + pos_in_seq);
          query_index.push_back(seg_local_base + j);
          ++real_count;
        } else {
          shard_index.push_back(-1);
        }
      }
      if (real_count > 0) {
        segment_seq_indices.push_back(static_cast<int64_t>(seq_id));
        // Causal prefix ends exactly at the last real query position + 1
        // = seg_start + real_count (segment is a contiguous real range).
        const int64_t prefix_len = seg_start + real_count;
        segment_kv_seq_lens.push_back(prefix_cache_length + prefix_len);
        q_run += real_count;
        q_cu_seqlens.push_back(q_run);
        for (int64_t p = 0; p < prefix_len; ++p) {
          kv_gather_index.push_back(global_offset + p);
        }
        kv_run += prefix_len;
        kv_cu_seqlens.push_back(kv_run);
      }
    }

    global_offset += length;
    local_offset += 2 * chunk_len;
  }

  const int64_t total_local = local_offset;

  // restore_index: for every global (real) row, where it lands in the
  // rank-major all-gather output [cp_size * total_local]. Its final size is the
  // total real-token count, which the first pass accumulated into
  // global_offset.
  std::vector<int64_t> restore_index;
  restore_index.reserve(global_offset);
  for (size_t s = 0; s < q_seq_lens.size(); ++s) {
    const int64_t length = q_seq_lens[s];
    const int64_t chunk_len = chunk_lens[s];
    const int64_t seg_offset = local_seg_offsets[s];
    for (int64_t pos_in_seq = 0; pos_in_seq < length; ++pos_in_seq) {
      const int64_t chunk_id = pos_in_seq / chunk_len;
      const int64_t row_in_chunk = pos_in_seq % chunk_len;
      int64_t owner_rank;
      int64_t local_pos;
      if (chunk_id < cp_size) {
        owner_rank = chunk_id;
        local_pos = seg_offset + row_in_chunk;
      } else {
        owner_rank = num_chunks - 1 - chunk_id;
        local_pos = seg_offset + chunk_len + row_in_chunk;
      }
      restore_index.push_back(owner_rank * total_local + local_pos);
    }
  }

  auto shard_tensor = torch::tensor(shard_index, torch::dtype(torch::kInt64));
  auto valid_mask = shard_tensor >= 0;
  auto gather_index =
      torch::where(valid_mask, shard_tensor, torch::zeros_like(shard_tensor));

  return std::make_tuple(shard_tensor.to(device),
                         gather_index.to(device),
                         valid_mask.to(device),
                         copy_indices(restore_index, device),
                         copy_indices(query_index, device),
                         copy_indices(kv_gather_index, device),
                         q_cu_seqlens,
                         kv_cu_seqlens,
                         copy_indices(segment_seq_indices, device),
                         segment_kv_seq_lens,
                         total_local);
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "build_cp_context(int[] q_seq_lens, int[] kv_seq_lens, int cp_size, "
      "int cp_rank, Device device) -> (Tensor shard_index, Tensor "
      "shard_gather_index, Tensor "
      "shard_valid_mask, Tensor restore_index, Tensor query_index, Tensor "
      "kv_gather_index, int[] q_cu_seqlens, int[] kv_cu_seqlens, Tensor "
      "segment_seq_indices, int[] segment_kv_seq_lens, int total_local)",
      TORCH_FN(build_cp_context));
}

}  // namespace
}  // namespace xllm_kernel
