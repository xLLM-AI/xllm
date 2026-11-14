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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "models/dit/transformers/minimax_h3_tp_layout.h"

namespace xllm {

// CPU characterization helpers for canonical checkpoint sharding. Runtime
// QKV uses the unpadded logical shard and pads only activations; runtime output
// projections retain the full replicated weight. The padded weight helpers
// below are test oracles for the dummy-head geometry, not resident loaders.

namespace minimax_h3_tp_weights_detail {

inline void check_tp_args(int64_t tp_rank, int64_t tp_size) {
  CHECK(tp_size > 0) << "MiniMax-H3 tp_size must be positive, got " << tp_size;
  CHECK(tp_rank >= 0 && tp_rank < tp_size)
      << "MiniMax-H3 tp_rank " << tp_rank << " out of range [0, " << tp_size
      << ")";
}

// Head-major grouped-QKV tensors store dim 0 as global_heads blocks of
// (3 * head_dim) rows (per-head Q/K/V). Slice this rank's contiguous logical
// head block -- identical to ColumnParallelLinear's dim-0 chunk -- then append
// dummy_heads zero blocks so the head count becomes physical (28 -> 32).
inline torch::Tensor slice_grouped_qkv_heads(const torch::Tensor& grouped,
                                             int64_t tp_rank,
                                             int64_t tp_size) {
  check_tp_args(tp_rank, tp_size);
  constexpr int64_t kHeadBlock = 3 * kMiniMaxH3HeadDim;
  constexpr int64_t kFullRows = kMiniMaxH3GlobalHeads * kHeadBlock;
  CHECK(grouped.dim() >= 1)
      << "MiniMax-H3 grouped QKV needs at least 1 dimension, got "
      << grouped.dim();
  CHECK(grouped.size(0) == kFullRows)
      << "MiniMax-H3 grouped QKV expects dim0 " << kFullRows
      << " (3*56*128), got " << grouped.size(0);
  CHECK(kMiniMaxH3GlobalHeads % tp_size == 0)
      << "MiniMax-H3 global heads 56 not divisible by tp_size " << tp_size;
  const int64_t logical_heads = kMiniMaxH3GlobalHeads / tp_size;
  CHECK(logical_heads == kMiniMaxH3LogicalHeadsPerTp)
      << "MiniMax-H3 head padding is defined for the TP2 contract ("
      << kMiniMaxH3LogicalHeadsPerTp << " logical heads), got "
      << logical_heads;
  const int64_t logical_rows = logical_heads * kHeadBlock;
  const torch::Tensor logical =
      grouped.narrow(0, tp_rank * logical_rows, logical_rows);
  std::vector<int64_t> dummy_sizes = logical.sizes().vec();
  dummy_sizes[0] = kMiniMaxH3DummyHeads * kHeadBlock;
  return torch::cat({logical, torch::zeros(dummy_sizes, grouped.options())}, 0);
}

}  // namespace minimax_h3_tp_weights_detail

// Grouped-QKV weight [3*56*128, hidden] -> this rank's [3*32*128, hidden] shard
// (28 logical heads in head-major Q/K/V order + 4 zero dummy heads).
inline torch::Tensor minimax_h3_slice_grouped_qkv_weight(
    const torch::Tensor& weight,
    int64_t tp_rank,
    int64_t tp_size = kMiniMaxH3TpSize) {
  CHECK(weight.dim() == 2)
      << "MiniMax-H3 grouped QKV weight must be 2D [rows, hidden], got "
      << weight.dim() << "D";
  return minimax_h3_tp_weights_detail::slice_grouped_qkv_heads(
      weight, tp_rank, tp_size);
}

// Grouped-QKV bias [3*56*128] -> this rank's [3*32*128] shard (dummy heads 0).
// ColumnParallelLinear shards bias on dim 0 with the same head-major layout.
inline torch::Tensor minimax_h3_slice_grouped_qkv_bias(
    const torch::Tensor& bias,
    int64_t tp_rank,
    int64_t tp_size = kMiniMaxH3TpSize) {
  CHECK(bias.dim() == 1)
      << "MiniMax-H3 grouped QKV bias must be 1D [rows], got " << bias.dim()
      << "D";
  return minimax_h3_tp_weights_detail::slice_grouped_qkv_heads(
      bias, tp_rank, tp_size);
}

// out_proj (row-parallel) weight [hidden, 56*128] -> this rank's
// [hidden, 32*128]: the 28 logical-head columns (contiguous dim-1 chunk,
// exactly RowParallelLinear's shard) followed by 4 zero dummy-head columns. The
// out_proj bias is replicated (not sharded) by RowParallelLinear, so it needs
// no adapter.
inline torch::Tensor minimax_h3_slice_out_proj_weight(
    const torch::Tensor& weight,
    int64_t tp_rank,
    int64_t tp_size = kMiniMaxH3TpSize) {
  minimax_h3_tp_weights_detail::check_tp_args(tp_rank, tp_size);
  constexpr int64_t kInner = kMiniMaxH3GlobalHeads * kMiniMaxH3HeadDim;
  CHECK(weight.dim() == 2)
      << "MiniMax-H3 out_proj weight must be 2D [hidden, inner], got "
      << weight.dim() << "D";
  CHECK(weight.size(1) == kInner)
      << "MiniMax-H3 out_proj expects dim1 " << kInner << " (56*128), got "
      << weight.size(1);
  CHECK(kMiniMaxH3GlobalHeads % tp_size == 0)
      << "MiniMax-H3 global heads 56 not divisible by tp_size " << tp_size;
  const int64_t logical_heads = kMiniMaxH3GlobalHeads / tp_size;
  CHECK(logical_heads == kMiniMaxH3LogicalHeadsPerTp)
      << "MiniMax-H3 head padding is defined for the TP2 contract ("
      << kMiniMaxH3LogicalHeadsPerTp << " logical heads), got "
      << logical_heads;
  const int64_t logical_cols = logical_heads * kMiniMaxH3HeadDim;
  const torch::Tensor logical =
      weight.narrow(1, tp_rank * logical_cols, logical_cols);
  std::vector<int64_t> dummy_sizes = logical.sizes().vec();
  dummy_sizes[1] = kMiniMaxH3DummyHeads * kMiniMaxH3HeadDim;
  return torch::cat({logical, torch::zeros(dummy_sizes, weight.options())}, 1);
}

// MLP gate/up (column-parallel) fused weight [2*ffn, hidden] -> this rank's
// [2*ffn/tp, hidden]. The two consecutive ffn halves (e.g. gate then up) are
// each sliced along dim 0 to their contiguous ffn shard and re-concatenated in
// the same order, matching a merged column-parallel linear with two equal
// shards.
inline torch::Tensor minimax_h3_slice_gate_up_weight(
    const torch::Tensor& weight,
    int64_t tp_rank,
    int64_t tp_size = kMiniMaxH3TpSize) {
  minimax_h3_tp_weights_detail::check_tp_args(tp_rank, tp_size);
  CHECK(weight.dim() == 2)
      << "MiniMax-H3 gate/up weight must be 2D [2*ffn, hidden], got "
      << weight.dim() << "D";
  CHECK(weight.size(0) % 2 == 0)
      << "MiniMax-H3 gate/up dim0 must be even (two halves), got "
      << weight.size(0);
  const int64_t ffn = weight.size(0) / 2;
  CHECK(ffn % tp_size == 0) << "MiniMax-H3 ffn dim " << ffn
                            << " not divisible by tp_size " << tp_size;
  const int64_t shard = ffn / tp_size;
  const torch::Tensor first =
      weight.narrow(0, 0, ffn).narrow(0, tp_rank * shard, shard);
  const torch::Tensor second =
      weight.narrow(0, ffn, ffn).narrow(0, tp_rank * shard, shard);
  return torch::cat({first, second}, 0);
}

// MLP down (row-parallel) weight [hidden, ffn] -> this rank's [hidden, ffn/tp]
// contiguous dim-1 shard, exactly RowParallelLinear's shard.
inline torch::Tensor minimax_h3_slice_down_weight(
    const torch::Tensor& weight,
    int64_t tp_rank,
    int64_t tp_size = kMiniMaxH3TpSize) {
  minimax_h3_tp_weights_detail::check_tp_args(tp_rank, tp_size);
  CHECK(weight.dim() == 2)
      << "MiniMax-H3 down weight must be 2D [hidden, ffn], got " << weight.dim()
      << "D";
  CHECK(weight.size(1) % tp_size == 0)
      << "MiniMax-H3 down dim1 " << weight.size(1)
      << " not divisible by tp_size " << tp_size;
  const int64_t shard = weight.size(1) / tp_size;
  return weight.narrow(1, tp_rank * shard, shard).contiguous();
}

}  // namespace xllm
