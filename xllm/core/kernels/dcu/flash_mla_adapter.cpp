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

#include "kernels/dcu/flash_mla_adapter.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

// flash-mla native extension symbols. Keep these declarations at global scope
// to match the unqualified symbols exported by the .so.

// Returns {tile_scheduler_metadata, num_splits}.
std::vector<torch::Tensor> get_mla_decoding_metadata_dense_fp8(
    torch::Tensor& seqlens_k,
    int32_t num_heads_per_head_k,
    int32_t num_heads_k);

// Split-q BF16/FP16 MLA decode.
std::vector<torch::Tensor> mha_fwd_kvcache_mla_nope_pe(
    torch::Tensor& q_nope,
    torch::Tensor& q_pe,
    const torch::Tensor& kcache,
    std::optional<const torch::Tensor>& vcache,
    int32_t head_size_v,
    const torch::Tensor& seqlens_k,
    const torch::Tensor& block_table,
    float softmax_scale,
    bool is_causal,
    const torch::Tensor& tile_scheduler_metadata,
    const torch::Tensor& num_splits);

namespace xllm {
namespace kernel {
namespace dcu {
namespace flash_mla {

namespace {

void check_dense_decode_inputs(const DenseDecodeParams& params) {
  CHECK(params.q_nope.defined()) << "flash_mla: q_nope must be defined";
  CHECK(params.q_pe.defined()) << "flash_mla: q_pe must be defined";
  CHECK(params.k_cache.defined()) << "flash_mla: k_cache must be defined";
  CHECK(params.seqlens_k.defined()) << "flash_mla: seqlens_k must be defined";
  CHECK(params.block_table.defined())
      << "flash_mla: block_table must be defined";

  const torch::Device kDev = params.k_cache.device();
  const torch::Device q_nope_dev = params.q_nope.device();
  const torch::Device q_pe_dev = params.q_pe.device();
  const torch::Device seqlens_dev = params.seqlens_k.device();
  const torch::Device block_table_dev = params.block_table.device();
  CHECK(q_nope_dev == kDev) << "flash_mla: q_nope must be on the DCU device";
  CHECK(q_pe_dev == kDev) << "flash_mla: q_pe must be on the DCU device";
  CHECK(seqlens_dev == kDev)
      << "flash_mla: seqlens_k must be on the DCU device";
  CHECK(block_table_dev == kDev)
      << "flash_mla: block_table must be on the DCU device";

  CHECK_EQ(params.q_nope.dim(), 4)
      << "flash_mla: q_nope must be 4D [B,S_q,H_q,D]";
  CHECK_EQ(params.q_pe.dim(), 4) << "flash_mla: q_pe must be 4D [B,S_q,H_q,D]";
  CHECK_EQ(params.k_cache.dim(), 4)
      << "flash_mla: k_cache must be 4D [blocks,page,1,D]";
  CHECK_EQ(params.k_cache.size(1), 64)
      << "flash_mla: page/block size must be 64, got "
      << params.k_cache.size(1);
  CHECK_EQ(params.k_cache.size(2), 1)
      << "flash_mla: MLA expects a single KV head (num_heads_k == 1)";
  CHECK_EQ(params.seqlens_k.dim(), 1) << "flash_mla: seqlens_k must be 1D [B]";
  CHECK_EQ(params.block_table.dim(), 2)
      << "flash_mla: block_table must be 2D [B,max_blocks]";

  const int64_t q_nope_dim = params.q_nope.size(3);
  const int64_t q_pe_dim = params.q_pe.size(3);
  CHECK_EQ(params.k_cache.size(3), q_nope_dim + q_pe_dim)
      << "flash_mla: k_cache last dim must equal q_nope.dim + q_pe.dim";
  CHECK_EQ(params.q_nope.size(0), params.q_pe.size(0))
      << "flash_mla: q_nope/q_pe batch must match";
  CHECK_EQ(params.q_nope.size(1), params.q_pe.size(1))
      << "flash_mla: q_nope/q_pe seq must match";
  CHECK_EQ(params.q_nope.size(2), params.q_pe.size(2))
      << "flash_mla: q_nope/q_pe head count must match";
  CHECK_GT(params.head_size_v, 0) << "flash_mla: head_size_v must be positive";
}

}  // namespace

torch::Tensor dense_decode(DenseDecodeParams& params) {
  CHECK(params.kind == DenseDecodeKind::kQNopePe)
      << "flash_mla: only DenseDecodeKind::kQNopePe (BF16/FP16 split-q) is "
         "supported in phase 1";
  check_dense_decode_inputs(params);

  torch::Tensor seqlens_k = params.seqlens_k;
  if (seqlens_k.scalar_type() != torch::kInt32) {
    seqlens_k = seqlens_k.to(torch::kInt32);
  }
  torch::Tensor block_table = params.block_table;
  if (block_table.scalar_type() != torch::kInt32) {
    block_table = block_table.to(torch::kInt32);
  }

  const int64_t seq_q = params.q_nope.size(1);
  const int64_t heads_q = params.q_nope.size(2);
  const int32_t num_heads_k = 1;
  const int32_t num_heads_per_head_k =
      static_cast<int32_t>(seq_q * heads_q / num_heads_k);

  auto metadata = get_mla_decoding_metadata_dense_fp8(
      seqlens_k, num_heads_per_head_k, num_heads_k);
  CHECK_EQ(metadata.size(), 2u)
      << "flash_mla: expected {tile_scheduler_metadata, num_splits}";
  const torch::Tensor& tile_scheduler_metadata = metadata[0];
  const torch::Tensor& num_splits = metadata[1];

  CHECK_GT(params.softmax_scale, 0.0F)
      << "flash_mla: softmax_scale must be set by the MLA attention layer";
  const float softmax_scale = params.softmax_scale;

  // MLA latent cache is key-only.
  std::optional<const torch::Tensor> vcache = std::nullopt;

  auto out_lse = mha_fwd_kvcache_mla_nope_pe(
      /*q_nope=*/params.q_nope,
      /*q_pe=*/params.q_pe,
      /*kcache=*/params.k_cache,
      /*vcache=*/vcache,
      /*head_size_v=*/static_cast<int32_t>(params.head_size_v),
      /*seqlens_k=*/seqlens_k,
      /*block_table=*/block_table,
      /*softmax_scale=*/softmax_scale,
      /*is_causal=*/params.is_causal,
      /*tile_scheduler_metadata=*/tile_scheduler_metadata,
      /*num_splits=*/num_splits);
  CHECK(!out_lse.empty()) << "flash_mla: decode returned no output";
  return out_lse[0];
}

}  // namespace flash_mla
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
