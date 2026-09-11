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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>
#include <string_view>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_GLM52_FP8_SPARSE_MLA_ATTENTION_REGISTRY_INC
#error "XLLM_TL_GLM52_FP8_SPARSE_MLA_ATTENTION_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kLatentDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kTopk = 2048;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kCoreNum = 24;
constexpr int64_t kHeadTile = 16;
constexpr int64_t kKvTile = 64;
constexpr int64_t kMaxNumQueries = 1024;
constexpr int64_t kMaxCacheBlocks = 32768;
constexpr int64_t kMaxBlockTableLen = 32768;

#include XLLM_TL_GLM52_FP8_SPARSE_MLA_ATTENTION_REGISTRY_INC

void check_npu(const torch::Tensor& tensor, std::string_view name) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK_EQ(tensor.device().type(), torch::kPrivateUse1)
      << name << " must be on NPU";
}

void check_npu_contiguous(const torch::Tensor& tensor, std::string_view name) {
  check_npu(tensor, name);
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

void check_same_device(const torch::Tensor& tensor,
                       const torch::Tensor& reference,
                       std::string_view name) {
  CHECK(tensor.device() == reference.device())
      << name << " must be on the same NPU device as q_latent";
}

void check_workspace(const torch::Tensor& workspace,
                     std::string_view name,
                     torch::ScalarType dtype,
                     int64_t core_slots,
                     int64_t dim_1,
                     int64_t dim_2) {
  check_npu_contiguous(workspace, name);
  CHECK_EQ(workspace.dtype(), dtype) << name << " dtype mismatch";
  CHECK_EQ(workspace.dim(), 3) << name << " must be 3D";
  CHECK_EQ(workspace.size(0), core_slots)
      << name << " core slot dimension mismatch";
  CHECK_EQ(workspace.size(1), dim_1) << name << " tile dimension mismatch";
  CHECK_EQ(workspace.size(2), dim_2) << name << " feature dimension mismatch";
}

void check_supported(const torch::Tensor& q_latent,
                     const torch::Tensor& q_rope,
                     const torch::Tensor& nope_cache,
                     const torch::Tensor& rope_cache,
                     const torch::Tensor& topk_indices,
                     const torch::Tensor& block_table,
                     const torch::Tensor& actual_seq_lengths_kv,
                     const torch::Tensor& e4m3_decode_table,
                     const torch::Tensor& output,
                     const torch::Tensor& workspace_k,
                     const torch::Tensor& workspace_k_rope,
                     const torch::Tensor& workspace_scores,
                     const torch::Tensor& workspace_probs,
                     const torch::Tensor& workspace_output,
                     const torch::Tensor& workspace_q,
                     const torch::Tensor& workspace_q_rope) {
  check_npu(q_latent, "TileLang GLM-5.2 FP8 MLA: q_latent");
  check_npu(q_rope, "TileLang GLM-5.2 FP8 MLA: q_rope");
  check_npu_contiguous(nope_cache, "TileLang GLM-5.2 FP8 MLA: nope_cache");
  check_npu_contiguous(rope_cache, "TileLang GLM-5.2 FP8 MLA: rope_cache");
  check_npu_contiguous(topk_indices, "TileLang GLM-5.2 FP8 MLA: topk_indices");
  check_npu_contiguous(block_table, "TileLang GLM-5.2 FP8 MLA: block_table");
  check_npu_contiguous(actual_seq_lengths_kv,
                       "TileLang GLM-5.2 FP8 MLA: actual_seq_lengths_kv");
  check_npu_contiguous(e4m3_decode_table,
                       "TileLang GLM-5.2 FP8 MLA: e4m3_decode_table");
  check_npu_contiguous(output, "TileLang GLM-5.2 FP8 MLA: output");

  CHECK_EQ(q_latent.dtype(), torch::kBFloat16)
      << "TileLang GLM-5.2 FP8 MLA: q_latent must be bfloat16";
  CHECK_EQ(q_rope.dtype(), q_latent.dtype())
      << "TileLang GLM-5.2 FP8 MLA: q_rope dtype mismatch";
  CHECK_EQ(nope_cache.dtype(), torch::kUInt8)
      << "TileLang GLM-5.2 FP8 MLA: nope_cache must contain raw uint8 E4M3 "
         "bytes";
  CHECK_EQ(rope_cache.dtype(), torch::kUInt8)
      << "TileLang GLM-5.2 FP8 MLA: rope_cache must contain raw uint8 E4M3 "
         "bytes";
  CHECK_EQ(topk_indices.dtype(), torch::kInt32)
      << "TileLang GLM-5.2 FP8 MLA: topk_indices must be int32";
  CHECK_EQ(block_table.dtype(), torch::kInt32)
      << "TileLang GLM-5.2 FP8 MLA: block_table must be int32";
  CHECK_EQ(actual_seq_lengths_kv.dtype(), torch::kInt32)
      << "TileLang GLM-5.2 FP8 MLA: actual_seq_lengths_kv must be int32";
  CHECK_EQ(e4m3_decode_table.dtype(), torch::kFloat32)
      << "TileLang GLM-5.2 FP8 MLA: E4M3 decode table must be float32";
  CHECK_EQ(output.dtype(), q_latent.dtype())
      << "TileLang GLM-5.2 FP8 MLA: output dtype mismatch";

  CHECK_EQ(q_latent.dim(), 3)
      << "TileLang GLM-5.2 FP8 MLA: q_latent must be [T, H, 512]";
  CHECK_EQ(q_rope.dim(), 3)
      << "TileLang GLM-5.2 FP8 MLA: q_rope must be [T, H, 64]";
  CHECK_EQ(q_latent.size(2), kLatentDim)
      << "TileLang GLM-5.2 FP8 MLA: latent dimension must be 512";
  CHECK_EQ(q_rope.size(2), kRopeDim)
      << "TileLang GLM-5.2 FP8 MLA: rope dimension must be 64";
  CHECK_EQ(q_rope.size(0), q_latent.size(0))
      << "TileLang GLM-5.2 FP8 MLA: query token mismatch";
  CHECK_EQ(q_rope.size(1), q_latent.size(1))
      << "TileLang GLM-5.2 FP8 MLA: query head mismatch";
  CHECK_EQ(q_latent.stride(2), 1)
      << "TileLang GLM-5.2 FP8 MLA: q_latent last dimension must be contiguous";
  CHECK_EQ(q_rope.stride(2), 1)
      << "TileLang GLM-5.2 FP8 MLA: q_rope last dimension must be contiguous";
  CHECK_GT(q_latent.stride(0), 0)
      << "TileLang GLM-5.2 FP8 MLA: q_latent token stride must be positive";
  CHECK_GT(q_latent.stride(1), 0)
      << "TileLang GLM-5.2 FP8 MLA: q_latent head stride must be positive";
  CHECK_GT(q_rope.stride(0), 0)
      << "TileLang GLM-5.2 FP8 MLA: q_rope token stride must be positive";
  CHECK_GT(q_rope.stride(1), 0)
      << "TileLang GLM-5.2 FP8 MLA: q_rope head stride must be positive";
  CHECK(q_latent.size(1) == 4 || q_latent.size(1) == 8 ||
        q_latent.size(1) == 16)
      << "TileLang GLM-5.2 FP8 MLA: supported local head counts are 4, 8, 16";
  CHECK_LE(q_latent.size(0), kMaxNumQueries)
      << "TileLang GLM-5.2 FP8 MLA: query count exceeds compile limit";

  CHECK_EQ(nope_cache.dim(), 4)
      << "TileLang GLM-5.2 FP8 MLA: nope_cache must be [B, 128, 1, 512]";
  CHECK_EQ(rope_cache.dim(), 4)
      << "TileLang GLM-5.2 FP8 MLA: rope_cache must be [B, 128, 1, 64]";
  CHECK_EQ(nope_cache.size(0), rope_cache.size(0))
      << "TileLang GLM-5.2 FP8 MLA: cache block count mismatch";
  CHECK_GT(nope_cache.size(0), 0)
      << "TileLang GLM-5.2 FP8 MLA: cache must contain at least one block";
  CHECK_LE(nope_cache.size(0), kMaxCacheBlocks)
      << "TileLang GLM-5.2 FP8 MLA: cache block count exceeds compile limit";
  CHECK_EQ(nope_cache.size(1), kBlockSize)
      << "TileLang GLM-5.2 FP8 MLA: cache block size must be 128";
  CHECK_EQ(rope_cache.size(1), kBlockSize)
      << "TileLang GLM-5.2 FP8 MLA: rope cache block size must be 128";
  CHECK_EQ(nope_cache.size(2), 1)
      << "TileLang GLM-5.2 FP8 MLA: nope_cache must have one KV head";
  CHECK_EQ(rope_cache.size(2), 1)
      << "TileLang GLM-5.2 FP8 MLA: rope_cache must have one KV head";
  CHECK_EQ(nope_cache.size(3), kLatentDim)
      << "TileLang GLM-5.2 FP8 MLA: nope_cache dimension must be 512";
  CHECK_EQ(rope_cache.size(3), kRopeDim)
      << "TileLang GLM-5.2 FP8 MLA: rope_cache dimension must be 64";

  CHECK_EQ(topk_indices.dim(), 3)
      << "TileLang GLM-5.2 FP8 MLA: topk_indices must be [T, 1, 2048]";
  CHECK_EQ(topk_indices.size(0), q_latent.size(0))
      << "TileLang GLM-5.2 FP8 MLA: topk query count mismatch";
  CHECK_EQ(topk_indices.size(1), 1)
      << "TileLang GLM-5.2 FP8 MLA: topk_indices must have one KV head";
  CHECK_EQ(topk_indices.size(2), kTopk)
      << "TileLang GLM-5.2 FP8 MLA: topk count must be 2048";

  CHECK_EQ(block_table.dim(), 2)
      << "TileLang GLM-5.2 FP8 MLA: block_table must be 2D";
  CHECK_EQ(block_table.size(0), q_latent.size(0))
      << "TileLang GLM-5.2 FP8 MLA: block_table batch mismatch";
  CHECK_GT(block_table.size(1), 0) << "TileLang GLM-5.2 FP8 MLA: block_table "
                                      "must contain at least one block";
  CHECK_LE(block_table.size(1), kMaxBlockTableLen)
      << "TileLang GLM-5.2 FP8 MLA: block_table stride exceeds compile limit";
  CHECK_EQ(actual_seq_lengths_kv.dim(), 1)
      << "TileLang GLM-5.2 FP8 MLA: actual_seq_lengths_kv must be 1D";
  CHECK_EQ(actual_seq_lengths_kv.size(0), q_latent.size(0))
      << "TileLang GLM-5.2 FP8 MLA: sequence length batch mismatch";
  CHECK_EQ(e4m3_decode_table.dim(), 1)
      << "TileLang GLM-5.2 FP8 MLA: E4M3 decode table must be 1D";
  CHECK_EQ(e4m3_decode_table.size(0), 256)
      << "TileLang GLM-5.2 FP8 MLA: E4M3 decode table must have 256 entries";
  CHECK_EQ(output.sizes(), q_latent.sizes())
      << "TileLang GLM-5.2 FP8 MLA: output shape mismatch";

  check_workspace(workspace_k,
                  "TileLang GLM-5.2 FP8 MLA: workspace_k",
                  torch::kBFloat16,
                  kCoreNum,
                  kKvTile,
                  kLatentDim);
  check_workspace(workspace_k_rope,
                  "TileLang GLM-5.2 FP8 MLA: workspace_k_rope",
                  torch::kBFloat16,
                  kCoreNum,
                  kKvTile,
                  kRopeDim);
  check_workspace(workspace_scores,
                  "TileLang GLM-5.2 FP8 MLA: workspace_scores",
                  torch::kFloat32,
                  kCoreNum,
                  kHeadTile,
                  kKvTile);
  check_workspace(workspace_probs,
                  "TileLang GLM-5.2 FP8 MLA: workspace_probs",
                  torch::kBFloat16,
                  kCoreNum,
                  kHeadTile,
                  kKvTile);
  check_workspace(workspace_output,
                  "TileLang GLM-5.2 FP8 MLA: workspace_output",
                  torch::kFloat32,
                  kCoreNum,
                  kHeadTile,
                  kLatentDim);
  check_workspace(workspace_q,
                  "TileLang GLM-5.2 FP8 MLA: workspace_q",
                  torch::kBFloat16,
                  kCoreNum,
                  kHeadTile,
                  kLatentDim);
  check_workspace(workspace_q_rope,
                  "TileLang GLM-5.2 FP8 MLA: workspace_q_rope",
                  torch::kBFloat16,
                  kCoreNum,
                  kHeadTile,
                  kRopeDim);

  check_same_device(q_rope, q_latent, "TileLang GLM-5.2 FP8 MLA: q_rope");
  check_same_device(
      nope_cache, q_latent, "TileLang GLM-5.2 FP8 MLA: nope_cache");
  check_same_device(
      rope_cache, q_latent, "TileLang GLM-5.2 FP8 MLA: rope_cache");
  check_same_device(
      topk_indices, q_latent, "TileLang GLM-5.2 FP8 MLA: topk_indices");
  check_same_device(
      block_table, q_latent, "TileLang GLM-5.2 FP8 MLA: block_table");
  check_same_device(actual_seq_lengths_kv,
                    q_latent,
                    "TileLang GLM-5.2 FP8 MLA: actual_seq_lengths_kv");
  check_same_device(e4m3_decode_table,
                    q_latent,
                    "TileLang GLM-5.2 FP8 MLA: e4m3_decode_table");
  check_same_device(output, q_latent, "TileLang GLM-5.2 FP8 MLA: output");
  check_same_device(
      workspace_k, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_k");
  check_same_device(
      workspace_k_rope, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_k_rope");
  check_same_device(
      workspace_scores, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_scores");
  check_same_device(
      workspace_probs, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_probs");
  check_same_device(
      workspace_output, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_output");
  check_same_device(
      workspace_q, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_q");
  check_same_device(
      workspace_q_rope, q_latent, "TileLang GLM-5.2 FP8 MLA: workspace_q_rope");
}

}  // namespace

void glm52_fp8_sparse_mla_attention(const torch::Tensor& q_latent,
                                    const torch::Tensor& q_rope,
                                    const torch::Tensor& nope_cache,
                                    const torch::Tensor& rope_cache,
                                    const torch::Tensor& topk_indices,
                                    const torch::Tensor& block_table,
                                    const torch::Tensor& actual_seq_lengths_kv,
                                    const torch::Tensor& e4m3_decode_table,
                                    torch::Tensor& output,
                                    torch::Tensor& workspace_k,
                                    torch::Tensor& workspace_k_rope,
                                    torch::Tensor& workspace_scores,
                                    torch::Tensor& workspace_probs,
                                    torch::Tensor& workspace_output,
                                    torch::Tensor& workspace_q,
                                    torch::Tensor& workspace_q_rope,
                                    float softmax_scale) {
  check_supported(q_latent,
                  q_rope,
                  nope_cache,
                  rope_cache,
                  topk_indices,
                  block_table,
                  actual_seq_lengths_kv,
                  e4m3_decode_table,
                  output,
                  workspace_k,
                  workspace_k_rope,
                  workspace_scores,
                  workspace_probs,
                  workspace_output,
                  workspace_q,
                  workspace_q_rope);
  if (q_latent.size(0) == 0) {
    return;
  }

  CHECK_LE(q_latent.size(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: query count exceeds int32 range";
  CHECK_LE(block_table.size(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: block_table stride exceeds int32 range";
  CHECK_LE(q_latent.stride(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: q_latent token stride exceeds int32 range";
  CHECK_LE(q_latent.stride(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: q_latent head stride exceeds int32 range";
  CHECK_LE(q_rope.stride(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: q_rope token stride exceeds int32 range";
  CHECK_LE(q_rope.stride(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang GLM-5.2 FP8 MLA: q_rope head stride exceeds int32 range";

  const auto specialization =
      make_glm52_fp8_sparse_mla_attention_specialization(
          Glm52Fp8SparseMlaAttentionNumHeads{
              static_cast<int32_t>(q_latent.size(1))},
          Glm52Fp8SparseMlaAttentionDType{
              to_tilelang_dtype(q_latent.scalar_type())});
  const auto* entry =
      find_glm52_fp8_sparse_mla_attention_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang GLM-5.2 FP8 MLA: no compiled variant. Available variants: "
      << available_glm52_fp8_sparse_mla_attention_variant_keys();

  const int32_t device_id = q_latent.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(q_latent.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(q_rope.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(nope_cache.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(rope_cache.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(topk_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(block_table.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(actual_seq_lengths_kv.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(e4m3_decode_table.data_ptr())),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_k.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_k_rope.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_scores.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_probs.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_output.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_q.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace_q_rope.data_ptr()),
      static_cast<int32_t>(q_latent.stride(0)),
      static_cast<int32_t>(q_latent.stride(1)),
      static_cast<int32_t>(q_rope.stride(0)),
      static_cast<int32_t>(q_rope.stride(1)),
      static_cast<int32_t>(q_latent.size(0)),
      static_cast<int32_t>(block_table.size(1)),
      softmax_scale,
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
