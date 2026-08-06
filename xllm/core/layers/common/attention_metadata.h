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

#pragma once

#include <torch/torch.h>

#include <vector>

#if defined(USE_CUDA) || defined(USE_MUSA)
#include <tvm/ffi/container/array.h>
namespace ffi = tvm::ffi;
#endif

#include <memory>
#include <optional>
#include <string>

#include "dsa_metadata.h"

namespace xllm::layer {

#if defined(USE_CUDA) || defined(USE_MUSA)
struct PlanInfo {
  int32_t layer_id = -1;
  ffi::Array<int64_t> plan_info;
  std::string uri;
};

// Cache for xattention two-stage decode.
struct XAttentionTwoStageDecodeCache {
  // Output tensors (shape fixed, values computed per layer)
  torch::Tensor shared_lse;    // [total_beam, num_heads, 1], float32
  torch::Tensor shared_o;      // [total_beam, num_heads, head_dim]
  torch::Tensor unshared_lse;  // [total_beam, num_heads, 1], float32
  torch::Tensor unshared_o;    // [total_beam, num_heads, head_dim]

  // Fixed tensors (values don't change for the same batch/shape)
  torch::Tensor q_cu_seq_lens_shared;             // [batch_size + 1], int32
  torch::Tensor qo_indptr_expanded;               // [total_beam + 1], int32
  torch::Tensor paged_kv_indptr_expanded;         // [total_beam + 1], int32
  torch::Tensor paged_kv_indices_expanded;        // [total_beam], int32
  torch::Tensor paged_kv_last_page_len_expanded;  // [total_beam], int32

  // Cached parameters for validation / reuse
  int32_t cached_batch_size = -1;
  int32_t cached_beam_size = -1;
  int32_t cached_num_heads = -1;
  int32_t cached_head_size = -1;
  int32_t cached_max_decode_step = -1;
  int32_t cached_step = -1;
};
#endif

// AttentionMetadata contains batch-level information shared across all
// attention layers. It is built once at the beginning of model forward pass and
// reused by all layers. This avoids redundant computation and memory allocation
// for metadata that is identical across layers (e.g., sequence lengths, paged
// KV cache indices, plan_info). Use
// AttentionMetadataBuilder to build instances from ModelInputParams.
struct AttentionMetadata {
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_seq_lens;
  std::vector<int32_t> kv_seq_lens_vec;
  std::vector<int32_t> q_seq_lens_vec;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
  int64_t max_query_len;
  int64_t max_seq_len;
  int64_t total_kv_len = 0;
  std::string compute_dtype;
  bool is_prefill;
  bool is_chunked_prefill;
  bool is_dummy;
  // Whether to apply causal mask. Default: true.
  bool is_causal = true;
#if defined(USE_DCU)
  bool use_dense_flash_attention = false;
#endif

  // Spec-verify ACL graph can run full attention as expanded decode while GDN
  // layers keep the original spec-verify metadata.
  bool use_expanded_decode_for_spec_verify_attention = false;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_block_table;
  torch::Tensor expanded_paged_attention_tiling_data;
  torch::Tensor expanded_kv_seq_lens_host;

  // for mrope
  torch::Tensor mrope_cos;
  torch::Tensor mrope_sin;

  // for flashinfer
  // Index pointer for paged KV cache, similar to row_splits in ragged tensor.
  // paged_kv_indptr[i] is the start index of sequence i in paged_kv_indices,
  // paged_kv_indptr[i+1] is the end index (exclusive). paged_kv_indptr[0] = 0.
  // Shape: [batch_size + 1]. Type: int32.
  torch::Tensor paged_kv_indptr;
  // Page indices (block IDs) of the paged KV cache for all sequences.
  // Contains all block/page IDs used by all sequences, flattened into a 1D
  // array. Shape: [total_num_blocks]. Type: int32.
  torch::Tensor paged_kv_indices;
  // Number of valid entries in the last page of each sequence in the paged KV
  // cache. Since pages are fixed-size (block_size), the last page may be
  // partially filled. Shape: [batch_size]. Type: int32.
  torch::Tensor paged_kv_last_page_len;
  // Query/Output index pointer tensor for decode mode with tensor core.
  // Similar to row_splits in ragged tensor: cumulative sum of sequence lengths.
  // qo_indptr[i] is the start index of sequence i in the packed query/output
  // tensor, qo_indptr[i+1] is the end index (exclusive). qo_indptr[0] = 0,
  // qo_indptr[batch_size] = total_tokens. Shape: [batch_size + 1]. Type: int32.
  // Used when use_tensor_core=true. If not defined (use .defined() to check),
  // will be created internally in batch_decode.
  std::optional<torch::Tensor> qo_indptr;
  // FlashInfer execution plan information for attention computation.
  // Contains kernel URI and plan tensor that specifies how to execute the
  // attention kernel. Only updated at layer 0 (shared across all layers). The
  // plan_info tensor contains pre-computed execution parameters optimized for
  // the current batch configuration.
#if defined(USE_CUDA) || defined(USE_MUSA)
  std::shared_ptr<PlanInfo> plan_info;
  // Separate plan info for the shared stage in xattention two-stage decode.
  std::shared_ptr<PlanInfo> shared_plan_info;
  // Separate plan info for the unshared stage in xattention two-stage decode.
  std::shared_ptr<PlanInfo> unshared_plan_info;

  // for xattention two-stage decode cache (layer 0 only)
  std::optional<XAttentionTwoStageDecodeCache>
      xattention_two_stage_decode_cache;
#endif
  // for CUDA graph - CPU tensors for plan_info update (avoid .to(CPU) during
  // graph capture) torch::Tensor q_cu_seq_lens_host;      // Prefill mode:
  // q_cu_seq_lens on CPU torch::Tensor kv_cu_seq_lens_host;    // Prefill mode:
  // kv_cu_seq_lens on CPU torch::Tensor paged_kv_indptr_host;    // Decode
  // mode: paged_kv_indptr on CPU torch::Tensor kv_seq_lens_host;        //
  // Decode mode (tensor_core) / NPU: kv_seq_lens on CPU for CUDA graph
  bool enable_cuda_graph = false;

  // for xattention
  torch::Tensor full_k_cache;
  torch::Tensor full_v_cache;
  torch::Tensor unshared_k_cache;
  torch::Tensor unshared_v_cache;
  torch::Tensor step_tensor;

  // for GDN attention
  torch::Tensor chunk_indices;
  torch::Tensor batch;
  torch::Tensor token_block_offset;
  torch::Tensor has_initial_states;
  int32_t tot = 0;

  // custom attention mask
  torch::Tensor attn_mask;

  // DeepSeek V4 sparse attention metadata (optional).
  // Built by DSAMetadataBuilder and shared across all layers.
  std::shared_ptr<DSAMetadata> dsa_metadata;

#if defined(USE_NPU)
  // for npu
  bool is_spec_verify = false;
  torch::Tensor q_seq_lens_host;
  torch::Tensor kv_seq_lens_host;
  // For ACL graph execution - fixed-address device tiling data for
  // CustomPagedAttention replay.
  torch::Tensor paged_attention_tiling_data;
  // Pre-computed attention mask for npu_fused_infer_attention.
  torch::Tensor fia_attn_mask;
  // Host vectors for npu_fused_infer_attention (kernel requires host memory).
  std::vector<int64_t> q_cu_seq_lens_host_vec;
  std::vector<int64_t> kv_cu_seq_lens_host_vec;
  // Non-cumulative per-sequence lengths for chunked_prefill mode.
  std::vector<int64_t> kv_seq_lens_host_vec;
#endif
};

}  // namespace xllm::layer
