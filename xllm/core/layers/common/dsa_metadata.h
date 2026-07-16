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

#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// DSA cache type enum for DeepSeek V4 multi-cache management
enum class DSACacheType : int32_t {
  TOKEN = 0,           // block allocated by token count / ratio
  SEQUENCE = 1,        // one block per sequence
  SLIDING_WINDOW = 2,  // sliding window, fixed number of blocks per seq
};

// Per-cache metadata within a layer
struct DSACacheInfo {
  int32_t group_id;    // which block manager group this cache belongs to
  DSACacheType type;   // cache type
  int32_t ratio;       // compression ratio
  int32_t block_size;  // block size for this cache
};

// Group-level info
struct DSAGroupInfo {
  DSACacheType type;
  int32_t ratio;
  int32_t block_size;
};

namespace layer {

struct DSACompressedAttentionMetadata {
  torch::Tensor context_lens;
  torch::Tensor block_table_for_attn;
  int64_t max_context_len = 0;
};

// Context-parallel (DSA-CP) per-rank metadata. Aligned with vllm-ascend
// DSACPMetadata (vllm_ascend/attention/context_parallel/dsa_cp.py). Produced
// by DSAMetadataBuilder::build_cp_local_metadata when cp_size > 1.
//
// The query token stream is split evenly across the CP group. Each field
// below describes the CURRENT rank's local token slice:
//   - local_query_start_loc: (num_reqs+1,) rebased cumulative query starts
//     for the local slice, always starting at 0.
//   - local_seq_lens:        (num_reqs,)   per-request KV length visible to
//     the local slice (0 for requests with no local query tokens).
//   - local_start / local_end: [start, end) token interval owned by this rank
//     in the padded global token stream.
//   - tokens_per_rank:       num_tokens_pad / cp_size.
//   - num_tokens_pad:        num_input_tokens padded to a multiple of
//     cp_size.
//   - local_cos / local_sin: RoPE tables sliced to [local_start, local_end).
//     Undefined when no full RoPE table is supplied to the builder.
struct DSACPMetadata {
  torch::Tensor local_query_start_loc;
  torch::Tensor local_seq_lens;
  int64_t local_start = 0;
  int64_t local_end = 0;
  int64_t tokens_per_rank = 0;
  int64_t num_tokens_pad = 0;
  torch::Tensor local_cos;
  torch::Tensor local_sin;
};

// DSAMetadata contains DeepSeek V4 sparse attention specific metadata,
// aligned with Python DSAMetadata(AttentionMetadata) class.
// It is built once at the beginning of model forward pass and reused by
// all layers. Use DSAMetadataBuilder to build instances from ModelInputParams.
struct DSAMetadata {
  // ===== Fields from Python AttentionMetadata base class =====
  // seq_lens: kv sequence lengths (context_length)
  torch::Tensor seq_lens;
  // seq_lens_q: query sequence lengths
  torch::Tensor seq_lens_q;
  // attn_mask: attention mask
  torch::Tensor attn_mask;
  // cos_table / sin_table: base RoPE cos/sin tables
  torch::Tensor cos_table;
  torch::Tensor sin_table;
  // inverse_sin_table: precomputed -sin_table for inverse RoPE.
  torch::Tensor inverse_sin_table;

  // ===== DSA-specific fields =====
  // layer_id: current layer (Python per-layer, C++ shared across layers)
  int32_t layer_id = 0;
  // num_speculative_tokens: number of speculative decoding tokens
  int32_t num_speculative_tokens = 0;
  // True when the metadata is consumed by ACL graph forward. Debug paths must
  // not perform host/device copies in this mode.
  bool is_acl_graph = false;

  // cp_input_dict: context-parallel inputs placeholder (reserved, optional).
  // Kept for backward compatibility with any callers that still read it;
  // structured CP metadata lives in cp_metadata below.
  std::unordered_map<std::string, torch::Tensor> cp_input_dict;

  // ===== DSA-CP (context-parallel) fields =====
  // cp_enabled: true when cp_size > 1 and DSA-CP is active for this forward.
  bool cp_enabled = false;
  int32_t cp_size = 1;
  int32_t cp_rank = 0;
  // Per-rank local token metadata; only meaningful when cp_enabled == true.
  DSACPMetadata cp_metadata;

  // NPU-only DSA metadata fields.
  // RoPE caches selected for the current layer's q/kv/output RoPE.
  torch::Tensor cos;
  torch::Tensor sin;
  // RoPE caches for compressor/indexer paths, indexed by compressed positions.
  torch::Tensor c4_cos;
  torch::Tensor c4_sin;
  torch::Tensor c128_cos;
  torch::Tensor c128_sin;
  // Main q/kv RoPE tensors for compressed layers at input-token length.
  torch::Tensor c4_input_cos;
  torch::Tensor c4_input_sin;
  torch::Tensor c128_input_cos;
  torch::Tensor c128_input_sin;
  torch::Tensor start_pos;

  // Multi-manager block tables and slot mappings
  // Indexed as [layer_id][cache_idx] after expansion by build_forward_context.
  // Same-group caches share the same underlying tensor (no copy).
  std::vector<std::vector<torch::Tensor>> block_tables;
  std::vector<std::vector<torch::Tensor>> slot_mappings;

  // Host-side max lengths cached alongside the tensors so graph code can
  // avoid scalar reads from device tensors.
  int64_t max_query_len = 0;
  int64_t max_seq_len = 0;

  // Sequence length metadata
  // actual_seq_lengths_kv: (batch_size,) — per-seq kv context length
  torch::Tensor actual_seq_lengths_kv;
  // actual_seq_lengths_query: (batch_size+1,) — cumsum of per-seq query lengths
  //   prefill: pad(cumsum(context_length), (1,0), 0)
  //   decode:  pad(cumsum(ones(batch_size)), (1,0), 0)
  torch::Tensor actual_seq_lengths_query;
  // kv_cu_seq_lens: (batch_size+1,) — pad(cumsum(actual_seq_lengths_kv),
  // (1,0)). Built once per forward and reused by all layers so the per-layer
  // indexer metadata builder does not recompute a host-side cumsum on every DSA
  // layer.
  torch::Tensor kv_cu_seq_lens;
  // max_seqlen_kv / max_seqlen_q: max sequence lengths
  torch::Tensor max_seqlen_kv;
  torch::Tensor max_seqlen_q;

  // Compressed positions
  // input_positions: (total_tokens,) — token position IDs
  torch::Tensor input_positions;
  // c4_pad_positions: positions for C4 compressed RoPE
  torch::Tensor c4_pad_positions;
  // c128_pad_positions: positions for C128 compressed RoPE
  torch::Tensor c128_pad_positions;

  // Precomputed sparse/indexer metadata tensors (Python forward aligned).
  // Built once per model forward before layer iteration.
  torch::Tensor c1_metadata;
  torch::Tensor c4_metadata;
  torch::Tensor c128_metadata;
  torch::Tensor qli_metadata;

  // hadamard: Hadamard transform matrix
  torch::Tensor hadamard;

  // Owns the device storage for non-graph DSA metadata tensors packed into a
  // single host-to-device transfer. Individual metadata tensors may be views
  // into this buffer.
  torch::Tensor packed_metadata_buffer;

  std::unordered_map<int64_t, torch::Tensor> cmp_slots_dict;

  // Host-side batch metadata for MLU small operators.
  // query_start_offsets: [0, cumsum(seq_lens_q)].
  std::vector<int64_t> query_start_offsets;
  // start_pos_vec[i] = actual_seq_lengths_kv[i] - seq_lens_q[i].
  std::vector<int64_t> start_pos_vec;
  // SWA window plan derived from start_pos_vec and model window_size.
  // swa_start_pos_vec[i] is the absolute first token retained for sequence i.
  std::vector<int64_t> swa_start_pos_vec;
  // swa_history_lens: per-sequence persisted SWA history to read before q.
  torch::Tensor swa_history_lens;
  // swa_context_lens: per-query-token local context length from swa_start.
  torch::Tensor swa_context_lens;
  int64_t swa_max_history_len = 0;
  int64_t swa_max_context_len = 0;

  // MLU-only canonical query sequence lengths consumed by DSA operators.
  // q_cu_seq_lens: (batch_size+1,) with leading 0.
  // (kv_cu_seq_lens is declared above in the general sequence-length section.)
  torch::Tensor q_cu_seq_lens;
  // kv_seq_lens / q_seq_lens: (batch_size,) per-sequence lengths.
  torch::Tensor kv_seq_lens;
  torch::Tensor q_seq_lens;
  // index_c4_seq_lens: (batch_size,) compressed kv lengths for indexer.
  torch::Tensor index_c4_seq_lens;
  int64_t index_total_c4_len = 0;
  int64_t index_max_c4_len = 0;

  // MLU-only ratio=128 compressed attention final decode inputs.
  // Built once per model forward before layer iteration.
  DSACompressedAttentionMetadata c128_attn_metadata;

  // MLU-only RoPE cos/sin tables (shared across all layers, set per forward).
  torch::Tensor compressed_cos_table;
  torch::Tensor compressed_sin_table;
  // compressed_inverse_sin_table: precomputed -compressed_sin_table.
  torch::Tensor compressed_inverse_sin_table;

  // Cache spec per layer
  // caches_info[layer_id][cache_idx] = {group_id, type, ratio, block_size}
  // Points to model-owned data; valid for the lifetime of the model.
  const std::vector<std::vector<DSACacheInfo>>* caches_info = nullptr;
};

}  // namespace layer
}  // namespace xllm
