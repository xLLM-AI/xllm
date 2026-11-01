/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <torch/types.h>

#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/framework/eplb/eplb_info.h"
#include "core/kv_cache/block/block.h"
#include "core/kv_cache/linear_state_types.h"
#include "core/platform/layer_synchronizer.h"
#if defined(USE_NPU)
#include "core/platform/npu/npu_layer_synchronizer.h"
#endif
#if defined(USE_MLU)
#include "core/platform/mlu/mlu_layer_synchronizer.h"
#endif
#if defined(USE_DCU)
#include "core/platform/dcu/dcu_layer_synchronizer.h"
#endif

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/parallel_state/npu_cp_plan.h"
#include "core/framework/parallel_state/npu_dp_ep_padding.h"
#include "core/util/hash_util.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// CPU page tables keyed by cache structure, independent of the consuming model.
using CacheBlockTables = std::map<BlockType, torch::Tensor>;

template <typename Embedding>
void materialize_linear_state_indices(Embedding& embedding,
                                      const torch::Device& device) {
  if (embedding.linear_state_ids.empty()) {
    return;
  }
  if (!embedding.linear_state_indices.defined()) {
    embedding.linear_state_indices =
        make_cpu_tensor(embedding.linear_state_ids).to(device);
  }
  if (!embedding.linear_state_read_indices.defined()) {
    const auto& read_ids = embedding.linear_state_read_ids.empty()
                               ? embedding.linear_state_ids
                               : embedding.linear_state_read_ids;
    CHECK_EQ(read_ids.size(), embedding.linear_state_ids.size());
    embedding.linear_state_read_indices =
        read_ids == embedding.linear_state_ids
            ? embedding.linear_state_indices
            : make_cpu_tensor(read_ids).to(device);
  }
}

class PythonAttentionMetadata;
namespace npu {
struct AclGraphTaskUpdateContext;
}  // namespace npu
namespace layer {
struct AttentionMetadata;
}  // namespace layer

struct LlmAttentionHostInput final {
  std::vector<int32_t> kpool_query_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;
  std::vector<int32_t> new_cache_slots;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> ring_cur_seqlen;
  std::vector<int32_t> ring_cache_seqlen;
  torch::Tensor block_tables;

  const int32_t* graph_q_seq_lens_data = nullptr;
  const int32_t* graph_kv_seq_lens_data = nullptr;
};

struct VlmAttentionHostInput final {
  std::vector<int32_t> kpool_query_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;
  std::vector<int32_t> new_cache_slots;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> ring_cur_seqlen;
  std::vector<int32_t> ring_cache_seqlen;
  torch::Tensor block_tables;

  const int32_t* graph_q_seq_lens_data = nullptr;
  const int32_t* graph_kv_seq_lens_data = nullptr;
};

struct RecAttentionHostInput final {
  std::vector<int32_t> kpool_query_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;
  std::vector<int32_t> new_cache_slots;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> ring_cur_seqlen;
  std::vector<int32_t> ring_cache_seqlen;
  torch::Tensor block_tables;

  const int32_t* graph_q_seq_lens_data = nullptr;
  const int32_t* graph_kv_seq_lens_data = nullptr;
};

namespace detail {
template <typename Input>
Input attention_device_to(const Input& source, const torch::Device& device) {
  Input out;
  out.q_seq_lens = safe_to(source.q_seq_lens, device, true);
  out.kv_seq_lens = safe_to(source.kv_seq_lens, device, true);
#if !defined(USE_CUDA) && !defined(USE_MUSA)
  out.q_cu_seq_lens = safe_to(source.q_cu_seq_lens, device, true);
#else
  out.q_cu_seq_lens = source.q_cu_seq_lens;
#endif
  out.new_cache_slots = safe_to(source.new_cache_slots, device, true);
  out.block_tables = safe_to(source.block_tables, device, true);
  out.paged_kv_indptr = safe_to(source.paged_kv_indptr, device);
  out.paged_kv_indices = safe_to(source.paged_kv_indices, device);
  out.paged_kv_last_page_len = safe_to(source.paged_kv_last_page_len, device);
  out.new_cache_slot_offsets = safe_to(source.new_cache_slot_offsets, device);
  out.kv_cache_start_offsets = safe_to(source.kv_cache_start_offsets, device);
  out.kv_cache_tokens_nums = safe_to(source.kv_cache_tokens_nums, device);
  out.history_compressed_kv = safe_to(source.history_compressed_kv, device);
  out.history_k_rope = safe_to(source.history_k_rope, device);
  out.ring_cur_seqlen = safe_to(source.ring_cur_seqlen, device);
  out.ring_cache_seqlen = safe_to(source.ring_cache_seqlen, device);
  out.in_prefix_slots = safe_to(source.in_prefix_slots, device, true);
  return out;
}
}  // namespace detail

class LlmAttentionDeviceInput final {
 public:
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor new_cache_slot_offsets;
  torch::Tensor kv_cache_start_offsets;
  torch::Tensor kv_cache_tokens_nums;
  torch::Tensor history_compressed_kv;
  torch::Tensor history_k_rope;
  torch::Tensor ring_cur_seqlen;
  torch::Tensor ring_cache_seqlen;

  // Per-rank prefix slot indices for KV-split prefix AllGather. NpuCpPlan
  // supplies this graph input with the rest of the CP attention metadata.
  torch::Tensor in_prefix_slots;

  LlmAttentionDeviceInput to(const torch::Device& device) const {
    return detail::attention_device_to(*this, device);
  }
};

class VlmAttentionDeviceInput final {
 public:
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor new_cache_slot_offsets;
  torch::Tensor kv_cache_start_offsets;
  torch::Tensor kv_cache_tokens_nums;
  torch::Tensor history_compressed_kv;
  torch::Tensor history_k_rope;
  torch::Tensor ring_cur_seqlen;
  torch::Tensor ring_cache_seqlen;

  // Per-rank prefix slot indices for KV-split prefix AllGather. NpuCpPlan
  // supplies this graph input with the rest of the CP attention metadata.
  torch::Tensor in_prefix_slots;

  VlmAttentionDeviceInput to(const torch::Device& device) const {
    return detail::attention_device_to(*this, device);
  }
};

class RecAttentionDeviceInput final {
 public:
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor new_cache_slot_offsets;
  torch::Tensor kv_cache_start_offsets;
  torch::Tensor kv_cache_tokens_nums;
  torch::Tensor history_compressed_kv;
  torch::Tensor history_k_rope;
  torch::Tensor ring_cur_seqlen;
  torch::Tensor ring_cache_seqlen;

  // Per-rank prefix slot indices for KV-split prefix AllGather. NpuCpPlan
  // supplies this graph input with the rest of the CP attention metadata.
  torch::Tensor in_prefix_slots;

  RecAttentionDeviceInput to(const torch::Device& device) const {
    return detail::attention_device_to(*this, device);
  }
};

enum class TransferType : uint8_t {
  G2H = 0,    // global memory(KVCache store) to host memory(DRAM)
  H2D = 1,    // host memory(DRAM) to device memory(HBM)
  D2G = 2,    // device memory(HBM) to global memory(KVCache store)
  G2D = 3,    // global memory(KVCache store) to device memory(HBM)
  D2H2G = 4,  // device memory(HBM) to host memory(DRAM) to global
              // memory(KVCache store)
};

class BlockTransferInfo final {
 public:
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  BlockType block_type = BlockType::KV;
  // The two-argument constructor is also used for local swap metadata where
  // no transport direction exists. Keep that state explicitly invalid so a
  // transport codec cannot read an uninitialized enum value.
  TransferType transfer_type = static_cast<TransferType>(UINT8_MAX);

  BlockTransferInfo(int32_t src_block_id, int32_t dst_block_id) {
    this->src_block_id = src_block_id;
    this->dst_block_id = dst_block_id;
  }

  BlockTransferInfo(int32_t src_id,
                    int32_t dst_id,
                    const uint8_t* key,
                    TransferType type,
                    BlockType btype = BlockType::KV)
      : src_block_id(src_id),
        dst_block_id(dst_id),
        block_type(btype),
        transfer_type(type) {
    memcpy(hash_key, key, XXH3_128BITS_HASH_VALUE_LEN);
  }

  BlockTransferInfo(const BlockTransferInfo& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        block_type(other.block_type),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
  }

  BlockTransferInfo(BlockTransferInfo&& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        block_type(other.block_type),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
  }

  BlockTransferInfo& operator=(const BlockTransferInfo& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    block_type = other.block_type;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
    return *this;
  }

  BlockTransferInfo& operator=(BlockTransferInfo&& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    block_type = other.block_type;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
    return *this;
  }

  std::string to_string() const {
    std::string rt = ", has_key:";
    for (int32_t i = 0; i < 16; ++i) {
      rt += std::to_string(static_cast<int64_t>(hash_key[i])) + " ";
    }
    return std::to_string(src_block_id) + "->" + std::to_string(dst_block_id) +
           ", " + std::to_string(static_cast<uint32_t>(transfer_type)) + rt;
  }
};

struct ExecutionBatchMetadata {
  int32_t num_reqs = 0;
  int64_t num_tokens = 0;
  std::vector<int32_t> num_scheduled_tokens;
  std::vector<int32_t> num_computed_tokens;
  std::vector<int32_t> query_start_loc;
  std::vector<uint8_t> is_prefilling;
};

inline void fill_request_layout(
    ExecutionBatchMetadata& batch,
    std::optional<std::span<const int32_t>> token_positions = std::nullopt) {
  auto& starts = batch.query_start_loc;
  const size_t num_reqs = batch.num_scheduled_tokens.size();
  if (token_positions) {
    batch.num_computed_tokens.resize(num_reqs);
    starts.resize(num_reqs + 1);
    starts[0] = 0;
    for (size_t row = 0; row < num_reqs; ++row) {
      const int32_t begin = starts[row];
      batch.num_computed_tokens[row] = (*token_positions)[begin];
      starts[row + 1] = begin + batch.num_scheduled_tokens[row];
    }
    return;
  }
  starts.reserve(num_reqs + 1);
  starts.assign(1, 0);
  std::partial_sum(batch.num_scheduled_tokens.begin(),
                   batch.num_scheduled_tokens.end(),
                   std::back_inserter(starts));
}

struct BatchInputMeta {
  BatchForwardType batch_forward_type;
  int32_t num_sequences = 0;
  int32_t actual_num_sequences = 0;
  int32_t kv_max_seq_len = 0;
  int32_t q_max_seq_len = 0;
  ExecutionBatchMetadata execution_batch;
  uint64_t batch_id = 0;
  bool is_graph_warmup = false;
  // Engine-selected MTP depth for this task; zero keeps configured maximum.
  // Transported with the input, never inferred independently on TP ranks.
  uint32_t mtp_draft_steps = 0;
  // Startup-only padded target shape for measuring the full MTP cost surface.
  uint32_t mtp_target_graph_rows = 0;
};

class LlmEmbeddingInput final {
 public:
  // input embedding
  mutable torch::Tensor input_embedding;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // linear state ids of each sequence
  std::vector<int32_t> linear_state_ids;

  // IntTensor: [n_seq]
  torch::Tensor linear_state_indices;

  std::vector<int32_t> linear_state_read_ids;
  torch::Tensor linear_state_read_indices;

  // request ids of each sequence, used by suffix decoding request identity
  std::vector<std::string> request_ids;

  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;

  // Precomputed shifted token ids for MTP prefill, aligned with tokens.
  torch::Tensor mtp_shifted_token_ids;

  // Pending PD handoff bootstrap rows for the first MTP decode step.
  std::vector<int32_t> mtp_bootstrap_row_idxes;
  torch::Tensor mtp_bootstrap_embeddings;

  LlmEmbeddingInput to(const torch::Device& device) const {
    LlmEmbeddingInput out;
    out.input_embedding = safe_to(input_embedding, device);
    out.embedding_ids = embedding_ids;
    out.linear_state_ids = linear_state_ids;
    out.linear_state_indices = safe_to(linear_state_indices, device, true);
    out.linear_state_read_ids = linear_state_read_ids;
    out.linear_state_read_indices =
        safe_to(linear_state_read_indices, device, true);
    out.request_ids = request_ids;
    out.extra_token_ids = extra_token_ids;
    out.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    out.mtp_bootstrap_row_idxes = mtp_bootstrap_row_idxes;
    out.mtp_bootstrap_embeddings =
        safe_to(mtp_bootstrap_embeddings, device, true);
    return out;
  }
};

class BlockCopyInput final {
 public:
  // swap
  std::vector<BlockTransferInfo> swap_blocks;

  // block copy kernel
  torch::Tensor src_block_indices;
  torch::Tensor dst_block_indices;
  torch::Tensor cum_sum;

  BlockCopyInput to(const torch::Device& device) const {
    BlockCopyInput out;
    out.swap_blocks = swap_blocks;
    out.src_block_indices = safe_to(src_block_indices, device, true);
    out.dst_block_indices = safe_to(dst_block_indices, device, true);
    out.cum_sum = safe_to(cum_sum, device, true);
    return out;
  }
};

class VlmVisionInput final {
 public:
  // multimodal
  mutable MMBatchData mm_data;

  // deep_stack for Qwen3-VL
  mutable std::vector<torch::Tensor> deep_stacks;

  VlmVisionInput to(const torch::Device& device) const {
    VlmVisionInput out;
    out.mm_data = MMBatchData::to(mm_data, device);
    out.deep_stacks = deep_stacks;
    return out;
  }
};

class ParallelInput final {
 public:
  // num tokens of all workers, mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  // Logical sequence counts before speculative/MTP rows are expanded. This
  // remains stable when one request contributes a repair row and token counts
  // are scaled for graph/collective execution.
  std::vector<int32_t> dp_global_sequence_nums;
  // Original DP token counts before empty ranks are padded to one fake token.
  // Attention/FFN paths may need the padded counts, while lm_head output
  // compaction must skip true empty DP ranks.
  std::vector<int32_t> raw_dp_global_token_nums;
  // Per-DP-shard generation derived from the local batch identity. Every shard
  // receives the full vector so speculative prelaunch reuse decisions remain
  // collective-order consistent when any shard changes its batch.
  std::vector<uint64_t> dp_global_batch_generations;
  // max kv seq len of all dp shards. Graph key generation uses this so empty
  // DP decode ranks pick the same graph as ranks with real decode tokens.
  std::vector<int32_t> dp_global_kv_max_seq_lens;
  // Replicated per-DP-shard JSON grammar presence for collective-safe MTP
  // prelaunch admission. Missing metadata disables the prelaunch.
  std::vector<int32_t> dp_global_json_object_active;
  std::vector<int32_t> dp_is_decode;
  // Request-ordered, n-2 DSpark feedback replicated by the engine. This is
  // Host planning metadata, not a Device input or an additional readback.
  std::vector<float> dp_dspark_confidence;
  std::vector<uint64_t> dp_dspark_rounds_since_full_width;

  // Persistent int8 mask for real MegaMoE rows in a captured graph bucket.
  torch::Tensor mega_active_mask;

  DpEpPaddingData dp_ep_padding_data;
  NpuCpPlan cp_plan;

#if defined(USE_MLU)
  std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer = nullptr;
#elif defined(USE_DCU)
  std::shared_ptr<DCULayerSynchronizerImpl> layer_synchronizer = nullptr;
#elif defined(USE_NPU)
  std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer = nullptr;
#endif
  uint32_t layers_per_event = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<LayerSynchronizer> layer_wise_load_synchronizer = nullptr;
  std::optional<uint32_t> draft_load_event_index;
#if defined(USE_NPU) || defined(USE_MUSA)
  std::vector<int64_t> query_start_loc;
#endif

  ParallelInput to(const torch::Device& device) const {
    ParallelInput out;
    out.dp_global_token_nums = dp_global_token_nums;
    out.dp_global_sequence_nums = dp_global_sequence_nums;
    out.raw_dp_global_token_nums = raw_dp_global_token_nums;
    out.dp_global_batch_generations = dp_global_batch_generations;
    out.dp_global_kv_max_seq_lens = dp_global_kv_max_seq_lens;
    out.dp_global_json_object_active = dp_global_json_object_active;
    out.dp_is_decode = dp_is_decode;
    out.dp_dspark_confidence = dp_dspark_confidence;
    out.dp_dspark_rounds_since_full_width = dp_dspark_rounds_since_full_width;
    out.mega_active_mask = mega_active_mask;
    out.dp_ep_padding_data = dp_ep_padding_data;
    out.cp_plan = cp_plan.to(device);
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    out.layer_synchronizer = layer_synchronizer;
#endif
    out.layers_per_event = layers_per_event;
    out.layer_wise_load_synchronizer = layer_wise_load_synchronizer;
    out.draft_load_event_index = draft_load_event_index;
#if defined(USE_NPU) || defined(USE_MUSA)
    out.query_start_loc = query_start_loc;
#endif
    return out;
  }
};

// EPLB mask growth is left to the caller: its factor is not always the
// token-count multiplier.
inline void scale_parallel_token_counts(ParallelInput& parallel,
                                        int32_t multiplier) {
  for (int32_t& token_num : parallel.dp_global_token_nums) {
    token_num *= multiplier;
  }
  for (int32_t& token_num : parallel.raw_dp_global_token_nums) {
    token_num *= multiplier;
  }
}

class ExpertInput final {
 public:
  torch::Tensor expert_load_data;
  torch::Tensor expert_array;
  EplbInfo eplb_info;
  torch::Tensor eplb_decode_token_mask;

  ExpertInput to(const torch::Device& device) const {
    ExpertInput out;
    out.expert_load_data = expert_load_data;
    out.expert_array = expert_array;
    out.eplb_info = eplb_info;
    out.eplb_decode_token_mask =
        safe_to(eplb_decode_token_mask, device, /*non_blocking=*/true);
    return out;
  }
};

// Borrowed fixed-capacity sources for in-graph adaptive DSpark packing. The
// control word stores the active request count in its low 32 bits and a
// startup/profiling KV-write suppression bit in its high 32 bits. No control
// value or compact-arena offset is captured as a Host kernel argument.
struct DSparkAdaptiveGraphInput {
  torch::Tensor arena;
  torch::Tensor control;
  torch::Tensor anchors;
  torch::Tensor base_positions;
  torch::Tensor base_kv_lengths;
  torch::Tensor base_block_table;
  torch::Tensor offsets;
  torch::Tensor owners;
  torch::Tensor columns;
  torch::Tensor linear_state_indices;
  torch::Tensor num_accepted_tokens;
  uint32_t block_size = 0;
  uint32_t draft_width = 7;
};

namespace detail {
inline DSparkAdaptiveGraphInput dspark_adaptive_graph_input_to(
    const DSparkAdaptiveGraphInput& input,
    const torch::Device& device) {
  DSparkAdaptiveGraphInput out = input;
  for (torch::Tensor* tensor : {&out.arena,
                                &out.control,
                                &out.anchors,
                                &out.base_positions,
                                &out.base_kv_lengths,
                                &out.base_block_table,
                                &out.offsets,
                                &out.owners,
                                &out.columns,
                                &out.linear_state_indices,
                                &out.num_accepted_tokens}) {
    if (tensor->defined()) {
      *tensor = safe_to(*tensor, device, /*non_blocking=*/true);
    }
  }
  return out;
}
}  // namespace detail

class LlmGraphInput final {
 public:
  torch::Tensor attn_mask;
  torch::Tensor tiling_data;
#if defined(USE_DCU)
  bool use_dense_flash_attention = false;
#endif
  bool use_expanded_decode_for_spec_verify_attention = false;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_block_tables;
  torch::Tensor expanded_paged_kv_indptr;
  torch::Tensor expanded_paged_kv_indices;
  torch::Tensor expanded_paged_kv_last_page_len;
  torch::Tensor expanded_tiling_data;
  std::vector<int32_t> expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext> acl_graph_task_update_context;
#endif
  torch::Tensor input_tokens_override;
  // Device token sources produced by a speculative proposer. A matching
  // backend may fuse them into graph-owned target-verify storage.
  std::vector<torch::Tensor> spec_verify_draft_token_sources;
  // All dynamic target-verify source tensors retain their backing addresses
  // across replay generations. When true, the ACL graph records those
  // addresses separately from its graph key/task signature and validates them
  // before each replay.
  bool spec_verify_source_addresses_stable = false;
  // All ready events for the current static causal-conv task signature have
  // already been recorded on the signal stream. Replay can skip cold-path
  // signaling on the final-draft-to-target critical path.
  bool spec_verify_static_graph_tasks_prepared = false;
  DSparkAdaptiveGraphInput dspark_adaptive;

  LlmGraphInput to(const torch::Device& device) const {
    LlmGraphInput out;
    out.attn_mask = safe_to(attn_mask, device, true);
    out.tiling_data = safe_to(tiling_data, device, true);
#if defined(USE_DCU)
    out.use_dense_flash_attention = use_dense_flash_attention;
#endif
    out.use_expanded_decode_for_spec_verify_attention =
        use_expanded_decode_for_spec_verify_attention;
    out.expanded_kv_seq_lens = safe_to(expanded_kv_seq_lens, device, true);
    out.expanded_block_tables = safe_to(expanded_block_tables, device, true);
    out.expanded_paged_kv_indptr =
        safe_to(expanded_paged_kv_indptr, device, true);
    out.expanded_paged_kv_indices =
        safe_to(expanded_paged_kv_indices, device, true);
    out.expanded_paged_kv_last_page_len =
        safe_to(expanded_paged_kv_last_page_len, device, true);
    out.expanded_tiling_data = safe_to(expanded_tiling_data, device, true);
    out.expanded_kv_seq_lens_vec = expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
    out.acl_graph_task_update_context = acl_graph_task_update_context;
#endif
    out.input_tokens_override =
        safe_to(input_tokens_override, device, /*non_blocking=*/true);
    out.spec_verify_draft_token_sources.reserve(
        spec_verify_draft_token_sources.size());
    for (const auto& token : spec_verify_draft_token_sources) {
      out.spec_verify_draft_token_sources.push_back(
          safe_to(token, device, /*non_blocking=*/true));
    }
    out.spec_verify_source_addresses_stable =
        spec_verify_source_addresses_stable;
    out.spec_verify_static_graph_tasks_prepared =
        spec_verify_static_graph_tasks_prepared;
    out.dspark_adaptive =
        detail::dspark_adaptive_graph_input_to(dspark_adaptive, device);
    return out;
  }
};

class VlmEmbeddingInput final {
 public:
  // input embedding
  mutable torch::Tensor input_embedding;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // linear state ids of each sequence
  std::vector<int32_t> linear_state_ids;

  // IntTensor: [n_seq]
  torch::Tensor linear_state_indices;

  std::vector<int32_t> linear_state_read_ids;
  torch::Tensor linear_state_read_indices;

  // request ids of each sequence, used by suffix decoding request identity
  std::vector<std::string> request_ids;

  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;

  // Precomputed shifted token ids for MTP prefill, aligned with tokens.
  torch::Tensor mtp_shifted_token_ids;

  // Pending PD handoff bootstrap rows for the first MTP decode step.
  std::vector<int32_t> mtp_bootstrap_row_idxes;
  torch::Tensor mtp_bootstrap_embeddings;

  VlmEmbeddingInput to(const torch::Device& device) const {
    VlmEmbeddingInput out;
    out.input_embedding = safe_to(input_embedding, device);
    out.embedding_ids = embedding_ids;
    out.linear_state_ids = linear_state_ids;
    out.linear_state_indices = safe_to(linear_state_indices, device, true);
    out.linear_state_read_ids = linear_state_read_ids;
    out.linear_state_read_indices =
        safe_to(linear_state_read_indices, device, true);
    out.request_ids = request_ids;
    out.extra_token_ids = extra_token_ids;
    out.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    out.mtp_bootstrap_row_idxes = mtp_bootstrap_row_idxes;
    out.mtp_bootstrap_embeddings =
        safe_to(mtp_bootstrap_embeddings, device, true);
    return out;
  }
};

class VlmGraphInput final {
 public:
  torch::Tensor attn_mask;
  torch::Tensor tiling_data;
#if defined(USE_DCU)
  bool use_dense_flash_attention = false;
#endif
  bool use_expanded_decode_for_spec_verify_attention = false;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_block_tables;
  torch::Tensor expanded_paged_kv_indptr;
  torch::Tensor expanded_paged_kv_indices;
  torch::Tensor expanded_paged_kv_last_page_len;
  torch::Tensor expanded_tiling_data;
  std::vector<int32_t> expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext> acl_graph_task_update_context;
#endif
  torch::Tensor input_tokens_override;
  // Device token sources produced by a speculative proposer. A matching
  // backend may fuse them into graph-owned target-verify storage.
  std::vector<torch::Tensor> spec_verify_draft_token_sources;
  // All dynamic target-verify source tensors retain their backing addresses
  // across replay generations. When true, the ACL graph records those
  // addresses separately from its graph key/task signature and validates them
  // before each replay.
  bool spec_verify_source_addresses_stable = false;
  // All ready events for the current static causal-conv task signature have
  // already been recorded on the signal stream. Replay can skip cold-path
  // signaling on the final-draft-to-target critical path.
  bool spec_verify_static_graph_tasks_prepared = false;
  DSparkAdaptiveGraphInput dspark_adaptive;

  VlmGraphInput to(const torch::Device& device) const {
    VlmGraphInput out;
    out.attn_mask = safe_to(attn_mask, device, true);
    out.tiling_data = safe_to(tiling_data, device, true);
#if defined(USE_DCU)
    out.use_dense_flash_attention = use_dense_flash_attention;
#endif
    out.use_expanded_decode_for_spec_verify_attention =
        use_expanded_decode_for_spec_verify_attention;
    out.expanded_kv_seq_lens = safe_to(expanded_kv_seq_lens, device, true);
    out.expanded_block_tables = safe_to(expanded_block_tables, device, true);
    out.expanded_paged_kv_indptr =
        safe_to(expanded_paged_kv_indptr, device, true);
    out.expanded_paged_kv_indices =
        safe_to(expanded_paged_kv_indices, device, true);
    out.expanded_paged_kv_last_page_len =
        safe_to(expanded_paged_kv_last_page_len, device, true);
    out.expanded_tiling_data = safe_to(expanded_tiling_data, device, true);
    out.expanded_kv_seq_lens_vec = expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
    out.acl_graph_task_update_context = acl_graph_task_update_context;
#endif
    out.input_tokens_override =
        safe_to(input_tokens_override, device, /*non_blocking=*/true);
    out.spec_verify_draft_token_sources.reserve(
        spec_verify_draft_token_sources.size());
    for (const auto& token : spec_verify_draft_token_sources) {
      out.spec_verify_draft_token_sources.push_back(
          safe_to(token, device, /*non_blocking=*/true));
    }
    out.spec_verify_source_addresses_stable =
        spec_verify_source_addresses_stable;
    out.spec_verify_static_graph_tasks_prepared =
        spec_verify_static_graph_tasks_prepared;
    out.dspark_adaptive =
        detail::dspark_adaptive_graph_input_to(dspark_adaptive, device);
    return out;
  }
};

}  // namespace xllm
