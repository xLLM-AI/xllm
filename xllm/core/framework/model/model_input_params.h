/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <memory>
#include <optional>
#include <string>
#include <variant>

#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#include "framework/batch/batch_forward_type.h"
#include "framework/request/mm_batch_data.h"
#include "npu_cp_ep_padding.h"
#include "npu_cp_prepare.h"
#include "npu_dp_ep_padding.h"
#include "runtime/dit_forward_params.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace layer {
struct AttentionMetadata;
}  // namespace layer

struct OneRecModelInputParams {
  enum class RecStage {
    PREFILL,
    DECODE,
  };

  RecStage rec_stage = RecStage::PREFILL;
  bool is_hybrid_mode = false;
  bool is_encoder_forward = false;
  bool has_encoder_output = false;
  std::vector<int32_t> encoder_seq_lens;
  torch::Tensor encoder_seq_lens_tensor;
  int32_t encoder_max_seq_len = 0;

  bool is_first_prefill = true;
  int32_t bs = 0;
  int32_t group_width = 0;
  int32_t seq_len = 0;
  std::vector<std::vector<int32_t>> generated_tokens;
  torch::Tensor encoder_sparse_embedding;
  torch::Tensor decoder_context_embedding;

  torch::Tensor cross_attn_kv_cu_seq_lens;
  torch::Tensor cross_attn_new_cache_slots;
  torch::Tensor cross_attn_block_tables;
  std::vector<int32_t> cross_attn_kv_cu_seq_lens_vec;

  torch::Tensor encoder_token_ids;
  torch::Tensor encoder_positions;

  OneRecModelInputParams to(const c10::Device& device) const {
    OneRecModelInputParams result = *this;

    if (encoder_seq_lens_tensor.defined()) {
      result.encoder_seq_lens_tensor = encoder_seq_lens_tensor.to(device);
    }
    if (encoder_sparse_embedding.defined()) {
      result.encoder_sparse_embedding = encoder_sparse_embedding.to(device);
    }
    if (decoder_context_embedding.defined()) {
      result.decoder_context_embedding = decoder_context_embedding.to(device);
    }
    if (cross_attn_kv_cu_seq_lens.defined()) {
      result.cross_attn_kv_cu_seq_lens = cross_attn_kv_cu_seq_lens.to(device);
    }
    if (cross_attn_new_cache_slots.defined()) {
      result.cross_attn_new_cache_slots = cross_attn_new_cache_slots.to(device);
    }
    if (cross_attn_block_tables.defined()) {
      result.cross_attn_block_tables = cross_attn_block_tables.to(device);
    }
    if (encoder_token_ids.defined()) {
      result.encoder_token_ids = encoder_token_ids.to(device);
    }
    if (encoder_positions.defined()) {
      result.encoder_positions = encoder_positions.to(device);
    }

    return result;
  }

  void print() const {
    LOG(INFO) << "OneRecModelInputParams:"
              << " rec_stage: "
              << (rec_stage == RecStage::PREFILL ? "PREFILL" : "DECODE")
              << " is_hybrid_mode: " << is_hybrid_mode
              << " is_encoder_forward: " << is_encoder_forward
              << " has_encoder_output: " << has_encoder_output
              << " encoder_max_seq_len: " << encoder_max_seq_len
              << " is_first_prefill: " << is_first_prefill << " bs: " << bs
              << " group_width: " << group_width << " seq_len: " << seq_len
              << " encoder_seq_lens size: " << encoder_seq_lens.size()
              << " cross_attn_kv_cu_seq_lens_vec size: "
              << cross_attn_kv_cu_seq_lens_vec.size()
              << " generated_tokens size: " << generated_tokens.size();
    if (encoder_seq_lens_tensor.defined()) {
      LOG(INFO) << " encoder_seq_lens_tensor shape: "
                << encoder_seq_lens_tensor.sizes();
    }
    if (encoder_sparse_embedding.defined()) {
      LOG(INFO) << " encoder_sparse_embedding shape: "
                << encoder_sparse_embedding.sizes();
    }
    if (decoder_context_embedding.defined()) {
      LOG(INFO) << " decoder_context_embedding shape: "
                << decoder_context_embedding.sizes();
    }
    if (cross_attn_kv_cu_seq_lens.defined()) {
      LOG(INFO) << " cross_attn_kv_cu_seq_lens shape: "
                << cross_attn_kv_cu_seq_lens.sizes();
    }
    if (cross_attn_new_cache_slots.defined()) {
      LOG(INFO) << " cross_attn_new_cache_slots shape: "
                << cross_attn_new_cache_slots.sizes();
    }
    if (cross_attn_block_tables.defined()) {
      LOG(INFO) << " cross_attn_block_tables shape: "
                << cross_attn_block_tables.sizes();
    }
    if (encoder_token_ids.defined()) {
      LOG(INFO) << " encoder_token_ids shape: " << encoder_token_ids.sizes();
    }
    if (encoder_positions.defined()) {
      LOG(INFO) << " encoder_positions shape: " << encoder_positions.sizes();
    }
  }
};

struct OneRecXAttentionParams : public OneRecModelInputParams {
  std::vector<torch::Tensor> unshared_k_caches;
  std::vector<torch::Tensor> unshared_v_caches;
  std::vector<torch::Tensor> shared_k_caches;
  std::vector<torch::Tensor> shared_v_caches;
  torch::Tensor beam_width_tensor;
  torch::Tensor current_round_tensor;
  torch::Tensor debug_selected_token_idxes;
  std::vector<int64_t> debug_selected_token_idxes_expected;

  OneRecXAttentionParams to(const c10::Device& device) const {
    OneRecXAttentionParams result = *this;
    static_cast<OneRecModelInputParams&>(result) =
        OneRecModelInputParams::to(device);
    result.unshared_k_caches.clear();
    result.unshared_v_caches.clear();
    result.shared_k_caches.clear();
    result.shared_v_caches.clear();
    result.unshared_k_caches.reserve(unshared_k_caches.size());
    result.unshared_v_caches.reserve(unshared_v_caches.size());
    result.shared_k_caches.reserve(shared_k_caches.size());
    result.shared_v_caches.reserve(shared_v_caches.size());
    for (const auto& t : unshared_k_caches) {
      result.unshared_k_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : unshared_v_caches) {
      result.unshared_v_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : shared_k_caches) {
      result.shared_k_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : shared_v_caches) {
      result.shared_v_caches.emplace_back(safe_to(t, device));
    }
    if (beam_width_tensor.defined()) {
      result.beam_width_tensor = safe_to(beam_width_tensor, device, true);
    }
    if (current_round_tensor.defined()) {
      result.current_round_tensor = safe_to(current_round_tensor, device, true);
    }
    if (debug_selected_token_idxes.defined()) {
      result.debug_selected_token_idxes =
          safe_to(debug_selected_token_idxes, device);
    }
    return result;
  }

  void print() const {
    LOG(INFO) << "OneRecXAttentionParams:";
    OneRecModelInputParams::print();
    LOG(INFO) << " unshared_k_caches size: " << unshared_k_caches.size()
              << " unshared_v_caches size: " << unshared_v_caches.size()
              << " shared_k_caches size: " << shared_k_caches.size()
              << " shared_v_caches size: " << shared_v_caches.size();
    if (beam_width_tensor.defined()) {
      LOG(INFO) << " beam_width_tensor shape: " << beam_width_tensor.sizes();
    }
    if (current_round_tensor.defined()) {
      LOG(INFO) << " current_round_tensor shape: "
                << current_round_tensor.sizes();
    }
  }
};

// Parameters for LLM Rec multi-round mode (device loop, beam search).
struct LlmRecMultiRoundParams {
  // full kv caches provided by engine for step-level decode, per layer
  std::vector<torch::Tensor> full_k_caches;
  std::vector<torch::Tensor> full_v_caches;
  std::vector<torch::Tensor> unshared_k_caches;
  std::vector<torch::Tensor> unshared_v_caches;
  std::vector<torch::Tensor> shared_k_caches;
  std::vector<torch::Tensor> shared_v_caches;
  std::vector<torch::Tensor> decode_positions_tensor_list;
  // beam width for step-level decode
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  torch::Tensor beam_width_tensor;
  // current round for step-level decode
  torch::Tensor current_round_tensor;
  int32_t total_round = 0;

  // xattention two-stage decode cache tensors prepared by RecWorker.
  torch::Tensor two_stage_shared_lse;
  torch::Tensor two_stage_shared_o;
  torch::Tensor two_stage_unshared_lse;
  torch::Tensor two_stage_unshared_o;
  torch::Tensor two_stage_q_cu_seq_lens_shared;
  torch::Tensor two_stage_shared_lse_kernel;
  torch::Tensor two_stage_decode_slot_mapping;
  torch::Tensor two_stage_unshared_seq_lens;
  torch::Tensor two_stage_qo_indptr_expanded;
  torch::Tensor two_stage_paged_kv_indptr_expanded;
  torch::Tensor two_stage_paged_kv_indices_expanded;
  torch::Tensor two_stage_paged_kv_last_page_len_expanded;

  LlmRecMultiRoundParams to(const torch::Device& device) const {
    LlmRecMultiRoundParams result = *this;

    result.full_k_caches.clear();
    result.full_v_caches.clear();
    for (const auto& t : full_k_caches) {
      result.full_k_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : full_v_caches) {
      result.full_v_caches.emplace_back(safe_to(t, device));
    }
    result.unshared_k_caches.clear();
    result.unshared_v_caches.clear();
    result.shared_k_caches.clear();
    result.shared_v_caches.clear();
    for (const auto& t : unshared_k_caches) {
      result.unshared_k_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : unshared_v_caches) {
      result.unshared_v_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : shared_k_caches) {
      result.shared_k_caches.emplace_back(safe_to(t, device));
    }
    for (const auto& t : shared_v_caches) {
      result.shared_v_caches.emplace_back(safe_to(t, device));
    }

    if (beam_width_tensor.defined()) {
      result.beam_width_tensor = safe_to(beam_width_tensor, device, true);
    }
    if (current_round_tensor.defined()) {
      result.current_round_tensor = safe_to(current_round_tensor, device, true);
    }

    if (two_stage_shared_lse.defined()) {
      result.two_stage_shared_lse = safe_to(two_stage_shared_lse, device);
    }
    if (two_stage_shared_o.defined()) {
      result.two_stage_shared_o = safe_to(two_stage_shared_o, device);
    }
    if (two_stage_unshared_lse.defined()) {
      result.two_stage_unshared_lse = safe_to(two_stage_unshared_lse, device);
    }
    if (two_stage_unshared_o.defined()) {
      result.two_stage_unshared_o = safe_to(two_stage_unshared_o, device);
    }
    if (two_stage_q_cu_seq_lens_shared.defined()) {
      result.two_stage_q_cu_seq_lens_shared =
          safe_to(two_stage_q_cu_seq_lens_shared, device);
    }
    if (two_stage_shared_lse_kernel.defined()) {
      result.two_stage_shared_lse_kernel =
          safe_to(two_stage_shared_lse_kernel, device);
    }
    if (two_stage_decode_slot_mapping.defined()) {
      result.two_stage_decode_slot_mapping =
          safe_to(two_stage_decode_slot_mapping, device);
    }
    if (two_stage_unshared_seq_lens.defined()) {
      result.two_stage_unshared_seq_lens =
          safe_to(two_stage_unshared_seq_lens, device);
    }
    if (two_stage_qo_indptr_expanded.defined()) {
      result.two_stage_qo_indptr_expanded =
          safe_to(two_stage_qo_indptr_expanded, device);
    }
    if (two_stage_paged_kv_indptr_expanded.defined()) {
      result.two_stage_paged_kv_indptr_expanded =
          safe_to(two_stage_paged_kv_indptr_expanded, device);
    }
    if (two_stage_paged_kv_indices_expanded.defined()) {
      result.two_stage_paged_kv_indices_expanded =
          safe_to(two_stage_paged_kv_indices_expanded, device);
    }
    if (two_stage_paged_kv_last_page_len_expanded.defined()) {
      result.two_stage_paged_kv_last_page_len_expanded =
          safe_to(two_stage_paged_kv_last_page_len_expanded, device);
    }

    result.decode_positions_tensor_list.clear();
    for (const auto& t : decode_positions_tensor_list) {
      result.decode_positions_tensor_list.emplace_back(safe_to(t, device));
    }

    return result;
  }
};

using RecModelInputParams = std::variant<std::monostate,
                                         OneRecModelInputParams,
                                         OneRecXAttentionParams,
                                         LlmRecMultiRoundParams>;

enum class TransferType : uint8_t {
  G2H = 0,  // global memory(KVCache store) to host memory(DRAM)
  H2D = 1,  // host memory(DRAM) to device memory(HBM)
  D2G = 2,  // host memory(DRAM) to global memory(KVCache store)
  G2D = 3   // global memory(KVCache store) to device memory(HBM)
};

struct BlockTransferInfo {
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN];
  TransferType transfer_type;

  BlockTransferInfo(int32_t src_block_id, int32_t dst_block_id) {
    this->src_block_id = src_block_id;
    this->dst_block_id = dst_block_id;
  }

  BlockTransferInfo(int32_t src_id,
                    int32_t dst_id,
                    const uint8_t* key,
                    TransferType type)
      : src_block_id(src_id), dst_block_id(dst_id), transfer_type(type) {
    memcpy(hash_key, key, XXH3_128BITS_HASH_VALUE_LEN);
  }

  BlockTransferInfo(const BlockTransferInfo& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
  }

  BlockTransferInfo(BlockTransferInfo&& other)
      : src_block_id(other.src_block_id),
        dst_block_id(other.dst_block_id),
        transfer_type(other.transfer_type) {
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
  }

  BlockTransferInfo& operator=(const BlockTransferInfo& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
    return *this;
  }

  BlockTransferInfo& operator=(BlockTransferInfo&& other) {
    src_block_id = other.src_block_id;
    dst_block_id = other.dst_block_id;
    transfer_type = other.transfer_type;
    memcpy(hash_key, other.hash_key, XXH3_128BITS_HASH_VALUE_LEN);

    other.src_block_id = -1;
    other.dst_block_id = -1;
    return *this;
  }

  std::string to_string() const {
    std::string rt = ", has_key:";
    for (int i = 0; i < 16; i++) {
      rt += std::to_string(int64_t(hash_key[i])) + " ";
    }
    return std::to_string(src_block_id) + "->" + std::to_string(dst_block_id) +
           ", " + std::to_string(uint32_t(transfer_type)) + rt;
  }
};

struct ModelInputParams {
  ModelInputParams to(const torch::Device& device) const {
    ModelInputParams params;
    params.batch_forward_type = batch_forward_type;
    params.num_sequences = num_sequences;
    params.kv_max_seq_len = kv_max_seq_len;
    params.q_max_seq_len = q_max_seq_len;
    params.enable_mla = enable_mla;

    params.kv_seq_lens = safe_to(kv_seq_lens, device, true);
    params.q_seq_lens = safe_to(q_seq_lens, device, true);
    params.q_cu_seq_lens = safe_to(q_cu_seq_lens, device, true);

    params.new_cache_slots = safe_to(new_cache_slots, device, true);
    params.block_tables = safe_to(block_tables, device, true);
    params.kv_seq_lens_vec = kv_seq_lens_vec;
    params.q_seq_lens_vec = q_seq_lens_vec;

    params.input_embedding = safe_to(input_embedding, device);

    params.deep_stacks = deep_stacks;
    params.visual_pos_masks = visual_pos_masks;
    params.mm_data = MMBatchData::to(mm_data, device);
    params.dp_global_token_nums = dp_global_token_nums;
    params.dp_is_decode = dp_is_decode;
    params.embedding_ids = std::move(embedding_ids);
    params.request_ids = std::move(request_ids);
    params.extra_token_ids = std::move(extra_token_ids);
    params.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    params.dp_ep_padding_data = dp_ep_padding_data;
    params.cp_ep_padding_data
        .attn_padding_idx(
            safe_to(cp_ep_padding_data.attn_padding_idx(), device, true))
        .attn_unpadding_idx(
            safe_to(cp_ep_padding_data.attn_unpadding_idx(), device, true))
        .ffn_padding_idx(
            safe_to(cp_ep_padding_data.ffn_padding_idx(), device, true))
        .ffn_unpadding_idx(
            safe_to(cp_ep_padding_data.ffn_unpadding_idx(), device, true))
        .lm_head_skip_padding_token_indices(
            safe_to(cp_ep_padding_data.lm_head_skip_padding_token_indices(),
                    device,
                    true))
        .gather_prenorm_idx(
            safe_to(cp_ep_padding_data.gather_prenorm_idx(), device, true))
        .padding_idx(safe_to(cp_ep_padding_data.padding_idx(), device, true))
        .un_padding_idx(
            safe_to(cp_ep_padding_data.un_padding_idx(), device, true))
        .dynamic_ep_idx(
            safe_to(cp_ep_padding_data.dynamic_ep_idx(), device, true))
        .moe_idx(safe_to(cp_ep_padding_data.moe_idx(), device, true))
        .expert_array(safe_to(cp_ep_padding_data.expert_array(), device, true));

    params.kv_cache_tokens_nums_host = std::move(kv_cache_tokens_nums_host);
    params.kv_cache_tokens_nums = safe_to(kv_cache_tokens_nums, device);
    params.history_compressed_kv = safe_to(history_compressed_kv, device);
    params.history_k_rope = safe_to(history_k_rope, device);
    params.ring_cur_seqlen = safe_to(ring_cur_seqlen, device);
    params.ring_cur_seqlen_host = ring_cur_seqlen_host;
    params.ring_cache_seqlen = safe_to(ring_cache_seqlen, device);
    params.ring_cache_seqlen_host = ring_cache_seqlen_host;
#if defined(USE_NPU)
    params.layer_synchronizer = layer_synchronizer;
#endif
    params.expert_load_data = expert_load_data;
    params.expert_array = expert_array;

    params.swap_blocks = std::move(swap_blocks);

    params.src_block_indices = safe_to(src_block_indices, device, true);
    params.dst_block_indices = safe_to(dst_block_indices, device, true);
    params.cum_sum = safe_to(cum_sum, device, true);

    // params for continuous kvcache
    params.new_cache_slot_offsets = safe_to(new_cache_slot_offsets, device);
    params.kv_cache_start_offsets = safe_to(kv_cache_start_offsets, device);

    // Copy graph_buffer to device
    // params.graph_buffer = safe_to(graph_buffer, device, true);
    params.graph_buffer.attn_mask =
        safe_to(graph_buffer.attn_mask, device, true);
    params.graph_buffer.tiling_data =
        safe_to(graph_buffer.tiling_data, device, true);

    // params for flashinfer
    params.paged_kv_indptr = safe_to(paged_kv_indptr, device);
    params.paged_kv_indices = safe_to(paged_kv_indices, device);
    params.paged_kv_last_page_len = safe_to(paged_kv_last_page_len, device);

    params.batch_id = batch_id;

    params.dit_forward_input = dit_forward_input.to(device);

    // rec_params device conversion for both OneRec and LLM-Rec variants
    if (const auto* onerec_xattn = onerec_xattention_params()) {
      params.rec_params = onerec_xattn->to(device);
    } else if (const auto* onerec = onerec_params()) {
      params.rec_params = onerec->to(device);
    } else if (const auto* llmrec = llmrec_params()) {
      params.rec_params = llmrec->to(device);
    }

    params.cp_prefill_inputs = cp_prefill_inputs.to(device);

    return params;
  }

  void print() const {
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << batch_forward_type.to_string() << " , num_sequences is "
              << num_sequences << " , kv_max_seq_len is " << kv_max_seq_len
              << " , q_max_seq_len is " << q_max_seq_len;
    LOG(INFO) << "ModelInputParams: kv_seq_lens_vec is " << kv_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: q_seq_lens_vec is " << q_seq_lens_vec;
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << batch_forward_type.to_string();
    print_tensor(kv_seq_lens, "ModelInputParams: kv_seq_lens", 4);
    print_tensor(q_seq_lens, "ModelInputParams: q_seq_lens", 4);
    print_tensor(q_cu_seq_lens, "ModelInputParams: q_cu_seq_lens", 4);
    print_tensor(new_cache_slots, "ModelInputParams: new_cache_slots", 4);
    print_tensor(block_tables, "ModelInputParams: block_tables", 4);
    LOG(INFO) << "ModelInputParams: dp_global_token_nums is "
              << dp_global_token_nums << ", dp_is_decode: " << dp_is_decode;

    if (const auto* onerec_xattn = onerec_xattention_params()) {
      LOG(INFO) << "ModelInputParams: has onerec_xattention_params";
      onerec_xattn->print();
    } else if (const auto* onerec = onerec_params()) {
      LOG(INFO) << "ModelInputParams: has onerec_params";
      onerec->print();
    } else if (const auto* llmrec = llmrec_params()) {
      LOG(INFO) << "ModelInputParams: has llm_rec_multi_round_params"
                << ", beam_width=" << llmrec->beam_width
                << ", total_round=" << llmrec->total_round;
    }
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
#if defined(USE_NPU)
    CHECK(seq_idx < q_seq_lens_vec.size()) << "seq_idx out of range";
    return q_seq_lens_vec[seq_idx];
#else
    CHECK(seq_idx < q_seq_lens_vec.size() - 1) << "seq_idx out of range";
    return q_seq_lens_vec[seq_idx + 1] - q_seq_lens_vec[seq_idx];
#endif
  }

  bool synchronize_layer(uint32_t layer_idx) const {
#if defined(USE_NPU)
    if (layer_wise_load_synchronizer != nullptr &&
        layer_idx % layers_per_bacth_copy == 0) {
      if (!layer_wise_load_synchronizer->synchronize_layer(
              layer_idx / layers_per_bacth_copy)) {
        return false;
      }
    }
#endif
    return true;
  }

  // Compatibility fallback for metadata builders that are not wired with
  // ModelArgs yet.
  bool enable_mla = false;

  BatchForwardType batch_forward_type;

  // total number of sequences in the batch
  int32_t num_sequences = 0;

  // max length for qkv.
  int32_t kv_max_seq_len = 0;
  int32_t q_max_seq_len = 0;

  uint64_t batch_id = 0;

  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  std::vector<int> kv_seq_lens_vec;
  std::vector<int> q_seq_lens_vec;

  // IntTensor: [n_tokens]
  torch::Tensor new_cache_slots;

  // IntTensor: [n_seq, max_n_blocks]
  torch::Tensor block_tables;

  // the indptr of the paged kv-cache
  // used in flashinfer
  // IntTensor: [n_seq + 1]
  torch::Tensor paged_kv_indptr;

  // the page indices of the paged kv cache
  // used in flashinfer
  torch::Tensor paged_kv_indices;

  // the number of entries in the last page of each request in
  // the paged kv cache
  // used in flashinfer
  // IntTensor: [n_seq]
  torch::Tensor paged_kv_last_page_len;

  // new slot offsets for continuous kvcache
  // used to store kv-cache to right position
  // IntTensor: [n_tokens]
  torch::Tensor new_cache_slot_offsets;

  // kvcache offset of sequence in the xtensor for all layers
  // IntTensor: [n_seq]
  torch::Tensor kv_cache_start_offsets;

  // input embedding
  mutable torch::Tensor input_embedding;

  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  std::vector<int32_t> dp_is_decode;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // request ids of each sequence, used by suffix decoding request identity
  std::vector<std::string> request_ids;

  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;
  // precomputed shifted token ids for mtp prefill.
  torch::Tensor mtp_shifted_token_ids;

  // swap
  std::vector<BlockTransferInfo> swap_blocks;

  // block copy kernel
  torch::Tensor src_block_indices;
  torch::Tensor dst_block_indices;
  torch::Tensor cum_sum;

  // multimodal
  mutable MMBatchData mm_data;

  // deep_stack for Qwen3-VL
  mutable std::vector<torch::Tensor> deep_stacks;
  // visual pos mask for Qwen3-VL
  mutable torch::Tensor visual_pos_masks;

#if defined(USE_NPU)
  std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer = nullptr;
  uint32_t layers_per_bacth_copy = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<NPULayerSynchronizerImpl> layer_wise_load_synchronizer =
      nullptr;
#endif

  DpEpPaddingData dp_ep_padding_data;
  CpEpPaddingData cp_ep_padding_data;
  CpPrefillInputs cp_prefill_inputs;

  torch::Tensor expert_load_data;
  torch::Tensor expert_array;

  torch::Tensor kv_cache_tokens_nums;
  std::vector<int32_t> kv_cache_tokens_nums_host;
  torch::Tensor history_compressed_kv;
  torch::Tensor history_k_rope;
  torch::Tensor ring_cur_seqlen;
  std::vector<int32_t> ring_cur_seqlen_host;
  torch::Tensor ring_cache_seqlen;
  std::vector<int32_t> ring_cache_seqlen_host;

  RecModelInputParams rec_params;

  // dit input data
  DiTForwardInput dit_forward_input;

  const OneRecModelInputParams* onerec_params() const {
    if (const auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return params;
    }
    if (const auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return static_cast<const OneRecModelInputParams*>(params);
    }
    return nullptr;
  }

  bool has_onerec_params() const { return onerec_params() != nullptr; }

  OneRecModelInputParams& mutable_onerec_params() {
    if (auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return *params;
    }
    if (auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return static_cast<OneRecModelInputParams&>(*params);
    }
    if (!has_onerec_params()) {
      rec_params.emplace<OneRecModelInputParams>();
    }
    return std::get<OneRecModelInputParams>(rec_params);
  }

  const OneRecXAttentionParams* onerec_xattention_params() const {
    return std::get_if<OneRecXAttentionParams>(&rec_params);
  }

  bool has_onerec_xattention_params() const {
    return onerec_xattention_params() != nullptr;
  }

  OneRecXAttentionParams& mutable_onerec_xattention_params() {
    if (!has_onerec_xattention_params()) {
      rec_params.emplace<OneRecXAttentionParams>();
    }
    return std::get<OneRecXAttentionParams>(rec_params);
  }

  // Accessors for LLM Rec multi-round params inside rec_params variant
  const LlmRecMultiRoundParams* llmrec_params() const {
    return std::get_if<LlmRecMultiRoundParams>(&rec_params);
  }

  bool has_llmrec_params() const { return llmrec_params() != nullptr; }

  LlmRecMultiRoundParams& mutable_llmrec_params() {
    if (!has_llmrec_params()) {
      rec_params.emplace<LlmRecMultiRoundParams>();
    }
    return std::get<LlmRecMultiRoundParams>(rec_params);
  }

  struct GraphBuffer {
    torch::Tensor attn_mask;
    torch::Tensor tiling_data;
  };
  GraphBuffer graph_buffer;

  // Optional attention metadata, built by executor
  // Using shared_ptr with forward declaration to avoid circular dependency
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;

  // Flag for CUDA graph capture mode
  bool enable_cuda_graph = false;
};

}  // namespace xllm
