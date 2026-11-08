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
#include <variant>
#include <vector>

#include "core/util/tensor_helper.h"

namespace xllm {

class OneRecModelInputParams {
 public:
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

  OneRecModelInputParams to(const torch::Device& device) const {
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

class OneRecXAttentionParams final : public OneRecModelInputParams {
 public:
  std::vector<torch::Tensor> unshared_k_caches;
  std::vector<torch::Tensor> unshared_v_caches;
  std::vector<torch::Tensor> shared_k_caches;
  std::vector<torch::Tensor> shared_v_caches;
  torch::Tensor beam_width_tensor;
  torch::Tensor current_round_tensor;
  torch::Tensor debug_selected_token_idxes;
  std::vector<int64_t> debug_selected_token_idxes_expected;

  OneRecXAttentionParams to(const torch::Device& device) const {
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
class LlmRecMultiRoundParams final {
 public:
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
  torch::Tensor two_stage_qo_indptr_expanded;
  torch::Tensor two_stage_paged_kv_indptr_expanded;
  torch::Tensor two_stage_paged_kv_indices_expanded;
  torch::Tensor two_stage_paged_kv_last_page_len_expanded;

  LlmRecMultiRoundParams to(const torch::Device& device) const {
    LlmRecMultiRoundParams result = *this;

    result.full_k_caches.clear();
    result.full_v_caches.clear();
    result.full_k_caches.reserve(full_k_caches.size());
    result.full_v_caches.reserve(full_v_caches.size());
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
    result.decode_positions_tensor_list.reserve(
        decode_positions_tensor_list.size());
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

}  // namespace xllm
