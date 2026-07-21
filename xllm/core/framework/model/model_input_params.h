/* Copyright 2025-2026 The xLLM Authors.
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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "common/types.h"
#include "framework/block/block.h"
#include "platform/layer_synchronizer.h"
#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#if defined(USE_MLU)
#include "platform/mlu/mlu_layer_synchronizer.h"
#endif
#if defined(USE_DCU)
#include "platform/dcu/dcu_layer_synchronizer.h"
#endif

#include "core/framework/multimodal/mm_batch_data.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/parallel_state/npu_cp_ep_padding.h"
#include "framework/parallel_state/npu_cp_prepare.h"
#include "framework/parallel_state/npu_dp_ep_padding.h"
#include "runtime/dit_forward_params.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace npu {
struct AclGraphTaskUpdateContext;
}  // namespace npu
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

struct AttentionHostInput {
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;
  std::vector<int32_t> new_cache_slots;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> ring_cur_seqlen;
  std::vector<int32_t> ring_cache_seqlen;
  torch::Tensor block_tables;
};

struct AttentionDeviceInput {
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

  // Per-rank prefix slot index for KV-split prefix AllGather (see
  // WorkerImpl::compute_in_prefix_slots). Must propagate in to(device) because
  // nested worker paths skip recomputation when cp_partitioned is true.
  torch::Tensor in_prefix_slots;

  AttentionDeviceInput to(const torch::Device& device) const {
    AttentionDeviceInput out;
    out.q_seq_lens = safe_to(q_seq_lens, device, true);
    out.kv_seq_lens = safe_to(kv_seq_lens, device, true);
#if !defined(USE_CUDA)
    out.q_cu_seq_lens = safe_to(q_cu_seq_lens, device, true);
#else
    out.q_cu_seq_lens = q_cu_seq_lens;
#endif
    out.new_cache_slots = safe_to(new_cache_slots, device, true);
    out.block_tables = safe_to(block_tables, device, true);
    out.paged_kv_indptr = safe_to(paged_kv_indptr, device);
    out.paged_kv_indices = safe_to(paged_kv_indices, device);
    out.paged_kv_last_page_len = safe_to(paged_kv_last_page_len, device);
    out.new_cache_slot_offsets = safe_to(new_cache_slot_offsets, device);
    out.kv_cache_start_offsets = safe_to(kv_cache_start_offsets, device);
    out.kv_cache_tokens_nums = safe_to(kv_cache_tokens_nums, device);
    out.history_compressed_kv = safe_to(history_compressed_kv, device);
    out.history_k_rope = safe_to(history_k_rope, device);
    out.ring_cur_seqlen = safe_to(ring_cur_seqlen, device);
    out.ring_cache_seqlen = safe_to(ring_cache_seqlen, device);
    out.in_prefix_slots = safe_to(in_prefix_slots, device, true);
    return out;
  }
};

struct AttentionInput {
  AttentionHostInput host;
  AttentionDeviceInput device;
  torch::Tensor attention_host_buffer;
  torch::Tensor attention_device_buffer;
  uint64_t attention_buffer_bytes = 0;
  uint64_t attention_buffer_capacity = 0;
  std::shared_ptr<int> attention_buffer_owner = std::make_shared<int>(0);

  AttentionInput to(const torch::Device& target_device) const {
    AttentionInput out;
    out.host = host;
    out.device = device.to(target_device);
    out.attention_host_buffer = attention_host_buffer;
    out.attention_device_buffer = attention_device_buffer;
    out.attention_buffer_bytes = attention_buffer_bytes;
    out.attention_buffer_capacity = attention_buffer_capacity;
    out.attention_buffer_owner = attention_buffer_owner;
    return out;
  }

  bool rebuild_device_buffer(const torch::Device& target_device) {
    struct Entry {
      const void* source = nullptr;
      std::vector<int64_t> sizes;
      torch::ScalarType dtype = torch::kUInt8;
      torch::Tensor* target = nullptr;
      uint64_t offset = 0;
      uint64_t bytes = 0;
      uint64_t aligned_bytes = 0;
    };

    auto align_up = [](uint64_t value, uint64_t alignment) {
      return ((value + alignment - 1) / alignment) * alignment;
    };

    CHECK_EQ(host.q_seq_lens.empty(), host.q_cu_seq_lens.empty())
        << "q_seq_lens and q_cu_seq_lens must be provided together";

    std::vector<Entry> entries;
    std::vector<torch::Tensor> tensor_sources;
    tensor_sources.reserve(16);

    auto add_raw = [&entries](const void* source,
                              std::vector<int64_t> sizes,
                              torch::ScalarType dtype,
                              uint64_t bytes,
                              torch::Tensor* target) {
      if (source == nullptr || bytes == 0) {
        return;
      }
      entries.push_back(
          Entry{source, std::move(sizes), dtype, target, 0, bytes, 0});
    };

    auto add_int_vector = [&add_raw](const std::vector<int32_t>& values,
                                     torch::Tensor* target) {
      if (values.empty()) {
        return;
      }
      add_raw(values.data(),
              {static_cast<int64_t>(values.size())},
              torch::kInt,
              static_cast<uint64_t>(values.size() * sizeof(int32_t)),
              target);
    };

    auto add_cpu_tensor = [&entries, &tensor_sources](
                              const torch::Tensor& tensor,
                              torch::Tensor* target) {
      if (!tensor.defined()) {
        return;
      }
      if (!tensor.device().is_cpu()) {
        return;
      }
      tensor_sources.emplace_back(tensor.contiguous());
      const torch::Tensor& source = tensor_sources.back();
      const uint64_t bytes =
          static_cast<uint64_t>(source.numel() * source.element_size());
      if (bytes == 0) {
        return;
      }
      entries.push_back(Entry{source.data_ptr(),
                              source.sizes().vec(),
                              source.scalar_type(),
                              target,
                              0,
                              bytes,
                              0});
    };

    add_int_vector(host.q_seq_lens, &device.q_seq_lens);
    add_int_vector(host.kv_seq_lens, &device.kv_seq_lens);
    add_int_vector(host.q_cu_seq_lens, &device.q_cu_seq_lens);
    add_int_vector(host.new_cache_slots, &device.new_cache_slots);
    add_cpu_tensor(host.block_tables, &device.block_tables);
    add_int_vector(host.kv_cache_tokens_nums, &device.kv_cache_tokens_nums);
    add_int_vector(host.ring_cur_seqlen, &device.ring_cur_seqlen);
    add_int_vector(host.ring_cache_seqlen, &device.ring_cache_seqlen);

    add_cpu_tensor(device.paged_kv_indptr, &device.paged_kv_indptr);
    add_cpu_tensor(device.paged_kv_indices, &device.paged_kv_indices);
    add_cpu_tensor(device.paged_kv_last_page_len,
                   &device.paged_kv_last_page_len);
    add_cpu_tensor(device.new_cache_slot_offsets,
                   &device.new_cache_slot_offsets);
    add_cpu_tensor(device.kv_cache_start_offsets,
                   &device.kv_cache_start_offsets);
    add_cpu_tensor(device.history_compressed_kv, &device.history_compressed_kv);
    add_cpu_tensor(device.history_k_rope, &device.history_k_rope);

    if (entries.empty()) {
      attention_buffer_bytes = 0;
      return true;
    }

    constexpr uint64_t kAlignment = 16;
    uint64_t total_bytes = 0;
    for (auto& entry : entries) {
      total_bytes = align_up(total_bytes, kAlignment);
      entry.offset = total_bytes;
      entry.aligned_bytes = align_up(entry.bytes, kAlignment);
      total_bytes += entry.aligned_bytes;
    }
    if (total_bytes == 0) {
      attention_buffer_bytes = 0;
      return true;
    }

    detach_attention_buffer_if_shared();
    ensure_attention_buffer_capacity(total_bytes, target_device);
    attention_buffer_bytes = total_bytes;

    auto* host_base = static_cast<char*>(attention_host_buffer.data_ptr());
    for (const auto& entry : entries) {
      if (entry.bytes == 0) {
        continue;
      }
      std::memcpy(host_base + entry.offset, entry.source, entry.bytes);
      if (entry.aligned_bytes > entry.bytes) {
        std::memset(host_base + entry.offset + entry.bytes,
                    0,
                    static_cast<size_t>(entry.aligned_bytes - entry.bytes));
      }
    }

    attention_device_buffer.narrow(0, 0, static_cast<int64_t>(total_bytes))
        .copy_(attention_host_buffer.narrow(
                   0, 0, static_cast<int64_t>(total_bytes)),
               /*non_blocking=*/true);
    const char* device_base =
        static_cast<const char*>(attention_device_buffer.data_ptr());
    for (const auto& entry : entries) {
      if (entry.target == nullptr) {
        continue;
      }
      const void* ptr = device_base + entry.offset;
#if defined(USE_CUDA) || defined(USE_DCU)
      if (target_device.type() == torch::kCUDA) {
        *entry.target = get_tensor_from_blob(
            entry.sizes, entry.dtype, ptr, attention_device_buffer);
        continue;
      }
#endif
#if defined(USE_MLU)
      if (target_device.type() == torch::kPrivateUse1) {
        *entry.target = get_tensor_from_blob(
            entry.sizes, entry.dtype, ptr, attention_device_buffer);
        continue;
      }
#endif
#if defined(USE_NPU)
      *entry.target = get_tensor_from_blob(entry.sizes, entry.dtype, ptr);
#else
      (void)ptr;
#endif
    }
    return true;
  }

 private:
  void detach_attention_buffer_if_shared() {
    if (attention_buffer_owner == nullptr) {
      attention_buffer_owner = std::make_shared<int>(0);
    }
    if (attention_buffer_owner.use_count() <= 1) {
      return;
    }
    attention_buffer_owner = std::make_shared<int>(0);
    attention_host_buffer = torch::Tensor();
    attention_device_buffer = torch::Tensor();
    attention_buffer_bytes = 0;
    attention_buffer_capacity = 0;
  }

  void ensure_attention_buffer_capacity(uint64_t total_bytes,
                                        const torch::Device& target_device) {
    if (attention_host_buffer.defined() && attention_device_buffer.defined() &&
        attention_host_buffer.device().is_cpu() &&
        attention_device_buffer.device() == target_device &&
        attention_buffer_capacity >= total_bytes) {
      return;
    }

    const uint64_t new_capacity = std::max(
        total_bytes, std::max<uint64_t>(attention_buffer_capacity * 2, 1024));
    attention_host_buffer = torch::empty({static_cast<int64_t>(new_capacity)},
                                         torch::TensorOptions()
                                             .dtype(torch::kUInt8)
                                             .device(torch::kCPU)
                                             .pinned_memory(true));
    attention_device_buffer = torch::empty(
        {static_cast<int64_t>(new_capacity)},
        torch::TensorOptions().dtype(torch::kUInt8).device(target_device));
    attention_buffer_capacity = new_capacity;
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

struct BlockTransferInfo {
  int32_t src_block_id = -1;
  int32_t dst_block_id = -1;
  uint8_t hash_key[XXH3_128BITS_HASH_VALUE_LEN];
  BlockType block_type = BlockType::KV;
  TransferType transfer_type;

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
    for (int i = 0; i < 16; i++) {
      rt += std::to_string(int64_t(hash_key[i])) + " ";
    }
    return std::to_string(src_block_id) + "->" + std::to_string(dst_block_id) +
           ", " + std::to_string(uint32_t(transfer_type)) + rt;
  }
};

struct BatchInputMeta {
  BatchForwardType batch_forward_type;
  int32_t num_sequences = 0;
  int32_t actual_num_sequences = 0;
  int32_t kv_max_seq_len = 0;
  int32_t q_max_seq_len = 0;
  uint64_t batch_id = 0;
};

struct ModelEmbeddingInput {
  // input embedding
  mutable torch::Tensor input_embedding;

  // embedding ids of each sequence
  std::vector<int32_t> embedding_ids;

  // linear state ids of each sequence
  std::vector<int32_t> linear_state_ids;

  // IntTensor: [n_seq]
  torch::Tensor linear_state_indices;

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

  ModelEmbeddingInput to(const torch::Device& device) const {
    ModelEmbeddingInput out;
    out.input_embedding = safe_to(input_embedding, device);
    out.embedding_ids = embedding_ids;
    out.linear_state_ids = linear_state_ids;
    out.linear_state_indices = safe_to(linear_state_indices, device, true);
    out.request_ids = request_ids;
    out.extra_token_ids = extra_token_ids;
    out.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    out.mtp_bootstrap_row_idxes = mtp_bootstrap_row_idxes;
    out.mtp_bootstrap_embeddings =
        safe_to(mtp_bootstrap_embeddings, device, true);
    return out;
  }
};

struct BlockCopyInput {
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

struct MultiModalInput {
  // multimodal
  mutable MMBatchData mm_data;

  // deep_stack for Qwen3-VL
  mutable std::vector<torch::Tensor> deep_stacks;

  MultiModalInput to(const torch::Device& device) const {
    MultiModalInput out;
    out.mm_data = MMBatchData::to(mm_data, device);
    out.deep_stacks = deep_stacks;
    return out;
  }
};

struct ParallelInput {
  // num tokens of all workers, mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  std::vector<int32_t> dp_is_decode;

  DpEpPaddingData dp_ep_padding_data;
  CpEpPaddingData cp_ep_padding_data;
  CpPrefillInputs cp_prefill_inputs;

#if defined(USE_MLU)
  std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer = nullptr;
#elif defined(USE_DCU)
  std::shared_ptr<DCULayerSynchronizerImpl> layer_synchronizer = nullptr;
#elif defined(USE_NPU)
  std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer = nullptr;
#endif
  uint32_t layers_per_bacth_copy = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<LayerSynchronizer> layer_wise_load_synchronizer = nullptr;
#if defined(USE_NPU)
  std::vector<int64_t> query_start_loc;
  std::vector<int64_t> has_initial_state;
#endif

  ParallelInput to(const torch::Device& device) const {
    ParallelInput out;
    out.dp_global_token_nums = dp_global_token_nums;
    out.dp_is_decode = dp_is_decode;
    out.dp_ep_padding_data = dp_ep_padding_data;
    out.cp_ep_padding_data
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
    out.cp_prefill_inputs = cp_prefill_inputs.to(device);
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    out.layer_synchronizer = layer_synchronizer;
#endif
    out.layers_per_bacth_copy = layers_per_bacth_copy;
    out.layer_wise_load_synchronizer = layer_wise_load_synchronizer;
#if defined(USE_NPU)
    out.query_start_loc = query_start_loc;
    out.has_initial_state = has_initial_state;
#endif
    return out;
  }
};

using LinearStatePrefixHash = PrefixHash;

struct LinearStateCacheOp {
  // Live slot the sequence advances its recurrent state in.
  int32_t linear_state_id = -1;
  // Restore request flag and the checkpoint slot the scheduler resolved it to.
  // The worker copies `restore_src_slot_id` -> `linear_state_id`. This mirrors
  // KV, which sends the worker only the resolved block-swap descriptor and
  // never the prefix hash. Kept as a bool (not derived from
  // `restore_src_slot_id >= 0`) so the "restore requested but the scheduler
  // could not resolve a source -> force cold start" state survives the IPC
  // boundary; the worker's copy-in relies on that bit
  // (linear_state_restore.cpp).
  bool restore_requested = false;
  int32_t restore_src_slot_id = -1;
};

struct ExpertInput {
  torch::Tensor expert_load_data;
  torch::Tensor expert_array;
  EplbInfo eplb_info;

  ExpertInput to(const torch::Device& device) const {
    ExpertInput out;
    out.expert_load_data = expert_load_data;
    out.expert_array = expert_array;
    out.eplb_info = eplb_info;
    return out;
  }
};

struct GraphInput {
  torch::Tensor attn_mask;
  torch::Tensor tiling_data;
#if defined(USE_DCU)
  bool use_dense_flash_attention = false;
#endif
  bool use_expanded_decode_for_spec_verify_attention = false;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_block_tables;
  torch::Tensor expanded_tiling_data;
  std::vector<int32_t> expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext> acl_graph_task_update_context;
#endif
  torch::Tensor input_tokens_override;

  GraphInput to(const torch::Device& device) const {
    GraphInput out;
    out.attn_mask = safe_to(attn_mask, device, true);
    out.tiling_data = safe_to(tiling_data, device, true);
#if defined(USE_DCU)
    out.use_dense_flash_attention = use_dense_flash_attention;
#endif
    out.use_expanded_decode_for_spec_verify_attention =
        use_expanded_decode_for_spec_verify_attention;
    out.expanded_kv_seq_lens = safe_to(expanded_kv_seq_lens, device, true);
    out.expanded_block_tables = safe_to(expanded_block_tables, device, true);
    out.expanded_tiling_data = safe_to(expanded_tiling_data, device, true);
    out.expanded_kv_seq_lens_vec = expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
    out.acl_graph_task_update_context = acl_graph_task_update_context;
#endif
    out.input_tokens_override =
        safe_to(input_tokens_override, device, /*non_blocking=*/true);
    return out;
  }
};

struct ModelInputParams {
  ModelInputParams to(const torch::Device& device) const {
    ModelInputParams params;
    params.meta = meta;
    params.attention = attention.to(device);
    params.embedding = embedding.to(device);
    params.block_copy = block_copy.to(device);
    params.multimodal = multimodal.to(device);
    params.parallel = parallel.to(device);
    params.expert = expert.to(device);
    params.graph = graph.to(device);
    params.dit_forward_input = dit_forward_input.to(device);
    params.linear_state_cache_ops = linear_state_cache_ops;
    params.is_spec_verify = is_spec_verify;
    params.num_accepted_tokens = safe_to(num_accepted_tokens, device, true);
    params.num_accepted_tokens_host = num_accepted_tokens_host;
    params.dsa_topk_indices = safe_to(dsa_topk_indices, device, true);
    for (const auto& table : multi_block_tables) {
      params.multi_block_tables.push_back(
          safe_to(table, table.options().device(torch::kCPU), true));
    }
    params.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    if (!params.embedding.linear_state_indices.defined() &&
        !params.embedding.linear_state_ids.empty()) {
      params.embedding.linear_state_indices =
          torch::tensor(params.embedding.linear_state_ids, torch::kInt)
              .to(device);
    }

    // rec_params device conversion for both OneRec and LLM-Rec variants
    if (const auto* onerec_xattn = onerec_xattention_params()) {
      params.rec_params = onerec_xattn->to(device);
    } else if (const auto* onerec = onerec_params()) {
      params.rec_params = onerec->to(device);
    } else if (const auto* llmrec = llmrec_params()) {
      params.rec_params = llmrec->to(device);
    }

    return params;
  }

  void print() const {
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << meta.batch_forward_type.to_string() << " , num_sequences is "
              << meta.num_sequences << " , kv_max_seq_len is "
              << meta.kv_max_seq_len << " , q_max_seq_len is "
              << meta.q_max_seq_len;
    LOG(INFO) << "ModelInputParams: attention.host.kv_seq_lens is "
              << attention.host.kv_seq_lens;
    LOG(INFO) << "ModelInputParams: attention.host.q_seq_lens is "
              << attention.host.q_seq_lens;
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << meta.batch_forward_type.to_string();
    print_tensor(
        attention.device.kv_seq_lens, "ModelInputParams: kv_seq_lens", 4);
    print_tensor(
        attention.device.q_seq_lens, "ModelInputParams: q_seq_lens", 4);
    print_tensor(
        attention.device.q_cu_seq_lens, "ModelInputParams: q_cu_seq_lens", 4);
    print_tensor(attention.device.new_cache_slots,
                 "ModelInputParams: new_cache_slots",
                 4);
    print_tensor(
        attention.device.block_tables, "ModelInputParams: block_tables", 4);
    LOG(INFO) << "ModelInputParams: dp_global_token_nums is "
              << parallel.dp_global_token_nums
              << ", dp_is_decode: " << parallel.dp_is_decode;
    LOG(INFO) << "ModelInputParams: is_spec_verify is " << is_spec_verify;
    print_tensor(num_accepted_tokens,
                 "ModelInputParams: num_accepted_tokens",
                 /*max_elements=*/4);

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
    CHECK(seq_idx < attention.host.q_seq_lens.size()) << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx];
#else
    CHECK(seq_idx < attention.host.q_seq_lens.size() - 1)
        << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx + 1] -
           attention.host.q_seq_lens[seq_idx];
#endif
  }

  bool synchronize_layer(uint32_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer != nullptr &&
        layer_idx % parallel.layers_per_bacth_copy == 0) {
      if (!parallel.layer_wise_load_synchronizer->synchronize_layer(
              layer_idx / parallel.layers_per_bacth_copy)) {
        return false;
      }
    }
    return true;
  }

  bool record_layer(uint32_t layer_idx, const torch::Device& device) const {
#if defined(USE_MLU) || defined(USE_DCU)
    if (parallel.layer_synchronizer != nullptr) {
      return parallel.layer_synchronizer->record_current(layer_idx,
                                                         device.index());
    }
#else
    (void)layer_idx;
    (void)device;
#endif
    return true;
  }

  BatchInputMeta meta;
  AttentionInput attention;
  ModelEmbeddingInput embedding;
  ParallelInput parallel;
  BlockCopyInput block_copy;
  MultiModalInput multimodal;
  ExpertInput expert;
  GraphInput graph;

  // Multi block manager block tables for DeepSeek V4.
  // Each tensor is [batch_size, max_block_len] for one manager.
  std::vector<torch::Tensor> multi_block_tables;

  // Shifted target token ids for MTP training/evaluation paths.
  torch::Tensor mtp_shifted_token_ids;

  // Structured per-row linear-state cache operations.
  std::vector<LinearStateCacheOp> linear_state_cache_ops;

  bool is_spec_verify = false;
  torch::Tensor num_accepted_tokens;
  torch::Tensor dsa_topk_indices;
  std::vector<int64_t> num_accepted_tokens_host;

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

  // Optional attention metadata, built by executor
  // Using shared_ptr with forward declaration to avoid circular dependency
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;

  // Flag for graph capture/replay mode.
  bool enable_graph = false;
};

}  // namespace xllm
