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

#pragma once

#include <algorithm>
#include <cstring>
#include <utility>

#include "core/framework/model/model_input_types.h"

namespace xllm {

enum class AttentionBufferReusePolicy {
  COPY_ON_WRITE,
  GROWABLE,
  FIXED_CAPACITY,
};

struct PackedAttentionIntInput {
  const std::vector<int32_t>* values = nullptr;
  torch::Tensor* host_view = nullptr;
  torch::Tensor* device_view = nullptr;
};

class AttentionHostInputView final {
 public:
  template <typename Owner>
  explicit AttentionHostInputView(Owner& owner)
      : kpool_query_lens(owner.kpool_query_lens),
        q_seq_lens(owner.q_seq_lens),
        q_cu_seq_lens(owner.q_cu_seq_lens),
        kv_seq_lens(owner.kv_seq_lens),
        kv_cu_seq_lens(owner.kv_cu_seq_lens),
        new_cache_slots(owner.new_cache_slots),
        kv_cache_tokens_nums(owner.kv_cache_tokens_nums),
        ring_cur_seqlen(owner.ring_cur_seqlen),
        ring_cache_seqlen(owner.ring_cache_seqlen),
        block_tables(owner.block_tables),
        graph_q_seq_lens_data(owner.graph_q_seq_lens_data),
        graph_kv_seq_lens_data(owner.graph_kv_seq_lens_data) {}

  std::vector<int32_t>& kpool_query_lens;
  std::vector<int32_t>& q_seq_lens;
  std::vector<int32_t>& q_cu_seq_lens;
  std::vector<int32_t>& kv_seq_lens;
  std::vector<int32_t>& kv_cu_seq_lens;
  std::vector<int32_t>& new_cache_slots;
  std::vector<int32_t>& kv_cache_tokens_nums;
  std::vector<int32_t>& ring_cur_seqlen;
  std::vector<int32_t>& ring_cache_seqlen;
  torch::Tensor& block_tables;
  const int32_t*& graph_q_seq_lens_data;
  const int32_t*& graph_kv_seq_lens_data;
};

class AttentionDeviceInputView final {
 public:
  template <typename Owner>
  explicit AttentionDeviceInputView(Owner& owner)
      : q_seq_lens(owner.q_seq_lens),
        kv_seq_lens(owner.kv_seq_lens),
        q_cu_seq_lens(owner.q_cu_seq_lens),
        new_cache_slots(owner.new_cache_slots),
        block_tables(owner.block_tables),
        paged_kv_indptr(owner.paged_kv_indptr),
        paged_kv_indices(owner.paged_kv_indices),
        paged_kv_last_page_len(owner.paged_kv_last_page_len),
        new_cache_slot_offsets(owner.new_cache_slot_offsets),
        kv_cache_start_offsets(owner.kv_cache_start_offsets),
        kv_cache_tokens_nums(owner.kv_cache_tokens_nums),
        history_compressed_kv(owner.history_compressed_kv),
        history_k_rope(owner.history_k_rope),
        ring_cur_seqlen(owner.ring_cur_seqlen),
        ring_cache_seqlen(owner.ring_cache_seqlen),
        in_prefix_slots(owner.in_prefix_slots) {}

  torch::Tensor& q_seq_lens;
  torch::Tensor& kv_seq_lens;
  torch::Tensor& q_cu_seq_lens;
  torch::Tensor& new_cache_slots;
  torch::Tensor& block_tables;
  torch::Tensor& paged_kv_indptr;
  torch::Tensor& paged_kv_indices;
  torch::Tensor& paged_kv_last_page_len;
  torch::Tensor& new_cache_slot_offsets;
  torch::Tensor& kv_cache_start_offsets;
  torch::Tensor& kv_cache_tokens_nums;
  torch::Tensor& history_compressed_kv;
  torch::Tensor& history_k_rope;
  torch::Tensor& ring_cur_seqlen;
  torch::Tensor& ring_cache_seqlen;
  torch::Tensor& in_prefix_slots;
};

// Borrows storage from a domain owner. No tensor or vector is copied.
class AttentionInputView final {
 public:
  using BufferReusePolicy = AttentionBufferReusePolicy;
  using PackedIntInput = PackedAttentionIntInput;

  template <typename Owner>
  explicit AttentionInputView(Owner& owner)
      : host(owner.host),
        device(owner.device),
        attention_host_buffer(owner.attention_host_buffer),
        attention_device_buffer(owner.attention_device_buffer),
        attention_buffer_bytes(owner.attention_buffer_bytes),
        attention_buffer_capacity(owner.attention_buffer_capacity),
        attention_buffer_owner(owner.attention_buffer_owner) {}

  bool rebuild_device_buffer(
      const torch::Device& target_device,
      const std::vector<PackedIntInput>& extra_int_inputs = {},
      BufferReusePolicy reuse_policy = BufferReusePolicy::COPY_ON_WRITE) {
    struct Entry {
      const void* source = nullptr;
      std::vector<int64_t> sizes;
      torch::ScalarType dtype = torch::kUInt8;
      torch::Tensor* host_target = nullptr;
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
    entries.reserve(extra_int_inputs.size() + 16);
    std::vector<torch::Tensor> tensor_sources;
    tensor_sources.reserve(16);
    auto add_raw = [&entries](const void* source,
                              std::vector<int64_t> sizes,
                              torch::ScalarType dtype,
                              uint64_t bytes,
                              torch::Tensor* host_target,
                              torch::Tensor* target) {
      if (source == nullptr || bytes == 0) {
        return;
      }
      entries.emplace_back(Entry{
          source, std::move(sizes), dtype, host_target, target, 0, bytes, 0});
    };
    auto add_int_vector = [&add_raw](const std::vector<int32_t>& values,
                                     torch::Tensor* host_target,
                                     torch::Tensor* target) {
      if (values.empty()) {
        return;
      }
      add_raw(values.data(),
              {static_cast<int64_t>(values.size())},
              torch::kInt,
              static_cast<uint64_t>(values.size() * sizeof(int32_t)),
              host_target,
              target);
    };
    auto add_cpu_tensor = [&entries, &tensor_sources](
                              const torch::Tensor& tensor,
                              torch::Tensor* target) {
      if (!tensor.defined() || !tensor.device().is_cpu()) {
        return;
      }
      tensor_sources.emplace_back(tensor.contiguous());
      const torch::Tensor& source = tensor_sources.back();
      const uint64_t bytes =
          static_cast<uint64_t>(source.numel() * source.element_size());
      if (bytes == 0) {
        return;
      }
      entries.emplace_back(Entry{source.data_ptr(),
                                 source.sizes().vec(),
                                 source.scalar_type(),
                                 nullptr,
                                 target,
                                 0,
                                 bytes,
                                 0});
    };

    for (const PackedIntInput& extra : extra_int_inputs) {
      CHECK(extra.values != nullptr);
      add_int_vector(*extra.values, extra.host_view, extra.device_view);
    }
    add_int_vector(host.q_seq_lens, nullptr, &device.q_seq_lens);
    add_int_vector(host.kv_seq_lens, nullptr, &device.kv_seq_lens);
    add_int_vector(host.q_cu_seq_lens, nullptr, &device.q_cu_seq_lens);
    add_int_vector(host.new_cache_slots, nullptr, &device.new_cache_slots);
    add_cpu_tensor(host.block_tables, &device.block_tables);
    add_int_vector(
        host.kv_cache_tokens_nums, nullptr, &device.kv_cache_tokens_nums);
    add_int_vector(host.ring_cur_seqlen, nullptr, &device.ring_cur_seqlen);
    add_int_vector(host.ring_cache_seqlen, nullptr, &device.ring_cache_seqlen);
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
    if (reuse_policy == BufferReusePolicy::COPY_ON_WRITE) {
      detach_attention_buffer_if_shared();
    }
    if (reuse_policy == BufferReusePolicy::FIXED_CAPACITY) {
      CHECK(attention_host_buffer.defined() &&
            attention_device_buffer.defined());
      CHECK_GE(attention_buffer_capacity, total_bytes)
          << "fixed attention buffer cannot grow after graph capture";
    } else {
      ensure_attention_buffer_capacity(total_bytes, target_device);
    }
    attention_buffer_bytes = total_bytes;

    auto* host_base = static_cast<char*>(attention_host_buffer.data_ptr());
    for (const auto& entry : entries) {
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
      if (entry.host_target != nullptr) {
        *entry.host_target = torch::from_blob(
            host_base + entry.offset,
            entry.sizes,
            torch::TensorOptions().dtype(entry.dtype).device(torch::kCPU));
      }
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
#if defined(USE_MLU) || defined(USE_MUSA)
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

  void reserve_device_buffer_capacity(uint64_t capacity,
                                      const torch::Device& target_device) {
    ensure_attention_buffer_capacity(capacity, target_device);
  }

  AttentionHostInputView host;
  AttentionDeviceInputView device;
  torch::Tensor& attention_host_buffer;
  torch::Tensor& attention_device_buffer;
  uint64_t& attention_buffer_bytes;
  uint64_t& attention_buffer_capacity;
  std::shared_ptr<int>& attention_buffer_owner;

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

namespace detail {
template <typename Source, typename Target>
void copy_attention_host_input(const Source& source, Target& target) {
  target.kpool_query_lens = source.kpool_query_lens;
  target.q_seq_lens = source.q_seq_lens;
  target.q_cu_seq_lens = source.q_cu_seq_lens;
  target.kv_seq_lens = source.kv_seq_lens;
  target.kv_cu_seq_lens = source.kv_cu_seq_lens;
  target.new_cache_slots = source.new_cache_slots;
  target.kv_cache_tokens_nums = source.kv_cache_tokens_nums;
  target.ring_cur_seqlen = source.ring_cur_seqlen;
  target.ring_cache_seqlen = source.ring_cache_seqlen;
  target.block_tables = source.block_tables;
  target.graph_q_seq_lens_data = source.graph_q_seq_lens_data;
  target.graph_kv_seq_lens_data = source.graph_kv_seq_lens_data;
}

template <typename Source, typename Target>
void copy_attention_device_input(const Source& source, Target& target) {
  target.q_seq_lens = source.q_seq_lens;
  target.kv_seq_lens = source.kv_seq_lens;
  target.q_cu_seq_lens = source.q_cu_seq_lens;
  target.new_cache_slots = source.new_cache_slots;
  target.block_tables = source.block_tables;
  target.paged_kv_indptr = source.paged_kv_indptr;
  target.paged_kv_indices = source.paged_kv_indices;
  target.paged_kv_last_page_len = source.paged_kv_last_page_len;
  target.new_cache_slot_offsets = source.new_cache_slot_offsets;
  target.kv_cache_start_offsets = source.kv_cache_start_offsets;
  target.kv_cache_tokens_nums = source.kv_cache_tokens_nums;
  target.history_compressed_kv = source.history_compressed_kv;
  target.history_k_rope = source.history_k_rope;
  target.ring_cur_seqlen = source.ring_cur_seqlen;
  target.ring_cache_seqlen = source.ring_cache_seqlen;
  target.in_prefix_slots = source.in_prefix_slots;
}

template <typename Owner>
Owner attention_to(const Owner& source, const torch::Device& device) {
  Owner output;
  output.host = source.host;
  output.device = source.device.to(device);
  output.attention_host_buffer = source.attention_host_buffer;
  output.attention_device_buffer = source.attention_device_buffer;
  output.attention_buffer_bytes = source.attention_buffer_bytes;
  output.attention_buffer_capacity = source.attention_buffer_capacity;
  output.attention_buffer_owner = source.attention_buffer_owner;
  return output;
}
}  // namespace detail

class LlmAttentionInput final {
 public:
  using BufferReusePolicy = AttentionBufferReusePolicy;
  using PackedIntInput = PackedAttentionIntInput;
  LlmAttentionInput to(const torch::Device& target) const {
    return detail::attention_to(*this, target);
  }
  bool rebuild_device_buffer(
      const torch::Device& target,
      const std::vector<PackedIntInput>& extra = {},
      BufferReusePolicy policy = BufferReusePolicy::COPY_ON_WRITE) {
    return AttentionInputView(*this).rebuild_device_buffer(
        target, extra, policy);
  }
  void reserve_device_buffer_capacity(uint64_t bytes,
                                      const torch::Device& target) {
    AttentionInputView(*this).reserve_device_buffer_capacity(bytes, target);
  }
  LlmAttentionHostInput host;
  LlmAttentionDeviceInput device;
  torch::Tensor attention_host_buffer;
  torch::Tensor attention_device_buffer;
  uint64_t attention_buffer_bytes = 0;
  uint64_t attention_buffer_capacity = 0;
  std::shared_ptr<int> attention_buffer_owner = std::make_shared<int>(0);
};

class VlmAttentionInput final {
 public:
  using BufferReusePolicy = AttentionBufferReusePolicy;
  using PackedIntInput = PackedAttentionIntInput;
  VlmAttentionInput to(const torch::Device& target) const {
    return detail::attention_to(*this, target);
  }
  bool rebuild_device_buffer(
      const torch::Device& target,
      const std::vector<PackedIntInput>& extra = {},
      BufferReusePolicy policy = BufferReusePolicy::COPY_ON_WRITE) {
    return AttentionInputView(*this).rebuild_device_buffer(
        target, extra, policy);
  }
  void reserve_device_buffer_capacity(uint64_t bytes,
                                      const torch::Device& target) {
    AttentionInputView(*this).reserve_device_buffer_capacity(bytes, target);
  }
  VlmAttentionHostInput host;
  VlmAttentionDeviceInput device;
  torch::Tensor attention_host_buffer;
  torch::Tensor attention_device_buffer;
  uint64_t attention_buffer_bytes = 0;
  uint64_t attention_buffer_capacity = 0;
  std::shared_ptr<int> attention_buffer_owner = std::make_shared<int>(0);
};

class RecAttentionInput final {
 public:
  using BufferReusePolicy = AttentionBufferReusePolicy;
  using PackedIntInput = PackedAttentionIntInput;
  RecAttentionInput to(const torch::Device& target) const {
    return detail::attention_to(*this, target);
  }
  bool rebuild_device_buffer(
      const torch::Device& target,
      const std::vector<PackedIntInput>& extra = {},
      BufferReusePolicy policy = BufferReusePolicy::COPY_ON_WRITE) {
    return AttentionInputView(*this).rebuild_device_buffer(
        target, extra, policy);
  }
  void reserve_device_buffer_capacity(uint64_t bytes,
                                      const torch::Device& target) {
    AttentionInputView(*this).reserve_device_buffer_capacity(bytes, target);
  }
  RecAttentionHostInput host;
  RecAttentionDeviceInput device;
  torch::Tensor attention_host_buffer;
  torch::Tensor attention_device_buffer;
  uint64_t attention_buffer_bytes = 0;
  uint64_t attention_buffer_capacity = 0;
  std::shared_ptr<int> attention_buffer_owner = std::make_shared<int>(0);
};

}  // namespace xllm
