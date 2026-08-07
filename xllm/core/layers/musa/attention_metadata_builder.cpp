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

#include "layers/musa/attention_metadata_builder.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"

namespace xllm::layer::musa {

namespace {

torch::Tensor to_host_contiguous(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return {};
  }
  torch::Tensor result =
      tensor.device().is_cpu() ? tensor : tensor.to(torch::kCPU);
  if (result.scalar_type() != torch::kInt32) {
    result = result.to(torch::kInt32);
  }
  return result.contiguous();
}

torch::Tensor to_device_int32_contiguous(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return {};
  }
  torch::Tensor result = tensor;
  if (result.scalar_type() != torch::kInt32) {
    result = result.to(torch::kInt32);
  }
  return result.contiguous();
}

torch::Tensor prepend_zero(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return {};
  }
  auto zero = torch::zeros({1}, tensor.options());
  return torch::cat({zero, tensor}, 0).contiguous();
}

void convert_cumulative_lengths(std::vector<int32_t>& lengths,
                                int64_t sequence_count) {
  if (lengths.empty() ||
      lengths.size() == static_cast<size_t>(sequence_count)) {
    return;
  }
  CHECK_EQ(lengths.front(), 0);

  std::vector<int32_t> per_sequence_lengths;
  per_sequence_lengths.reserve(lengths.size() - 1);
  for (std::size_t i = 1; i < lengths.size(); ++i) {
    per_sequence_lengths.emplace_back(lengths[i] - lengths[i - 1]);
  }
  lengths = std::move(per_sequence_lengths);
}

void populate_attention_metadata(
    AttentionMetadata& attn_metadata,
    const ModelInputParams& params,
    const std::optional<torch::Tensor>& attn_mask) {
  const bool q_lengths_are_cumulative =
      params.attention.host.q_seq_lens.size() > 1 &&
      params.attention.host.q_seq_lens.front() == 0;
  const bool kv_lengths_are_cumulative =
      params.attention.host.kv_seq_lens.size() > 1 &&
      params.attention.host.kv_seq_lens.front() == 0;
  const int64_t q_sequence_count =
      params.attention.host.q_seq_lens.empty()
          ? params.meta.num_sequences
          : (q_lengths_are_cumulative
                 ? params.attention.host.q_seq_lens.size() - 1
                 : params.attention.host.q_seq_lens.size());
  const int64_t kv_sequence_count =
      params.attention.host.kv_seq_lens.empty()
          ? params.meta.num_sequences
          : (kv_lengths_are_cumulative
                 ? params.attention.host.kv_seq_lens.size() - 1
                 : params.attention.host.kv_seq_lens.size());
  convert_cumulative_lengths(attn_metadata.q_seq_lens_vec, q_sequence_count);
  convert_cumulative_lengths(attn_metadata.kv_seq_lens_vec, kv_sequence_count);

  const bool q_device_lengths_are_cumulative =
      q_lengths_are_cumulative ||
      (params.attention.device.q_seq_lens.defined() &&
       params.attention.device.q_seq_lens.numel() == q_sequence_count + 1);
  const bool kv_device_lengths_are_cumulative =
      kv_lengths_are_cumulative ||
      (params.attention.device.kv_seq_lens.defined() &&
       params.attention.device.kv_seq_lens.numel() == kv_sequence_count + 1);
  torch::Tensor q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  if (q_cu_seq_lens.defined() && q_cu_seq_lens.numel() == q_sequence_count) {
    q_cu_seq_lens = prepend_zero(q_lengths_are_cumulative
                                     ? q_cu_seq_lens
                                     : torch::cumsum(q_cu_seq_lens, 0));
  }
  if (!q_cu_seq_lens.defined()) {
    q_cu_seq_lens = params.attention.device.q_cu_seq_lens;
    if (q_cu_seq_lens.defined() && q_cu_seq_lens.numel() == q_sequence_count) {
      q_cu_seq_lens = prepend_zero(q_cu_seq_lens);
    }
  }
  if (!q_cu_seq_lens.defined() &&
      params.attention.device.q_seq_lens.defined()) {
    q_cu_seq_lens = params.attention.device.q_seq_lens;
    if (!q_device_lengths_are_cumulative) {
      q_cu_seq_lens = prepend_zero(torch::cumsum(q_cu_seq_lens, 0));
    }
  }
  attn_metadata.q_cu_seq_lens = to_device_int32_contiguous(q_cu_seq_lens);
  torch::Tensor kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  bool kv_cu_is_cumulative = kv_cu_seq_lens.defined() &&
                             (kv_device_lengths_are_cumulative ||
                              kv_cu_seq_lens.numel() == kv_sequence_count + 1);
  if (!kv_cu_seq_lens.defined()) {
    kv_cu_seq_lens = params.attention.device.kv_cu_seq_lens;
    if (kv_cu_seq_lens.defined() &&
        kv_cu_seq_lens.numel() == kv_sequence_count) {
      kv_cu_seq_lens = prepend_zero(kv_cu_seq_lens);
      kv_cu_is_cumulative = true;
    } else if (kv_cu_seq_lens.defined()) {
      kv_cu_is_cumulative = true;
    }
  }
  if (!kv_cu_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.defined()) {
    kv_cu_seq_lens = params.attention.device.kv_seq_lens;
    if (!kv_device_lengths_are_cumulative) {
      kv_cu_seq_lens = prepend_zero(torch::cumsum(kv_cu_seq_lens, 0));
    }
    kv_cu_is_cumulative = true;
  }
  attn_metadata.kv_cu_seq_lens = to_device_int32_contiguous(kv_cu_seq_lens);
  if (attn_metadata.kv_cu_seq_lens.defined() && !kv_cu_is_cumulative) {
    attn_metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, attn_metadata.kv_cu_seq_lens.options()),
                    torch::cumsum(attn_metadata.kv_cu_seq_lens, 0)},
                   0)
            .contiguous();
  }

  if (!attn_metadata.paged_kv_indptr_host.defined()) {
    attn_metadata.paged_kv_indptr_host =
        to_host_contiguous(params.attention.device.paged_kv_indptr);
  }
  if (!attn_metadata.paged_kv_indices_host.defined()) {
    attn_metadata.paged_kv_indices_host =
        to_host_contiguous(params.attention.device.paged_kv_indices);
  }
  if (!attn_metadata.paged_kv_last_page_len_host.defined()) {
    attn_metadata.paged_kv_last_page_len_host =
        to_host_contiguous(params.attention.device.paged_kv_last_page_len);
  }

  if (attn_mask.has_value() && attn_mask->dim() == 1) {
    attn_metadata.attn_mask = attn_mask.value();
  }

  if (params.attention.device.block_tables.defined()) {
    attn_metadata.block_table =
        to_device_int32_contiguous(params.attention.device.block_tables);
  }
  if (params.attention.device.q_seq_lens.defined()) {
    attn_metadata.q_seq_lens =
        q_device_lengths_are_cumulative
            ? to_device_int32_contiguous(
                  torch::diff(params.attention.device.q_seq_lens))
            : to_device_int32_contiguous(params.attention.device.q_seq_lens);
  }
  if (params.attention.device.kv_seq_lens.defined()) {
    attn_metadata.kv_seq_lens =
        kv_device_lengths_are_cumulative
            ? to_device_int32_contiguous(
                  torch::diff(params.attention.device.kv_seq_lens))
            : to_device_int32_contiguous(params.attention.device.kv_seq_lens);
  }
}

void finalize_attention_metadata(AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_causal && !attn_metadata.enable_cuda_graph &&
      attn_metadata.q_cu_seq_lens.defined()) {
    attn_metadata.qo_indptr =
        attn_metadata.q_cu_seq_lens.to(torch::kPrivateUse1);
  }
}

}  // namespace

AttentionMetadata build_attention_metadata(
    const ModelInputParams& params,
    bool enable_mla,
    const std::optional<torch::Tensor>& attn_mask,
    const std::optional<torch::Device>& device) {
  AttentionMetadata attn_metadata{};
  if (params.attn_metadata != nullptr) {
    const AttentionMetadata& seed =
        get_attention_metadata(*params.attn_metadata);
    attn_metadata.paged_kv_indptr_host = seed.paged_kv_indptr_host;
    attn_metadata.paged_kv_indices_host = seed.paged_kv_indices_host;
    attn_metadata.paged_kv_last_page_len_host =
        seed.paged_kv_last_page_len_host;
    attn_metadata.share_fa3_scheduler_metadata =
        seed.share_fa3_scheduler_metadata;
    attn_metadata.fa3_scheduler_metadata = seed.fa3_scheduler_metadata;
  }

  static_cast<::xllm::layer::AttentionMetadata&>(attn_metadata) =
      ::xllm::layer::AttentionMetadataBuilder::build(
          params, enable_mla, attn_mask, device);
  populate_attention_metadata(attn_metadata, params, attn_mask);
  finalize_attention_metadata(attn_metadata);
  attn_metadata.initialized = true;
  return attn_metadata;
}

}  // namespace xllm::layer::musa
