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

#include "layers/dcu/torch_attention.h"

#include <ATen/ops/scaled_dot_product_attention.h>
#include <glog/logging.h>

#include "framework/kv_cache/kv_cache.h"
#include "kernels/dcu/attention_runner.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

TorchAttentionImpl::TorchAttentionImpl(int64_t num_heads,
                                       int64_t head_size,
                                       float scale,
                                       int64_t num_kv_heads,
                                       int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window) {}

std::pair<torch::Tensor, torch::Tensor> TorchAttentionImpl::expand_kv_for_mqa(
    const torch::Tensor& key,
    const torch::Tensor& value) {
  // key: [seq_len, num_kv_heads, head_size]
  // value: [seq_len, num_kv_heads, head_size]
  if (num_kv_heads_ == num_heads_) {
    return {key, value};
  }
  CHECK_GT(num_kv_heads_, 0) << "num_kv_heads must be positive";
  CHECK_EQ(num_heads_ % num_kv_heads_, 0)
      << "num_heads must be divisible by num_kv_heads";

  const int64_t expansion_factor = num_heads_ / num_kv_heads_;
  const int64_t seq_len = key.size(0);

  torch::Tensor key_expanded =
      key.unsqueeze(2)
          .expand({seq_len, num_kv_heads_, expansion_factor, head_size_})
          .reshape({seq_len, num_heads_, head_size_});
  torch::Tensor value_expanded =
      value.unsqueeze(2)
          .expand({seq_len, num_kv_heads_, expansion_factor, head_size_})
          .reshape({seq_len, num_heads_, head_size_});

  return {key_expanded, value_expanded};
}

torch::Tensor TorchAttentionImpl::compute_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool is_causal,
    const std::optional<torch::Tensor>& attn_mask) {
  torch::Tensor query_4d = query.unsqueeze(0).permute({0, 2, 1, 3});
  torch::Tensor key_4d = key.unsqueeze(0).permute({0, 2, 1, 3});
  torch::Tensor value_4d = value.unsqueeze(0).permute({0, 2, 1, 3});

  std::optional<torch::Tensor> sdpa_mask = std::nullopt;
  bool sdpa_is_causal = is_causal;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    torch::Tensor key_mask = attn_mask.value()
                                 .to(query.device())
                                 .to(torch::kBool)
                                 .view({1, 1, 1, key.size(0)});
    torch::Tensor combined_mask =
        key_mask.expand({1, 1, query.size(0), key.size(0)});
    if (is_causal) {
      CHECK_LE(query.size(0), key.size(0))
          << "masked causal torch attention expects q_len <= kv_len";
      auto causal_mask =
          torch::ones(
              {query.size(0), key.size(0)},
              torch::TensorOptions().dtype(torch::kBool).device(query.device()))
              .tril(key.size(0) - query.size(0))
              .view({1, 1, query.size(0), key.size(0)});
      combined_mask = combined_mask.logical_and(causal_mask);
      sdpa_is_causal = false;
    }
    sdpa_mask = combined_mask;
  }

  torch::Tensor attn_output =
      at::scaled_dot_product_attention(query_4d,
                                       key_4d,
                                       value_4d,
                                       sdpa_mask,
                                       /*dropout_p=*/0.0,
                                       sdpa_is_causal);

  return attn_output.permute({0, 2, 1, 3}).squeeze(0);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
TorchAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                            torch::Tensor& query,
                            torch::Tensor& key,
                            torch::Tensor& value,
                            torch::Tensor& output,
                            KVCache& kv_cache) {
  std::optional<at::Tensor> output_lse = std::nullopt;

  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  // Reshape inputs from [seq_len, num_heads * head_size] to
  // [seq_len, num_heads, head_size].
  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // Reshape and store to cache if k_cache is properly initialized.
  if (k_cache.defined() && k_cache.dim() >= 2) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (attn_metadata.is_prefill) {
    AttentionMetadata attn_metadata_copy = attn_metadata;

    torch::Tensor query_saved = query;
    torch::Tensor key_saved = key;
    torch::Tensor value_saved = value;

    // Output is [T, num_heads, head_size] for attention writes.
    torch::Tensor output_3d_saved = output;

    // This flat tensor is returned to the following o_proj / linear layer.
    torch::Tensor output_flat = output.view({-1, num_heads_ * head_size_});

    return ::xllm::kernel::dcu::prefill_with_optional_piecewise_capture(
        [this,
         attn_metadata_copy,
         query_saved,
         key_saved,
         value_saved,
         output_3d_saved](
            const ::xllm::kernel::dcu::AttentionReplayParams& params) mutable {
          const AttentionMetadata& replay_metadata =
              params.attn_metadata ? *params.attn_metadata : attn_metadata_copy;

          torch::Tensor query = query_saved;
          torch::Tensor key = key_saved;
          torch::Tensor value = value_saved;

          // Attention still writes a 3D output internally.
          torch::Tensor output = output_3d_saved;

          std::optional<at::Tensor> output_lse = std::nullopt;

          torch::Tensor q_cu_seq_lens =
              replay_metadata.q_cu_seq_lens.cpu().to(torch::kInt64);
          torch::Tensor kv_cu_seq_lens =
              replay_metadata.kv_cu_seq_lens.cpu().to(torch::kInt64);

          int64_t batch_size = q_cu_seq_lens.size(0) - 1;

          std::pair<torch::Tensor, torch::Tensor> expanded_kv =
              expand_kv_for_mqa(key, value);
          torch::Tensor key_expanded = expanded_kv.first;
          torch::Tensor value_expanded = expanded_kv.second;

          for (int64_t i = 0; i < batch_size; ++i) {
            int64_t q_start = q_cu_seq_lens[i].item<int64_t>();
            int64_t q_end = q_cu_seq_lens[i + 1].item<int64_t>();
            int64_t kv_start = kv_cu_seq_lens[i].item<int64_t>();
            int64_t kv_end = kv_cu_seq_lens[i + 1].item<int64_t>();

            torch::Tensor q_seq = query.slice(0, q_start, q_end);
            torch::Tensor k_seq = key_expanded.slice(0, kv_start, kv_end);
            torch::Tensor v_seq = value_expanded.slice(0, kv_start, kv_end);
            std::optional<torch::Tensor> attn_mask_seq = std::nullopt;
            if (replay_metadata.attn_mask.defined()) {
              CHECK_GE(replay_metadata.attn_mask.numel(), kv_end)
                  << "attention mask is shorter than kv sequence";
              attn_mask_seq =
                  replay_metadata.attn_mask.slice(0, kv_start, kv_end);
            }

            torch::Tensor attn_out = compute_attention(
                q_seq, k_seq, v_seq, replay_metadata.is_causal, attn_mask_seq);

            output.slice(0, q_start, q_end).copy_(attn_out);
          }

          // Replay also returns flat output.
          torch::Tensor output_flat =
              output.view({-1, num_heads_ * head_size_});
          return std::make_tuple(output_flat, output_lse);
        },
        output_flat);
  } else if (attn_metadata.is_chunked_prefill) {
    // Chunked prefill: use paged KV cache for longer sequences.
    // Reconstruct full KV sequences from paged cache.
    torch::Tensor paged_kv_indptr =
        attn_metadata.paged_kv_indptr.cpu().to(torch::kInt64);
    torch::Tensor paged_kv_indices =
        attn_metadata.paged_kv_indices.cpu().to(torch::kInt64);
    const int64_t block_size =
        k_cache.defined() && k_cache.dim() >= 2 ? k_cache.size(1) : 1;

    const int64_t batch_size = paged_kv_indptr.size(0) - 1;
    torch::Tensor q_cu_seq_lens =
        attn_metadata.q_cu_seq_lens.cpu().to(torch::kInt64);

    // Expand KV heads to match query heads.
    std::pair<torch::Tensor, torch::Tensor> expanded_kv =
        expand_kv_for_mqa(key, value);
    torch::Tensor key_expanded = expanded_kv.first;
    torch::Tensor value_expanded = expanded_kv.second;

    // For each batch element.
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t q_start = q_cu_seq_lens[i].item<int64_t>();
      int64_t q_end = q_cu_seq_lens[i + 1].item<int64_t>();

      torch::Tensor q_seq = query.slice(0, q_start, q_end);

      // Gather pages for this batch from paged KV cache.
      std::vector<torch::Tensor> kv_pages_k, kv_pages_v;
      int64_t page_start = paged_kv_indptr[i].item<int64_t>();
      int64_t page_end = paged_kv_indptr[i + 1].item<int64_t>();

      for (int64_t p = page_start; p < page_end; ++p) {
        int64_t page_idx = paged_kv_indices[p].item<int64_t>();
        if (page_idx < k_cache.size(0)) {
          if (p == page_end - 1) {
            // Last page: use only the sequence-specific paged_kv_last_page_len.
            int64_t last_page_len =
                attn_metadata.paged_kv_last_page_len[i].item<int64_t>();
            kv_pages_k.push_back(k_cache[page_idx].narrow(0, 0, last_page_len));
            kv_pages_v.push_back(v_cache[page_idx].narrow(0, 0, last_page_len));
          } else {
            // Full page.
            kv_pages_k.push_back(k_cache[page_idx].narrow(0, 0, block_size));
            kv_pages_v.push_back(v_cache[page_idx].narrow(0, 0, block_size));
          }
        }
      }

      if (!kv_pages_k.empty()) {
        torch::Tensor k_seq = torch::cat(kv_pages_k, 0);
        torch::Tensor v_seq = torch::cat(kv_pages_v, 0);
        std::pair<torch::Tensor, torch::Tensor> seq_expanded_kv =
            expand_kv_for_mqa(
                k_seq.view({k_seq.size(0), num_kv_heads_, head_size_}),
                v_seq.view({v_seq.size(0), num_kv_heads_, head_size_}));

        torch::Tensor attn_out = compute_attention(q_seq,
                                                   seq_expanded_kv.first,
                                                   seq_expanded_kv.second,
                                                   attn_metadata.is_causal);
        output.slice(0, q_start, q_end).copy_(attn_out);
      }
    }
  } else {
    // Decode phase: single token attention against all cached KV.
    // key and value contain the current token for each batch item.
    // k_cache and v_cache contain all previous tokens in paged format.

    torch::Tensor paged_kv_indptr =
        attn_metadata.paged_kv_indptr.cpu().to(torch::kInt64);
    torch::Tensor paged_kv_indices =
        attn_metadata.paged_kv_indices.cpu().to(torch::kInt64);
    const int64_t block_size =
        k_cache.defined() && k_cache.dim() >= 2 ? k_cache.size(1) : 1;
    const int64_t batch_size = paged_kv_indptr.size(0) - 1;
    torch::Tensor q_cu_seq_lens =
        attn_metadata.q_cu_seq_lens.cpu().to(torch::kInt64);

    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t q_start = q_cu_seq_lens[i].item<int64_t>();
      int64_t q_end = q_cu_seq_lens[i + 1].item<int64_t>();

      // Gather pages from paged KV cache to reconstruct full sequence.
      std::vector<torch::Tensor> kv_pages_k, kv_pages_v;
      int64_t page_start = paged_kv_indptr[i].item<int64_t>();
      int64_t page_end = paged_kv_indptr[i + 1].item<int64_t>();

      for (int64_t p = page_start; p < page_end; ++p) {
        int64_t page_idx = paged_kv_indices[p].item<int64_t>();
        if (page_idx < k_cache.size(0)) {
          if (p == page_end - 1) {
            int64_t last_page_len =
                attn_metadata.paged_kv_last_page_len[i].item<int64_t>();
            kv_pages_k.push_back(k_cache[page_idx].narrow(0, 0, last_page_len));
            kv_pages_v.push_back(v_cache[page_idx].narrow(0, 0, last_page_len));
          } else {
            kv_pages_k.push_back(k_cache[page_idx].narrow(0, 0, block_size));
            kv_pages_v.push_back(v_cache[page_idx].narrow(0, 0, block_size));
          }
        }
      }

      // Concatenate all pages to form full KV cache sequence.
      torch::Tensor full_k, full_v;
      if (!kv_pages_k.empty()) {
        full_k = torch::cat(kv_pages_k, 0);
        full_v = torch::cat(kv_pages_v, 0);
      } else {
        // Empty cache, use current key/value only.
        full_k = key.slice(0, q_start, q_end);
        full_v = value.slice(0, q_start, q_end);
      }

      // Expand KV heads.
      std::pair<torch::Tensor, torch::Tensor> expanded_kv =
          expand_kv_for_mqa(full_k, full_v);
      torch::Tensor key_expanded = expanded_kv.first;
      torch::Tensor value_expanded = expanded_kv.second;

      // Single query token attends to all KV.
      torch::Tensor q_seq = query.slice(0, q_start, q_end);
      torch::Tensor attn_out = compute_attention(
          q_seq, key_expanded, value_expanded, attn_metadata.is_causal);
      output.slice(0, q_start, q_end).copy_(attn_out);
    }
  }

  // Reshape output back to [seq_len, num_heads * head_size].
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
