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

#include "layers/dcu/attention.h"

#include "core/util/rec_model_utils.h"
#include "layers/dcu/base_attention_impl.h"
#include "layers/dcu/flash_attention.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window) {
  // Select implementation based on mode. Use polymorphism via base class
  // pointer to manage different implementations.
  VLOG(1) << "## scale " << scale << " num_heads " << num_heads << " head_size "
          << head_size << " num_kv_heads " << num_kv_heads << " sliding_window "
          << sliding_window << " is_rec_multi_round_mode "
          << is_rec_multi_round_mode();

  attention_impl_ = std::make_shared<FlashAttentionImpl>(
      num_heads, head_size, scale, num_kv_heads, sliding_window);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // Create output tensor internally to unify the interface with other devices
  torch::Tensor output = torch::empty_like(query);

  VLOG(1) << "AttentionImpl::forward [CUDA graph mode]:   "
          << attn_metadata.enable_cuda_graph;

  if (!attn_metadata.enable_cuda_graph) {
    VLOG(1)
        << "AttentionImpl::forward"
        << " is_prefill=" << attn_metadata.is_prefill
        << " batch_size=" << (attn_metadata.kv_cu_seq_lens.size(0) - 1)
        << " max_seq_len=" << attn_metadata.max_seq_len
        << " max_query_len=" << attn_metadata.max_query_len
        << " q_shape=" << query.sizes() << " k_shape=" << key.sizes()
        << " v_shape=" << value.sizes() << " output_shape=" << output.sizes()
        << " kv_cache k_cache=" << kv_cache.get_k_cache().sizes()
        << " kv_cache v_cache=" << kv_cache.get_v_cache().sizes()
        << " full_k_cache size =" << attn_metadata.full_k_cache.sizes()
        << " full_v_cache=" << attn_metadata.full_v_cache.sizes()
        << " unshared_k_cache size =" << attn_metadata.unshared_k_cache.sizes()
        << " unshared_v_cache size =" << attn_metadata.unshared_v_cache.sizes()
        << " q_cu_seq_lens=" << attn_metadata.q_cu_seq_lens
        << " kv_cu_seq_lens=" << attn_metadata.kv_cu_seq_lens
        << " block_table=" << attn_metadata.block_table.sizes()
        << " paged_kv_indptr size =" << attn_metadata.paged_kv_indptr.sizes()
        << " paged_kv_indices size =" << attn_metadata.paged_kv_indices.sizes()
        << " paged_kv_last_page_len=" << attn_metadata.paged_kv_last_page_len
        << " slot_mapping size =" << attn_metadata.slot_mapping.sizes()
        << " q_seq_lens=" << attn_metadata.q_seq_lens
        << " kv_seq_lens=" << attn_metadata.kv_seq_lens;
  } else {
    // CUDA graph capture mode: cannot call tensor methods (sizes(), defined(),
    // numel()) but can print scalar fields and tensor objects directly
    VLOG(1) << "AttentionImpl::forward [CUDA graph mode]"
            << " kv_cache k_cache=" << kv_cache.get_k_cache().sizes()
            << " kv_cache v_cache=" << kv_cache.get_v_cache().sizes()
            << " is_prefill=" << attn_metadata.is_prefill
            << " max_seq_len=" << attn_metadata.max_seq_len
            << " max_query_len=" << attn_metadata.max_query_len
            << " total_kv_len=" << attn_metadata.total_kv_len
            << " is_causal=" << attn_metadata.is_causal
            << " is_chunked_prefill=" << attn_metadata.is_chunked_prefill
            << " is_dummy=" << attn_metadata.is_dummy
            << " compute_dtype=" << attn_metadata.compute_dtype
            << " enable_cuda_graph=" << attn_metadata.enable_cuda_graph;
  }

  // Use polymorphism to dispatch to the appropriate implementation,
  // making the code elegant and type-safe.
  return attention_impl_->forward(
      attn_metadata, query, key, value, output, kv_cache);
}

}  // namespace layer
}  // namespace xllm
