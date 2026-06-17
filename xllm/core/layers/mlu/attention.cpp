/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "attention.h"

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(head_size),
      use_fused_mla_qkv_(false),
      enable_lighting_indexer_(false),
      enable_mla_(false),
      sliding_window_(sliding_window) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             int64_t num_kv_heads,
                             int64_t v_head_dim,
                             int64_t sliding_window,
                             float scale,
                             bool use_fused_mla_qkv,
                             bool enable_lighting_indexer,
                             bool enable_mla)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(v_head_dim),
      use_fused_mla_qkv_(use_fused_mla_qkv),
      enable_lighting_indexer_(enable_lighting_indexer),
      enable_mla_(enable_mla),
      sliding_window_(sliding_window) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  const bool enable_mla = enable_mla_;
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output;
  if (enable_mla) {
    output = torch::empty({query.size(0), num_heads_ * v_head_dim_},
                          query.options());
  } else {
    output = torch::empty_like(query);
  }
  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  int64_t num_kv_heads = (enable_mla && !only_prefill) ? 1 : num_kv_heads_;
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> v_cache;
  std::optional<torch::Tensor> v;
  if (!enable_mla) {
    v = value.view({-1, num_kv_heads, head_size_});
    v_cache = kv_cache.get_v_cache();
  }

  // Check if KV cache quantization is enabled by checking scale tensors
  std::optional<torch::Tensor> k_cache_scale = kv_cache.get_k_cache_scale();
  std::optional<torch::Tensor> v_cache_scale = kv_cache.get_v_cache_scale();

  bool skip_process_cache = enable_mla && (only_prefill || use_fused_mla_qkv_);
  if (!skip_process_cache) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key.view({-1, num_kv_heads, head_size_});
    reshape_paged_cache_params.value = v;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;

    if (k_cache_scale.has_value()) {
      // Use quant_to_paged_cache for INT8 quantization
      reshape_paged_cache_params.k_cache_scale = k_cache_scale;
      reshape_paged_cache_params.v_cache_scale = v_cache_scale;
      xllm::kernel::quant_to_paged_cache(reshape_paged_cache_params);
    } else {
      // Use standard reshape_paged_cache
      xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
    }
  }

  if (enable_lighting_indexer_ || !only_prefill) {
    // This is a trick for better performance on extracting k cache on sparse
    // attention
    if (enable_lighting_indexer_ && k_cache.defined() && k_cache.dim() == 4 &&
        k_cache.size(2) != 1) {
      // we must explicitly make sure the k_cache is contiguous after reshaping
      k_cache = k_cache.reshape({-1, k_cache.size(1), 1, k_cache.size(3)})
                    .contiguous();
      if (k_cache_scale.has_value()) {
        auto scale = k_cache_scale.value();
        k_cache_scale = scale.reshape({-1, scale.size(1), 1}).contiguous();
      }
    }

    decoder_forward(query,
                    output,
                    k_cache,
                    v_cache,
                    attn_metadata,
                    k_cache_scale,
                    v_cache_scale);
  } else {
    prefill_forward(query,
                    key,
                    value,
                    output,
                    k_cache,
                    v_cache,
                    attn_metadata,
                    k_cache_scale,
                    v_cache_scale);
  }

  int64_t head_size = enable_mla ? v_head_dim_ : head_size_;
  output = output.view({-1, num_heads_ * head_size});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& output,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    const AttentionMetadata& attn_metadata,
    const std::optional<torch::Tensor>& k_cache_scale,
    const std::optional<torch::Tensor>& v_cache_scale) {
  const bool enable_mla = enable_mla_;
  int64_t head_size_v = enable_mla ? v_head_dim_ : head_size_;
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_v});

  std::optional<torch::Tensor> block_tables = std::nullopt;
  std::optional<torch::Tensor> output_lse = std::nullopt;

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_v});
  } else if (attn_metadata.is_chunked_prefill) {
    // For chunked prefill with quantized KV cache, we need to dequantize first
    if (k_cache_scale.has_value() && k_cache_scale->defined() &&
        k_cache_scale->numel() > 0) {
      // Quantized KV cache path - dequantize before flash attention

      // Reuse the host-side total length cached in attention metadata to avoid
      // synchronizing the device cu_seq tensor back to CPU.
      int64_t total_seqlens = attn_metadata.total_kv_len;

      // Allocate dequantized output tensors
      torch::Tensor key_dequant = torch::zeros(
          {total_seqlens, num_kv_heads_, head_size_}, query.options());

      torch::Tensor value_dequant;
      std::optional<torch::Tensor> value_cache_for_dequant = v_cache;
      std::optional<torch::Tensor> value_cache_scale_for_dequant =
          v_cache_scale;
      if (enable_mla_) {
        // MLA stores latent cache only in k_cache, but the MLU dequant API
        // requires both key/value carriers. Reuse the latent cache as a dummy
        // value carrier and slice the real V view from key_dequant below.
        value_dequant = torch::zeros_like(key_dequant);
        value_cache_for_dequant = k_cache;
        value_cache_scale_for_dequant = k_cache_scale;
      }
      if (v_cache_scale.has_value() && v_cache_scale->defined() &&
          v_cache_scale->numel() > 0) {
        value_dequant = torch::zeros(
            {total_seqlens, num_kv_heads_, head_size_v}, query.options());
      }

      // Call dequant_from_paged_cache
      xllm::kernel::ReshapeFromCacheParams dequant_params;
      dequant_params.key = key_dequant;
      dequant_params.value = value_dequant;
      dequant_params.key_cache = k_cache;
      dequant_params.value_cache = value_cache_for_dequant;
      dequant_params.key_cache_quant_scale = k_cache_scale;
      dequant_params.value_cache_quant_scale = value_cache_scale_for_dequant;
      dequant_params.context_lengths = attn_metadata.kv_seq_lens;
      dequant_params.max_context_len = attn_metadata.max_seq_len;
      dequant_params.context_seq_offset = std::nullopt;
      dequant_params.block_tables = attn_metadata.block_table;
      dequant_params.quant_mode = 1;  // per-token quantization
      dequant_params.quant_bit = 8;   // only support INT8 for now.

      xllm::kernel::dequant_from_paged_cache(dequant_params);

      // Use dequantized tensors for flash attention
      key = key_dequant;
      value = enable_mla ? key_dequant.slice(-1, 0, v_head_dim_).contiguous()
                         : value_dequant;
    } else {
      // Non-quantized KV cache path - use directly
      key = k_cache;
      value = enable_mla ? k_cache.slice(-1, 0, v_head_dim_).contiguous()
                         : v_cache.value();
      block_tables = attn_metadata.block_table;
    }
  }

  xllm::kernel::mlu::batch_prefill(query,
                                   key,
                                   value,
                                   output,
                                   output_lse,
                                   attn_metadata.q_cu_seq_lens,
                                   attn_metadata.kv_cu_seq_lens,
                                   /*alibi_slope=*/std::nullopt,
                                   /*alibi_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   /*out_quant_scale=*/std::nullopt,
                                   block_tables,
                                   attn_metadata.max_query_len,
                                   attn_metadata.max_seq_len,
                                   scale_,
                                   /*is_causal=*/true,
                                   sliding_window_,
                                   /*window_size_right=*/-1,
                                   /*compute_dtype=*/"float",
                                   /*return_lse=*/false);
}

void AttentionImpl::decoder_forward(
    torch::Tensor& query,
    torch::Tensor& output,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    const AttentionMetadata& attn_metadata,
    const std::optional<torch::Tensor>& k_cache_scale,
    const std::optional<torch::Tensor>& v_cache_scale) {
  int64_t head_size_v = enable_mla_ ? v_head_dim_ : head_size_;
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_v});

  std::optional<torch::Tensor> output_lse = std::nullopt;

  // Set quantization parameters if KV cache is quantized
  std::optional<torch::Tensor> k_cache_quant_scale;
  std::optional<torch::Tensor> v_cache_quant_scale;
  int64_t kv_cache_quant_bit_size = -1;
  if (k_cache_scale.has_value()) {
    k_cache_quant_scale = k_cache_scale;
    if (v_cache_scale.has_value()) {
      v_cache_quant_scale = v_cache_scale;
    }
    kv_cache_quant_bit_size = 8;  // INT8 quantization
  }

  xllm::kernel::mlu::batch_decode(query,
                                  k_cache,
                                  output,
                                  attn_metadata.block_table,
                                  attn_metadata.kv_seq_lens,
                                  v_cache,
                                  output_lse,
                                  /*q_quant_scale=*/std::nullopt,
                                  k_cache_quant_scale,
                                  v_cache_quant_scale,
                                  /*out_quant_scale=*/std::nullopt,
                                  /*alibi_slope=*/std::nullopt,
                                  /*mask=*/std::nullopt,
                                  attn_metadata.compute_dtype,
                                  attn_metadata.max_seq_len,
                                  sliding_window_,
                                  /*window_size_right=*/-1,
                                  scale_,
                                  /*return_lse=*/false,
                                  kv_cache_quant_bit_size);
}

}  // namespace layer
}  // namespace xllm
