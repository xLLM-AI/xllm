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

#include "xattention.h"

#include <glog/logging.h>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::layer {
namespace {

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  torch::ScalarType dtype,
                  int64_t dim) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(tensor.device().is_privateuseone()) << name << " must be on MLU";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " has invalid dtype";
  CHECK_EQ(tensor.dim(), dim) << name << " has invalid rank";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
}

void check_shape(const torch::Tensor& tensor,
                 const char* name,
                 at::IntArrayRef shape) {
  CHECK_EQ(tensor.sizes(), shape) << name << " shape mismatch, expected "
                                  << shape << ", got " << tensor.sizes();
}

}  // namespace

MluXAttentionImpl::MluXAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads,
                                     int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window) {
  CHECK_GT(num_heads_, 0);
  CHECK_GT(head_size_, 0);
  CHECK_GT(num_kv_heads_, 0);
  CHECK_EQ(num_heads_ % num_kv_heads_, 0);
  CHECK_EQ(sliding_window_, -1) << "sliding-window attention is not supported";
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
MluXAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           torch::Tensor& value,
                           KVCache& kv_cache) {
  static_cast<void>(kv_cache);
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    return {torch::empty_like(query), output_lse};
  }

  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";

  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();
  torch::Tensor output = torch::empty_like(query);
  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});
  check_common(attn_metadata, query, key, value, output);

  if (attn_metadata.is_prefill) {
    prefill(attn_metadata, query, key, value, output);
  } else {
    decode(attn_metadata, query, key, value, output);
  }

  return {output.view({-1, num_heads_ * head_size_}), output_lse};
}

void MluXAttentionImpl::check_common(const AttentionMetadata& attn_metadata,
                                     const torch::Tensor& query,
                                     const torch::Tensor& key,
                                     const torch::Tensor& value,
                                     const torch::Tensor& output) const {
  check_tensor(query, "query", torch::kBFloat16, 3);
  check_tensor(key, "key", torch::kBFloat16, 3);
  check_tensor(value, "value", torch::kBFloat16, 3);
  check_tensor(output, "output", torch::kBFloat16, 3);
  CHECK_EQ(query.device(), key.device());
  CHECK_EQ(query.device(), value.device());
  CHECK_EQ(query.size(1), num_heads_);
  CHECK_EQ(query.size(2), head_size_);
  CHECK_EQ(key.size(0), value.size(0));
  CHECK_EQ(key.size(1), num_kv_heads_);
  CHECK_EQ(value.size(1), num_kv_heads_);
  CHECK_EQ(key.size(2), head_size_);
  CHECK_EQ(value.size(2), head_size_);
  CHECK_EQ(attn_metadata.compute_dtype, "float");

  check_tensor(attn_metadata.full_k_cache, "full_k_cache", torch::kBFloat16, 3);
  check_tensor(attn_metadata.full_v_cache, "full_v_cache", torch::kBFloat16, 3);
  CHECK_EQ(attn_metadata.full_k_cache.device(), query.device());
  CHECK_EQ(attn_metadata.full_v_cache.device(), query.device());
  CHECK_EQ(attn_metadata.full_k_cache.sizes(),
           attn_metadata.full_v_cache.sizes());
  CHECK_EQ(attn_metadata.full_k_cache.size(1), num_kv_heads_);
  CHECK_EQ(attn_metadata.full_k_cache.size(2), head_size_);
}

void MluXAttentionImpl::prefill(const AttentionMetadata& attn_metadata,
                                torch::Tensor& query,
                                torch::Tensor& key,
                                torch::Tensor& value,
                                torch::Tensor& output) const {
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_LE(key.size(0), attn_metadata.full_k_cache.size(0));
  check_tensor(attn_metadata.q_cu_seq_lens, "q_cu_seq_lens", torch::kInt32, 1);
  check_tensor(
      attn_metadata.kv_cu_seq_lens, "kv_cu_seq_lens", torch::kInt32, 1);
  CHECK_EQ(attn_metadata.q_cu_seq_lens.device(), query.device());
  CHECK_EQ(attn_metadata.kv_cu_seq_lens.device(), query.device());
  CHECK_EQ(attn_metadata.q_cu_seq_lens.numel(),
           attn_metadata.kv_cu_seq_lens.numel());

  const int64_t shared_len = key.size(0);
  CHECK_EQ(attn_metadata.total_kv_len, shared_len);
  attn_metadata.full_k_cache.slice(0, 0, shared_len).copy_(key);
  attn_metadata.full_v_cache.slice(0, 0, shared_len).copy_(value);

  std::optional<torch::Tensor> output_lse = std::nullopt;
  xllm::kernel::mlu::batch_prefill(query,
                                   key,
                                   value,
                                   output,
                                   output_lse,
                                   attn_metadata.q_cu_seq_lens,
                                   attn_metadata.kv_cu_seq_lens,
                                   /*alibi_slope=*/std::nullopt,
                                   /*attn_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   /*out_quant_scale=*/std::nullopt,
                                   /*block_tables=*/std::nullopt,
                                   attn_metadata.max_query_len,
                                   attn_metadata.max_seq_len,
                                   scale_,
                                   /*is_causal=*/true,
                                   /*window_size_left=*/-1,
                                   /*window_size_right=*/-1,
                                   /*compute_dtype=*/"float",
                                   /*return_lse=*/false);
}

void MluXAttentionImpl::decode(const AttentionMetadata& attn_metadata,
                               torch::Tensor& query,
                               torch::Tensor& key,
                               torch::Tensor& value,
                               torch::Tensor& output) const {
  CHECK(attn_metadata.xattention_two_stage_decode_cache.has_value())
      << "two-stage decode cache must be initialized";
  auto cache = attn_metadata.xattention_two_stage_decode_cache.value();

  const int64_t batch_size = cache.q_cu_seq_lens_shared.numel() - 1;
  const int64_t total_beam = query.size(0);
  CHECK_GT(batch_size, 0);
  CHECK_EQ(total_beam % batch_size, 0);
  CHECK_EQ(key.size(0), total_beam);
  CHECK_EQ(value.size(0), total_beam);
  const int64_t beam_width = total_beam / batch_size;

  check_tensor(
      attn_metadata.unshared_k_cache, "unshared_k_cache", torch::kBFloat16, 5);
  check_tensor(
      attn_metadata.unshared_v_cache, "unshared_v_cache", torch::kBFloat16, 5);
  CHECK(attn_metadata.unshared_k_cache.is_contiguous());
  CHECK(attn_metadata.unshared_v_cache.is_contiguous());
  CHECK_EQ(attn_metadata.unshared_k_cache.sizes(),
           attn_metadata.unshared_v_cache.sizes());
  CHECK_EQ(attn_metadata.unshared_k_cache.device(), query.device());
  CHECK_EQ(attn_metadata.unshared_v_cache.device(), query.device());
  CHECK_EQ(attn_metadata.unshared_k_cache.size(0) *
               attn_metadata.unshared_k_cache.size(1),
           total_beam);
  CHECK_EQ(attn_metadata.unshared_k_cache.size(2), num_kv_heads_);
  CHECK_EQ(attn_metadata.unshared_k_cache.size(4), head_size_);
  const int64_t max_decode_steps = attn_metadata.unshared_k_cache.size(3);
  CHECK_GT(max_decode_steps, 0);
  CHECK_EQ(attn_metadata.unshared_k_cache.numel(),
           total_beam * num_kv_heads_ * max_decode_steps * head_size_);

  check_tensor(cache.shared_o, "shared_o", torch::kBFloat16, 3);
  check_tensor(cache.unshared_o, "unshared_o", torch::kBFloat16, 3);
  check_tensor(cache.shared_lse, "shared_lse", torch::kFloat32, 3);
  check_tensor(cache.unshared_lse, "unshared_lse", torch::kFloat32, 3);
  check_tensor(
      cache.shared_lse_kernel, "shared_lse_kernel", torch::kFloat32, 2);
  check_tensor(
      cache.q_cu_seq_lens_shared, "q_cu_seq_lens_shared", torch::kInt32, 1);
  check_tensor(
      cache.decode_slot_mapping, "decode_slot_mapping", torch::kInt32, 1);
  check_tensor(cache.unshared_seq_lens, "unshared_seq_lens", torch::kInt32, 1);
  check_tensor(attn_metadata.block_table, "block_table", torch::kInt32, 2);
  check_tensor(
      attn_metadata.kv_cu_seq_lens, "kv_cu_seq_lens", torch::kInt32, 1);

  check_shape(cache.shared_o, "shared_o", {total_beam, num_heads_, head_size_});
  check_shape(
      cache.unshared_o, "unshared_o", {total_beam, num_heads_, head_size_});
  check_shape(cache.shared_lse, "shared_lse", {total_beam, num_heads_, 1});
  check_shape(cache.unshared_lse, "unshared_lse", {total_beam, num_heads_, 1});
  check_shape(
      cache.shared_lse_kernel, "shared_lse_kernel", {num_heads_, total_beam});
  check_shape(
      cache.q_cu_seq_lens_shared, "q_cu_seq_lens_shared", {batch_size + 1});
  check_shape(cache.decode_slot_mapping, "decode_slot_mapping", {total_beam});
  check_shape(cache.unshared_seq_lens, "unshared_seq_lens", {total_beam});
  check_shape(attn_metadata.block_table, "block_table", {total_beam, 1});
  CHECK_EQ(cache.shared_o.device(), query.device());
  CHECK_EQ(cache.unshared_o.device(), query.device());
  CHECK_EQ(cache.shared_lse.device(), query.device());
  CHECK_EQ(cache.unshared_lse.device(), query.device());
  CHECK_EQ(cache.shared_lse_kernel.device(), query.device());
  CHECK_EQ(cache.q_cu_seq_lens_shared.device(), query.device());
  CHECK_EQ(cache.decode_slot_mapping.device(), query.device());
  CHECK_EQ(cache.unshared_seq_lens.device(), query.device());
  CHECK_EQ(attn_metadata.block_table.device(), query.device());
  CHECK_EQ(attn_metadata.kv_cu_seq_lens.device(), query.device());
  CHECK_EQ(attn_metadata.kv_cu_seq_lens.numel(), batch_size + 1);

  const int64_t shared_len = attn_metadata.total_kv_len;
  CHECK_GT(shared_len, 0);
  CHECK_LE(shared_len, attn_metadata.full_k_cache.size(0));
  auto shared_k = attn_metadata.full_k_cache.slice(0, 0, shared_len);
  auto shared_v = attn_metadata.full_v_cache.slice(0, 0, shared_len);
  auto unshared_k = attn_metadata.unshared_k_cache.view(
      {total_beam, num_kv_heads_, max_decode_steps, head_size_});
  auto unshared_v = attn_metadata.unshared_v_cache.view(
      {total_beam, num_kv_heads_, max_decode_steps, head_size_});

  std::optional<torch::Tensor> v = value;
  std::optional<torch::Tensor> v_cache = unshared_v;
  xllm::kernel::mlu::reshape_paged_cache(key,
                                         v,
                                         unshared_k,
                                         v_cache,
                                         cache.decode_slot_mapping,
                                         /*direction=*/false);

  std::optional<torch::Tensor> shared_lse_kernel = cache.shared_lse_kernel;
  xllm::kernel::mlu::batch_prefill(query,
                                   shared_k,
                                   shared_v,
                                   cache.shared_o,
                                   shared_lse_kernel,
                                   cache.q_cu_seq_lens_shared,
                                   attn_metadata.kv_cu_seq_lens,
                                   /*alibi_slope=*/std::nullopt,
                                   /*attn_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   /*out_quant_scale=*/std::nullopt,
                                   /*block_tables=*/std::nullopt,
                                   /*max_query_len=*/beam_width,
                                   attn_metadata.max_seq_len,
                                   scale_,
                                   /*is_causal=*/false,
                                   /*window_size_left=*/-1,
                                   /*window_size_right=*/-1,
                                   /*compute_dtype=*/"float",
                                   /*return_lse=*/true);
  cache.shared_lse.copy_(cache.shared_lse_kernel.transpose(0, 1).unsqueeze(-1));

  auto query_pad = query.view({total_beam, 1, num_heads_, head_size_});
  auto unshared_o_pad =
      cache.unshared_o.view({total_beam, 1, num_heads_, head_size_});
  std::optional<torch::Tensor> unshared_lse = cache.unshared_lse;
  xllm::kernel::mlu::batch_decode(query_pad,
                                  unshared_k,
                                  unshared_o_pad,
                                  attn_metadata.block_table,
                                  cache.unshared_seq_lens,
                                  v_cache,
                                  unshared_lse,
                                  /*q_quant_scale=*/std::nullopt,
                                  /*k_cache_quant_scale=*/std::nullopt,
                                  /*v_cache_quant_scale=*/std::nullopt,
                                  /*out_quant_scale=*/std::nullopt,
                                  /*alibi_slope=*/std::nullopt,
                                  /*mask=*/std::nullopt,
                                  /*compute_dtype=*/"float",
                                  max_decode_steps,
                                  /*window_size_left=*/-1,
                                  /*window_size_right=*/-1,
                                  scale_,
                                  /*return_lse=*/true,
                                  /*kv_cache_quant_bit_size=*/-1);

  output.copy_(cache.shared_o);
  auto output_pad = output.view({total_beam, 1, num_heads_, head_size_});
  xllm::kernel::mlu::update_out_and_lse(output_pad,
                                        cache.shared_lse,
                                        unshared_o_pad,
                                        cache.unshared_lse,
                                        /*seq_offsets=*/std::nullopt,
                                        /*cu_seqs=*/std::nullopt,
                                        /*block_cu_seqs=*/std::nullopt);
}

}  // namespace xllm::layer
