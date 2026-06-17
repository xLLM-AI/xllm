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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void reshape_paged_cache(torch::Tensor& key,
                         const std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         const std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction) {
  tmo::torch_api::reshape_paged_cache(
      key, value, k_cache, v_cache, slot_mapping, direction);
}

void reshape_from_cache(torch::Tensor& key,
                        const std::optional<torch::Tensor>& value,
                        const torch::Tensor& key_cache,
                        const std::optional<torch::Tensor>& value_cache,
                        const torch::Tensor& context_lengths,
                        const int64_t max_context_len,
                        const std::optional<torch::Tensor>& context_seq_offset,
                        const std::optional<torch::Tensor>& block_tables,
                        const std::optional<torch::Tensor>& cache_seq_offset) {
  tmo::torch_api::reshape_from_cache(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     context_lengths,
                                     max_context_len,
                                     context_seq_offset,
                                     block_tables,
                                     cache_seq_offset);
}

void quant_to_paged_cache(const torch::Tensor& k,
                          const std::optional<torch::Tensor>& v,
                          torch::Tensor& k_cache,
                          const std::optional<torch::Tensor>& v_cache,
                          torch::Tensor& k_cache_scale,
                          const std::optional<torch::Tensor>& v_cache_scale,
                          const torch::Tensor& slot_mapping) {
  tmo::torch_api::quant_to_paged_cache(
      k, v, k_cache, v_cache, k_cache_scale, v_cache_scale, slot_mapping);
}

void dequant_from_paged_cache(
    torch::Tensor& key,
    const std::optional<torch::Tensor>& value,
    const torch::Tensor& key_cache,
    const std::optional<torch::Tensor>& value_cache,
    const torch::Tensor& key_cache_quant_scale,
    const std::optional<torch::Tensor>& value_cache_quant_scale,
    const torch::Tensor& context_lengths,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& context_seq_offset,
    const torch::Tensor& block_tables,
    int64_t quant_mode,
    int64_t quant_bit) {
  tmo::torch_api::dequant_from_paged_cache(key,
                                           value,
                                           key_cache,
                                           value_cache,
                                           key_cache_quant_scale,
                                           value_cache_quant_scale,
                                           context_lengths,
                                           max_context_len,
                                           context_seq_offset,
                                           block_tables,
                                           quant_mode,
                                           quant_bit);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_table,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse) {
  tmo::torch_api::flash_attention(query,
                                  key,
                                  value,
                                  output,
                                  output_lse,
                                  q_cu_seq_lens,
                                  kv_cu_seq_lens,
                                  alibi_slope,
                                  attn_bias,
                                  q_quant_scale,
                                  k_quant_scale,
                                  v_quant_scale,
                                  out_quant_scale,
                                  block_table,
                                  max_query_len,
                                  max_seq_len,
                                  scale,
                                  is_causal,
                                  window_size_left,
                                  window_size_right,
                                  compute_dtype,
                                  return_lse);
}

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  int64_t kv_cache_quant_bit_size,
                  const std::optional<torch::Tensor>& cu_seq_q,
                  int64_t max_seq_q,
                  const std::optional<torch::Tensor>& sink) {
  tmo::torch_api::single_query_cached_kv_attn(query,
                                              k_cache,
                                              output,
                                              block_table,
                                              seq_lens,
                                              v_cache,
                                              output_lse,
                                              q_quant_scale,
                                              k_cache_quant_scale,
                                              v_cache_quant_scale,
                                              out_quant_scale,
                                              alibi_slope,
                                              mask,
                                              compute_dtype,
                                              max_seq_len,
                                              window_size_left,
                                              window_size_right,
                                              scale,
                                              return_lse,
                                              kv_cache_quant_bit_size,
                                              cu_seq_q,
                                              max_seq_q,
                                              sink);
}

void update_out_and_lse(torch::Tensor& out,
                        torch::Tensor& lse,
                        const torch::Tensor& block_out,
                        const torch::Tensor& block_lse,
                        const std::optional<torch::Tensor>& seq_offsets,
                        const std::optional<torch::Tensor>& cu_seqs,
                        const std::optional<torch::Tensor>& block_cu_seqs) {
  tmo::torch_api::update_out_and_lse(
      out, lse, block_out, block_lse, seq_offsets, cu_seqs, block_cu_seqs);
}

void masked_indexer_select_paged_kv(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const torch::Tensor& weights,
    const torch::Tensor& kv_cache_block_table,
    const std::optional<torch::Tensor>& cu_seq_q_lens,
    const std::optional<torch::Tensor>& cu_seq_k_lens,
    const std::optional<torch::Tensor>& k_context_lens,
    const std::optional<torch::Tensor>& k_cache_block_table,
    const bool is_prefill,
    const int64_t index_topk,
    const int64_t kv_cache_block_size,
    const double softmax_scale,
    const std::optional<torch::Tensor>& q_scale,
    const std::optional<torch::Tensor>& k_scale_cache,
    const torch::Tensor& sparse_block_table,
    const torch::Tensor& sparse_context_lens,
    bool is_score_float,
    int64_t compress_ratio,
    const std::optional<torch::Tensor>& kv_cache_block_table_offset) {
  // add one redundant dimension for future extension
  torch::Tensor weights_extended = weights.unsqueeze(-1);
  tmo::torch_api::masked_indexer_select_paged_kv(query,
                                                 k_cache,
                                                 weights_extended,
                                                 kv_cache_block_table,
                                                 cu_seq_q_lens,
                                                 cu_seq_k_lens,
                                                 k_context_lens,
                                                 k_cache_block_table,
                                                 is_prefill,
                                                 index_topk,
                                                 kv_cache_block_size,
                                                 softmax_scale,
                                                 q_scale,
                                                 k_scale_cache,
                                                 sparse_block_table,
                                                 sparse_context_lens,
                                                 is_score_float,
                                                 compress_ratio,
                                                 kv_cache_block_table_offset);
}

}  // namespace xllm::kernel::mlu
