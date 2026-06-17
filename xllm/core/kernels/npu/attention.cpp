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

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping) {
  atb::npu_reshape_and_cache(
      key, value.value(), k_cache, v_cache.value(), slot_mapping);
}

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   torch::Tensor& output) {
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = key.size(-2);
  atb::npu_flash_attention(
      query, key, value, mask, seq_len, scale, num_heads, num_kv_heads, output);
}

void batch_chunked_paged_prefill(const torch::Tensor& query,
                                 const torch::Tensor& k_cache,
                                 const torch::Tensor& v_cache,
                                 float scale,
                                 const torch::Tensor& block_table,
                                 const torch::Tensor& seq_lens,
                                 const torch::Tensor& attn_mask,
                                 const torch::Tensor& q_seq_lens,
                                 torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});
  atb::npu_chunked_paged_attention(q,
                                   k_cache,
                                   v_cache,
                                   num_kv_heads,
                                   num_heads,
                                   scale,
                                   block_table,
                                   seq_lens,
                                   attn_mask,
                                   q_seq_lens,
                                   o);
}

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});
  atb::npu_paged_attention(q,
                           k_cache,
                           v_cache,
                           num_kv_heads,
                           num_heads,
                           scale,
                           block_table,
                           seq_lens,
                           o);
}

void batch_decode_acl_graph(const torch::Tensor& query,
                            const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache,
                            float scale,
                            const torch::Tensor& block_table,
                            const torch::Tensor& seq_lens,
                            const torch::Tensor& tiling_data,
                            torch::Tensor& output) {
  int64_t head_size = query.size(-1);
  int64_t num_heads = query.size(-2);
  int64_t num_kv_heads = k_cache.size(-2);
  auto q = query.view({-1, num_heads, head_size});
  auto o = output.view({-1, num_heads, head_size});
  atb::npu_custom_paged_attention(q,
                                  k_cache,
                                  v_cache,
                                  num_kv_heads,
                                  num_heads,
                                  scale,
                                  block_table,
                                  seq_lens,
                                  tiling_data,
                                  o);
}

}  // namespace xllm::kernel::npu
