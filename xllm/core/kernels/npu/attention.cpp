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

#include <ATen/ops/scaled_dot_product_attention.h>
#include <glog/logging.h>

#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace {

void ascend950_paged_attention(const torch::Tensor& query,
                               const torch::Tensor& k_cache,
                               const torch::Tensor& v_cache,
                               float scale,
                               const torch::Tensor& block_table,
                               const torch::Tensor& seq_lens,
                               torch::Tensor& output) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_size = query.size(2);
  const int64_t block_size = k_cache.size(1);
  const int64_t num_kv_heads = k_cache.size(2);
  CHECK_EQ(seq_lens.numel(), batch_size);
  CHECK_EQ(block_table.size(0), batch_size);

  const torch::Tensor seq_lens_cpu =
      seq_lens.device().is_cpu() ? seq_lens : seq_lens.to(torch::kCPU);
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    const int64_t seq_len = seq_lens_cpu[batch_index].item<int64_t>();
    CHECK_GT(seq_len, 0);
    const int64_t num_blocks = (seq_len + block_size - 1) / block_size;
    torch::Tensor block_indices =
        block_table[batch_index].narrow(0, 0, num_blocks);
    block_indices = block_indices.to(k_cache.options().dtype(torch::kInt64));

    torch::Tensor key =
        k_cache.index_select(0, block_indices)
            .reshape({num_blocks * block_size, num_kv_heads, head_size})
            .narrow(0, 0, seq_len);
    torch::Tensor value =
        v_cache.index_select(0, block_indices)
            .reshape({num_blocks * block_size, num_kv_heads, head_size})
            .narrow(0, 0, seq_len);
    key = xllm::kernel::npu::expand_kv_heads(key, num_heads, num_kv_heads);
    value = xllm::kernel::npu::expand_kv_heads(value, num_heads, num_kv_heads);

    const torch::Tensor query_4d = query[batch_index].unsqueeze(0).unsqueeze(2);
    const torch::Tensor key_4d = key.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor value_4d = value.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor sequence_output =
        torch::scaled_dot_product_attention(
            query_4d,
            key_4d,
            value_4d,
            /*attn_mask=*/std::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/false,
            /*scale=*/static_cast<double>(scale))
            .squeeze(0)
            .squeeze(1);
    output[batch_index].copy_(sequence_output);
  }
}

}  // namespace

namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping) {
  CHECK(value.has_value()) << "NPU reshape_paged_cache requires value.";
  CHECK(v_cache.has_value()) << "NPU reshape_paged_cache requires v_cache.";
  if (is_ascend950()) {
    reshape_and_cache_a5(
        key, value.value(), k_cache, v_cache.value(), slot_mapping);
    return;
  }
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
  if (is_ascend950()) {
    ascend950_paged_attention(
        q, k_cache, v_cache, scale, block_table, seq_lens, o);
    return;
  }
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
