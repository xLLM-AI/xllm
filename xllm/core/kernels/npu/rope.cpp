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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

void apply_rotary_pos_emb(torch::Tensor& query,
                          torch::Tensor& key,
                          const torch::Tensor& cos,
                          const torch::Tensor& sin) {
  CHECK(cos.defined() && sin.defined());
  CHECK(cos.sizes() == sin.sizes());
  CHECK_GT(cos.dim(), 0);
  const int64_t rotary_dim = cos.size(-1);
  CHECK_GT(rotary_dim, 0);
  const int64_t num_tokens = cos.numel() / rotary_dim;
  CHECK_GT(num_tokens, 0);
  CHECK_EQ(query.numel() % (num_tokens * rotary_dim), 0);
  CHECK_EQ(key.numel() % (num_tokens * rotary_dim), 0);

  const std::vector<int64_t> query_shape = query.sizes().vec();
  const std::vector<int64_t> key_shape = key.sizes().vec();
  const int64_t num_query_heads = query.numel() / (num_tokens * rotary_dim);
  const int64_t num_key_heads = key.numel() / (num_tokens * rotary_dim);

  torch::Tensor query_view =
      query.contiguous().view({1, num_tokens, num_query_heads, rotary_dim});
  torch::Tensor key_view =
      key.contiguous().view({1, num_tokens, num_key_heads, rotary_dim});
  torch::Tensor cos_view =
      cos.contiguous().view({1, num_tokens, 1, rotary_dim});
  torch::Tensor sin_view =
      sin.contiguous().view({1, num_tokens, 1, rotary_dim});

  auto rotary_result = at_npu::native::custom_ops::npu_apply_rotary_pos_emb(
      query_view, key_view, cos_view, sin_view, "BSND");
  query = std::get<0>(rotary_result).view(query_shape);
  key = std::get<1>(rotary_result).view(key_shape);
}

void apply_rotary_pos_emb_atb(torch::Tensor& query,
                              torch::Tensor& key,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin) {
  const int64_t rotary_dim = query.size(-1) / 2;
  auto cos_input = cos.size(-1) == query.size(-1)
                       ? cos.slice(/*dim=*/-1, /*start=*/0, /*end=*/rotary_dim)
                       : cos;
  auto sin_input = sin.size(-1) == query.size(-1)
                       ? sin.slice(/*dim=*/-1, /*start=*/0, /*end=*/rotary_dim)
                       : sin;
  atb::npu_rotary_embedding_with_cos_sin(cos_input.contiguous(),
                                         sin_input.contiguous(),
                                         query,
                                         key,
                                         /*is_neox_style=*/true);
}

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& positions) {
  // FIXME: This computation of 'cos' and 'sin' should only be performed
  // for the first layer (or if the cache is empty). For subsequent layers,
  // the calculated 'cos' and 'sin' values from the first layer should be
  // reused/cached to avoid redundant computation.
  auto cos_sin = cos_sin_cache.index_select(0, positions);
  int64_t last_dim = cos_sin.size(-1);

  const int64_t rotary_dim = last_dim / 2;
  auto cos_sin_split = cos_sin.chunk(2, /*dim=*/-1);
  // Ensure tensors are contiguous for NPU operations
  auto cos = cos_sin_split[0].contiguous().view({1, -1, 1, rotary_dim});
  auto sin = cos_sin_split[1].contiguous().view({1, -1, 1, rotary_dim});

  q = q.view({1, q.size(0), -1, rotary_dim});
  k = k.view({1, k.size(0), -1, rotary_dim});

  apply_rotary_pos_emb(q, k, cos, sin);
}

std::pair<torch::Tensor, torch::Tensor> apply_npu_partial_rotary_embedding(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    int64_t rotary_dim,
    const torch::Tensor& cos_sin_cache,
    bool is_neox_style) {
  torch::IntArrayRef query_shape = query.sizes();
  torch::IntArrayRef key_shape = key.sizes();

  int64_t num_tokens = query.size(0);

  torch::Tensor query_reshaped = query.view({num_tokens, -1, head_size});
  torch::Tensor key_reshaped = key.view({num_tokens, -1, head_size});

  torch::Tensor q_rot = query_reshaped.slice(-1, 0, rotary_dim);
  torch::Tensor q_pass = query_reshaped.slice(-1, rotary_dim, head_size);
  torch::Tensor k_rot = key_reshaped.slice(-1, 0, rotary_dim);
  torch::Tensor k_pass = key_reshaped.slice(-1, rotary_dim, head_size);

  torch::Tensor q_rot_contig = q_rot.contiguous().view({num_tokens, -1});
  torch::Tensor k_rot_contig = k_rot.contiguous().view({num_tokens, -1});
  atb::npu_rotary_embedding(positions,
                            q_rot_contig,
                            k_rot_contig,
                            head_size,
                            cos_sin_cache,
                            is_neox_style);
  torch::Tensor q_rot_3d = q_rot_contig.view({num_tokens, -1, rotary_dim});
  torch::Tensor k_rot_3d = k_rot_contig.view({num_tokens, -1, rotary_dim});

  torch::Tensor q_concat = at::cat({q_rot_3d, q_pass}, -1);
  torch::Tensor q_final = q_concat.reshape(query_shape);

  torch::Tensor k_concat = at::cat({k_rot_3d, k_pass}, -1);
  torch::Tensor k_final = k_concat.reshape(key_shape);

  return {q_final, k_final};
}

}  // namespace xllm::kernel::npu
