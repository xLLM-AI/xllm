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

  at_npu::native::custom_ops::npu_apply_rotary_pos_emb(q, k, cos, sin);
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
