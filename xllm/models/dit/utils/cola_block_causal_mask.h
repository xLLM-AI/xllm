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

#pragma once

#include <torch/torch.h>

#include <limits>
#include <vector>

namespace xllm {

// Creates an additive block-causal attention mask for the NA (no-padding)
// flatten-concat layout. Returns (1, 1, L_q_total, L_k_total) with
// 0 at allowed positions and dtype::min elsewhere.
inline torch::Tensor create_block_causal_mask(
    const std::vector<int64_t>& k_lens,
    const std::vector<int64_t>& q_lens,
    int64_t block_size,
    torch::ScalarType dtype,
    torch::Device device) {
  int64_t B = k_lens.size();
  int64_t L_k = 0;
  int64_t L_q = 0;
  for (int64_t i = 0; i < B; ++i) {
    L_k += k_lens[i];
    L_q += q_lens[i];
  }

  // Compute cumulative sums
  std::vector<int64_t> k_cu(B + 1, 0);
  std::vector<int64_t> q_cu(B + 1, 0);
  for (int64_t i = 0; i < B; ++i) {
    k_cu[i + 1] = k_cu[i] + k_lens[i];
    q_cu[i + 1] = q_cu[i] + q_lens[i];
  }

  // Build index tensors
  auto k_sample = torch::zeros({L_k}, torch::kLong);
  auto k_local = torch::zeros({L_k}, torch::kLong);
  auto q_sample = torch::zeros({L_q}, torch::kLong);
  auto q_local = torch::zeros({L_q}, torch::kLong);

  for (int64_t b = 0; b < B; ++b) {
    int64_t k_len_b = k_lens[b];
    int64_t q_len_b = q_lens[b];
    if (k_len_b > 0) {
      k_sample.narrow(0, k_cu[b], k_len_b).fill_(b);
      // Use copy_() to in-place fill the view — plain = rebinds the local var.
      k_local.narrow(0, k_cu[b], k_len_b).copy_(torch::arange(k_len_b));
    }
    if (q_len_b > 0) {
      q_sample.narrow(0, q_cu[b], q_len_b).fill_(b);
      // Q refers to the LAST q_len_b positions of K within the same sample.
      q_local.narrow(0, q_cu[b], q_len_b)
          .copy_(torch::arange(k_len_b - q_len_b, k_len_b));
    }
  }

  // Integer block index — MUST use truncating division (//), not float
  // division. ``q_local / block_size`` on int64 promotes to float and wrongly
  // applies position-wise causality inside a block; official
  // create_na_block_causal_mask uses ``txt_local // block_size``.
  auto q_block = torch::div(q_local.unsqueeze(1), block_size, "trunc");
  auto k_block = torch::div(k_local.unsqueeze(0), block_size, "trunc");
  auto same_sample = q_sample.unsqueeze(1) == k_sample.unsqueeze(0);
  auto block_causal = q_block >= k_block;
  auto allowed = same_sample & block_causal;

  // Build additive mask in fp32, then cast to target dtype (matches official
  // torch.finfo(dtype).min for disallowed positions).
  auto mask =
      torch::full({L_q, L_k},
                  std::numeric_limits<float>::lowest(),
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  mask.masked_fill_(allowed.to(device), 0.0f);
  return mask.to(dtype).unsqueeze(0).unsqueeze(0);
}

}  // namespace xllm
