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

#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::cuda {

torch::Tensor random_sample(const torch::Tensor& probs) {
  CHECK(probs.dim() == 2 || probs.dim() == 3)
      << "probs must be a 2D or 3D tensor";

  torch::Tensor flat_probs =
      (probs.dim() == 3) ? probs.reshape({-1, probs.size(2)}) : probs;

  const torch::Device device = flat_probs.device();
  auto p = (flat_probs.scalar_type() == torch::kFloat32)
               ? flat_probs
               : flat_probs.to(torch::kFloat32);

  const int64_t batch_size = p.size(0);
  const int64_t vocab_size = p.size(1);

  auto cdf = p.cumsum(/*dim=*/-1);

  auto u =
      torch::rand({batch_size, 1},
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  u = u * cdf.narrow(/*dim=*/-1, vocab_size - 1, 1);

  auto samples = torch::searchsorted(cdf, u).squeeze(-1).to(torch::kInt64);
  samples = samples.clamp(/*min=*/0, /*max=*/vocab_size - 1);

  if (probs.dim() == 3) {
    return samples.reshape({probs.size(0), probs.size(1)});
  }
  return samples.flatten();
}

}  // namespace xllm::kernel::cuda
