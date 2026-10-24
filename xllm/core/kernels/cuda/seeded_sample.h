/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/types.h>

namespace xllm::kernel::cuda {

// Finite FP32 nonnegative probabilities [batch, vocab], with contiguous
// vocabulary and positive total mass on each random row.
// Contiguous per-row seeds/offsets are int64, on the same CUDA device.
// do_sample is a contiguous bool vector for mixed/greedy rows, or undefined
// when every row is random. Seed -1 denotes an unseeded row.
// has_unseeded_sample is derived on the host when building the batch; it must
// be true if any random row has seed -1. Fully seeded calls leave the default
// generator unchanged.
// A seed/offset pair is independent of batch composition. This categorical
// algorithm intentionally does not reproduce torch::multinomial's sequence.
// CUDA Graph capture records the host flag; changing it requires recapture.
torch::Tensor seeded_sample(const torch::Tensor& probs,
                            const torch::Tensor& seeds,
                            const torch::Tensor& offsets,
                            const torch::Tensor& do_sample,
                            bool has_unseeded_sample);

}  // namespace xllm::kernel::cuda
