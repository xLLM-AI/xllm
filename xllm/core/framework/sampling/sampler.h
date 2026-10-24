/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

SampleOutput sample(torch::Tensor& logits,
                    const SamplingParameters& params,
                    const torch::Tensor& filter_mask = torch::Tensor());

// Gumbel primitives: argmax(logits + g), g ~ Gumbel(0, 1), samples exactly
// categorical(softmax(logits)) without materializing probs. Noise buffers
// carry log(Exp(1)) draws; greedy rows carry the 0.0 sentinel (log(0) = -inf,
// so the raw buffer stays zero-free).

torch::Tensor gumbel_sample(const torch::Tensor& logits);

// The 0.0 greedy-row sentinel is already placed on mixed batches.
torch::Tensor gumbel_noise(const SamplingParameters& params,
                           torch::IntArrayRef sizes,
                           const torch::Device& device);

// Zero-noise argmax over the last dim; accepts -inf masks.
torch::Tensor greedy_sample(const torch::Tensor& logits);

// Argmax over pre-materialized noise, consuming the buffer as scratch.
torch::Tensor gumbel_argmax(const torch::Tensor& logits,
                            torch::Tensor gumbel_noise);

}  // namespace xllm
