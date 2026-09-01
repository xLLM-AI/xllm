/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cstdint>

#include "framework/sampling/sampling_params.h"

namespace xllm {

// Draws Gumbel(0, 1) noise shaped [batch_size, num_steps, num_classes] on
// `device`. Greedy batches get zero noise so the gumbel_argmax below collapses
// onto the plain argmax; mixed batches zero the greedy rows so their argmax
// stays deterministic, mirroring the sampler's where(do_sample, ...) select.
// The caller owns cross-rank consensus: per-rank RNG divergence would fork the
// sampled path, so under TP > 1 the caller must broadcast the result from a
// single root before feeding it to gumbel_argmax.
torch::Tensor sample_gumbel_noise(int64_t batch_size,
                                  int64_t num_steps,
                                  int64_t num_classes,
                                  const SamplingParameters& sampling_params,
                                  const torch::Device& device);

// Applies temperatures in-place to `logits` [batch, ..., num_classes] with the
// generic sampler's semantics: temperature 0 means greedy and must not divide,
// so it is substituted with 1 before scaling.
void apply_selector_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures,
                                 int64_t batch_size);

// Single-row Gumbel-max sampling over log-probabilities: argmax(log_probs + g)
// draws from exactly categorical(exp(log_probs)), so logits MUST already be
// normalized (e.g. log_softmax output). Returns sampled class indices
// [batch_size] on the logits device. Stays on the current stream and never
// blocks the host, which is what makes it ACL-graph safe.
torch::Tensor gumbel_argmax(const torch::Tensor& log_probs,
                            const torch::Tensor& gumbel_noise);

}  // namespace xllm
