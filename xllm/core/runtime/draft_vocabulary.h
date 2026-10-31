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

#include <cstdint>

namespace xllm {

class SampleOutput;

// Checkpoint d2t stores offsets: target_token_ids[draft_id] = draft_id +
// d2t[draft_id]. An undefined mapping keeps the shared vocabulary unchanged.
struct DraftVocabulary {
  DraftVocabulary(torch::Tensor d2t,
                  int64_t draft_vocab_size,
                  int64_t target_vocab_size,
                  const torch::Device& device);
  DraftVocabulary() = default;

  torch::Tensor target_token_ids;
  int64_t target_vocab_size = 0;

  void remap(SampleOutput& sample_output,
             torch::Tensor target_logits = torch::Tensor()) const;
};

}  // namespace xllm
