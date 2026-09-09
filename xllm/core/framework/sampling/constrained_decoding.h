/* Copyright 2025-2026 The xLLM Authors.

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
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace xllm {

// Constrained decoding is used to ensure that the generated content
// conforms to specific formats or rules.
class ConstrainedDecoding {
 public:
  virtual ~ConstrainedDecoding() = default;

  // Precompute and cache fixed constraint masks (e.g., static vocabulary
  // whitelists) to avoid redundant calculations during token generation.
  // Returns: true if cache built successfully, false otherwise
  virtual bool build_mask_cache() = 0;

  // Generate dynamic constraint mask based on already generated token
  // sequences. This mask will be applied to filter invalid tokens.
  //
  // Input: generated_token_list - 2D vector of token IDs, where each inner
  // vector represents the generated tokens for a single sequence in the batch
  // (format:[sequence_num][token_ids])
  // Output: tensor of shape [sequence_num, vocab_size], where 0.0f
  // indicates allowed tokens and a large negative number indicates forbidden
  // tokens for each sequence, the usage is to filter invalid tokens by adding
  // the mask to the model logits.
  virtual torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list) = 0;
};
}  // namespace xllm
