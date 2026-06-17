/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "common/macros.h"

namespace xllm {

class EmbeddingCache final {
 public:
  // Per-request decode state from the last target prefill/validate output. The
  // same token_id and position_offset drive both schedule-overlap correction
  // and next-step draft extend input construction.
  struct DecodeState {
    // False only for storage entries that have not received target output yet.
    // Public read APIs materialize these entries into fake decode states.
    bool valid = false;
    std::string request_id;

    // True only when accepted_len == num_speculative_tokens + 1. Non-DP draft
    // extend may use this to decide whether two rows are necessary.
    bool all_draft_accepted = false;

    // Last accepted target token and its correction offset relative to the
    // scheduler-provided decode row. All model input builders should derive the
    // current real position from this offset.
    int32_t token_id = -1;
    int32_t position_offset = 0;
    // Store detached views into target outputs instead of cloning. The views
    // keep their storage alive until replaced, trading short-lived HBM
    // retention for avoiding per-step embedding copies.
    torch::Tensor embedding;

    // Previous accepted target token. Its position is derived as
    // position_offset - 1 when a 2-row draft extend is required.
    int32_t prev_token_id = -1;
    torch::Tensor prev_embedding;

    int32_t correction_token_id = 0;  // accepted token for step correction
    int32_t correction_position_offset = 0;
  };

  EmbeddingCache(int32_t total_nums);

  ~EmbeddingCache() = default;

  // disable copy, move and assign
  DISALLOW_COPY_AND_ASSIGN(EmbeddingCache);

  // Writes target prefill output after target model generates the first token.
  // Draft prefill output is intentionally ignored and must not be written here.
  void write_prefill_target_context(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      const torch::Tensor& next_tokens,
      const torch::Tensor& embeddings,
      const torch::Tensor& selected_token_idxes = torch::Tensor());

  // Writes PD handoff bootstrap target context for the first MTP decode step.
  void write_mtp_bootstrap_context(int32_t embedding_id,
                                   const std::string& request_id,
                                   int32_t token_id,
                                   const torch::Tensor& embedding);

  // Writes target validate output after rejection sampling. accepted_tokens is
  // a contiguous accepted prefix padded by -1; accepted_embeddings keeps the
  // corresponding target hidden states for the next draft extend input.
  void write_target_context(const std::vector<int32_t>& embedding_ids,
                            const std::vector<std::string>& request_ids,
                            const torch::Tensor& accepted_tokens,
                            const torch::Tensor& accepted_embeddings,
                            int32_t num_speculative_tokens);

  // Algorithm-specific placeholder embedding for missing target context, e.g.
  // PD first decode. MTP uses hidden_size; Eagle3 uses 3 * hidden_size.
  void set_placeholder(const torch::Tensor& embedding_placeholder);
  const torch::Tensor& embedding_placeholder() const;

  // Non-failing read used by PD first decode. Missing entries are materialized
  // as fake target states so workers can follow the normal decode path.
  std::vector<DecodeState> read_decode_states(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  std::vector<int32_t> read_accepted_prefix_lengths(
      const std::vector<int32_t>& embedding_ids) const;

  void clear(const std::vector<int32_t>& embedding_ids);

 private:
  std::vector<DecodeState> decode_tails_;
  torch::Tensor embedding_placeholder_;

  DecodeState& mutable_tail(int32_t embedding_id);
  const DecodeState& get_tail(int32_t embedding_id) const;
};

}  // namespace xllm
