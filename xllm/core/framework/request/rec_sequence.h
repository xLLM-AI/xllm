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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"

namespace xllm {

// Result of a multi-round beam search run on the device (written by the OneRec
// x-attention and LlmRec multi-round pipelines, read when the request output
// is produced). Default-constructed it is empty (not ready).
class RecBeamSearchResult {
 public:
  RecBeamSearchResult() = default;

  // `beams` has `beam_width` rows of total_rounds * REC_TOKEN_SIZE tokens
  // each; `last_logprobs` has one entry per row (may be empty when the kernel
  // did not report them).
  RecBeamSearchResult(int32_t beam_width,
                      int32_t total_rounds,
                      std::vector<std::vector<int32_t>> beams,
                      std::vector<float> last_logprobs);

  bool ready() const {
    return beam_width_ > 0 && total_rounds_ > 0 && !beams_.empty();
  }

  int32_t beam_width() const { return beam_width_; }
  int32_t total_rounds() const { return total_rounds_; }
  const std::vector<std::vector<int32_t>>& beams() const { return beams_; }
  const std::vector<float>& last_logprobs() const { return last_logprobs_; }

 private:
  int32_t beam_width_ = 0;
  int32_t total_rounds_ = 0;
  std::vector<std::vector<int32_t>> beams_;
  std::vector<float> last_logprobs_;
};

// Common base of the recommendation sequences: carries the device-side
// multi-round beam result. A fork starts with an empty result; the beam
// pipelines re-populate it for the surviving sequence.
class RecSequence : public Sequence {
 public:
  RecSequence(size_t index,
              const std::vector<int32_t>& prompt_token_ids,
              torch::Tensor input_embedding,
              const MMData& mm_data,
              const IncrementalDecoder& incremental_decoder,
              const SequenceParams& seq_params);

  RecSequence(const RecSequence& other);
  RecSequence(const RecSequence& other, size_t index);

  // Typed access at REC-only entry points; CHECK-fails on a non-REC sequence.
  static RecSequence& from(Sequence& sequence);
  static const RecSequence& from(const Sequence& sequence);

  const RecBeamSearchResult& beam_search_result() const {
    return beam_search_result_;
  }
  void set_beam_search_result(RecBeamSearchResult result) {
    beam_search_result_ = std::move(result);
  }

 protected:
  // BOS-seeded decoder, for derived types that do not consume the prompt as
  // decoder tokens (OneRec).
  RecSequence(size_t index,
              const DecoderSeed& seed,
              torch::Tensor input_embedding,
              const MMData& mm_data,
              const IncrementalDecoder& incremental_decoder,
              const SequenceParams& seq_params,
              bool force_token_logprobs);

 private:
  RecBeamSearchResult beam_search_result_;
};

// Decoder-only recommendation model driven like an LLM: prompt-seeded decoder,
// standard detokenizing output.
class LlmRecSequence final : public RecSequence {
 public:
  LlmRecSequence(size_t index,
                 const std::vector<int32_t>& prompt_token_ids,
                 torch::Tensor input_embedding,
                 const MMData& mm_data,
                 const IncrementalDecoder& incremental_decoder,
                 const SequenceParams& seq_params);

  LlmRecSequence(const LlmRecSequence& other, size_t index);

  std::unique_ptr<Sequence> fork(size_t index) const override;
};

}  // namespace xllm
