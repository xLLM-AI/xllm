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

#include "core/framework/request/rec_sequence.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

namespace xllm {

RecBeamSearchResult::RecBeamSearchResult(
    int32_t beam_width,
    int32_t total_rounds,
    std::vector<std::vector<int32_t>> beams,
    std::vector<float> last_logprobs)
    : beam_width_(beam_width),
      total_rounds_(total_rounds),
      beams_(std::move(beams)),
      last_logprobs_(std::move(last_logprobs)) {}

RecSequence::RecSequence(size_t index,
                         const std::vector<int32_t>& prompt_token_ids,
                         torch::Tensor input_embedding,
                         const MMData& mm_data,
                         const IncrementalDecoder& incremental_decoder,
                         const SequenceParams& seq_params)
    : Sequence(index,
               prompt_token_ids,
               std::move(input_embedding),
               mm_data,
               incremental_decoder,
               seq_params) {}

RecSequence::RecSequence(size_t index,
                         const DecoderSeed& seed,
                         torch::Tensor input_embedding,
                         const MMData& mm_data,
                         const IncrementalDecoder& incremental_decoder,
                         const SequenceParams& seq_params,
                         bool force_token_logprobs)
    : Sequence(index,
               seed,
               std::move(input_embedding),
               mm_data,
               incremental_decoder,
               seq_params,
               force_token_logprobs) {}

RecSequence::RecSequence(const RecSequence& other)
    : RecSequence(other, other.index()) {}

// beam_search_result_ intentionally starts empty: the forked sequence is a new
// beam candidate and must not report its source's cached beams.
RecSequence::RecSequence(const RecSequence& other, size_t index)
    : Sequence(other, index) {}

RecSequence& RecSequence::from(Sequence& sequence) {
  auto* rec_sequence = dynamic_cast<RecSequence*>(&sequence);
  CHECK(rec_sequence != nullptr)
      << "sequence " << sequence.seq_id() << " is not a REC sequence";
  return *rec_sequence;
}

const RecSequence& RecSequence::from(const Sequence& sequence) {
  const auto* rec_sequence = dynamic_cast<const RecSequence*>(&sequence);
  CHECK(rec_sequence != nullptr)
      << "sequence " << sequence.seq_id() << " is not a REC sequence";
  return *rec_sequence;
}

LlmRecSequence::LlmRecSequence(size_t index,
                               const std::vector<int32_t>& prompt_token_ids,
                               torch::Tensor input_embedding,
                               const MMData& mm_data,
                               const IncrementalDecoder& incremental_decoder,
                               const SequenceParams& seq_params)
    : RecSequence(index,
                  prompt_token_ids,
                  std::move(input_embedding),
                  mm_data,
                  incremental_decoder,
                  seq_params) {}

LlmRecSequence::LlmRecSequence(const LlmRecSequence& other, size_t index)
    : RecSequence(other, index) {}

std::unique_ptr<Sequence> LlmRecSequence::fork(size_t index) const {
  return std::make_unique<LlmRecSequence>(*this, index);
}

}  // namespace xllm
