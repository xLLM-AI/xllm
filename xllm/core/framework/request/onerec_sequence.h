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
#include <optional>
#include <string>
#include <vector>

#include "core/framework/request/rec_sequence.h"

namespace xllm {

// OneRec encoder/decoder sequence. The prompt (token ids or a sparse embedding
// carried in mm_data) feeds the encoder; the decoder is seeded with a BOS token
// or with pre-computed context embeddings and generates REC token triples. The
// output carries raw generated ids (optionally with per-token SKU logprobs and
// the decoded item ids) instead of detokenized text.
class OneRecSequence final : public RecSequence {
 public:
  // Keys under which OneRec inputs travel in MMData.
  static const std::string kEncoderSparseEmbeddingName;
  static const std::string kDecoderContextEmbeddingName;

  OneRecSequence(size_t index,
                 const std::vector<int32_t>& prompt_token_ids,
                 torch::Tensor input_embedding,
                 const MMData& mm_data,
                 const IncrementalDecoder& incremental_decoder,
                 const SequenceParams& seq_params);

  OneRecSequence(const OneRecSequence& other, size_t index);

  // Typed access at OneRec-only entry points; CHECK-fails otherwise.
  static OneRecSequence& from(Sequence& sequence);
  static const OneRecSequence& from(const Sequence& sequence);

  std::unique_ptr<Sequence> fork(size_t index) const override;

  // Encoder input: the prompt token ids, or empty when the encoder input is the
  // sparse embedding.
  const std::vector<int32_t>& encoder_tokens() const { return encoder_tokens_; }
  // Encoder sequence length: token count or sparse embedding rows.
  size_t encoder_seq_len() const { return num_encoder_tokens_; }
  // Rows of decoder context embedding that precede the generated tokens; 0 when
  // the decoder is seeded with BOS instead.
  size_t num_decoder_embeddings() const { return num_decoder_embeddings_; }

  using Sequence::generate_output;
  std::optional<SequenceOutput> generate_streaming_output(
      size_t size,
      const Tokenizer& tokenizer) override;
  SequenceOutput generate_output(const Tokenizer& tokenizer) override;

 protected:
  // OneRec appends to the decoder before any KV cache is filled and cannot
  // finish before it generated anything.
  bool allows_append_before_prefill() const override { return true; }
  bool requires_generated_token_to_finish() const override { return true; }

 private:
  // Encoder/decoder layout derived from the prompt and mm_data before the
  // base is constructed.
  struct Layout {
    size_t num_encoder_tokens = 0;
    size_t num_decoder_embeddings = 0;
    std::vector<int32_t> encoder_tokens;
    DecoderSeed decoder_seed;
  };
  static Layout derive_layout(const std::vector<int32_t>& prompt_token_ids,
                              const MMData& mm_data);

  OneRecSequence(size_t index,
                 Layout layout,
                 torch::Tensor input_embedding,
                 const MMData& mm_data,
                 const IncrementalDecoder& incremental_decoder,
                 const SequenceParams& seq_params);

  Slice<int32_t> generated_ids() const;

  size_t num_encoder_tokens_ = 0;
  size_t num_decoder_embeddings_ = 0;
  std::vector<int32_t> encoder_tokens_;
};

}  // namespace xllm
