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

#include "core/framework/request/onerec_sequence.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <random>
#include <unordered_set>
#include <utility>

#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/tokenizer/rec_tokenizer.h"

namespace xllm {

namespace {
constexpr size_t kDecoderBosTokenCount = 1;
constexpr size_t kDecoderMaxTokenCount = kRecTotalSteps + kDecoderBosTokenCount;

// Shuffle seed for capping the converted item list: deterministic per sequence
// when a random seed is configured, otherwise random.
uint32_t item_shuffle_seed(size_t sequence_index) {
  const int32_t random_seed = ExecutionConfig::get_instance().random_seed();
  if (random_seed >= 0) {
    return static_cast<uint32_t>(random_seed) +
           static_cast<uint32_t>(sequence_index);
  }
  return std::random_device{}();
}

std::vector<int64_t> normalize_rec_item_ids(const std::vector<int64_t>& raw_ids,
                                            size_t sequence_index) {
  std::vector<int64_t> item_ids;
  item_ids.reserve(raw_ids.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const int64_t item_id : raw_ids) {
    if (seen_item_ids.insert(item_id).second) {
      item_ids.emplace_back(item_id);
    }
  }

  const int32_t each_threshold =
      RecConfig::get_instance().each_conversion_threshold();
  if (each_threshold > 0 &&
      static_cast<int32_t>(item_ids.size()) > each_threshold) {
    std::mt19937 generator(item_shuffle_seed(sequence_index));
    std::shuffle(item_ids.begin(), item_ids.end(), generator);
    item_ids.resize(each_threshold);
  }

  return item_ids;
}

std::vector<RecItemInfo> normalize_rec_item_infos(
    const std::vector<RecItemInfo>& raw_item_infos,
    size_t sequence_index) {
  std::vector<RecItemInfo> item_infos;
  item_infos.reserve(raw_item_infos.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const RecItemInfo& item_info : raw_item_infos) {
    if (seen_item_ids.insert(item_info.item_id).second) {
      item_infos.emplace_back(item_info);
    }
  }

  const int32_t each_threshold =
      RecConfig::get_instance().each_conversion_threshold();
  if (each_threshold > 0 &&
      static_cast<int32_t>(item_infos.size()) > each_threshold) {
    std::mt19937 generator(item_shuffle_seed(sequence_index));
    std::shuffle(item_infos.begin(), item_infos.end(), generator);
    item_infos.resize(each_threshold);
  }

  return item_infos;
}
}  // namespace

const std::string OneRecSequence::kEncoderSparseEmbeddingName =
    "sparse_embedding";
const std::string OneRecSequence::kDecoderContextEmbeddingName =
    "decoder_context_embedding";

OneRecSequence::Layout OneRecSequence::derive_layout(
    const std::vector<int32_t>& prompt_token_ids,
    const MMData& mm_data) {
  Layout layout;
  if (!prompt_token_ids.empty()) {
    layout.encoder_tokens = prompt_token_ids;
    layout.num_encoder_tokens = prompt_token_ids.size();
  } else {
    const auto encoder_sparse_embedding =
        mm_data.get<torch::Tensor>(kEncoderSparseEmbeddingName);
    CHECK(encoder_sparse_embedding.has_value())
        << "encoder sparse embedding not found in mm_data";
    layout.num_encoder_tokens = encoder_sparse_embedding.value().size(0);
  }

  const auto decoder_context_embedding =
      mm_data.get<torch::Tensor>(kDecoderContextEmbeddingName);
  if (decoder_context_embedding.has_value()) {
    // The context embeddings play the role of the decoder prompt; the token
    // buffer only needs room for the generated triples.
    layout.num_decoder_embeddings = decoder_context_embedding.value().size(0);
    layout.decoder_seed.num_bos_tokens = 0;
    layout.decoder_seed.capacity = layout.num_decoder_embeddings +
                                   kDecoderMaxTokenCount -
                                   kDecoderBosTokenCount;
  } else {
    layout.decoder_seed.num_bos_tokens = kDecoderBosTokenCount;
    layout.decoder_seed.capacity = kDecoderMaxTokenCount;
  }
  return layout;
}

OneRecSequence::OneRecSequence(size_t index,
                               const std::vector<int32_t>& prompt_token_ids,
                               torch::Tensor input_embedding,
                               const MMData& mm_data,
                               const IncrementalDecoder& incremental_decoder,
                               const SequenceParams& seq_params)
    : OneRecSequence(index,
                     derive_layout(prompt_token_ids, mm_data),
                     std::move(input_embedding),
                     mm_data,
                     incremental_decoder,
                     seq_params) {}

// OneRec emits per-token logprobs in its output when SKU logprobs are enabled,
// independent of the request's sampling flags, so the buffer is forced on.
OneRecSequence::OneRecSequence(size_t index,
                               Layout layout,
                               torch::Tensor input_embedding,
                               const MMData& mm_data,
                               const IncrementalDecoder& incremental_decoder,
                               const SequenceParams& seq_params)
    : RecSequence(index,
                  layout.decoder_seed,
                  std::move(input_embedding),
                  mm_data,
                  incremental_decoder,
                  seq_params,
                  RecConfig::get_instance().enable_output_sku_logprobs()),
      num_encoder_tokens_(layout.num_encoder_tokens),
      num_decoder_embeddings_(layout.num_decoder_embeddings),
      encoder_tokens_(std::move(layout.encoder_tokens)) {}

OneRecSequence::OneRecSequence(const OneRecSequence& other, size_t index)
    : RecSequence(other, index),
      num_encoder_tokens_(other.num_encoder_tokens_),
      num_decoder_embeddings_(other.num_decoder_embeddings_),
      encoder_tokens_(other.encoder_tokens_) {}

OneRecSequence& OneRecSequence::from(Sequence& sequence) {
  auto* onerec_sequence = dynamic_cast<OneRecSequence*>(&sequence);
  CHECK(onerec_sequence != nullptr)
      << "sequence " << sequence.seq_id() << " is not a OneRec sequence";
  return *onerec_sequence;
}

const OneRecSequence& OneRecSequence::from(const Sequence& sequence) {
  const auto* onerec_sequence = dynamic_cast<const OneRecSequence*>(&sequence);
  CHECK(onerec_sequence != nullptr)
      << "sequence " << sequence.seq_id() << " is not a OneRec sequence";
  return *onerec_sequence;
}

std::unique_ptr<Sequence> OneRecSequence::fork(size_t index) const {
  return std::make_unique<OneRecSequence>(*this, index);
}

Slice<int32_t> OneRecSequence::generated_ids() const {
  return tokens().slice(num_prompt_tokens(), num_valid_tokens());
}

std::optional<SequenceOutput> OneRecSequence::generate_streaming_output(
    size_t /*size*/,
    const Tokenizer& /*tokenizer*/) {
  AUTO_COUNTER(detokenization_latency_seconds_stream);
  SequenceOutput output;
  output.index = index();
  output.token_ids = generated_ids();
  return output;
}

SequenceOutput OneRecSequence::generate_output(const Tokenizer& tokenizer) {
  AUTO_COUNTER(detokenization_latency_seconds_non_stream);
  const auto& rec_config = RecConfig::get_instance();

  SequenceOutput output;
  output.index = index();
  if (output_embedding().defined()) {
    output.embedding = output_embedding();
  }
  if (finish_reason() != FinishReason::NONE) {
    output.finish_reason = finish_reason().to_string();
  }
  output.token_ids = generated_ids();

  if (rec_config.enable_output_sku_logprobs()) {
    const auto& token_logprobs = logprob_state()->get_logprobs();
    const size_t begin = num_prompt_tokens();
    const size_t end = begin + output.token_ids.size();
    output.token_ids_logprobs.reserve(output.token_ids.size());
    for (size_t i = begin; i < end; ++i) {
      if (i < token_logprobs.size()) {
        output.token_ids_logprobs.emplace_back(token_logprobs[i]);
      } else {
        output.token_ids_logprobs.emplace_back();
      }
    }
  }

  if (!rec_config.enable_convert_tokens_to_item() ||
      output.token_ids.size() != static_cast<size_t>(REC_TOKEN_SIZE)) {
    return output;
  }

  const Slice<int32_t> token_slice{output.token_ids.data(),
                                   output.token_ids.size()};
  if (rec_config.enable_extended_item_info()) {
    const auto* rec_tokenizer = dynamic_cast<const RecTokenizer*>(&tokenizer);
    if (rec_tokenizer == nullptr) {
      return output;
    }
    std::vector<RecItemInfo> item_infos;
    const bool ok = rec_tokenizer->decode_item_infos(token_slice, &item_infos);
    if (!ok || item_infos.empty()) {
      return output;
    }
    output.item_infos_list = normalize_rec_item_infos(item_infos, index());
    output.item_ids_list.reserve(output.item_infos_list.size());
    for (const RecItemInfo& item_info : output.item_infos_list) {
      output.item_ids_list.emplace_back(item_info.item_id);
    }
    if (!output.item_infos_list.empty()) {
      output.item_ids = output.item_ids_list.front();
      output.item_info = output.item_infos_list.front();
    }
    return output;
  }

  std::vector<int64_t> item_ids;
  const bool ok = tokenizer.decode(
      token_slice, sequence_params().skip_special_tokens, &item_ids);
  if (ok && !item_ids.empty()) {
    output.item_ids_list = normalize_rec_item_ids(item_ids, index());
    if (!output.item_ids_list.empty()) {
      output.item_ids = output.item_ids_list.front();
    }
  }
  return output;
}

}  // namespace xllm
