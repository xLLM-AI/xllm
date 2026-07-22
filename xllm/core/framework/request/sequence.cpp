/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "sequence.h"

#include <absl/strings/match.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/multimodal/embedding_output.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "core/framework/tokenizer/rec_tokenizer.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"
#include "core/util/tensor_helper.h"
#include "rec_type.h"

namespace xllm {

namespace {
constexpr size_t kDecoderBosTokenCount = 1;
constexpr size_t kDecoderMaxTokenCount = kRecTotalSteps + kDecoderBosTokenCount;
constexpr char kEmptyLogprobsFinishReason[] = "empty_logprobs";

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
      ::xllm::RecConfig::get_instance().each_conversion_threshold();
  if (each_threshold > 0 &&
      static_cast<int32_t>(item_ids.size()) > each_threshold) {
    uint32_t seed =
        ::xllm::ExecutionConfig::get_instance().random_seed() >= 0
            ? static_cast<uint32_t>(
                  ::xllm::ExecutionConfig::get_instance().random_seed()) +
                  static_cast<uint32_t>(sequence_index)
            : std::random_device{}();
    std::mt19937 generator(seed);
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
      ::xllm::RecConfig::get_instance().each_conversion_threshold();
  if (each_threshold > 0 &&
      static_cast<int32_t>(item_infos.size()) > each_threshold) {
    uint32_t seed =
        ::xllm::ExecutionConfig::get_instance().random_seed() >= 0
            ? static_cast<uint32_t>(
                  ::xllm::ExecutionConfig::get_instance().random_seed()) +
                  static_cast<uint32_t>(sequence_index)
            : std::random_device{}();
    std::mt19937 generator(seed);
    std::shuffle(item_infos.begin(), item_infos.end(), generator);
    item_infos.resize(each_threshold);
  }

  return item_infos;
}
}  // namespace

const std::string Sequence::ENCODER_SPARSE_EMBEDDING_NAME = "sparse_embedding";
const std::string Sequence::DECODER_CONTEXT_EMBEDDING_NAME =
    "decoder_context_embedding";

void Sequence::init_onerec_sequence(
    const std::vector<int32_t>& prompt_token_ids,
    torch::Tensor input_embedding) {
  auto& onerec_state = onerec_state_.emplace();
  if (!prompt_token_ids.empty()) {
    onerec_state.encoder_tokens.assign(prompt_token_ids.begin(),
                                       prompt_token_ids.end());
    onerec_state.num_encoder_tokens = prompt_token_ids.size();
  } else {
    auto encoder_sparse_embedding =
        mm_data_.get<torch::Tensor>(ENCODER_SPARSE_EMBEDDING_NAME);
    CHECK(encoder_sparse_embedding.has_value())
        << "encoder sparse embedding not found in mm_data";
    onerec_state.num_encoder_tokens = encoder_sparse_embedding.value().size(0);
  }

  auto decoder_context_embedding =
      mm_data_.get<torch::Tensor>(DECODER_CONTEXT_EMBEDDING_NAME);

  size_t capacity = kDecoderMaxTokenCount;
  if (decoder_context_embedding.has_value()) {
    num_prompt_tokens_ = 0;
    onerec_state.num_decoder_embeddings =
        decoder_context_embedding.value().size(0);
    capacity =
        onerec_state.num_decoder_embeddings + capacity - kDecoderBosTokenCount;
  } else {
    num_prompt_tokens_ = kDecoderBosTokenCount;
  }

  tokens_.resize(capacity);
  for (size_t i = 0; i < num_prompt_tokens_; ++i) {
    tokens_[num_tokens_++] = sequence_params_.bos_token_id;
    token_to_count_map_[sequence_params_.bos_token_id]++;
  }
  volatile_num_prompt_tokens_ = num_prompt_tokens_;
  input_embedding_ = std::move(input_embedding);
  cur_generated_token_idx_ = num_prompt_tokens_;
  logprob_state_ = std::make_unique<LogprobState>(num_prompt_tokens_, capacity);
}

void Sequence::generate_onerec_streaming_output(const Slice<int32_t>& ids,
                                                size_t size,
                                                SequenceOutput& output) const {
  output.index = index_;
  output.token_ids = ids.slice(num_prompt_tokens_, size);
}

void Sequence::generate_onerec_output(const Slice<int32_t>& ids,
                                      size_t size,
                                      const Tokenizer& tokenizer,
                                      SequenceOutput& output) const {
  output.index = index_;
  if (output_embedding_.defined()) {
    output.embedding = output_embedding_;
  }
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = finish_reason_.to_string();
  }
  output.token_ids = ids.slice(num_prompt_tokens_, size);
  if (::xllm::RecConfig::get_instance().enable_output_sku_logprobs() &&
      logprob_state_ != nullptr) {
    const auto& token_logprobs = logprob_state_->get_logprobs();
    output.token_ids_logprobs.reserve(output.token_ids.size());
    for (size_t i = num_prompt_tokens_; i < size; ++i) {
      if (i < token_logprobs.size()) {
        output.token_ids_logprobs.emplace_back(token_logprobs[i]);
      } else {
        output.token_ids_logprobs.emplace_back();
      }
    }
  }
  const size_t rec_token_size = static_cast<size_t>(REC_TOKEN_SIZE);
  if (::xllm::RecConfig::get_instance().enable_convert_tokens_to_item() &&
      output.token_ids.size() == rec_token_size) {
    const Slice<int32_t> token_slice{output.token_ids.data(),
                                     output.token_ids.size()};
    if (::xllm::RecConfig::get_instance().enable_extended_item_info()) {
      const auto* rec_tokenizer = dynamic_cast<const RecTokenizer*>(&tokenizer);
      if (rec_tokenizer != nullptr) {
        std::vector<RecItemInfo> item_infos;
        const bool ok =
            rec_tokenizer->decode_item_infos(token_slice, &item_infos);
        if (ok && !item_infos.empty()) {
          output.item_infos_list = normalize_rec_item_infos(item_infos, index_);
          output.item_ids_list.reserve(output.item_infos_list.size());
          for (const RecItemInfo& item_info : output.item_infos_list) {
            output.item_ids_list.emplace_back(item_info.item_id);
          }
          if (!output.item_infos_list.empty()) {
            output.item_ids = output.item_ids_list.front();
            output.item_info = output.item_infos_list.front();
          }
        }
      }
    } else {
      std::vector<int64_t> item_ids;
      const bool ok = tokenizer.decode(
          token_slice, sequence_params_.skip_special_tokens, &item_ids);
      if (ok && !item_ids.empty()) {
        output.item_ids_list = normalize_rec_item_ids(item_ids, index_);
        if (!output.item_ids_list.empty()) {
          output.item_ids = output.item_ids_list.front();
        }
      }
    }
  }
}

Sequence::Sequence(size_t index,
                   const std::vector<int32_t>& prompt_token_ids,
                   torch::Tensor input_embedding,
                   const MMData& mm_data,
                   const IncrementalDecoder& decoder,
                   const SequenceParams& seq_params)
    : index_(index),
      mm_data_(mm_data),
      latest_generate_time_(absl::Now()),
      sequence_params_(seq_params),
      decoder_(std::move(decoder)),
      termination_flag_(std::make_shared<std::atomic<int32_t>>(INT32_MAX)),
      request_id_(seq_params.request_id) {
  if (is_onerec_model()) {
    init_onerec_sequence(prompt_token_ids, std::move(input_embedding));
    return;
  }

  CHECK(!prompt_token_ids.empty()) << "empty prompt token ids";
  auto capacity = sequence_params_.seq_capacity;
  CHECK_GT(capacity, prompt_token_ids.size()) << "capacity too small";

  num_prompt_tokens_ = prompt_token_ids.size();
  volatile_num_prompt_tokens_ = num_prompt_tokens_;
  tokens_.resize(capacity);

  // init logprob state
  logprob_state_ = std::make_unique<LogprobState>(num_prompt_tokens_, capacity);

  if (sequence_params_.sampling_param->frequency_penalty != 0 ||
      sequence_params_.sampling_param->presence_penalty != 0 ||
      sequence_params_.sampling_param->repetition_penalty != 1) {
    need_unique_tokens_ = true;
  }

  // add the prompt tokens
  for (const auto token_id : prompt_token_ids) {
    tokens_[num_tokens_++] = token_id;
    if (need_unique_tokens_) {
      token_to_count_map_[token_id] = 0;
    }
  }
  // need one token to padding even dont need token count
  token_to_count_map_[prompt_token_ids.back()] = 0;
  input_embedding_ = input_embedding;
  cur_generated_token_idx_ = num_prompt_tokens_;
}

Sequence::Sequence(const Sequence& other)
    : index_(other.index_),
      kv_state_(other.kv_state_),
      host_kv_state_(other.host_kv_state_),
      latest_generate_time_(other.latest_generate_time_),
      time_to_first_token_latency_seconds_(
          other.time_to_first_token_latency_seconds_),
      is_first_token_(other.is_first_token_),
      is_cache_block_for_prefill_(other.is_cache_block_for_prefill_),
      sequence_params_(other.sequence_params_),
      decoder_(other.decoder_),
      tokens_(other.tokens_),
      input_embedding_(other.input_embedding_),
      mm_data_(other.mm_data_),
      mrope_position_delta_(other.mrope_position_delta_),
      output_embedding_(other.output_embedding_),
      mtp_bootstrap_embedding_(other.mtp_bootstrap_embedding_),
      num_tokens_(other.num_tokens_),
      token_to_count_map_(other.token_to_count_map_),
      num_prompt_tokens_(other.num_prompt_tokens_),
      block_hashes_(other.block_hashes_),
      hash_block_size_(other.hash_block_size_),
      linear_state_hashes_(other.linear_state_hashes_),
      linear_hash_stride_(other.linear_hash_stride_),
      onerec_state_(other.onerec_state_),
      volatile_num_prompt_tokens_(other.volatile_num_prompt_tokens_),
      request_id_(other.request_id_),
      finished_(other.finished_),
      finish_status_invalidated_(other.finish_status_invalidated_),
      finish_reason_(other.finish_reason_),
      closed_(other.closed_),
      dp_rank_(other.dp_rank_),
      cur_generated_token_idx_(other.cur_generated_token_idx_),
      first_token_(other.first_token_),
      is_pre_scheduled_step_prefill_(other.is_pre_scheduled_step_prefill_),
      updated_since_last_beam_search_(other.updated_since_last_beam_search_),
      termination_flag_(std::make_shared<std::atomic<int32_t>>(INT32_MAX)) {
  logprob_state_ = std::make_unique<LogprobState>(*other.logprob_state_);
  // A forked sequence (beam / best_of) shares the prompt KV prefix by
  // ref-counting those blocks, but its linear-state / embedding resource block
  // is private: drop the copied Single and Linear blocks so this sequence
  // allocates its own on the next allocate. Preserves the pre-map behavior
  // where single_block_ was never copied by this constructor.
  kv_state_.erase_blocks(BlockType::SINGLE);
  kv_state_.erase_blocks(BlockType::LINEAR);
  host_kv_state_.erase_blocks(BlockType::SINGLE);
  host_kv_state_.erase_blocks(BlockType::LINEAR);
}

// The first token will be only used in disagg pd mode.
void Sequence::record_first_token(const Token& token) {
  if (!::xllm::DisaggPDConfig::get_instance().enable_disagg_pd() ||
      !is_first_token_) {
    return;
  }
  RemoteToken t;
  t.token_id = token.id;
  if (token.logprob.has_value()) {
    t.token_logprob = token.logprob.value();
  }
  t.token_top_tokens = token.top_tokens;
  t.token_top_logprobs = token.top_logprobs;
  first_token_ = std::move(t);
}

void Sequence::append_token(const Token& token) {
  CHECK_LT(num_tokens_, tokens_.size())
      << "exceed the token capacity of the sequence";
  CHECK(!finished_) << "cannot append token to a finished sequence";
  if (!is_onerec_model()) {
    CHECK(kv_state_.kv_cache_tokens_num() > 0 && !is_chunked_prefill_stage())
        << "cannot append token to a prefill sequence";
  }

  // The real token was generated in function
  // `Sequence::update_last_step_token` when enable_schedule_overlap.
  // So here we only consider the case when we disable enable_schedule_overlap.
  if (!sequence_params_.enable_schedule_overlap) {
    // check if the token is the first token after the prompt
    is_first_token_ = num_tokens_ == num_prompt_tokens_;
  }
  record_first_token(token);

  // append the token id and update the token count
  const auto cur_idx = num_tokens_++;
  kv_state_.set_kv_cache_tokens_num(cur_idx);
  const int32_t token_id = static_cast<int32_t>(token.id);
  tokens_[cur_idx] = token_id;

  // skip update in enable_schedule_overlap
  if (sequence_params_.enable_schedule_overlap && token_id < 0) {
    finish_status_invalidated_ = true;
    return;
  }

  // A real token was committed (overlap-fake placeholders returned above).
  ++generated_tokens_since_latency_;

  if (need_unique_tokens_) {
    token_to_count_map_[token_id]++;
  }
  // update logprobs if needed
  if (sequence_params_.sampling_param->logprobs) {
    logprob_state_->update_logprob(
        cur_idx, token, sequence_params_.sampling_param->top_logprobs);
  }

  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = true;
  updated_since_last_beam_search_ = true;
}

void Sequence::update_last_step_token(const Token& token, size_t token_offset) {
  CHECK(sequence_params_.enable_schedule_overlap)
      << "update_last_step_token should only be called when "
         "enable_schedule_overlap";
  // check if the token is the first token
  is_first_token_ = cur_generated_token_idx_ == num_prompt_tokens_;
  record_first_token(token);

  // for mtp, currently only support multi-nodes task.
  if (token_offset > 0) {
    // Skip MTP token processing if sequence has no KV cache blocks.
    // This happens when the sequence was preempted during schedule_request(),
    // causing its KV cache to be deallocated (reset), but it's still in
    // last_batch_ being processed by update_last_step_result().
    // Composite KV managers can have capacity without exposing local blocks.
    if (kv_state_.current_max_tokens_capacity() == 0) {
      return;
    }
    kv_state_.incr_kv_cache_tokens_num(1);
    num_tokens_++;
    // when enable speculative decoding, fake token id will be covered.
    tokens_[cur_generated_token_idx_ + 2] =
        tokens_[cur_generated_token_idx_ + 1];
    tokens_[cur_generated_token_idx_ + 1] = tokens_[cur_generated_token_idx_];
  }

  // A real token is committed here (one per call, including the extra accepted
  // MTP token when token_offset > 0); preempted MTP steps returned above.
  ++generated_tokens_since_latency_;

  const int32_t token_id = static_cast<int32_t>(token.id);
  tokens_[cur_generated_token_idx_] = token_id;
  // Overlap/MTP may rewrite tokens at decode positions; drop any cached block
  // hash from this position onward so it is recomputed when next needed.
  invalidate_block_hashes_from(cur_generated_token_idx_);
  invalidate_linear_state_hashes_from(cur_generated_token_idx_);
  if (need_unique_tokens_) {
    token_to_count_map_[token_id]++;
  }
  // update logprobs if needed
  if (sequence_params_.sampling_param->logprobs) {
    logprob_state_->update_logprob(
        cur_generated_token_idx_,
        token,
        sequence_params_.sampling_param->top_logprobs);
  }
  ++cur_generated_token_idx_;
  finish_status_invalidated_ = true;
  updated_since_last_beam_search_ = true;
}

void Sequence::update_token(size_t index, const Token& token) {
  // TODO: not record in non-disagg pd mode.
  record_first_token(token);

  const int32_t origin_token_id = tokens_[index];
  const int32_t token_id = static_cast<int32_t>(token.id);
  tokens_[index] = token_id;
  // A rewritten token invalidates the cached hash of its block and all
  // subsequent blocks; recompute lazily on the next update_block_hashes().
  invalidate_block_hashes_from(index);
  invalidate_linear_state_hashes_from(index);
  if (need_unique_tokens_) {
    --token_to_count_map_[origin_token_id];
    ++token_to_count_map_[token_id];
  }
  // update logprobs if needed
  if (sequence_params_.sampling_param->logprobs) {
    logprob_state_->update_logprob(
        index, token, sequence_params_.sampling_param->top_logprobs);
  }
  // logprobs_[index] = token.logprob;
  finish_status_invalidated_ = true;
}

void Sequence::update_mm_embeddings(
    const std::vector<torch::Tensor>& mm_embeddings) {
  // cannot update embeddings to a finished sequence
  if (finished_) {
    return;
  }
  output_mm_embeddings_ = mm_embeddings;
  CHECK(sequence_params_.sampling_param->is_embeddings);
  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = false;
  finished_ = true;
  finish_reason_ = FinishReason::STOP;
}

void Sequence::update_embeddings(const torch::Tensor& embeddings) {
  // cannot update embeddings to a finished sequence
  if (finished_) {
    return;
  }
  if (embeddings.defined()) {
    output_embedding_ = embeddings;
  }
  if (sequence_params_.sampling_param->is_embeddings) {
    // invalidate the finish status once a new token is appended
    finish_status_invalidated_ = false;
    finished_ = true;
    finish_reason_ = FinishReason::STOP;
  } else {
    if (output_embedding_.dim() == 1) {
      output_embedding_ = output_embedding_.unsqueeze(0);
    }
  }
}

void Sequence::update_mtp_bootstrap_embedding(const torch::Tensor& embedding) {
  if (embedding.defined()) {
    mtp_bootstrap_embedding_ = embedding.detach().clone();
  }
}

std::optional<SequenceOutput> Sequence::generate_streaming_output(
    size_t size,
    const Tokenizer& tokenizer) {
  // figure out the valid generated token
  // because there might be fake token -1 if enable_schedule_overlap
  for (auto i = num_tokens_ - 1; i >= 0; --i) {
    if (tokens_[i] >= 0) {
      size = i + 1;
      break;
    }
  }
  CHECK_LE(size, num_tokens_);
  AUTO_COUNTER(detokenization_latency_seconds_stream);
  const auto ids = Slice<int32_t>(tokens_, size);

  SequenceOutput output;
  if (is_onerec_model()) {
    generate_onerec_streaming_output(ids, size, output);
    return output;
  }

  // record the start index of token ids
  const size_t start = decoder_.output_offset();
  auto delta = decoder_.decode(ids, tokenizer);
  // NOTE:
  // There is a incomprehensible logic here: we use a thread pool to handle
  // request callbacks in response handler, which means that the main thread and
  // the tasks processing callbacks execute concurrently. This gives rise to a
  // scenario where the main thread finish forwarding, but the callbacks of some
  // previous steps have not yet been executed. However, the main thread
  // forwarding operation modifies sequence information such as um_tokens, which
  // may cause callback handle a previous step to process all accumulated tokens
  // directly when executing "generate_streaming_output" in a streaming
  // scenario. Example: output-1:
  // - step1: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   I'm"}]}
  // - step2: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   trying"}]}
  // - step3: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   to"}]}
  // output-2:
  // - step1: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   I'm trying to"}]}
  // - step2: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"","finish_reason":"length"}]}
  // - step3: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"","finish_reason":"length"}]}
  //
  // We consider both of these cases to be valid,
  // subsequent callbacks only need to skip to return tokens.
  //
  if (delta.empty()) {
    return std::nullopt;
  }

  output.index = index_;
  output.text = std::move(delta);

  const size_t end = decoder_.output_offset();
  output.token_ids = ids.slice(start, end);
  generate_output_tokens_logprobs(start, end, tokenizer, output.logprobs);

  return output;
}

SequenceOutput Sequence::generate_output() {
  SequenceOutput output;
  output.index = index_;
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = finish_reason_.to_string();
  }

  return output;
}

void Sequence::generate_sample_outputs(std::vector<SequenceOutput>& outputs,
                                       const Tokenizer& tokenizer) {
  const auto& slots = sample_slots();
  if (slots.empty()) {
    outputs.push_back(generate_output(tokenizer));
    return;
  }

  outputs.reserve(outputs.size() + slots.size());
  for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
    SequenceOutput output;
    output.index = slots[slot_idx].sample_id;

    const size_t token_idx = num_prompt_tokens_ + slot_idx;
    if (token_idx >= num_tokens_ || tokens_[token_idx] < 0) {
      output.finish_reason = kEmptyLogprobsFinishReason;
      outputs.push_back(std::move(output));
      continue;
    }

    output.token_ids.push_back(tokens_[token_idx]);
    generate_output_tokens_logprobs(
        token_idx, token_idx + 1, tokenizer, output.logprobs);
    if (!output.logprobs.has_value() || output.logprobs->empty()) {
      output.token_ids.clear();
      output.finish_reason = kEmptyLogprobsFinishReason;
      outputs.push_back(std::move(output));
      continue;
    }

    output.text = output.logprobs->front().token;
    outputs.push_back(std::move(output));
  }
}

SequenceOutputType Sequence::output_type() {
  // EMBEDDINGS or MM_EMBEDDINGS
  if (sequence_params_.sampling_param->is_embeddings) {
    if (output_mm_embeddings_.size() > 0) {
      return SequenceOutputType::MM_EMBEDDINGS;
    }
    return SequenceOutputType::EMBEDDINGS;
  }
  return SequenceOutputType::TOKENS;
}

void Sequence::generate_embeddings_output(SequenceOutput& output) {
  output.index = index_;
  Slice<float> embedding_slice = {
      output_embedding_.data_ptr<float>(),
      static_cast<size_t>(output_embedding_.size(0))};
  output.embeddings = embedding_slice;
}

void Sequence::generate_mm_embeddings_output(SequenceOutput& output) {
  output.index = index_;
  std::vector<EmbeddingOutput> embedding_outputs;
  embedding_outputs.reserve(output_mm_embeddings_.size());
  std::unordered_map<MMKey, std::vector<torch::Tensor>> metadata;
  CollectItemTensorVisitor visitor(metadata, {"pixel_values"});
  mm_data_.foreach (visitor);
  for (int i = 0; i < output_mm_embeddings_.size(); i++) {
    const auto& output_mm_embedding = output_mm_embeddings_[i];
    EmbeddingOutput embedding_output;
    embedding_output.embedding = output_mm_embedding;
    for (const auto& [key, value] : metadata) {
      embedding_output.metadata[key] = value[i];
    }
    embedding_outputs.push_back(embedding_output);
  };
  output.mm_embeddings = embedding_outputs;
}

SequenceOutput Sequence::generate_output(const Tokenizer& tokenizer) {
  AUTO_COUNTER(detokenization_latency_seconds_non_stream);

  SequenceOutputType seq_output_type = output_type();

  SequenceOutput output;
  // 1. return mm embeddings for output
  if (seq_output_type == SequenceOutputType::MM_EMBEDDINGS) {
    generate_mm_embeddings_output(output);
    return output;
  }

  // 2. return embeddings for output
  if (seq_output_type == SequenceOutputType::EMBEDDINGS) {
    generate_embeddings_output(output);
    return output;
  }

  // NOTE: enable_schedule_overlap will generate an extra '-1' token.
  // we need to ignore these '-1' tokens.
  const auto ids = tokens();
  size_t size;
  for (auto i = num_tokens_ - 1; i >= 0; --i) {
    if (tokens_[i] >= 0) {
      size = i + 1;
      break;
    }
  }

  // 3. generate onerec output
  if (is_onerec_model()) {
    generate_onerec_output(ids, size, tokenizer, output);
    return output;
  }

  // 4. generate tokens output
  output.index = index_;
  if (output_embedding_.defined()) {
    output.embedding = output_embedding_;
  }
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = finish_reason_.to_string();
  }

  // record the start index of token ids
  const size_t start = decoder_.output_offset();

  // decide which position to start incremental decoding
  // leave 6 tokens for potential unfinished byte sequence
  size_t incremental_start = size <= 6 ? 0 : size - 6;
  // at least start from the first generated token
  if (incremental_start < num_prompt_tokens_) {
    incremental_start = num_prompt_tokens_;
  }
  // incrementally decode tokens between [incremental_start, size)
  std::stringstream ss;
  for (size_t end = incremental_start; end <= size; ++end) {
    ss << decoder_.decode(ids.slice(0, end), tokenizer);
  }

  output.text = ss.str();

  const size_t end = decoder_.output_offset();
  output.token_ids = ids.slice(start, end);
  generate_output_tokens_logprobs(start, end, tokenizer, output.logprobs);

  return output;
}

void Sequence::add_blocks(BlockType type, const std::vector<Block>& blocks) {
  kv_state_.add_blocks(type, blocks);
}

void Sequence::add_host_blocks(BlockType type,
                               const std::vector<Block>& blocks) {
  host_kv_state_.add_blocks(type, blocks);
}

size_t Sequence::num_prefix_cache_tokens() const {
  size_t cached_tokens = std::max(kv_state_.shared_tokens_num(),
                                  host_kv_state_.shared_tokens_num());
  DCHECK_LE(cached_tokens, num_prompt_tokens_);
  return cached_tokens;
}

// release all cache blocks
void Sequence::reset() {
  kv_state_.reset();
  host_kv_state_.reset();
  timer_.reset();
  is_timeout_set_ = false;
  volatile_num_prompt_tokens_ = num_tokens_;
}

void Sequence::add_shared_blocks(BlockType type, std::vector<Block>&& blocks) {
  kv_state_.add_shared_blocks(type, std::move(blocks), num_tokens_);
}

void Sequence::add_shared_host_blocks(BlockType type,
                                      std::vector<Block>&& blocks) {
  host_kv_state_.add_shared_blocks(type, std::move(blocks), num_tokens_);
}

void Sequence::update_block_hashes(uint32_t block_size,
                                   BlockHasherType hasher_type) {
  if (block_size == 0) {
    return;
  }
  hash_block_size_ = block_size;
  extend_prefix_hashes(hasher_type,
                       mm_data_,
                       this->tokens(),
                       block_size,
                       /*boundary_blocks=*/num_tokens_ / block_size,
                       block_hashes_);
}

void Sequence::invalidate_block_hashes_from(size_t token_index) {
  if (block_hashes_.empty() || hash_block_size_ == 0) {
    return;
  }
  const size_t first_stale_block = token_index / hash_block_size_;
  if (first_stale_block < block_hashes_.size()) {
    block_hashes_.resize(first_stale_block);
  }
}

void Sequence::update_linear_state_hashes(uint32_t chunk_stride) {
  if (chunk_stride == 0) {
    return;
  }
  linear_hash_stride_ = chunk_stride;
  // Cover only whole chunks; the trailing partial chunk carries no checkpoint.
  // Fold multimodal content into the chunk digest whenever this sequence
  // carries it, so a linear-state checkpoint keyed on a chunk that spans image
  // tokens cannot collide with a text-only chunk (or a different image) at the
  // same token boundary. The digest is chosen from mm_data_ rather than an
  // engine-bound type because the linear hash is only ever computed here, in
  // the sequence's own context, where mm_data_ is in hand -- and with an empty
  // mm_data_ the MM hasher is byte-identical to TEXT, so text-only sequences
  // are unaffected.
  const BlockHasherType hasher_type =
      mm_data_.valid() ? BlockHasherType::MM : BlockHasherType::TEXT;
  extend_prefix_hashes(hasher_type,
                       mm_data_,
                       this->tokens(),
                       chunk_stride,
                       /*boundary_blocks=*/num_tokens_ / chunk_stride,
                       linear_state_hashes_);
}

void Sequence::invalidate_linear_state_hashes_from(size_t token_index) {
  if (linear_state_hashes_.empty() || linear_hash_stride_ == 0) {
    return;
  }
  const size_t first_stale_chunk = token_index / linear_hash_stride_;
  if (first_stale_chunk < linear_state_hashes_.size()) {
    linear_state_hashes_.resize(first_stale_chunk);
  }
}

bool Sequence::finished() const {
  // return the cached finish status
  if (!finish_status_invalidated_) {
    return finished_;
  }

  if (is_onerec_model() && num_tokens_ == num_prompt_tokens_) {
    return false;
  }

  // Embedding sequence never be finished until it updates its embeddings
  if (finish_status_invalidated_ &&
      sequence_params_.sampling_param->is_embeddings) {
    return false;
  }

  // reset the finish status invalidation flag
  finish_status_invalidated_ = false;

  auto finish_reason =
      sequence_params_.stopping_checker->check(tokens(), num_prompt_tokens_);
  if (finish_reason != FinishReason::NONE) {
    finish_reason_ = finish_reason;
    finished_ = true;
    return true;
  }
  return false;
}

int64_t Sequence::tbt(const absl::Time& now) {
  const int64_t latency =
      absl::ToInt64Milliseconds(now - latest_generate_time_);
  latest_generate_time_ = now;
  // Reset the committed-token counter so the next tbt() interval amortizes only
  // the tokens generated within that interval.
  generated_tokens_since_latency_ = 0;
  return latency;
}

float Sequence::get_acc_logprob() {
  return logprob_state_->get_acc_logprob(num_tokens_);
}

float Sequence::get_base_logprob() {
  return logprob_state_->get_base_logprob(num_tokens_);
}

void Sequence::generate_output_tokens_logprobs(
    size_t start_idx,
    size_t end_idx,
    const Tokenizer& tokenizer,
    std::optional<std::vector<LogProb>>& out_logprobs) {
  if (!sequence_params_.logprobs || start_idx >= end_idx) {
    return;
  }

  logprob_state_->generate_output_tokens_logprobs(
      start_idx,
      end_idx,
      tokenizer,
      out_logprobs,
      sequence_params_.skip_special_tokens,
      tokens_);
}

Slice<int32_t> Sequence::get_generated_tokens() const {
  // Return a slice of generated token IDs (excluding prompt tokens)
  if (num_tokens_ > num_prompt_tokens_) {
    return {tokens_.data() + num_prompt_tokens_,
            num_tokens_ - num_prompt_tokens_};
  }
  return {tokens_.data(), 0};
}

bool Sequence::update_prefetch_result(uint32_t timeout, uint32_t& success_cnt) {
  if (prefetch_results_.empty()) {
    return true;
  }

  if (timeout != 0 && termination_flag_->load(std::memory_order_acquire) > 0) {
    if (!is_timeout_set_) {
      timer_.reset();
      is_timeout_set_ = true;
      return false;
    }

    if (timer_.elapsed_milliseconds() < timeout) {
      return false;
    }
  }

  termination_flag_->store(0, std::memory_order_release);
  success_cnt = host_kv_state_.blocks(BlockType::KV).size();
  for (auto& cnt : prefetch_results_) {
    success_cnt = std::min(success_cnt, cnt->load());
  }
  if (success_cnt > 0) {
    host_kv_state_.incr_kv_cache_tokens_num(
        success_cnt * host_kv_state_.blocks(BlockType::KV)[0].size());
    host_kv_state_.incr_shared_blocks_num(BlockType::KV, success_cnt);
  }
  prefetch_results_.clear();
  return true;
}

void Sequence::finish() {
  finished_ = true;
  finish_status_invalidated_ = false;
  if (finish_reason_ == FinishReason::NONE) {
    finish_reason_ = FinishReason::STOP;
  }
}

void Sequence::reset_finish_state_for_beam_search() {
  finished_ = false;
  finish_reason_ = FinishReason::NONE;
  finish_status_invalidated_ = true;
  finished();
}

}  // namespace xllm
