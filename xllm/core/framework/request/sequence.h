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

#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/futures/Future.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"
#include "finish_reason.h"
#include "framework/block/block.h"
#include "incremental_decoder.h"
#include "rec_type.h"
#include "request_output.h"
#include "sample_slot.h"
#include "sequence_kv_state.h"
#include "sequence_logprob_state.h"
#include "stopping_checker.h"
#include "util/timer.h"

namespace xllm {

enum class SequenceOutputType : int8_t {
  TOKENS = 0,
  EMBEDDINGS = 1,
  MM_EMBEDDINGS = 2,
};

enum class SequenceStage : int8_t {
  // Prefill without using kv cache.
  PREFILL = 0,
  // Chunked prefill using kv cache.
  CHUNKED_PREFILL = 1,
  // Decode one token.
  DECODE = 2
};

struct SequenceParams {
  // max tokens count in the sequence.
  size_t seq_capacity = 0;

  // whether to skip special tokens in the output text. default = true.
  bool skip_special_tokens = true;

  // whether to echo the prompt in the output text. default = false.
  bool echo = false;

  // whether to return the log probabilities of the tokens. default = false.
  bool logprobs = false;

  // the num of outputs.
  size_t n = 1;

  // the num of sequences to generate for each prompt and select the best
  // among.
  size_t best_of = 1;

  // whether the request is streaming
  bool streaming = false;

  // enable_schedule_overlap or not. default = false.
  bool enable_schedule_overlap = false;

  RecType rec_type = RecType::kNone;

  int32_t bos_token_id = 0;

  // request id for suffix-decoding request identity
  std::string request_id;

  const std::vector<SampleSlot>* sample_slots = nullptr;

  // sampling params
  // reference from request
  RequestSamplingParam* sampling_param;  // not owned

  // stopping checker
  // reference from request
  StoppingChecker* stopping_checker;  // not owned
};

class Sequence final {
 public:
  Sequence(size_t index,
           const std::vector<int32_t>& prompt_token_ids,
           torch::Tensor input_embedding,
           const MMData& mm_data,
           const IncrementalDecoder& incremental_decoder,
           const SequenceParams& seq_params);

  Sequence(const Sequence& other);

  // get mm data
  const MMData& mm_data() const { return mm_data_; }
  MMData& mutable_mm_data() { return mm_data_; }
  void set_mrope_position_delta(int val) { mrope_position_delta_ = val; }
  int get_mrope_position_delta() { return mrope_position_delta_; }

  // get token ids to count map
  const std::unordered_map<int32_t, int32_t>& token_to_count_map() const {
    return token_to_count_map_;
  }

  // check if in prefill stage
  bool is_chunked_prefill_stage() const {
    return stage() == SequenceStage::CHUNKED_PREFILL;
  }
  bool is_prefill_stage() const { return stage() != SequenceStage::DECODE; }
  // get the sequence stage
  SequenceStage stage() const {
    if (kv_state_.kv_cache_tokens_num() <
        std::max(volatile_num_prompt_tokens_, num_prompt_tokens())) {
      if (kv_state_.kv_cache_tokens_num() > 0) {
        return SequenceStage::CHUNKED_PREFILL;
      }
      return SequenceStage::PREFILL;
    }
    return SequenceStage::DECODE;
  }

  // whether the new added token is the first token
  bool is_first_token() const { return is_first_token_; }
  std::optional<RemoteToken>& first_token() { return first_token_; }
  // get the total number of tokens
  size_t num_tokens() const { return num_tokens_; }
  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }
  // get the number of generated tokens
  // returns 0 in prefill stage
  size_t num_generated_tokens() const {
    return num_tokens_ - num_prompt_tokens_;
  }
  Slice<int32_t> tokens() const { return {tokens_, num_tokens_}; }
  // get tokens in kv cache
  Slice<int32_t> cached_tokens() const {
    return {tokens_, kv_state_.kv_cache_tokens_num()};
  }

  // get token ids in host kv cache
  Slice<int32_t> cached_host_tokens() const {
    return {tokens_, host_kv_state_.kv_cache_tokens_num()};
  }

  // get the number of tokens need compute
  size_t num_need_compute_tokens() const {
    return num_tokens_ - std::max(kv_state_.kv_cache_tokens_num(),
                                  host_kv_state_.kv_cache_tokens_num());
  }

  size_t kv_cache_tokens_num() const {
    return std::max(kv_state_.kv_cache_tokens_num(),
                    host_kv_state_.kv_cache_tokens_num());
  }

  size_t num_prefix_cache_tokens() const;

  // add a new token id to the sequence and update the count
  // the token would be discarded if the sequence is still in prefill stage
  void append_token(const Token& token);
  void append_token(int64_t token_id) { append_token(Token(token_id)); }
  void update_token(size_t index, const Token& token);
  void update_last_step_token(const Token& token, size_t token_offset = 0);
  bool has_new_tokens_generated() const {
    return num_tokens_ > decoder_.output_offset();
  }

  // update mm embeddings to the sequence
  void update_mm_embeddings(const std::vector<torch::Tensor>& mm_embeddings);
  // update embeddings to the sequence
  void update_embeddings(const torch::Tensor& embedding);
  void update_mtp_bootstrap_embedding(const torch::Tensor& embedding);
  torch::Tensor get_mtp_bootstrap_embedding() const {
    return mtp_bootstrap_embedding_;
  }
  void clear_mtp_bootstrap_embedding() {
    mtp_bootstrap_embedding_ = torch::Tensor();
  }
  // Single per-sequence resource block id (linear-state / embedding), or -1.
  int32_t get_single_block_id() const {
    return kv_state_.get_single_block_id();
  }

  // Linear-state (Qwen3.5 GDN) live slot, stored in composite_blocks_ under
  // BlockType::LINEAR. Drawn from the dedicated LinearStateBlockManager id
  // space.
  bool has_linear_state_slot() const {
    return kv_state_.get_linear_block_id() >= 0;
  }
  int32_t get_linear_state_slot_id() const {
    return kv_state_.get_linear_block_id();
  }

  void set_pending_linear_save(const XXH3Key& hash) {
    kv_state_.set_pending_linear_save(hash);
  }
  std::optional<XXH3Key> take_pending_linear_save() {
    return kv_state_.take_pending_linear_save();
  }
  bool has_pending_linear_save() const {
    return kv_state_.has_pending_linear_save();
  }
  void set_linear_restore_src_block(Block&& block) {
    kv_state_.set_linear_restore_src_block(std::move(block));
  }
  bool has_linear_restore_src_block() const {
    return kv_state_.has_linear_restore_src_block();
  }
  std::optional<Block> take_linear_restore_src_block() {
    return kv_state_.take_linear_restore_src_block();
  }
  Block copy_block(BlockType type) const { return kv_state_.copy_block(type); }
  const std::string& request_id() const { return request_id_; }
  // get input embedding
  torch::Tensor get_input_embedding() const { return input_embedding_; }

  void add_blocks(BlockType type, const std::vector<Block>& blocks);
  void add_host_blocks(BlockType type, const std::vector<Block>& blocks);
  void add_shared_blocks(BlockType type, std::vector<Block>&& blocks);
  void add_shared_host_blocks(BlockType type, std::vector<Block>&& blocks);

  // Precomputed chained block hashes used by the prefix cache. Covers all full
  // blocks of the current tokens; reused by match()/insert() so the hash is
  // computed once per sequence instead of recomputed on every call. Returns the
  // chain for the stride set by the most recent update_block_hashes() call.
  Slice<XXH3Key> block_hashes() const;

  // Extend the per-stride chain for `block_size` to cover any newly completed
  // full blocks, and select it as the one block_hashes() returns. Cheap (no-op)
  // when no new full block is available, so it is safe to call before every
  // match()/cache(). DSV4 admission probes SWA / C4 / C128 back-to-back with
  // different strides (base / 4*base / 128*base); each stride keeps its own
  // chain so switching strides no longer discards and rebuilds the whole chain.
  void update_block_hashes(uint32_t block_size, BlockHasherType hasher_type);

  // Precomputed chained per-chunk hashes for the linear-state checkpoint index.
  // Separate hash domain from the KV block-hash chains: the stride is one
  // prefill chunk (a multiple of the KV block size), not a KV block, so a
  // linear checkpoint is a sparse overlay on the per-block KV cache. Consumed
  // by the batch builder (save/restore boundaries) and the LINEAR leaf's match
  // probe.
  Slice<XXH3Key> linear_state_hashes() const { return linear_state_hashes_; }

  // Extend `linear_state_hashes_` to cover any newly completed full chunks at
  // the given stride. Cheap (no-op) when no new full chunk is available, so it
  // is safe to call before every match()/save-boundary check.
  void update_linear_state_hashes(uint32_t chunk_stride);

  // whether the prefill stage has been cached.
  bool if_cache_block_for_prefill() {
    bool if_cache =
        !is_cache_block_for_prefill_ && num_tokens() > num_prompt_tokens();
    is_cache_block_for_prefill_ |= if_cache;
    return if_cache;
  }

  FinishReason finish_reason() const { return finish_reason_; }
  // check finish status, use cached value if not invalidated
  bool finished() const;
  // mark sequence as finished (used by rec model multi-round decoding)
  void finish();

  // get the output of the sequence until the specified number of tokens,
  // returns nullopt if no delta text and not finished
  std::optional<SequenceOutput> generate_streaming_output(
      size_t size,
      const Tokenizer& tokenizer);
  // get the full output of the sequence
  SequenceOutput generate_output(const Tokenizer& tokenizer);
  SequenceOutput generate_output();
  void generate_sample_outputs(std::vector<SequenceOutput>& outputs,
                               const Tokenizer& tokenizer);

  // get the sampling parameters
  const RequestSamplingParam* sampling_param() const {
    return sequence_params_.sampling_param;
  }

  // get the stopping criteria
  const StoppingChecker* stopping_checker() const {
    return sequence_params_.stopping_checker;
  }

  // close the sequence once all outputs have been sent
  void close() { closed_ = true; }
  bool is_closed() const { return closed_; }

  // time between two tokens
  int64_t tbt(const absl::Time& now);

  // Number of real tokens committed to this sequence since the last tbt()
  // reset. Speculative/MTP steps commit multiple tokens per step, so this is
  // used to amortize the per-step inter-token latency into a per-token value.
  size_t generated_tokens_since_latency() const {
    return generated_tokens_since_latency_;
  }

  void set_wait_time_ms() {
    wait_time_ms_ = static_cast<int32_t>(
        absl::ToDoubleSeconds(absl::Now() - latest_generate_time_) * 1000);
  }
  int32_t get_wait_time_ms() const { return wait_time_ms_; }

  // set sequence ttft
  void set_time_to_first_token_latency_seconds(
      double time_to_first_token_latency_seconds) {
    time_to_first_token_latency_seconds_ = time_to_first_token_latency_seconds;
  }
  double time_to_first_token_latency_seconds() const {
    return time_to_first_token_latency_seconds_;
  }

  void set_dp_rank(int32_t dp_rank) { dp_rank_ = dp_rank; }

  int32_t dp_rank() const { return dp_rank_; }

  void enable_checking_prefill_token() {
    decoder_.enable_checking_prefill_token();
  }

  // get all generated token IDs (excluding prompt tokens)
  Slice<int32_t> get_generated_tokens() const;

  std::queue<bool>& pre_scheduled_step_prefill_queue() {
    return is_pre_scheduled_step_prefill_;
  }

  KVCacheState& kv_state() { return kv_state_; }

  KVCacheState& host_kv_state() { return host_kv_state_; }

  // for generated tokens
  float get_acc_logprob();
  // Returns the beam base score: accumulated logprob excluding last token.
  float get_base_logprob();
  void generate_output_tokens_logprobs(
      size_t start_idx,
      size_t end_idx,
      const Tokenizer& tokenizer,
      std::optional<std::vector<LogProb>>& out_logprobs);

  std::shared_ptr<std::atomic<int32_t>> get_termination_flag() {
    return termination_flag_;
  }
  std::vector<std::shared_ptr<std::atomic<uint32_t>>>* get_prefetch_results() {
    return &prefetch_results_;
  }

  bool update_prefetch_result(uint32_t timeout, uint32_t& success_cnt);

  void reset();

  bool check_beam_search() {
    return sequence_params_.sampling_param->beam_width > 1;
  }

  const std::vector<SampleSlot>& sample_slots() const {
    static const std::vector<SampleSlot> kEmpty;
    return sequence_params_.sample_slots == nullptr
               ? kEmpty
               : *sequence_params_.sample_slots;
  }

  bool check_need_unique_tokens() { return need_unique_tokens_; }

  // True if this sequence received token updates after the previous
  // SequencesGroup::process_beam_search() round.
  bool updated_since_last_beam_search() const {
    return updated_since_last_beam_search_;
  }

  void clear_updated_since_last_beam_search() {
    updated_since_last_beam_search_ = false;
  }

  // Beam search may clone a finished sequence and then rewrite tokens to a
  // non-finished candidate. Reset cached finish state so finished() is
  // re-evaluated from current tokens.
  void reset_finish_state_for_beam_search();

  // Multi-round beam search result caching
  void set_beam_result(int32_t bw,
                       int32_t total_rounds,
                       const std::vector<std::vector<int32_t>>& flat,
                       const std::vector<float>& last_logprobs) {
    beam_width_cached_ = bw;
    total_rounds_cached_ = total_rounds;
    beam_seq_group_flat_ = flat;
    beam_last_logprobs_ = last_logprobs;
  }
  bool has_beam_result() const {
    return beam_width_cached_ > 0 && total_rounds_cached_ > 0 &&
           !beam_seq_group_flat_.empty();
  }
  const std::vector<std::vector<int32_t>>& beam_seq_group_flat() const {
    return beam_seq_group_flat_;
  }
  const std::vector<float>& beam_last_logprobs() const {
    return beam_last_logprobs_;
  }
  int32_t beam_width_cached() const { return beam_width_cached_; }
  int32_t total_rounds_cached() const { return total_rounds_cached_; }

  LogprobState* logprob_state() { return logprob_state_.get(); }
  void set_estimated_latency(double estimated_latency) {
    estimated_latency_ = estimated_latency;
  }

  double estimated_latency() const { return estimated_latency_; }

  // set sequence id
  void set_seq_id(int32_t seq_id) { seq_id_ = seq_id; }

  // get sequence id
  int32_t seq_id() const { return seq_id_; }

  const std::vector<int32_t>& encoder_tokens() const {
    static const std::vector<int32_t> kEmpty;
    if (!onerec_state_.has_value()) {
      return kEmpty;
    }
    return onerec_state_->encoder_tokens;
  }

  size_t encoder_seq_len() const {
    return onerec_state_.has_value() ? onerec_state_->num_encoder_tokens : 0;
  }

  size_t num_decoder_embeddings() const {
    return onerec_state_.has_value() ? onerec_state_->num_decoder_embeddings
                                     : 0;
  }

  RecType rec_type() const { return sequence_params_.rec_type; }
  bool is_onerec_model() const {
    return sequence_params_.rec_type == RecType::kOneRec;
  }

  static const std::string ENCODER_SPARSE_EMBEDDING_NAME;
  static const std::string DECODER_CONTEXT_EMBEDDING_NAME;

  void set_cancel() { cancelled_.store(true, std::memory_order_relaxed); }

  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  void handle_last_token() {
    last_token_handled_.store(true, std::memory_order_relaxed);
  }

  bool last_token_handled() const {
    return last_token_handled_.load(std::memory_order_relaxed);
  }

 private:
  void record_first_token(const Token& token);

  // Drop cached block hashes that may be stale after the token at
  // `token_index` was rewritten (beam search / speculative / disagg PD).
  void invalidate_block_hashes_from(size_t token_index);

  // Same as above for the linear-state hash domain: drop cached chunk hashes
  // from the chunk containing `token_index` onward.
  void invalidate_linear_state_hashes_from(size_t token_index);

  SequenceOutputType output_type();
  void generate_embeddings_output(SequenceOutput& output);
  void generate_mm_embeddings_output(SequenceOutput& output);

  void init_onerec_sequence(const std::vector<int32_t>& prompt_token_ids,
                            torch::Tensor input_embedding);

  void generate_onerec_streaming_output(const Slice<int32_t>& ids,
                                        size_t size,
                                        SequenceOutput& output) const;

  void generate_onerec_output(const Slice<int32_t>& ids,
                              size_t size,
                              const Tokenizer& tokenizer,
                              SequenceOutput& output) const;

  struct OneRecState {
    size_t num_encoder_tokens = 0;
    size_t num_decoder_embeddings = 0;
    std::vector<int32_t> encoder_tokens;
  };
  int32_t wait_time_ms_ = 0;
  double estimated_latency_ = 0.0;

  // the index of the sequence in the request
  size_t index_ = 0;

  KVCacheState kv_state_;

  KVCacheState host_kv_state_;

  std::unique_ptr<LogprobState> logprob_state_;

  // latest token generate time
  absl::Time latest_generate_time_;

  // number of real tokens committed since the last tbt() reset
  size_t generated_tokens_since_latency_ = 0;

  // sequence ttft latency
  double time_to_first_token_latency_seconds_ = 0.0;

  // whether the added token is the first generated token
  bool is_first_token_ = false;

  // whether the prefill stage has been cached.
  bool is_cache_block_for_prefill_ = false;

  SequenceParams sequence_params_;

  // incremental decoder to decode the tokens
  IncrementalDecoder decoder_;

  // token ids generated for the sequence
  std::vector<int32_t> tokens_;

  torch::Tensor input_embedding_;

  MMData mm_data_;
  int mrope_position_delta_ = 0;

  // mm embedding of the sequence
  std::vector<torch::Tensor> output_mm_embeddings_;

  // embeddings of the sequence
  torch::Tensor output_embedding_;

  // temporary PD handoff bootstrap hidden state for first MTP decode.
  torch::Tensor mtp_bootstrap_embedding_;

  // number of tokens in the sequence
  size_t num_tokens_ = 0;

  // the count of each token id
  std::unordered_map<int32_t, int32_t> token_to_count_map_;
  bool need_unique_tokens_ = false;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // Precomputed chained block hashes covering all full blocks of `tokens_`,
  // keyed by block-size stride. DSV4 admission probes multiple strides (base /
  // 4*base / 128*base) per tick; each keeps its own chain so a stride switch
  // extends incrementally instead of discarding and rebuilding. Extended
  // incrementally; consumed by the prefix cache. std::map node storage is
  // pointer-stable, so a Slice handed out by block_hashes() survives inserts of
  // other strides.
  std::map<uint32_t, std::vector<XXH3Key>> block_hashes_by_stride_;

  // Stride selected by the most recent update_block_hashes() call; keys
  // block_hashes() into `block_hashes_by_stride_` (0 until first computed).
  uint32_t hash_block_size_ = 0;

  // Precomputed chained per-chunk hashes for the linear-state checkpoint index
  // (own hash domain, chunk-strided). Extended incrementally, invalidated on
  // token rewrite; see linear_state_hashes()/update_linear_state_hashes().
  std::vector<XXH3Key> linear_state_hashes_;

  // Chunk stride used to compute `linear_state_hashes_` (0 until first
  // computed).
  uint32_t linear_hash_stride_ = 0;

  std::optional<OneRecState> onerec_state_;

  // NOTE: MUST FIXME Later
  // record all tokens num in last turn when the request is
  // interrupted due to the lack of kv cache capacity.
  // All block tables are released when request be interrupted,
  // but the generated tokens are retained for the next execution.
  // In the next execution, we should treat these generated tokens as prompts.
  size_t volatile_num_prompt_tokens_ = 0;

  // is the sequence finished
  mutable bool finished_ = false;

  // is the finish status invalidated
  mutable bool finish_status_invalidated_ = true;

  // the reason why the sequence is finished
  mutable FinishReason finish_reason_ = FinishReason::NONE;

  // is the sequence closed.
  bool closed_ = false;

  // dp_rank
  int32_t dp_rank_ = -1;

  // seq id in the batch
  int32_t seq_id_ = -1;

  // for enable_schedule_overlap case
  uint32_t cur_generated_token_idx_;

  // record first token in disaggregated PD mode.
  std::optional<RemoteToken> first_token_;

  // when enable_schedule_overlap, use this to record whether is the
  // "prefill" stage at the execution of appending the fake token id.
  // In the appending stage, we push the state into the queue, in the
  // update stage, we pop the state from the queue.
  // 2 valid elements at most, maximum 2 steps pre scheduled.
  std::queue<bool> is_pre_scheduled_step_prefill_;

  std::atomic<bool> cancelled_{false};

  // kvcache store copy async result
  std::shared_ptr<std::atomic<int32_t>> termination_flag_;
  std::vector<std::shared_ptr<std::atomic<uint32_t>>> prefetch_results_;

  Timer timer_;
  bool is_timeout_set_ = false;

  // whether the last token is handled
  std::atomic<bool> last_token_handled_{false};

  std::string request_id_;

  // Multi-round beam search result caching
  int32_t beam_width_cached_ = 0;
  int32_t total_rounds_cached_ = 0;
  std::vector<std::vector<int32_t>> beam_seq_group_flat_;
  std::vector<float> beam_last_logprobs_;

  // Mark whether the sequence has new token updates in current decode step.
  // This is only consumed by software beam search to distinguish:
  // 1) beams that were already finished before current step and
  // 2) beams that just became finished in current step.
  bool updated_since_last_beam_search_ = false;
};

}  // namespace xllm
