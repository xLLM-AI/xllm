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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/speculative/embedding_cache.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/model/model_args.h"
#include "runtime/speculative_worker_impl.h"
#include "util/utils.h"

namespace xllm {

namespace dflash_detail {

inline int32_t decode_draft_width(int32_t num_speculative_tokens,
                                  bool sample_from_anchor) {
  return sample_from_anchor ? num_speculative_tokens
                            : num_speculative_tokens + 1;
}

inline void invalidate_draft_model_geometry(ModelInputParams& input_params) {
  // Attention metadata is model-owned: DeepSeek-V4 bakes DSA group layout and
  // sparse tiling values such as ori_win_left into opaque tensors. A draft
  // input copied from the target must rebuild those tensors for draft geometry.
  input_params.attn_metadata.reset();
}

enum class DSparkSasMode : uint8_t {
  NOT_DSPARK,
  COMPATIBILITY,
  NATIVE,
};

inline DSparkSasMode classify_dspark_sas_mode(const ModelArgs& draft_args,
                                              bool sample_from_anchor) {
  if (!sample_from_anchor ||
      !util::is_deepseek_v4_dspark_model_type(draft_args.model_type())) {
    return DSparkSasMode::NOT_DSPARK;
  }
  return draft_args.dspark_use_native_sas() ? DSparkSasMode::NATIVE
                                            : DSparkSasMode::COMPATIBILITY;
}

}  // namespace dflash_detail

class DFlashWorkerImpl : public SpeculativeWorkerImpl {
 public:
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DFlashWorkerImpl() override = default;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;

  // Draft produces all speculative tokens of a block in one forward, so its
  // output is a single [batch, num_speculative_tokens] block rather than the
  // per-step outputs an autoregressive drafter (e.g. MTP) yields.
  struct DraftBlock {
    torch::Tensor token_ids;
    torch::Tensor probs;
    // Optional acceptance-probability estimate, [batch, num_speculative_tokens]
    // fp32 in [0, 1]. Populated by DSpark's ConfidenceHead when available;
    // consumed by the adaptive-speculative pruning controller. When undefined,
    // the controller falls back to `probs` (sampler-gathered softmax scores).
    torch::Tensor confidence_probs;
    // Adaptive lag-confidence pruning decision, computed in step_decode from
    // the PREVIOUS step's confidence (overlapping this step's draft forward)
    // and consumed by run_validate. Per-seq validate prefix length; empty when
    // lag confidence is off. Not produced by run_decode_draft.
    std::vector<int32_t> lagged_prefix_lengths;
    // Set when run_decode_draft already built the pruned varlen validate batch
    // in the draft-overlap window (lag confidence on + a pruning decision).
    // run_validate then skips its own metadata rebuild and consumes these
    // directly. varlen_prebuilt=false means the dense batch is in
    // validate_input and run_validate takes the legacy this-step decision +
    // rebuild path.
    bool varlen_prebuilt = false;
    std::vector<int32_t> per_seq_val_tokens;
    int32_t max_val_tokens = 0;
    // No-sync draft inputs must outlive validation's stream sync.
    std::vector<std::shared_ptr<ForwardInput>> retained_inputs;
  };

  // virtual: DSpark overrides the draft sampling (parallel block sample ->
  // one forward + sequential Markov-head sampling loop).
  //
  // lagged_prefix_lengths (lag confidence only): the pre-draft prune decision.
  // When it prunes, run_decode_draft builds the pruned varlen validate batch in
  // the draft-overlap window (setting DraftBlock.varlen_prebuilt) so the whole
  // rebuild overlaps the in-flight draft instead of sitting on run_validate's
  // critical path. Empty => build the dense batch as before.
  virtual DraftBlock run_decode_draft(
      const ForwardInput& input,
      ForwardInput& validate_input,
      const std::vector<int32_t>& lagged_prefix_lengths = {});

  // Block layout hook: false (DFlash) -> query_width N+1, slot 0 is the
  // un-selected anchor; true (DSpark) -> query_width N, every position predicts
  // and slot 0 predicts the first draft token. prepare_query_inputs and the
  // query row builder read this so the whole (helper-heavy) query-build logic
  // stays here and a subclass flips one bit.
  virtual bool sample_from_anchor() const { return false; }

  // Build the validate batch in the draft-overlap window and record the prune
  // decision into `draft_block`. Both DFlash and DSpark run_decode_draft call
  // this in place of the bare prepare_validate_inputs, so the lag overlap-build
  // covers both algorithms. When lagged_prefix_lengths prunes, builds the
  // pruned varlen batch (sets draft_block.varlen_prebuilt / per_seq_val_tokens
  // / max_val_tokens); otherwise builds the dense batch (varlen_prebuilt=false)
  // exactly as before.
  void prepare_overlap_validate_input(
      const ForwardInput& input,
      ForwardInput& validate_input,
      const std::vector<int32_t>& lagged_prefix_lengths,
      DraftBlock& draft_block);

  // Shared with subclasses (DSpark): build the N/N+1-wide draft query block and
  // the target validate input. A DSpark override of run_decode_draft calls both
  // before its draft forward.
  //
  // per_seq_val_tokens: when non-empty, build a *pruned varlen* validate batch
  // (Σ per_seq_val_tokens[i] tokens) instead of the dense [batch, N+1] batch.
  void prepare_query_inputs(const ForwardInput& input,
                            ForwardInput& query_input);
  void prepare_validate_inputs(
      const ForwardInput& input,
      ForwardInput& validate_input,
      const std::vector<int32_t>& per_seq_val_tokens = {});

 private:
  bool draft_use_block_parallel_rows() const {
    return draft_sas_mode_ == dflash_detail::DSparkSasMode::COMPATIBILITY;
  }
  BatchForwardType draft_batch_forward_type() const {
    return draft_sas_mode_ == dflash_detail::DSparkSasMode::NATIVE
               ? BatchForwardType::DECODE
               : BatchForwardType::CHUNKED_PREFILL;
  }

  void fill_validate_input_from_draft_outputs(const DraftBlock& draft_block,
                                              ForwardInput& validate_input,
                                              Stream& compute_stream,
                                              int32_t effective_val_tokens);

  // Per-seq varlen variant. Handles the case where validate_input.token_ids
  // is a flat [Σ per_seq_val_tokens] layout (i.e. the base per_seq builder
  // produced it), copying draft tokens into the varlen slots per seq.
  void fill_validate_input_from_draft_outputs_varlen(
      const DraftBlock& draft_block,
      ForwardInput& validate_input,
      Stream& compute_stream,
      const std::vector<int32_t>& per_seq_val_tokens);

  std::optional<ForwardOutput> run_validate(const ForwardInput& input,
                                            const DraftBlock& draft_block,
                                            ForwardInput& validate_input);

  // `per_seq_val_tokens` (optional): when non-empty, the target output was
  // scattered from a per-seq varlen batch and each seq's bonus lives at
  // dense col (per_seq_val_tokens[i] - 1), not the fixed last column. Passing
  // it lets validate() gather the bonus token per-seq. Empty = uniform
  // batch-max width (bonus at col effective_val_tokens - 1 for every seq).
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const DraftBlock& draft_block,
                        const ForwardOutput& target_output,
                        int32_t effective_val_tokens,
                        const std::vector<int32_t>& per_seq_val_tokens);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output,
                        int32_t effective_val_tokens,
                        const std::vector<int32_t>& per_seq_val_tokens);

  // Adaptive-speculative helper: run controller on draft acceptance
  // probabilities (ConfidenceHead output if available, sampler-gathered probs
  // otherwise) and return per-seq prefix_len vector (each entry in
  // [0, num_speculative_tokens]). Returns empty vector when adaptive is off,
  // in which case the caller keeps the full draft block.
  std::vector<int32_t> compute_adaptive_prefix_lengths(
      const DraftBlock& draft_block,
      const ForwardInput& input);

  // Core of the adaptive decision, shared by the this-step and lag-confidence
  // paths: build per-seq kv lengths and run the controller on the given
  // [batch, num_speculative_tokens] probs. The cost model is reused verbatim.
  std::vector<int32_t> compute_prefix_lengths_from_probs(
      const torch::Tensor& probs_for_controller,
      const ForwardInput& input);

  // Lag-confidence decision (enable_lag_confidence): prune from the PREVIOUS
  // step's confidence read from the embedding cache, so the decision does not
  // data-depend on this step's draft forward. Requests with no fresh lagged
  // confidence (first step / recycled slot) fall back to full width. Empty when
  // adaptive is off.
  std::vector<int32_t> decide_lagged_prefix_lengths(const ForwardInput& input);

  // Convert a per-seq prefix_len vector into per-seq validate widths
  // (prefix_len + 1 bonus). Writes per_seq_val_tokens (empty when
  // prefix_lengths is empty) and max_val_tokens; returns did_prune (true iff
  // any seq's width is below the full N+1). Shared by the lag overlap-build
  // path (run_decode_draft) and the legacy this-step path (run_validate).
  bool prefix_lengths_to_val_tokens(const std::vector<int32_t>& prefix_lengths,
                                    int32_t batch_size,
                                    std::vector<int32_t>* per_seq_val_tokens,
                                    int32_t* max_val_tokens) const;

  // Record precise (draft, accepted) counters. Padded -1 slots at positions
  // past per_seq_val_tokens[i]-1 are excluded — the count only walks each
  // row up to its per-seq width. Passing an empty vector treats every row
  // as full width (static). Caller must ensure val_output.next_tokens is on
  // CPU (avoids a blocking device sync on the hot path).
  void record_validate_metrics(
      const SampleOutput& val_output,
      const std::vector<int32_t>& per_seq_val_tokens) const;

  void process_draft_sample_output(SampleOutput& sample_output);

  // Mirrors sampled tokens to rank 0 under schedule-overlap so every rank
  // commits the same prefix. No-op for a single rank.
  void maybe_broadcast_spec_tokens(torch::Tensor& tokens);

  void update_decode_step_input(
      ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  void write_context_kv(const ForwardInput& input,
                        const torch::Tensor& context_hidden,
                        const torch::Tensor& positions_device,
                        const torch::Tensor& new_cache_slots_device);

  void write_target_context_to_cache(
      const ForwardInput& input,
      const SampleOutput& validate_output,
      const torch::Tensor& confidence = torch::Tensor());

 protected:
  std::unique_ptr<LLMWorkerImpl> draft_impl_;
  std::shared_ptr<EmbeddingCache> embedding_cache_;
#if defined(USE_NPU) || defined(USE_MLU)
  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
#endif
  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
  dflash_detail::DSparkSasMode draft_sas_mode_ =
      dflash_detail::DSparkSasMode::NOT_DSPARK;
};

}  // namespace xllm
