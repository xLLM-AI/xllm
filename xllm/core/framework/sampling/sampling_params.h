/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/sampling/draft_sampling_mode.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// How a batch's sequences are sampled. MIXED is the only mode that carries a
// do_sample tensor for the per-row choice.
enum class SamplingMode : int8_t { GREEDY, RANDOM, MIXED };

struct Token;

struct RequestSamplingParam {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  float top_p = 1.0;
  int64_t top_k = 0;
  float min_p = 0.0;
  std::optional<int64_t> seed;
  uint32_t min_tokens = 0;
  int64_t vocab_size = 0;
  std::unordered_map<int32_t, float> logit_bias;
  std::optional<std::vector<int32_t>> allowed_token_ids;
  std::vector<std::vector<int32_t>> bad_words_token_ids;
  std::vector<int32_t> all_stop_token_ids;
  bool logprobs = false;
  int64_t top_logprobs = 0;
  bool is_embeddings = false;
  bool json_object = false;
  int32_t beam_width = 0;
  int32_t num_return_sequences = 0;

  // Turns on beam search with the given width and enforces what the beam
  // machinery needs. Both the on-device BeamSearcher kernel and the host-side
  // SequencesGroup::process_beam_search() pick the next beams from the
  // sampler's top-k candidates, and the sampler produces exactly top_logprobs
  // of them per sequence (the batch takes max(top_logprobs) as its top-k). So
  // logprobs must be on, and top_logprobs must be at least beam_width: the
  // search starts from a single sequence, so with fewer candidates the first
  // step can only fan out to top_logprobs beams and the lower-ranked initial
  // candidates are discarded for good, yielding a narrower search than asked
  // for. A smaller explicit value is therefore raised to the width (the client
  // still receives at least the logprob count it requested); an unset one is
  // derived from it, matching the RequestParams-level default. Every factory
  // that enables beam search goes through here so the invariant lives in one
  // place. Out-of-range counts are rejected up front by
  // RequestParams::verify_params. A width <= 1 just records the value (no beam
  // search).
  void enable_beam_search(int32_t width) {
    beam_width = width;
    if (beam_width > 1) {
      logprobs = true;
      top_logprobs =
          std::max<int64_t>(top_logprobs, static_cast<int64_t>(beam_width));
    }
  }

  // Whenever logprobs are on, the sampler runs topk(top_logprobs) over the
  // vocabulary, and torch::topk throws for k > vocab. The batch uses
  // max(top_logprobs) across its requests, so a single oversized request would
  // take down every request in the batch. RequestParams::verify_params only
  // knows the model-agnostic 2000 cap; this is the model-aware check for the
  // factories, which know the vocabulary. Returns an error message when the
  // effective top-k exceeds it, std::nullopt when it fits or the vocabulary is
  // unknown (vocab_size <= 0). Call after enable_beam_search so the beam-raised
  // value is what gets checked.
  std::optional<std::string> top_logprobs_vocab_error(
      int64_t vocab_size) const {
    if (!logprobs || vocab_size <= 0 || top_logprobs <= vocab_size) {
      return std::nullopt;
    }
    std::string error = "top_logprobs (" + std::to_string(top_logprobs) + ")";
    if (beam_width > 1 && top_logprobs == static_cast<int64_t>(beam_width)) {
      // The value came from beam_width (unset or smaller top_logprobs was
      // raised to the width), so point the client at the field it actually set.
      error = "beam_width (" + std::to_string(beam_width) + ")";
    }
    return error + " must not exceed the model vocabulary size (" +
           std::to_string(vocab_size) + ")";
  }
};

struct SamplingParameters {
 public:
  void init(const std::vector<const RequestSamplingParam*>& req_sampling_params,
            const std::vector<int32_t>& selected_token_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec,
            const std::vector<torch::Tensor>& filter_mask_rows = {});

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const;

  // selected tokens are tokens for sampling the next token,
  // including the generated tokens and the last prompt token
  // IntTensor
  torch::Tensor selected_token_idxes;

  // Dense additive mask for token-level structured output constraints. Zero
  // entries are allowed and negative entries are forbidden.
  torch::Tensor filter_mask;

  // Compact allowed-token bitmask [num_tokens, ceil(vocab/32)] int32. When
  // defined, Sampler prefers this over filter_mask (smaller H2D).
  torch::Tensor filter_bitmask;

  // [num_tokens] FloatTensor
  torch::Tensor frequency_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor presence_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor repetition_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor temperatures;

  // [num_tokens] FloatTensor
  torch::Tensor top_p;

  // Minimum probability relative to the largest probability in each row.
  torch::Tensor min_p;

  // Per-selected-row additive constraints and independent random streams.
  torch::Tensor logits_bias;
  torch::Tensor seeds;
  torch::Tensor seed_offsets;

  // [num_tokens] LongTensor
  torch::Tensor top_k;

  // the unique token id and count of each sequence in the batch.
  // [num_tokens, max_unique_tokens] LongTensor
  torch::Tensor unique_token_ids;

  // [num_tokens, max_unique_tokens] IntTensor
  torch::Tensor unique_token_counts;

  // the number of unique tokens in each sequence.
  // [num_tokens] IntTensor
  torch::Tensor unique_token_ids_lens;

  // whether to sample for each sequence.
  // [num_seqs] BoolTensor
  // Defined only for mixed batches (see init); use require_do_sample() for
  // per-row dispatch. Per-sequence row domain.
  torch::Tensor do_sample;

  // Beam search accumulated log probability.
  // [num_seq, 1] FloatTensor
  torch::Tensor acc_logprob;

  // How this batch's sequences are sampled (see SamplingMode).
  SamplingMode sampling_mode = SamplingMode::GREEDY;

  // whether to output logprobs for each generated token.
  bool logprobs = false;

  // whether to expose post-temperature sampling logits on SampleOutput.
  bool return_logits = false;

  // whether to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // max number of top logprobs in the batch.
  // only used when logprobs is true.
  int64_t max_top_logprobs = 0;

  // requested final beam result width for request-level beam search output.
  int32_t num_return_sequences = 0;

  // for beam search
  bool use_beam_search = false;

  int64_t num_sample_seqs() const {
    return selected_token_idxes.defined() ? selected_token_idxes.numel() : 0;
  }

  // Whether any post-penalty transform in Sampler::process_logits (top-k,
  // top-p, min-p, or the token filter masks) is active for this batch; fast
  // paths that only apply penalties consult this instead of restating the
  // list.
  bool has_post_penalty_transforms() const;

  // Whether any per-row sampling field (penalties, transforms, seeds, token
  // filters) is defined; a fully greedy batch with no per-row state has none.
  bool has_sampling_fields() const;

  // Fails fast unless the batch is mixed (see do_sample).
  const torch::Tensor& require_do_sample() const {
    CHECK(do_sample.defined())
        << "mixed sampling requires do_sample to be defined";
    return do_sample;
  }
};

// The single registry of SamplingParameters' tensor members that carry
// per-row (per-validate-row) sampling state, each tagged with the roles it
// plays. The derived lists below (repeat/clear/post-penalty) all iterate this
// registry — a new per-row member is added exactly once here, with the roles
// it participates in.
//
// Roles:
// - clear_on_greedy_draft: dead on a greedy draft leg and therefore cleared
//   by clear_sampling_transforms.
// - post_penalty_transform: a transform in Sampler::process_logits applied
//   after the penalties (top-k, top-p, min-p, or the token filter masks).
//
// Per-sequence members (selected_token_idxes, do_sample) are handled by
// their call sites; slot_buffer.cpp's kFields is a separate staging-side
// list.
struct SamplingFieldDesc {
  torch::Tensor SamplingParameters::* member;
  bool clear_on_greedy_draft;
  bool post_penalty_transform;
};

inline constexpr std::array<SamplingFieldDesc, 15> kSamplingFields = {
    {{&SamplingParameters::frequency_penalties, false, false},
     {&SamplingParameters::presence_penalties, false, false},
     {&SamplingParameters::repetition_penalties, false, false},
     {&SamplingParameters::temperatures, true, false},
     {&SamplingParameters::top_p, true, true},
     {&SamplingParameters::min_p, true, true},
     {&SamplingParameters::seed_offsets, true, false},
     {&SamplingParameters::seeds, true, false},
     {&SamplingParameters::logits_bias, false, false},
     {&SamplingParameters::top_k, true, true},
     {&SamplingParameters::unique_token_ids, false, false},
     {&SamplingParameters::unique_token_counts, false, false},
     {&SamplingParameters::unique_token_ids_lens, false, false},
     {&SamplingParameters::filter_mask, false, true},
     {&SamplingParameters::filter_bitmask, false, true}}};

// Out-of-line because it derives from kSamplingFields (below the struct).
inline bool SamplingParameters::has_post_penalty_transforms() const {
  for (const auto& field : kSamplingFields) {
    if (field.post_penalty_transform && (this->*field.member).defined()) {
      return true;
    }
  }
  return false;
}

// Out-of-line because it derives from kSamplingFields (below the struct).
inline bool SamplingParameters::has_sampling_fields() const {
  return std::any_of(kSamplingFields.begin(),
                     kSamplingFields.end(),
                     [this](const SamplingFieldDesc& field) {
                       return (this->*field.member).defined();
                     });
}

// Repeats every per-row sampling field across speculative rows while
// preserving undefined optional fields. Repeats can be a scalar count or a
// tensor of per-row counts.
template <typename Repeats>
inline void repeat_sampling_params(SamplingParameters& sampling_params,
                                   const Repeats& repeats) {
  for (const auto& field : kSamplingFields) {
    repeat_interleave_if_defined(sampling_params.*field.member, repeats);
  }
}

// Clears the per-row transform and seed fields that are dead on a greedy
// draft leg (see clear_on_greedy_draft above).
inline void clear_sampling_transforms(SamplingParameters& sampling_params) {
  for (const auto& field : kSamplingFields) {
    if (field.clear_on_greedy_draft) {
      sampling_params.*field.member = torch::Tensor();
    }
  }
}

// Expands per-sequence sampling state onto the validate row domain (one row
// per speculative token), so RejectionSampler and SlotBuffer see one tensor
// per validate row.
template <typename Repeats>
inline void expand_sampling_params_for_verify(
    SamplingParameters& sampling_params,
    const Repeats& repeats,
    int64_t total_num_val_tokens,
    const torch::Device& device) {
  sampling_params.selected_token_idxes =
      arange_indices(total_num_val_tokens, device);
  repeat_sampling_params(sampling_params, repeats);
  repeat_interleave_if_defined(sampling_params.do_sample, repeats);
}

// Per-sequence variant: uploads the row counts as an int64 device tensor and
// delegates to the tensor overload above.
inline void expand_sampling_params_for_verify(
    SamplingParameters& sampling_params,
    const std::vector<int32_t>& per_seq_val_tokens,
    int64_t total_num_val_tokens,
    const torch::Device& device) {
  torch::Tensor repeats_tensor =
      async_h2d_tensor(std::vector<int64_t>(per_seq_val_tokens.begin(),
                                            per_seq_val_tokens.end()),
                       device);
  expand_sampling_params_for_verify(
      sampling_params, repeats_tensor, total_num_val_tokens, device);
}

// Mode-aware draft sampling policy: clears logprob outputs and, in GREEDY
// mode, flips the batch to greedy; PROBABILISTIC exposes draft logits
// exactly when the verifier needs q.
void prepare_draft_sampling(DraftSamplingMode mode,
                            SamplingParameters& sampling_params);

// Draft proposals and acceptances executed by the verifier, before output
// truncation. Accepted drafts exclude the target replacement/bonus token.
struct SpeculativeTokenStats {
  int64_t accepted_tokens = 0;
  int64_t proposed_tokens = 0;
};

class SampleOutput final {
 public:
  // Host access only: tensors must be CPU-contiguous. next_tokens is int32 or
  // int64 with shape [num_seq] for ordinary sampling, or [num_seq, width] for
  // speculative sampling. Optional float32 logprobs match that shape; paired
  // int64 top_tokens/float32 top_logprobs append a top-k dimension.
  // Returned top-k slices borrow this output's tensor storage; keep it alive
  // and unchanged while using the Token. The overload must match token rank.
  Token token_at(int64_t output_idx) const;
  Token token_at(int64_t output_idx, int64_t token_idx) const;

  // [num_seq, ...] LongTensor
  torch::Tensor next_tokens;

  // [num_seq, ..., vocab] post-temperature logits when return_logits.
  torch::Tensor draft_logits;

  // [num_seq, ...] FloatTensor
  torch::Tensor logprobs;

  // [num_seq, ..., top_k] FloatTensor
  torch::Tensor top_logprobs;
  // [num_seq, ..., top_k] LongTensor
  torch::Tensor top_tokens;

  // [num_seq, ..., embed_dim] FloatTensor (full forward hidden on the
  // speculative target-prefill path; see `selected_embeddings`).
  torch::Tensor embeddings;

  // Per-sequence target-prefill hidden: one row per sequence, in
  // `logits_indices` order. Separate from `embeddings` because under CP
  // `embeddings` holds only the local shard, whose rows cannot be re-indexed
  // per-sequence by the all-gather-space `logits_indices`.
  torch::Tensor selected_embeddings;

  std::vector<std::vector<torch::Tensor>> mm_embeddings;
};

}  // namespace xllm
