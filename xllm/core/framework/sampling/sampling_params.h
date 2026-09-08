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
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

struct RequestSamplingParam {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 0.0;
  float top_p = 1.0;
  int64_t top_k = -1;
  bool logprobs = false;
  int64_t top_logprobs = 0;
  bool do_sample = false;
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
            const std::vector<int32_t>& sample_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec,
            const std::vector<torch::Tensor>& filter_mask_rows = {});

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const;

  // concat two SamplingParameters into one
  void concat(const SamplingParameters& param);

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

  // the last index of the selected tokens for sampling.
  // [num_seqs] IntTensor
  torch::Tensor sample_idxes;

  // whether to sample for each sequence.
  // [num_seqs] BoolTensor
  torch::Tensor do_sample;

  // Beam search accumulated log probability.
  // [num_seq, 1] FloatTensor
  torch::Tensor acc_logprob;

  bool all_random_sample = false;
  bool all_greedy_sample = true;

  // whether to output logprobs for each generated token.
  bool logprobs = false;

  // whether downstream validation needs sampled probabilities even when the
  // request itself does not ask for logprobs.
  bool return_probs = false;

  // whether to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // max number of top logprobs in the batch.
  // only used when logprobs is true.
  int64_t max_top_logprobs = 0;

  // requested final beam result width for request-level beam search output.
  int32_t num_return_sequences = 0;

  // for beam search
  bool use_beam_search = false;
};

struct SpeculativeTokenStats {
  int64_t accepted_tokens = 0;
  int64_t proposed_tokens = 0;
};

struct SampleOutput {
  // [num_seq, ...] LongTensor
  torch::Tensor next_tokens;

  // [num_seq, ...] FloatTensor
  torch::Tensor probs;

  // [num_seq, ...] FloatTensor
  torch::Tensor logprobs;

  // [num_seq, ..., top_k] FloatTensor
  torch::Tensor top_logprobs;
  // [num_seq, ..., top_k] LongTensor
  torch::Tensor top_tokens;

  // [num_seq, ..., embed_dim] FloatTensor
  torch::Tensor embeddings;

  // Per-sequence selected target hidden states for speculative decoding. Under
  // context parallelism the prefill `embeddings` above holds the full LOCAL
  // hidden shard (rows = local token count) for the draft input_embedding,
  // whose rows cannot be indexed by the CP all-gather-space selected indices.
  // This field carries the already-gathered per-sequence hidden (rows =
  // num_seq) produced by the LmHead, so the embedding cache can store it
  // directly without re-selecting. Only set on the CP target prefill path.
  torch::Tensor selected_embeddings;

  std::vector<std::vector<torch::Tensor>> mm_embeddings;
  std::vector<SpeculativeTokenStats> speculative_token_stats;
};

}  // namespace xllm
