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
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/util/tensor_helper.h"

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
  int32_t beam_width = 0;
  int32_t num_return_sequences = 0;
};

struct SamplingParameters {
 public:
  void init(const std::vector<const RequestSamplingParam*>& req_sampling_params,
            const std::vector<int32_t>& selected_token_idxes,
            const std::vector<int32_t>& sample_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec);

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const {
    SamplingParameters params;

    // selected/sample indices are tiny control tensors and
    // correctness-critical. Use blocking H2D copies to avoid consuming
    // partially transferred index buffers on NPU runtime paths.
    params.selected_token_idxes =
        selected_token_idxes.defined()
            ? safe_to(selected_token_idxes, device).contiguous()
            : selected_token_idxes;

    auto options = torch::device(device).dtype(dtype);
    params.frequency_penalties = safe_to(frequency_penalties, options, true);
    params.presence_penalties = safe_to(presence_penalties, options, true);
    params.repetition_penalties = safe_to(repetition_penalties, options, true);
    params.temperatures = safe_to(temperatures, options, true);
    params.top_p = safe_to(top_p, options, true);
    params.top_k = safe_to(top_k, device, true);

    params.unique_token_ids = safe_to(unique_token_ids, device, true);
    params.unique_token_counts = safe_to(unique_token_counts, device, true);
    params.unique_token_ids_lens = safe_to(unique_token_ids_lens, device, true);

    params.sample_idxes = sample_idxes.defined()
                              ? safe_to(sample_idxes, device).contiguous()
                              : sample_idxes;
    params.do_sample = safe_to(do_sample, device, true);
    params.acc_logprob = safe_to(acc_logprob, device, true);
    params.all_random_sample = all_random_sample;
    params.all_greedy_sample = all_greedy_sample;
    params.logprobs = logprobs;
    params.return_probs = return_probs;
    params.max_top_logprobs = max_top_logprobs;
    params.is_embeddings = is_embeddings;
    params.num_return_sequences = num_return_sequences;

    // for beam search
    params.use_beam_search = use_beam_search;
    return params;
  }

  // concat two SamplingParameters into one
  void concat(const SamplingParameters& param);

  // selected tokens are tokens for sampling the next token,
  // including the generated tokens and the last prompt token
  // IntTensor
  torch::Tensor selected_token_idxes;

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

  // wheteher to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // max number of top logprobs in the batch.
  // only used when logprobs is true.
  int64_t max_top_logprobs = 0;

  // requested final beam result width for request-level beam search output.
  int32_t num_return_sequences = 0;

  // for beam search
  bool use_beam_search = false;
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
};

}  // namespace xllm
