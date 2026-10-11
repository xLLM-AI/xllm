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

#include "core/framework/sampling/sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <mutex>

#include "common/global_flags.h"
#include "core/framework/config/model_config.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/logits_utils.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace {

uint64_t sampling_seed(int64_t seed, int64_t offset) {
  uint64_t value = static_cast<uint64_t>(seed) +
                   static_cast<uint64_t>(offset) * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

}  // namespace

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) const {
  const torch::Tensor& effective_filter_mask =
      filter_mask.defined() ? filter_mask : params.filter_mask;
  SampleOutput output;
  // vLLM reports probabilities from the unmodified model distribution.
  const torch::Tensor raw_logprobs =
      params.logprobs && !params.use_beam_search
          ? torch::log_softmax(logits, /*dim=*/-1, torch::kFloat32)
          : torch::Tensor();
  if (params.logits_bias.defined()) {
    logits.add_(params.logits_bias);
  }
  // Apply repetition before frequency and presence penalties.
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(logits,
                               params.unique_token_ids,
                               params.repetition_penalties,
                               params.unique_token_ids_lens);
  }

  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  torch::Tensor sample_logits = logits;
  torch::Tensor sample_temperatures = params.temperatures;
  torch::Tensor sample_top_k = params.top_k;
  torch::Tensor sample_top_p = params.top_p;
  torch::Tensor sample_min_p = params.min_p;
  torch::Tensor sample_filter_bitmask = params.filter_bitmask;
  const bool use_sample_indices =
      params.selected_token_idxes.numel() != params.sample_idxes.numel();
  if (use_sample_indices) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
    if (params.temperatures.defined()) {
      sample_temperatures =
          params.temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_k.defined()) {
      sample_top_k = params.top_k.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.top_p.defined()) {
      sample_top_p = params.top_p.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (params.min_p.defined()) {
      sample_min_p = params.min_p.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (sample_filter_bitmask.defined()) {
      sample_filter_bitmask =
          sample_filter_bitmask.index_select(/*dim=*/0, params.sample_idxes);
    }
  }

  if (sample_filter_bitmask.defined()) {
    apply_token_bitmask_inplace(sample_logits, sample_filter_bitmask);
  } else if (effective_filter_mask.defined()) {
    CHECK_EQ(effective_filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << effective_filter_mask.dim();
    CHECK_EQ(effective_filter_mask.size(0), sample_logits.size(0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << effective_filter_mask.size(0)
        << ", sample_logits.size(0)=" << sample_logits.size(0);
    CHECK_EQ(effective_filter_mask.size(1), sample_logits.size(1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << effective_filter_mask.size(1)
        << ", sample_logits.size(1)=" << sample_logits.size(1);
    sample_logits = sample_logits + effective_filter_mask;
  }

  if (params.all_greedy_sample && !params.logprobs && !params.return_probs &&
      !use_sample_indices && !filter_mask.defined()) {
    output.next_tokens = greedy_sample(sample_logits).to(torch::kLong);
    return output;
  }

  if (params.all_greedy_sample && !params.logprobs && params.return_probs &&
      !use_sample_indices && !filter_mask.defined()) {
    torch::Tensor sample_indices =
        greedy_sample(sample_logits).to(torch::kLong);
    torch::Tensor selected_logits =
        sample_logits.gather(/*dim=*/-1, sample_indices.view({-1, 1}))
            .to(torch::kFloat32);
    torch::Tensor log_probs =
        selected_logits - torch::logsumexp(sample_logits,
                                           /*dim=*/-1,
                                           /*keepdim=*/true);
    output.next_tokens = sample_indices;
    output.probs = log_probs.exp().view({-1}).to(logits.dtype());
    return output;
  }

  // apply temperatures, top-k and top-p
  if (sample_temperatures.defined()) {
    apply_temperatures(sample_logits, sample_temperatures);
  }
  apply_min_p(sample_logits, sample_min_p);
  apply_top_k_top_p(sample_logits, torch::Tensor(), sample_top_k, sample_top_p);
  if (use_sample_indices) {
    logits.index_copy_(/*dim=*/0, params.sample_idxes, sample_logits);
  }

  CHECK(params.do_sample.defined()) << "params.do_sample must be defined";
  CHECK_EQ(params.do_sample.dim(), 1)
      << "params.do_sample must be 1D [num_seqs], got "
      << params.do_sample.sizes();
  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto probs =
      torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  torch::Tensor samples;
  if (params.seeds.defined() && !params.all_greedy_sample) {
    torch::Tensor seeds = params.seeds;
    torch::Tensor offsets = params.seed_offsets;
    if (use_sample_indices) {
      seeds = seeds.index_select(0, params.sample_idxes);
      offsets = offsets.index_select(0, params.sample_idxes);
    }
    seeds = seeds.to(torch::kCPU);
    offsets = offsets.to(torch::kCPU);
    const torch::Tensor sample_rows = params.do_sample.to(torch::kCPU);
    std::vector<torch::Tensor> rows;
    rows.reserve(probs.size(0));
    for (int64_t row = 0; row < probs.size(0); ++row) {
      const torch::Tensor probabilities = probs.select(0, row);
      if (!sample_rows[row].item<bool>()) {
        rows.emplace_back(probabilities.argmax(-1).view({1}));
        continue;
      }
      const int64_t seed = seeds[row].item<int64_t>();
      if (seed == -1) {
        rows.emplace_back(random_sample(probabilities.unsqueeze(0)).view({1}));
        continue;
      }
      // Cloning keeps this request independent from the process-global RNG.
      auto default_generator =
          torch::globalContext().defaultGenerator(probs.device());
      auto generator = [&default_generator]() {
        std::lock_guard<std::mutex> lock(default_generator.mutex());
        return default_generator.clone();
      }();
      generator.set_current_seed(
          sampling_seed(seed, offsets[row].item<int64_t>()));
      rows.emplace_back(probabilities.multinomial(/*num_samples=*/1,
                                                  /*replacement=*/false,
                                                  generator));
    }
    samples = torch::cat(rows, /*dim=*/0);
  } else if (params.all_random_sample) {
    samples = random_sample(probs);
  } else if (params.all_greedy_sample) {
    samples = greedy_sample(probs);
  } else {
    // mixed sample, sample both then choose based on do_sample
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(params.do_sample, random, greedy);
  }
  auto sample_indices = samples.to(torch::kLong);
  output.probs = probs.to(logits.dtype());
  output.next_tokens = sample_indices;

  if (params.logprobs) {
    if (::xllm::ModelConfig::get_instance().enable_qwen3_reranker()) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      sample_logits = sample_logits.index_select(/*dim=*/1, indices);
      auto logprobs = torch::log_softmax(
          sample_logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const torch::Tensor logprobs =
        raw_logprobs.defined()
            ? (use_sample_indices
                   ? raw_logprobs.index_select(0, params.sample_idxes)
                   : raw_logprobs)
            : torch::log_softmax(sample_logits, /*dim=*/-1, torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs =
        logprobs.gather(/*dim=*/-1, sample_indices.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
      // max_top_logprobs is the batch-wide max, so one request asking for more
      // than the vocabulary would make topk throw and fail every request in the
      // batch. The factories reject such requests up front
      // (RequestSamplingParam::top_logprobs_vocab_error); clamp here as the
      // last line of defense so the sampler can never be the failure point.
      // Every consumer sizes by the returned row width, so a shorter row is
      // safe.
      const int64_t k =
          std::min<int64_t>(params.max_top_logprobs, logprobs.size(-1));
      auto [values, indices] = logprobs.topk(k, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
#if defined(USE_MLU) || defined(USE_CUDA) || defined(USE_DCU)
  xllm::kernel::RandomSampleParams params;
  params.logits = probs;
  return xllm::kernel::random_sample(params);
#endif
  if (probs.dim() == 3) {
    auto batch_size = probs.size(0);
    auto seq_len = probs.size(1);
    auto vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});
    auto sampled =
        flat_probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
    return sampled.reshape({batch_size, seq_len});
  } else {
    return probs.multinomial(/*num_samples=*/1, /*replacement=*/false)
        .flatten();
  }
}

}  // namespace xllm
