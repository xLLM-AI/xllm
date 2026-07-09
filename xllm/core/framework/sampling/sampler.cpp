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

#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "common/global_flags.h"
#include "core/framework/config/model_config.h"
#include "logits_utils.h"
#include "sampling_params.h"

namespace xllm {

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) const {
  SampleOutput output;
  // apply frequency and presence penalties
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  // apply repetition penalties
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  torch::Tensor sample_logits = logits;
  torch::Tensor sample_temperatures = params.temperatures;
  torch::Tensor sample_top_k = params.top_k;
  torch::Tensor sample_top_p = params.top_p;
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
  }

  if (filter_mask.defined()) {
    CHECK_EQ(filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << filter_mask.dim();
    CHECK_EQ(filter_mask.size(0), sample_logits.size(0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << filter_mask.size(0)
        << ", sample_logits.size(0)=" << sample_logits.size(0);
    CHECK_EQ(filter_mask.size(1), sample_logits.size(1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << filter_mask.size(1)
        << ", sample_logits.size(1)=" << sample_logits.size(1);
    sample_logits = sample_logits + filter_mask;
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
  apply_top_k_top_p(
      sample_logits, sample_temperatures, sample_top_k, sample_top_p);
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
  if (params.all_random_sample) {
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
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs =
        logprobs.gather(/*dim=*/-1, sample_indices.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
      auto [values, indices] =
          logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
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
