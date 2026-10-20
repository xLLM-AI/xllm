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

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#endif

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

torch::Generator make_sampling_generator(const torch::Device& device) {
#if defined(USE_NPU)
  if (device.type() == torch::kPrivateUse1) {
    // Native execution does not register the generic default-generator hook.
    return at_npu::detail::createNPUGenerator(device.index());
  }
#endif
  auto generator = torch::globalContext().defaultGenerator(device);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.clone();
}

}  // namespace

void Sampler::apply_sampling_penalties(torch::Tensor& logits,
                                       const SamplingParameters& params) {
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
}

void Sampler::process_logits(torch::Tensor& logits,
                             const SamplingParameters& params,
                             const torch::Tensor& filter_mask) {
  const torch::Tensor& effective_filter_mask =
      filter_mask.defined() ? filter_mask : params.filter_mask;
  apply_sampling_penalties(logits, params);

  // Penalties, logits_bias, and the filter masks can move the argmax, so
  // they always run. Temperature, min-p, top-k, and top-p are argmax
  // preserving (positive scaling and masking that keeps each row's max), so
  // a greedy leg that reports no distribution values skips them entirely.
  if (params.filter_bitmask.defined()) {
    apply_token_bitmask_inplace(logits, params.filter_bitmask);
  } else if (effective_filter_mask.defined()) {
    CHECK_EQ(effective_filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << effective_filter_mask.dim();
    CHECK_EQ(effective_filter_mask.size(0), logits.size(0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << effective_filter_mask.size(0)
        << ", logits.size(0)=" << logits.size(0);
    CHECK_EQ(effective_filter_mask.size(1), logits.size(1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << effective_filter_mask.size(1)
        << ", logits.size(1)=" << logits.size(1);
    logits.add_(effective_filter_mask);
  }
  if (params.sampling_mode == SamplingMode::GREEDY && !params.logprobs &&
      !params.return_logits) {
    return;
  }

  torch::Tensor top_p = params.top_p;
  if (params.top_k.defined() && params.top_p.defined()) {
    // Native top-k/top-p consumes top-p in the logits dtype.
    top_p = top_p.to(logits.scalar_type());
  }
  if (params.temperatures.defined()) {
    apply_temperatures(logits, params.temperatures);
  }
  apply_min_p(logits, params.min_p);
  apply_top_k_top_p(logits, torch::Tensor(), params.top_k, top_p);
}

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) const {
  SampleOutput output;
  // vLLM reports probabilities from the unmodified model distribution.
  const torch::Tensor raw_logprobs =
      params.logprobs && !params.use_beam_search
          ? torch::log_softmax(logits, /*dim=*/-1, torch::kFloat32)
          : torch::Tensor();

  process_logits(logits, params, filter_mask);

  // Keep the model dtype; consumers upcast where the reduction needs fp32.
  if (params.return_logits) {
    output.draft_logits = logits;
  }

  // do_sample is defined only for mixed batches (see
  // SamplingParameters::init); undefined batches never reach the per-row
  // branches below.
  if (params.sampling_mode == SamplingMode::MIXED) {
    const torch::Tensor& do_sample = params.require_do_sample();
    CHECK_EQ(do_sample.dim(), 1)
        << "params.do_sample must be 1D [num_seqs], got " << do_sample.sizes();
    // same batch size
    CHECK_EQ(logits.size(0), do_sample.size(0));
  }

  // The greedy legs argmax the logits directly; softmax feeds only the
  // random legs (argmax(softmax(x)) == argmax(x)).
  const bool seed_sampling =
      params.seeds.defined() && params.sampling_mode != SamplingMode::GREEDY;
  const bool need_probs = params.sampling_mode == SamplingMode::RANDOM ||
                          params.sampling_mode == SamplingMode::MIXED;
  torch::Tensor probs;
  if (need_probs) {
    probs = torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  }
  torch::Tensor samples;
  if (seed_sampling) {
    const torch::Tensor seeds = params.seeds.to(torch::kCPU);
    const torch::Tensor offsets = params.seed_offsets.to(torch::kCPU);
    const torch::Tensor do_sample =
        params.sampling_mode == SamplingMode::RANDOM
            ? torch::Tensor()
            : params.require_do_sample().to(torch::kCPU);
    std::vector<torch::Tensor> rows;
    rows.reserve(probs.size(0));
    for (int64_t row = 0; row < probs.size(0); ++row) {
      const torch::Tensor probabilities = probs.select(0, row);
      if (do_sample.defined() && !do_sample[row].item<bool>()) {
        rows.emplace_back(probabilities.argmax(-1).view({1}));
        continue;
      }
      const int64_t seed = seeds[row].item<int64_t>();
      if (seed == -1) {
        rows.emplace_back(random_sample(probabilities.unsqueeze(0)).view({1}));
        continue;
      }
      // A private generator keeps this request independent from the global RNG.
      auto generator = make_sampling_generator(probs.device());
      generator.set_current_seed(
          sampling_seed(seed, offsets[row].item<int64_t>()));
      rows.emplace_back(probabilities.multinomial(/*num_samples=*/1,
                                                  /*replacement=*/false,
                                                  generator));
    }
    samples = torch::cat(rows, /*dim=*/0);
  } else if (params.sampling_mode == SamplingMode::RANDOM) {
    samples = random_sample(probs);
  } else if (params.sampling_mode == SamplingMode::GREEDY) {
    samples = greedy_sample(logits);
  } else {
    auto random = random_sample(probs);
    auto greedy = greedy_sample(logits);
    samples = torch::where(params.require_do_sample(), random, greedy);
  }
  auto sample_indices = samples.to(torch::kLong);
  output.next_tokens = sample_indices;

  if (params.logprobs) {
    if (::xllm::ModelConfig::get_instance().enable_qwen3_reranker()) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      logits = logits.index_select(/*dim=*/1, indices);
      auto logprobs =
          torch::log_softmax(logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const torch::Tensor logprobs =
        raw_logprobs.defined()
            ? raw_logprobs
            : torch::log_softmax(logits, /*dim=*/-1, torch::kFloat32);
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
  if (!probs.device().is_cpu()) {
    xllm::kernel::RandomSampleParams params;
    params.logits = probs;
    return xllm::kernel::random_sample(params);
  }
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
