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

#include "rec_sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <tuple>

#include "common/global_flags.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/rec_config.h"
#include "logits_utils.h"
#include "sampler.h"
#if defined(USE_CUDA)
#include "kernels/cuda/cuda_ops_api.h"
#endif

namespace xllm {
namespace {

static inline bool use_air_log_softmax_env() {
  static const bool enabled = []() {
    const char* v = std::getenv("XLLM_USE_AIR_LOG_SOFTMAX");
    if (!v) {
      return false;
    }
    char c = v[0];
    return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
  }();
  return enabled;
}

// Check if fast path sampling can be used for multi-round pipeline.
static inline bool can_use_fast_path(const SamplingParameters& params) {
  return params.use_beam_search && params.logprobs &&
         ::xllm::RecConfig::get_instance().enable_rec_fast_sampler() &&
         params.max_top_logprobs > 0 && !params.top_p.defined() &&
         !::xllm::ModelConfig::get_instance().enable_qwen3_reranker() &&
         ::xllm::RecConfig::get_instance().max_decode_rounds() > 0;
}

static inline torch::Tensor log_softmax_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& temperatures) {
  const bool has_temps = temperatures.defined();
#if defined(USE_CUDA)
  if (input.is_cuda() && use_air_log_softmax_env()) {
    return kernel::cuda::air_log_softmax_last_dim(input, temperatures);
  }
#endif

  if (!has_temps) {
    return torch::log_softmax(input, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  }

  auto logits = input.to(torch::kFloat32);
  auto temps = temperatures.to(torch::kFloat32).to(input.device()).unsqueeze(1);
  temps = torch::where(temps == 0, torch::ones_like(temps), temps);
  logits.div_(temps);
  return torch::log_softmax(logits, /*dim=*/-1);
}

static inline void sample_top_candidates(const torch::Tensor& probs,
                                         const torch::Tensor& logprobs,
                                         int64_t top_count,
                                         torch::Tensor* top_tokens,
                                         torch::Tensor* top_logprobs) {
  CHECK(top_tokens != nullptr);
  CHECK(top_logprobs != nullptr);
  CHECK_EQ(probs.dim(), 2) << "probs must be 2D, got " << probs.sizes();
  CHECK_EQ(logprobs.dim(), 2)
      << "logprobs must be 2D, got " << logprobs.sizes();
  CHECK_EQ(probs.sizes(), logprobs.sizes())
      << "probs/logprobs shape mismatch, probs=" << probs.sizes()
      << ", logprobs=" << logprobs.sizes();
  CHECK_GT(top_count, 0) << "top_count must be positive";

  const int64_t batch_size = probs.size(0);
  auto device = probs.device();
  auto token_options =
      torch::TensorOptions().dtype(torch::kLong).device(device);
  auto logprob_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  *top_tokens = torch::empty({batch_size, top_count}, token_options);
  *top_logprobs = torch::empty({batch_size, top_count}, logprob_options);

  auto valid_counts = probs.gt(0).sum(/*dim=*/-1).to(torch::kCPU);
  for (int64_t row = 0; row < batch_size; ++row) {
    auto probs_row = probs[row];
    auto logprobs_row = logprobs[row];
    int64_t valid_count = valid_counts[row].item<int64_t>();
    if (valid_count >= top_count) {
      auto sampled = probs_row.multinomial(
          /*num_samples=*/top_count, /*replacement=*/false);
      auto sampled_logprobs = logprobs_row.gather(/*dim=*/-1, sampled);
      torch::Tensor sorted_values;
      torch::Tensor sorted_order;
      std::tie(sorted_values, sorted_order) = sampled_logprobs.sort(
          /*dim=*/-1, /*descending=*/true);
      auto sorted_tokens = sampled.gather(/*dim=*/-1, sorted_order);
      (*top_tokens)[row].copy_(sorted_tokens);
      (*top_logprobs)[row].copy_(sorted_values);
    } else {
      torch::Tensor topk_values;
      torch::Tensor topk_indices;
      std::tie(topk_values, topk_indices) = logprobs_row.topk(
          top_count, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
      (*top_tokens)[row].copy_(topk_indices);
      (*top_logprobs)[row].copy_(topk_values);
    }
  }
}

}  // namespace

RecSampler::RecSampler(RecPipelineType pipeline_type)
    : sampler_(std::make_unique<Sampler>()),
      strategy_(create_sampling_strategy(pipeline_type, *sampler_)) {
  LOG(INFO) << "RecSampler initialized with Sampler delegate.";
}

SampleOutput RecSampler::forward(torch::Tensor& logits,
                                 const SamplingParameters& params,
                                 const torch::Tensor& filter_mask,
                                 const RecSamplingContext* context) const {
  return strategy_->forward(logits, params, filter_mask, context);
}

// --- SamplingStrategy factory ---

std::unique_ptr<RecSampler::SamplingStrategy>
RecSampler::create_sampling_strategy(RecPipelineType type,
                                     const Sampler& sampler) {
  switch (type) {
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<MultiRoundFastPathSamplingStrategy>(sampler);
    case RecPipelineType::kLlmRecDefault:
    case RecPipelineType::kLlmRecWithMmData:
      return std::make_unique<DefaultSamplingStrategy>(sampler);
    case RecPipelineType::kOneRecDefault:
    case RecPipelineType::kOneRecXAttentionPipeline:
      return std::make_unique<OneRecConstrainedSamplingStrategy>(sampler);
    default:
      LOG(FATAL) << "Unknown RecPipelineType: " << static_cast<int32_t>(type);
      __builtin_unreachable();
  }
}

// --- DefaultSamplingStrategy ---

RecSampler::DefaultSamplingStrategy::DefaultSamplingStrategy(
    const Sampler& sampler)
    : sampler_(sampler) {}

SampleOutput RecSampler::DefaultSamplingStrategy::forward(
    torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& filter_mask,
    const RecSamplingContext* context) const {
  UNUSED_PARAMETER(context);
  return sampler_.forward(logits, params, filter_mask);
}

// --- OneRecConstrainedSamplingStrategy ---

RecSampler::OneRecConstrainedSamplingStrategy::
    OneRecConstrainedSamplingStrategy(const Sampler& sampler)
    : sampler_(sampler) {}

SampleOutput RecSampler::OneRecConstrainedSamplingStrategy::forward(
    torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& filter_mask,
    const RecSamplingContext* context) const {
  if (context != nullptr && context->device_constrained_sampler) {
    auto sampled = context->device_constrained_sampler(logits,
                                                       params,
                                                       context->sequence_group,
                                                       context->current_step,
                                                       context->beam_width);
    if (sampled.has_value()) {
      return sampled.value();
    }
  }

  if (!(params.use_beam_search && params.all_random_sample && params.logprobs &&
        params.max_top_logprobs > 0)) {
    return sampler_.forward(logits, params, filter_mask);
  }

  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

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

  apply_top_k_top_p(
      sample_logits, sample_temperatures, sample_top_k, sample_top_p);
  if (use_sample_indices) {
    logits.index_copy_(/*dim=*/0, params.sample_idxes, sample_logits);
  }

  CHECK(params.do_sample.defined()) << "params.do_sample must be defined";
  CHECK_EQ(params.do_sample.dim(), 1)
      << "params.do_sample must be 1D [num_seqs], got "
      << params.do_sample.sizes();
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  SampleOutput output;
  auto probs =
      torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  output.probs = probs.to(logits.dtype());
  auto logprobs =
      torch::log_softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

  const int64_t vocab_size = probs.size(-1);
  const int64_t top_count = std::min<int64_t>(params.max_top_logprobs,
                                              static_cast<int64_t>(vocab_size));
  sample_top_candidates(
      probs, logprobs, top_count, &output.top_tokens, &output.top_logprobs);
  output.next_tokens =
      output.top_tokens.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  output.logprobs =
      output.top_logprobs.select(/*dim=*/1, /*index=*/0).contiguous();
  return output;
}

// --- MultiRoundFastPathSamplingStrategy ---

RecSampler::MultiRoundFastPathSamplingStrategy::
    MultiRoundFastPathSamplingStrategy(const Sampler& sampler)
    : sampler_(sampler) {}

SampleOutput RecSampler::MultiRoundFastPathSamplingStrategy::forward(
    torch::Tensor& logits,
    const SamplingParameters& params,
    const torch::Tensor& filter_mask,
    const RecSamplingContext* context) const {
  (void)filter_mask;
  UNUSED_PARAMETER(context);
  const bool use_fast_path = can_use_fast_path(params);

  if (!use_fast_path) {
    return sampler_.forward(logits, params);
  }

  LOG_FIRST_N(INFO, 1) << "RecSampler fast path activated.";

  SampleOutput output;

  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto [topk_values, topk_indices] = sample_logits.topk(
      params.max_top_logprobs,
      /*dim=*/-1,
      /*largest=*/true,
      /*sorted=*/::xllm::BeamSearchConfig::get_instance().enable_topk_sorted());
  output.top_tokens = (topk_indices.scalar_type() == torch::kLong)
                          ? topk_indices
                          : topk_indices.to(torch::kLong);

  torch::Tensor temperatures;
  if (params.temperatures.defined()) {
    temperatures = params.temperatures;
    if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
      temperatures = temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    temperatures = temperatures.to(torch::kFloat32);
  }

  output.top_logprobs = log_softmax_last_dim(topk_values, temperatures);
  return output;
}

}  // namespace xllm
