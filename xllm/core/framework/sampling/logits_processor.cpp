/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/sampling/logits_processor.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>

#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

#include "core/framework/sampling/sampling_params.h"
#include "core/kernels/ops_api.h"

namespace xllm {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  const torch::Tensor penalty =
      unique_token_counts * frequency_penalties.unsqueeze(-1) +
      (unique_token_counts > 0) * presence_penalties.unsqueeze(-1);
  // Padded zero-count entries must never overwrite a real token's penalty.
  // dim=-1 serves both the flat and grouped domains; on the grouped domain
  // the index gains a width axis via a stride-0 expand (no materialization).
  const torch::Tensor src = (-penalty).to(logits.scalar_type());
  logits.scatter_add_(
      /*dim=*/-1,
      unique_token_ids,
      unique_token_ids.dim() == src.dim() + 1
          ? expand_batch_to_tokens(src, unique_token_ids.size(1))
          : src);
}

void apply_min_token_block(torch::Tensor& logits,
                           const torch::Tensor& ids,
                           const torch::Tensor& min_tokens_remaining) {
  CHECK(min_tokens_remaining.defined());
  CHECK_EQ(ids.dim(), 2);
  CHECK_EQ(ids.size(0), logits.size(0));
  CHECK_EQ(min_tokens_remaining.numel(), logits.size(0));
  const auto active =
      torch::logical_and(ids.ge(0), min_tokens_remaining.gt(0).unsqueeze(-1));
  const auto bias =
      torch::zeros(ids.sizes(), logits.options())
          .masked_fill(active, -std::numeric_limits<float>::infinity());
  // Additive, not scatter-assign: padding maps to index 0, and assignment
  // could race a real stop id 0 back to unblocked.
  torch::Tensor index = ids.clamp_min(0);
  torch::Tensor src = bias;
  if (logits.dim() == index.dim() + 1) {
    index = expand_batch_to_tokens(index, logits.size(1));
    src = expand_batch_to_tokens(bias, logits.size(1));
  }
  logits.scatter_add_(/*dim=*/-1, index, src);
}

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties,
                                const torch::Tensor& unique_token_ids_lens) {
  const torch::Tensor scores = logits.gather(/*dim=*/-1, unique_token_ids);
  const torch::Tensor factors = penalties.unsqueeze(-1);
  torch::Tensor delta =
      torch::where(scores < 0, scores * factors, scores / factors) - scores;
  delta.masked_fill_(~torch::isfinite(scores), 0);
  if (unique_token_ids_lens.defined()) {
    const torch::Tensor positions =
        torch::arange(unique_token_ids.size(-1), unique_token_ids.options());
    delta.masked_fill_(
        positions.unsqueeze(0) >= unique_token_ids_lens.unsqueeze(-1), 0);
  }
  logits.scatter_add_(
      /*dim=*/-1, unique_token_ids, delta.to(logits.scalar_type()));
}

// Rank-generic min-p ([batch] scalars broadcast like apply_temperatures).
void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p) {
  if (!min_p.defined()) {
    return;
  }
  const torch::Tensor max_logits =
      std::get<0>(logits.max(/*dim=*/-1, /*keepdim=*/true));
  const torch::Tensor threshold =
      max_logits +
      min_p.to(torch::kFloat32)
          .log()
          .reshape(batch_broadcast_shape(min_p.numel(), logits.dim()));
  logits.masked_fill_(logits < threshold,
                      -std::numeric_limits<float>::infinity());
}

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p) {
  if (!top_k.defined() && !top_p.defined()) {
    return;
  }

#if defined(USE_MLU)
  // The device kernels are flat-row [rows, vocab] contracts.
  if (!logits.device().is_cpu()) {
    xllm::kernel::TopKPParams params;
    params.logits = logits;
    params.top_k = top_k;
    params.top_p = top_p;
    logits = xllm::kernel::apply_top_k_top_p(params);
    return;
  }
#endif

  // k <= 0 means no limit (the full vocabulary); clamp into [1, vocab].
  const torch::Tensor normalized_k =
      top_k.defined() ? torch::where(top_k <= 0, logits.size(-1), top_k)
                            .clamp(1, logits.size(-1))
                      : torch::Tensor();

#if defined(USE_NPU)
  if (!logits.device().is_cpu() && top_k.defined() && top_p.defined()) {
    xllm::kernel::npu::top_k_top_p(
        logits, normalized_k.to(torch::kInt32), top_p);
    return;
  }
#endif

  const int64_t vocab_size = logits.size(-1);
  const float filter_value = -std::numeric_limits<float>::infinity();
  auto [sorted_logits, indices] = logits.sort(/*dim=*/-1, /*descending=*/false);
  if (top_k.defined()) {
    const torch::Tensor k = normalized_k.to(torch::kLong);
    const torch::Tensor threshold =
        sorted_logits.gather(/*dim=*/-1, (vocab_size - k).unsqueeze(-1));
    // Preserve ties at the kth largest logit, as vLLM does.
    sorted_logits.masked_fill_(sorted_logits < threshold, filter_value);
  }
  if (top_p.defined()) {
    const torch::Tensor probabilities =
        sorted_logits.softmax(-1, torch::kFloat32);
    torch::Tensor mask = probabilities.cumsum(-1) <= (1 - top_p.unsqueeze(-1));
    mask.select(/*dim=*/-1, /*index=*/vocab_size - 1).fill_(false);
    sorted_logits.masked_fill_(mask, filter_value);
  }
  logits.scatter_(/*dim=*/-1, indices, sorted_logits);
}

void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask) {
  CHECK(logits.defined()) << "logits must be defined";
  CHECK(bitmask.defined()) << "bitmask must be defined";
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D [batch, vocab]";
  CHECK_EQ(bitmask.dim(), 2) << "bitmask must be 2-D [batch, words]";
  CHECK_EQ(logits.size(0), bitmask.size(0))
      << "bitmask batch mismatch, logits.size(0)=" << logits.size(0)
      << ", bitmask.size(0)=" << bitmask.size(0);

  const int64_t vocab_size = logits.size(1);
  const int64_t expected_words = (vocab_size + 31) / 32;
  CHECK_EQ(bitmask.size(1), expected_words)
      << "bitmask word count mismatch, bitmask.size(1)=" << bitmask.size(1)
      << ", expected=" << expected_words;
  CHECK(bitmask.scalar_type() == torch::kInt32 ||
        bitmask.scalar_type() == torch::kInt64)
      << "bitmask must be int32/int64";

#if defined(USE_NPU)
  if (kernel::npu::tilelang::can_apply_token_bitmask_inplace(logits, bitmask)) {
    kernel::npu::tilelang::apply_token_bitmask_inplace(logits, bitmask);
    return;
  }
#endif

  // The word/bit index tables depend only on (vocab_size, device); cache
  // them like apply_temperatures' one_scalar so constrained decode does not
  // rebuild full-vocabulary temporaries every step.
  static thread_local torch::Tensor cached_word_indices;
  static thread_local torch::Tensor cached_bit_indices;
  static thread_local int64_t cached_vocab_size = -1;
  static thread_local torch::Device cached_device(torch::kCPU);
  const torch::Device device = logits.device();
  if (vocab_size != cached_vocab_size || device != cached_device) {
    const auto options =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    const torch::Tensor token_ids = torch::arange(vocab_size, options);
    cached_word_indices = torch::floor_divide(token_ids, /*other=*/32);
    cached_bit_indices = torch::remainder(token_ids, /*other=*/32);
    cached_vocab_size = vocab_size;
    cached_device = device;
  }
  const torch::Tensor& word_indices = cached_word_indices;
  const torch::Tensor& bit_indices = cached_bit_indices;
  torch::Tensor words = bitmask.to(torch::kInt64)
                            .bitwise_and(/*other=*/0xffffffffLL)
                            .index_select(/*dim=*/1, word_indices);
  const torch::Tensor allowed =
      torch::bitwise_and(torch::bitwise_right_shift(words, bit_indices),
                         /*other=*/1);
  const auto mask_options =
      torch::TensorOptions().dtype(logits.dtype()).device(logits.device());
  const torch::Tensor additive = torch::where(
      allowed.to(torch::kBool),
      torch::zeros({1}, mask_options),
      torch::full({1}, -std::numeric_limits<float>::infinity(), mask_options));
  logits.add_(additive);
}

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures) {
  auto unsqueezed_temperatures = temperatures.reshape(
      batch_broadcast_shape(temperatures.size(0), logits.dim()));
  // Cache device-side scalar to avoid synchronous H2D copy that forces
  // aclrtSynchronizeStream per forward.
  static thread_local torch::Tensor one_scalar;
  const torch::Device& target_device = unsqueezed_temperatures.device();
  if (!one_scalar.defined() || one_scalar.device() != target_device) {
    one_scalar =
        torch::full({}, 1.0, torch::TensorOptions().device(target_device));
  }
  unsqueezed_temperatures = torch::where(
      unsqueezed_temperatures == 0, one_scalar, unsqueezed_temperatures);

  logits.div_(unsqueezed_temperatures);
}

void apply_sampling_penalties(torch::Tensor& logits,
                              const SamplingParameters& params) {
  if (params.logits_bias.defined()) {
    logits.add_(params.logits_bias);
  }
  if (params.min_token_ids.defined()) {
    apply_min_token_block(
        logits, params.min_token_ids, params.min_tokens_remaining);
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

void process_logits(torch::Tensor& logits,
                    const SamplingParameters& params,
                    const torch::Tensor& filter_mask) {
  const torch::Tensor& effective_filter_mask =
      filter_mask.defined() ? filter_mask : params.filter_mask;
  apply_sampling_penalties(logits, params);

  // Only penalties/bias/filters can move the argmax; the rest is
  // argmax-preserving (see SamplingParameters::is_all_greedy).
  if (params.filter_bitmask.defined()) {
    apply_token_bitmask_inplace(logits, params.filter_bitmask);
  } else if (effective_filter_mask.defined()) {
    CHECK_EQ(effective_filter_mask.sizes(), logits.sizes())
        << "filter_mask must match the logits shape";
    logits.add_(effective_filter_mask);
  }
  if (params.is_all_greedy()) {
    return;
  }

  torch::Tensor top_p = params.top_p;
  if (params.top_k.defined() && params.top_p.defined()) {
    // Native top-k/top-p consumes top-p in the logits dtype.
    top_p = top_p.to(logits.scalar_type());
  }
  const bool has_transforms = params.temperatures.defined() ||
                              params.min_p.defined() ||
                              params.top_k.defined() || top_p.defined();
  if (has_transforms) {
    kernel::FusedLogitsTransformParams transform_params{
        logits, params.temperatures, params.min_p, params.top_k, top_p};
    if (kernel::supports_fused_logits_transform(transform_params)) {
      // Falling through would apply the same transforms a second time.
      kernel::fused_logits_transform(transform_params);
      return;
    }
  }
  if (params.temperatures.defined()) {
    apply_temperatures(logits, params.temperatures);
  }
  apply_min_p(logits, params.min_p);
  apply_top_k_top_p(logits, params.top_k, top_p);
}

}  // namespace xllm
