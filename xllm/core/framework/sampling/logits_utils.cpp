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

#include "core/framework/sampling/logits_utils.h"

#include <torch/torch.h>

#include <memory>

namespace xllm {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  const torch::Tensor penalty =
      unique_token_counts * frequency_penalties.unsqueeze(1) +
      (unique_token_counts > 0) * presence_penalties.unsqueeze(1);
  // Padded zero-count entries must never overwrite a real token's penalty.
  logits.scatter_add_(
      /*dim=*/1, unique_token_ids, (-penalty).to(logits.scalar_type()));
}

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties,
                                const torch::Tensor& unique_token_ids_lens) {
  const torch::Tensor scores = logits.gather(/*dim=*/1, unique_token_ids);
  const torch::Tensor factors = penalties.unsqueeze(1);
  torch::Tensor delta =
      torch::where(scores < 0, scores * factors, scores / factors) - scores;
  delta.masked_fill_(~torch::isfinite(scores), 0);
  if (unique_token_ids_lens.defined()) {
    const torch::Tensor positions =
        torch::arange(unique_token_ids.size(1), unique_token_ids.options());
    delta.masked_fill_(
        positions.unsqueeze(0) >= unique_token_ids_lens.unsqueeze(1), 0);
  }
  logits.scatter_add_(
      /*dim=*/1, unique_token_ids, delta.to(logits.scalar_type()));
}

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures) {
  auto unsqueezed_temperatures = temperatures.unsqueeze(1);
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

void apply_top_k_top_p_torch_impl(torch::Tensor& logits,
                                  const torch::Tensor& top_k,
                                  const torch::Tensor& top_p) {
  const int64_t vocab_size = logits.size(-1);
  const float filter_value = -std::numeric_limits<float>::infinity();
  auto [sorted_logits, indices] = logits.sort(/*dim=*/-1, /*descending=*/false);
  if (top_k.defined()) {
    const torch::Tensor k = torch::where(top_k <= 0, vocab_size, top_k)
                                .clamp(1, vocab_size)
                                .to(torch::kLong);
    const torch::Tensor threshold = sorted_logits.gather(
        /*dim=*/-1, (vocab_size - k).unsqueeze(-1));
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

void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p) {
  if (!min_p.defined()) {
    return;
  }
  const torch::Tensor max_logits =
      std::get<0>(logits.max(/*dim=*/-1, /*keepdim=*/true));
  const torch::Tensor threshold =
      max_logits + min_p.to(torch::kFloat32).log().unsqueeze(-1);
  logits.masked_fill_(logits < threshold,
                      -std::numeric_limits<float>::infinity());
}

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& temperatures,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p) {
  if (temperatures.defined()) {
    apply_temperatures(logits, temperatures);
  }
  if (!top_k.defined() && !top_p.defined()) {
    return;
  }

#if defined(USE_MLU)
  if (!logits.device().is_cpu() && (top_k.defined() || top_p.defined())) {
    xllm::kernel::TopKPParams params;
    params.logits = logits;
    params.top_k = top_k;
    params.top_p = top_p;
    logits = xllm::kernel::apply_top_k_top_p(params);
    return;
  }
#endif

#if defined(USE_NPU)
  if (!logits.device().is_cpu() && top_k.defined() && top_p.defined()) {
    const int64_t vocab_size = logits.size(-1);
    const torch::Tensor processed_top_k =
        torch::where(top_k <= 0, vocab_size, top_k)
            .clamp(1, vocab_size)
            .to(torch::kInt32);
    xllm::kernel::npu::top_k_top_p(logits, processed_top_k, top_p);
    return;
  }
#endif
  apply_top_k_top_p_torch_impl(logits, top_k, top_p);
}

}  // namespace xllm
