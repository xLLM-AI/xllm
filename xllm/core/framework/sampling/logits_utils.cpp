/* Copyright 2025-2026 The xLLM Authors.

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

#include "logits_utils.h"

#include <torch/torch.h>

#include <memory>

namespace xllm {

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);
  score.sub_(unique_token_counts * frequency_penalties.unsqueeze(1));
  score.sub_((unique_token_counts > 0) * presence_penalties.unsqueeze(1));

  logits.scatter_(/*dim=*/1,
                  /*index=*/unique_token_ids,
                  /*core=*/score.to(logits.scalar_type()));
}

void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties) {
  auto unsqueezed_penalties = penalties.unsqueeze(1);
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);
  logits.scatter_(
      /*dim=*/1,
      /*index=*/unique_token_ids,
      /*core=*/
      torch::where(
          score < 0, score * unsqueezed_penalties, score / unsqueezed_penalties)
          .to(logits.scalar_type()));
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
  const int64_t vocab = logits.size(-1);
  const float inf = -std::numeric_limits<float>::infinity();

  auto [sorted, idx] = logits.sort(-1, /*descending=*/true);

  // top-k
  auto k = top_k.unsqueeze(-1).clamp(1, vocab).to(torch::kLong);
  auto k_mask = torch::arange(vocab, logits.device()).expand_as(sorted) >= k;
  sorted.masked_fill_(k_mask, inf);

  // top-p
  auto p = top_p.unsqueeze(-1);
  auto probs = sorted.softmax(-1);
  auto cum = probs.cumsum(-1);
  auto p_mask = cum > p;
  // at least one
  p_mask.index_put_({torch::indexing::Ellipsis, 0}, false);
  sorted.masked_fill_(p_mask, inf);

  logits.scatter_(-1, idx, sorted.to(logits.scalar_type()));
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
  if (top_k.defined() || top_p.defined()) {
    xllm::kernel::TopKPParams params;
    params.logits = logits;
    params.top_k = top_k;
    params.top_p = top_p;
    logits = xllm::kernel::apply_top_k_top_p(params);
    return;
  }
#endif

  if (top_k.defined() && top_p.defined()) {
#if defined(USE_NPU)
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    static thread_local torch::Tensor max_value_scalar;
    const torch::Device& top_k_device = top_k.device();
    if (!max_value_scalar.defined() ||
        max_value_scalar.device() != top_k_device) {
      max_value_scalar = torch::full(
          {}, kMaxInt64, torch::TensorOptions().device(top_k_device));
    }

    auto processed_top_k =
        torch::where(top_k <= 0, max_value_scalar, top_k).to(torch::kInt32);
    xllm::kernel::npu::top_k_top_p(logits, processed_top_k, top_p);
#elif defined(USE_MLU)
    apply_top_k_top_p_torch_impl(logits, top_k, top_p);
#endif
  } else {
    auto [sorted_logits, logits_idx] =
        logits.sort(/*dim=*/-1, /*descending=*/true);

    float filter_value = -std::numeric_limits<float>::infinity();

    if (top_k.defined()) {
      auto processed_top_k = top_k.unsqueeze(1);
      constexpr int64_t kMaxInt64Else = std::numeric_limits<int64_t>::max();
      static thread_local torch::Tensor max_value_scalar_else;
      const torch::Device& top_k_device = processed_top_k.device();
      if (!max_value_scalar_else.defined() ||
          max_value_scalar_else.device() != top_k_device) {
        max_value_scalar_else = torch::full(
            {}, kMaxInt64Else, torch::TensorOptions().device(top_k_device));
      }

      processed_top_k = torch::where(
          processed_top_k <= 0, max_value_scalar_else, processed_top_k);

      auto vocab_size = logits.size(-1);
      auto top_k_mask = torch::arange(vocab_size, sorted_logits.device())
                            .expand_as(sorted_logits);
      top_k_mask = top_k_mask >= processed_top_k;
      sorted_logits.masked_fill_(top_k_mask, filter_value);
    }

    if (top_p.defined()) {
      auto processed_top_p = top_p.unsqueeze(1);

      auto probs = sorted_logits.softmax(/*dim=*/-1).to(torch::kFloat32);
      auto probs_sum = probs.cumsum(/*dim=*/-1);
      auto mask = (probs_sum - probs) > processed_top_p;

      sorted_logits.masked_fill_(mask, filter_value);
    }
    logits = torch::empty_like(sorted_logits)
                 .scatter_(/*dim=*/-1,
                           /*index=*/logits_idx,
                           /*core=*/sorted_logits.to(logits.scalar_type()));
  }
}

}  // namespace xllm
