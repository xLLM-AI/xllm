/* Copyright 2026 The xLLM Authors.
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

#include "core/framework/speculative/rejection_sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
#include <limits>
#include <optional>
#include <utility>

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#if defined(USE_NPU)
#include <c10/core/DeviceType.h>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

// [batch, X] -> [batch, width, X] grouped view (scatter/gather do not
// broadcast index tensors).
torch::Tensor per_seq_group(const torch::Tensor& tensor, int64_t width) {
  std::vector<int64_t> shape = tensor.sizes().vec();
  shape.insert(shape.begin() + 1, 1);
  std::vector<int64_t> expanded = shape;
  expanded[1] = width;
  return tensor.reshape(shape).expand(expanded);
}

// The verify transforms (process_logits's stages) on flat [batch * width,
// vocab] logits; per-sequence inputs broadcast on the grouped view, only the
// per-token grammar bitmask stays flat.
void apply_verify_transforms(torch::Tensor& logits,
                             const SamplingParameters& params,
                             int64_t batch_size,
                             int64_t width) {
  const int64_t vocab_size = logits.size(1);
  torch::Tensor grouped = logits.view({batch_size, width, vocab_size});

  if (params.logits_bias.defined()) {
    CHECK_EQ(params.logits_bias.sizes(),
             torch::IntArrayRef({batch_size, vocab_size}))
        << "verify logits_bias must be per-sequence [batch, vocab]";
    grouped.add_(per_seq_group(params.logits_bias, width));
  }
  if (params.min_token_ids.defined()) {
    apply_min_token_block(
        grouped, params.min_token_ids, params.min_tokens_remaining);
  }
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(grouped,
                               per_seq_group(params.unique_token_ids, width),
                               params.repetition_penalties.unsqueeze(1),
                               params.unique_token_ids_lens.unsqueeze(1));
  }
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(
        grouped,
        per_seq_group(params.unique_token_ids, width),
        params.unique_token_counts,
        params.frequency_penalties,
        params.presence_penalties);
  }

  // The grammar bitmask is per-token; it stays on the flat rows.
  if (params.filter_bitmask.defined()) {
    apply_token_bitmask_inplace(logits, params.filter_bitmask);
  } else if (params.filter_mask.defined()) {
    CHECK_EQ(params.filter_mask.sizes(),
             torch::IntArrayRef({batch_size, vocab_size}))
        << "verify filter_mask must be per-sequence [batch, vocab]";
    grouped.add_(per_seq_group(params.filter_mask, width));
  }
  if (params.skips_argmax_preserving_transforms()) {
    return;
  }

  // All argmax-preserving transforms run on the flat 2-D [batch * width,
  // vocab] view with the per-sequence scalars expanded to per-row: each row
  // is independent, so this is mathematically identical to the grouped
  // broadcast (verified), and the flat contract matches the fused device
  // kernels. The penalty stages above stay rank-agnostic (dim=-1 with
  // stride-0 per-sequence views).
  if (params.temperatures.defined()) {
    LogitsProcessor::apply_temperatures(
        logits, params.temperatures.repeat_interleave(width, /*dim=*/0));
  }
  if (params.min_p.defined()) {
    apply_min_p(logits, params.min_p.repeat_interleave(width, /*dim=*/0));
  }
  if (params.top_k.defined() || params.top_p.defined()) {
    apply_top_k_top_p(logits,
                      params.top_k.defined()
                          ? params.top_k.repeat_interleave(width, /*dim=*/0)
                          : params.top_k,
                      params.top_p.defined()
                          ? params.top_p.repeat_interleave(width, /*dim=*/0)
                          : params.top_p);
  }
}

}  // namespace

SampleOutput RejectionSampler::forward(const DraftProposal& draft_proposal,
                                       torch::Tensor target_logits) const {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const int64_t num_draft_tokens = draft_token_ids.size(1);
  CHECK_EQ(target_logits.dim(), 2)
      << "target logits must arrive flat [batch * n_spec + 1, vocab]";
  CHECK_EQ(target_logits.size(0) % (num_draft_tokens + 1), 0)
      << "target logits rows must be divisible by the validation width";
  const int64_t batch_size = target_logits.size(0) / (num_draft_tokens + 1);
  const int64_t num_val_tokens = num_draft_tokens + 1;
  // The verify transforms consume the per-sequence inputs directly on the
  // [batch, val_tokens, vocab] grouped view (see apply_verify_transforms) —
  // no row-domain materialization of the per-sequence fields. The grammar
  // bitmask is the one per-token field and stays on the flat tokens.
  apply_verify_transforms(
      target_logits, sampling_params_, batch_size, num_val_tokens);
  const torch::Tensor grouped_logits =
      target_logits.view({batch_size, num_val_tokens, target_logits.size(-1)});

  torch::Tensor bonus_token_ids;
  torch::Tensor full_greedy_token_ids;
  // Built once for the mixed legs below: both forward()'s bonus-row select
  // and verify()'s per-sequence dispatch select with it.
  const torch::Tensor greedy_rows =
      sampling_params_.sampling_mode == SamplingMode::MIXED
          ? ::xllm::greedy_mask(sampling_params_, grouped_logits.device())
          : torch::Tensor();
  SampleOutput output;
  if (sampling_params_.sampling_mode == SamplingMode::GREEDY) {
    full_greedy_token_ids = greedy_sample(grouped_logits);
    bonus_token_ids =
        full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
  } else {
    bonus_token_ids = gumbel_sample(grouped_logits.select(/*dim=*/1, -1))
                          .view({grouped_logits.size(0), 1});
    if (sampling_params_.sampling_mode != SamplingMode::RANDOM) {
      full_greedy_token_ids = greedy_sample(grouped_logits);
      const torch::Tensor greedy_bonus =
          full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
      bonus_token_ids =
          torch::where(greedy_rows, greedy_bonus, bonus_token_ids);
    }
  }

  if (num_draft_tokens == 0) {
    output.next_tokens = bonus_token_ids;
    return fill_logprob_outputs(output, grouped_logits);
  }
  // Only the random/mixed verification legs read the lse; greedy batches
  // (the default draft mode) skip the [B, K, V] fp32 reduction entirely.
  const torch::Tensor target_lse =
      sampling_params_.sampling_mode == SamplingMode::GREEDY
          ? torch::Tensor()
          : torch::logsumexp(
                grouped_logits
                    .narrow(
                        /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
                    .to(torch::kFloat32),
                /*dim=*/-1,
                /*keepdim=*/false);
  output.next_tokens = verify(draft_proposal,
                              grouped_logits,
                              target_lse,
                              bonus_token_ids,
                              full_greedy_token_ids,
                              greedy_rows);
  return fill_logprob_outputs(output, grouped_logits);
}

SampleOutput RejectionSampler::fill_logprob_outputs(
    SampleOutput output,
    const torch::Tensor& grouped_logits) const {
  if (sampling_params_.logprobs) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    // [batch_size, n_speculative_tokens + 1, vocab_size]
    auto target_logprobs = torch::log_softmax(
        grouped_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // -1 sentinels cannot index the gather; their logprobs are zeroed.
    auto selected_logprobs =
        index_select_2d(target_logprobs,
                        /*dim=*/-1,
                        output.next_tokens.clamp_min(0).to(torch::kLong));
    selected_logprobs.masked_fill_(output.next_tokens.lt(0), 0);
    output.logprobs = std::move(selected_logprobs);

    if (sampling_params_.max_top_logprobs > 0) {
      auto [values, indices] =
          target_logprobs.topk(sampling_params_.max_top_logprobs, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

torch::Tensor RejectionSampler::verify(
    const DraftProposal& draft_proposal,
    const torch::Tensor& target_logits,
    const torch::Tensor& target_lse,
    const torch::Tensor& bonus_token_ids,
    const torch::Tensor& target_greedy_token_ids,
    const torch::Tensor& greedy_mask) const {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const int64_t num_draft_tokens = draft_token_ids.size(1);
  draft_proposal.validate(
      /*expected_batch_size=*/target_logits.size(0),
      /*expected_num_speculative_tokens=*/target_logits.size(1) - 1);
  const torch::Tensor greedy_target_token_ids =
      target_greedy_token_ids.defined()
          ? target_greedy_token_ids.narrow(
                /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
          : torch::Tensor();

  auto random_sample_dispatch = [&]() -> torch::Tensor {
    const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
    const std::optional<torch::Tensor>& draft_logits = draft_proposal.logits();

    // Accept p/q > u in log space. −1 sentinel drafts clamp to column 0
    // and are re-masked by the acceptance term.
    const torch::Tensor draft_index =
        draft_token_ids.clamp_min(0).to(torch::kLong);
    const auto gathered_logit = [&](const torch::Tensor& logits) {
      return index_select_2d(logits, /*dim=*/-1, draft_index)
          .to(torch::kFloat32);
    };
    torch::Tensor target_log_prob =
        gathered_logit(target_logits).sub(target_lse);
    torch::Tensor draft_lse;
    if (draft_logits.has_value()) {
      draft_lse = torch::logsumexp(
          draft_logits->to(torch::kFloat32), /*dim=*/-1, /*keepdim=*/false);
      target_log_prob.sub_(gathered_logit(*draft_logits).sub(draft_lse));
    }
    torch::Tensor uniform_rand =
        torch::rand(draft_token_ids.sizes(),
                    target_logits.options().dtype(torch::kFloat32));
    auto accepted = (uniform_rand.log_().lt(target_log_prob))
                        .logical_and(draft_token_ids.ge(0));

    // All-accepted sequences clamp back into range; their recovered token
    // is masked out below.
    // First rejected draft position per sequence; all-accepted sequences point
    // past the last draft row (the bonus position).
    auto accepted_int64 = accepted.to(torch::kInt64);
    auto bonus_mask =
        torch::zeros({accepted.size(0), 1}, accepted_int64.options());
    const torch::Tensor first_reject =
        torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1)
            .argmin(/*dim=*/1, /*keepdim=*/true);
    const torch::Tensor draft_first_reject =
        first_reject.clamp_max(draft_token_ids.size(1) - 1);
    const torch::Tensor rejected_tokens =
        draft_token_ids.gather(/*dim=*/1, draft_first_reject);
    // Row draft_first_reject[i] of each [B, K, V] tensor as [B, V], kept as
    // log-probs (row logsumexp subtracted) — the residual recovery works in
    // log space.
    const auto row_log_probs = [&](const torch::Tensor& logits,
                                   const torch::Tensor& lse) {
      torch::Tensor gathered_logits =
          index_select_2d(logits,
                          /*dim=*/1,
                          draft_first_reject.expand(
                              {target_logits.size(0), target_logits.size(-1)}))
              .to(torch::kFloat32);
      return gathered_logits.sub_(lse.gather(/*dim=*/1, draft_first_reject));
    };
    torch::Tensor target_log_probs_at_reject =
        row_log_probs(target_logits, target_lse);
    torch::Tensor draft_log_probs_at_reject;
    if (draft_logits.has_value()) {
      draft_log_probs_at_reject = row_log_probs(*draft_logits, draft_lse);
    }
    // Recovered token from the residual (p - q)+ at each sequence's first
    // rejected step, in log space; all -inf rows argmax to a deterministic
    // index, masked out downstream. Sentinel (-1) drafts carry log q = -inf.
    torch::Tensor recovered_tokens;
    {
      torch::Tensor target_log_probs = std::move(target_log_probs_at_reject);
      torch::Tensor draft_log_probs = std::move(draft_log_probs_at_reject);
      if (draft_log_probs.defined()) {
        draft_log_probs.masked_fill_(rejected_tokens.lt(0),
                                     -std::numeric_limits<float>::infinity());
        draft_log_probs.sub_(target_log_probs)
            .nan_to_num_(/*nan=*/0.0)
            .clamp_max_(0)
            .exp_()
            .neg_()
            .log1p_();
        target_log_probs.add_(draft_log_probs);
      } else {
        const torch::Tensor clamped_rejected = rejected_tokens.clamp_min(0);
        const torch::Tensor values =
            target_log_probs.gather(/*dim=*/-1, clamped_rejected);
        values.masked_fill_(rejected_tokens.ge(0),
                            -std::numeric_limits<float>::infinity());
        target_log_probs.scatter_(/*dim=*/-1, clamped_rejected, values);
      }
      recovered_tokens = gumbel_sample(target_log_probs);
    }
    auto recovered_token_ids = recovered_tokens.unsqueeze(/*dim=*/-1);

    // Accepted rows carry non-sentinel ids, so the clamp in the
    // rejected-token gather is the identity there.
    auto combined =
        torch::where(accepted, draft_token_ids, recovered_token_ids);
    auto accepted_token_ids =
        torch::cat({combined, bonus_token_ids}, /*dim=*/-1);
    const auto first_reject_indices =
        arange_indices(accepted_token_ids.size(1), accepted_token_ids.device())
            .unsqueeze(/*dim=*/0);
    return torch::where(
        first_reject_indices <= first_reject, accepted_token_ids, -1);
  };

  auto greedy_sample_dispatch = [&]() -> torch::Tensor {
    CHECK_EQ(greedy_target_token_ids.sizes(), draft_token_ids.sizes())
        << "target and draft token shapes must match";
#if defined(USE_NPU)
    if (greedy_target_token_ids.device().type() ==
        c10::DeviceType::PrivateUse1) {
      return kernel::npu::tilelang::greedy_prefix_verify(
          draft_token_ids, greedy_target_token_ids, bonus_token_ids);
    }
#endif
    torch::Tensor greedy_accepted =
        torch::cat({greedy_target_token_ids, bonus_token_ids}, /*dim=*/-1);
    auto accepted = (greedy_target_token_ids == draft_token_ids);
    auto accepted_int64 = accepted.to(torch::kInt64);
    auto bonus_mask =
        torch::zeros({accepted.size(0), 1}, accepted_int64.options());
    const torch::Tensor first_reject =
        torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1)
            .argmin(/*dim=*/1, /*keepdim=*/true);
    const auto reject_indices =
        arange_indices(greedy_accepted.size(1), greedy_accepted.device())
            .unsqueeze(/*dim=*/0);
    return torch::where(reject_indices <= first_reject, greedy_accepted, -1);
  };

  if (sampling_params_.sampling_mode == SamplingMode::GREEDY) {
    return greedy_sample_dispatch();
  }
  if (sampling_params_.sampling_mode == SamplingMode::RANDOM) {
    return random_sample_dispatch();
  }
  // mixed sample: dispatch both legs and select per row.
  return torch::where(
      greedy_mask, greedy_sample_dispatch(), random_sample_dispatch());
}

}  // namespace xllm
