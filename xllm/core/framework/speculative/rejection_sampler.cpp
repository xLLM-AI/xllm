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

#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

// Position of the first rejected draft per sequence [B, 1] int32 — the
// number of accepted drafts. All-accepted rows argmin to the appended bonus
// column (== width), which is the correct full-width count; gather callers
// must clamp back into index range themselves.
torch::Tensor first_rejected_indices(const torch::Tensor& accepted) {
  const torch::Tensor accepted_int64 = accepted.to(torch::kInt64);
  const torch::Tensor bonus_mask =
      torch::zeros({accepted.size(0), 1}, accepted_int64.options());
  return torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1)
      .argmin(/*dim=*/1, /*keepdim=*/true)
      .to(torch::kInt32);
}

// The committed-prefix construction shared by the dispatch legs: `accepted`
// is per-position, but the committed row is the first-failure prefix — a
// draft at position i commits only when every earlier position accepted
// (all_accepted_before, from the inclusive cumprod shifted right one
// column), so the prefix, the correction token at the first failure, and
// the -1 sentinel tail come out of one elementwise where.
torch::Tensor all_accepted_before(const torch::Tensor& accepted) {
  const torch::Tensor prefix_so_far =
      torch::cumprod(accepted.to(torch::kInt32), /*dim=*/1) > 0;
  return torch::cat({torch::ones({accepted.size(0), 1}, accepted.options()),
                     prefix_so_far.narrow(/*dim=*/1,
                                          /*start=*/0,
                                          /*length=*/accepted.size(1) - 1)},
                    /*dim=*/-1);
}

// Selected + optional top-k logprobs from the grouped logits; -1 sentinels
// cannot index the gather, so their logprobs are zeroed.
void compute_logprobs(SampleOutput& output,
                      const torch::Tensor& grouped_logits,
                      const SamplingParameters& params) {
  if (!params.logprobs) {
    return;
  }
  auto target_logprobs =
      torch::log_softmax(grouped_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  auto selected_logprobs =
      index_select_2d(target_logprobs,
                      /*dim=*/-1,
                      output.next_tokens.clamp_min(0).to(torch::kLong));
  selected_logprobs.masked_fill_(output.next_tokens.lt(0), 0);
  output.logprobs = std::move(selected_logprobs);
  if (params.max_top_logprobs > 0) {
    auto [values, indices] =
        target_logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
    output.top_logprobs = values;
    output.top_tokens = indices;
  }
}

}  // namespace

SampleOutput rejection_sample(const SamplingParameters& params,
                              const DraftProposal& draft_proposal,
                              torch::Tensor target_logits) {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const int64_t num_draft_tokens = draft_token_ids.size(1);
  CHECK_EQ(target_logits.dim(), 2)
      << "target logits must arrive flat [batch * n_spec + 1, vocab]";
  CHECK_EQ(target_logits.size(0) % (num_draft_tokens + 1), 0)
      << "target logits rows must be divisible by the validation width";
  const int64_t batch_size = target_logits.size(0) / (num_draft_tokens + 1);
  const int64_t num_val_tokens = num_draft_tokens + 1;
  // Verify transforms: the penalty stages broadcast per-sequence inputs on
  // the grouped view; the grammar bitmask stays on the flat rows.
  const int64_t vocab_size = target_logits.size(1);
  torch::Tensor grouped_logits =
      target_logits.view({batch_size, num_val_tokens, vocab_size});
  const torch::Tensor raw_logits =
      params.logprobs ? grouped_logits.clone() : torch::Tensor();
  if (params.has_sampling_fields()) {
    for (int64_t column = 0; column < num_val_tokens; ++column) {
      SamplingParameters step_params = params;
      if (params.min_tokens_remaining.defined()) {
        step_params.min_tokens_remaining = params.min_tokens_remaining - column;
      }
      const torch::Tensor prefix = draft_token_ids.narrow(1, 0, column);
      step_params.prefix_tokens =
          params.prefix_tokens.defined()
              ? torch::cat({params.prefix_tokens, prefix}, 1)
              : prefix;
      if (params.bad_word_suffix.defined()) {
        step_params.bad_word_suffix =
            append_token_suffix(params.bad_word_suffix, prefix);
      }
      torch::Tensor row_logits = grouped_logits.select(1, column);
      apply_sampling_penalties(row_logits, step_params);
    }
  }
  // MTP installs the bitmask at full validate width; other workers pass a
  // per-sequence [batch, words] mask, repeated here per verify row.
  if (params.filter_bitmask.defined()) {
    const torch::Tensor bitmask =
        params.filter_bitmask.size(0) == batch_size
            ? params.filter_bitmask.repeat_interleave(num_val_tokens,
                                                      /*dim=*/0)
            : params.filter_bitmask;
    apply_token_bitmask_inplace(target_logits, bitmask);
  } else if (params.filter_mask.defined()) {
    CHECK_EQ(params.filter_mask.sizes(),
             torch::IntArrayRef({batch_size, vocab_size}))
        << "verify filter_mask must be per-sequence [batch, vocab]";
    grouped_logits.add_(
        expand_batch_to_tokens(params.filter_mask, num_val_tokens));
  }
  if (!params.is_all_greedy()) {
    if (params.temperatures.defined()) {
      apply_temperatures(grouped_logits, params.temperatures);
    }
    if (params.min_p.defined()) {
      apply_min_p(grouped_logits, params.min_p);
    }
    if (params.top_k.defined() || params.top_p.defined()) {
      apply_top_k_top_p(
          target_logits,
          params.top_k.defined()
              ? params.top_k.repeat_interleave(num_val_tokens, /*dim=*/0)
              : params.top_k,
          params.top_p.defined()
              ? params.top_p.repeat_interleave(num_val_tokens, /*dim=*/0)
              : params.top_p);
    }
  }
  if (params.seeds.defined() && !params.is_all_greedy()) {
    // Sampling the target at each logical token offset makes seeded requests
    // independent of batch position and the number of accepted MTP drafts.
    const torch::Tensor sampled = gumbel_argmax(
        grouped_logits,
        gumbel_noise(params, grouped_logits.sizes(), grouped_logits.device()));
    SampleOutput output;
    if (num_draft_tokens == 0) {
      output.next_tokens = sampled;
      output.num_accepted_tokens =
          torch::zeros({batch_size, 1}, sampled.options().dtype(torch::kInt32));
    } else {
      const torch::Tensor accepted = sampled.narrow(1, 0, num_draft_tokens)
                                         .eq(draft_token_ids)
                                         .logical_and(draft_token_ids.ge(0));
      const torch::Tensor valid =
          torch::cat({torch::ones({batch_size, 1}, accepted.options()),
                      accepted.to(torch::kInt32).cumprod(1).gt(0)},
                     1);
      output.next_tokens = sampled.masked_fill(valid.logical_not(), -1);
      output.num_accepted_tokens =
          valid.sum(/*dim=*/1, /*keepdim=*/true, torch::kInt32).sub_(1);
    }
    compute_logprobs(
        output, raw_logits.defined() ? raw_logits : grouped_logits, params);
    return output;
  }
  torch::Tensor bonus_token_ids;
  torch::Tensor full_greedy_token_ids;
  const torch::Tensor greedy_rows =
      params.sampling_mode == SamplingMode::MIXED
          ? ::xllm::greedy_mask(params, grouped_logits.device())
          : torch::Tensor();
  SampleOutput output;
  if (params.sampling_mode == SamplingMode::GREEDY) {
    full_greedy_token_ids = greedy_sample(grouped_logits);
    bonus_token_ids =
        full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
  } else {
    bonus_token_ids = gumbel_sample(grouped_logits.select(/*dim=*/1, -1))
                          .view({grouped_logits.size(0), 1});
    if (params.sampling_mode != SamplingMode::RANDOM) {
      full_greedy_token_ids = greedy_sample(grouped_logits);
      const torch::Tensor greedy_bonus =
          full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
      bonus_token_ids =
          torch::where(greedy_rows, greedy_bonus, bonus_token_ids);
    }
  }

  if (num_draft_tokens == 0) {
    output.next_tokens = bonus_token_ids;
    output.num_accepted_tokens = torch::zeros(
        {batch_size, 1}, bonus_token_ids.options().dtype(torch::kInt32));
    compute_logprobs(
        output, raw_logits.defined() ? raw_logits : grouped_logits, params);
    return output;
  }
  // Only the random/mixed verification legs read the lse; greedy batches
  // (the default draft mode) skip the [B, K, V] fp32 reduction entirely.
  const torch::Tensor target_lse =
      params.sampling_mode == SamplingMode::GREEDY
          ? torch::Tensor()
          : torch::logsumexp(
                grouped_logits
                    .narrow(
                        /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
                    .to(torch::kFloat32),
                /*dim=*/-1,
                /*keepdim=*/false);
  draft_proposal.validate(
      /*expected_batch_size=*/grouped_logits.size(0),
      /*expected_num_speculative_tokens=*/num_draft_tokens);
  const torch::Tensor greedy_target_token_ids =
      full_greedy_token_ids.defined()
          ? full_greedy_token_ids.narrow(
                /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
          : torch::Tensor();

  // The committed-prefix tail shared by both dispatch legs (prefix per
  // all_accepted_before): first failure carries `recovery`, the rest -1;
  // the bonus column commits only when the whole draft row accepted.
  auto commit_prefix = [&](const torch::Tensor& accepted,
                           const torch::Tensor& recovery) {
    const torch::Tensor prev = all_accepted_before(accepted);
    const torch::Tensor prefix = accepted.logical_and(prev);
    const torch::Tensor recovery_col = prev.logical_and(accepted.logical_not());
    const torch::Tensor committed = torch::where(
        prefix, draft_token_ids, torch::where(recovery_col, recovery, -1));
    return torch::cat(
        {committed,
         torch::where(
             accepted.all(/*dim=*/1, /*keepdim=*/true), bonus_token_ids, -1)},
        /*dim=*/-1);
  };

  auto random_sample_dispatch =
      [&]() -> std::pair<torch::Tensor, torch::Tensor> {
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
        gathered_logit(grouped_logits).sub(target_lse);
    torch::Tensor draft_lse;
    if (draft_logits.has_value()) {
      draft_lse = torch::logsumexp(
          draft_logits->to(torch::kFloat32), /*dim=*/-1, /*keepdim=*/false);
      target_log_prob.sub_(gathered_logit(*draft_logits).sub(draft_lse));
    }
    torch::Tensor uniform_rand =
        torch::rand(draft_token_ids.sizes(),
                    grouped_logits.options().dtype(torch::kFloat32));
    auto accepted = (uniform_rand.log_().lt(target_log_prob))
                        .logical_and(draft_token_ids.ge(0));

    const torch::Tensor first_reject = first_rejected_indices(accepted);
    const torch::Tensor gather_reject =
        first_reject.clamp_max(num_draft_tokens - 1);
    const torch::Tensor rejected_tokens =
        draft_token_ids.gather(/*dim=*/1, gather_reject);
    // Row first_reject[i] of each [B, K, V] tensor as [B, V], kept as
    // log-probs (row logsumexp subtracted) — the residual recovery works in
    // log space.
    const auto row_log_probs = [&](const torch::Tensor& logits,
                                   const torch::Tensor& lse) {
      torch::Tensor gathered_logits =
          index_select_2d(logits,
                          /*dim=*/1,
                          gather_reject.expand({grouped_logits.size(0),
                                                grouped_logits.size(-1)}))
              .to(torch::kFloat32);
      return gathered_logits.sub_(lse.gather(/*dim=*/1, gather_reject));
    };
    // Recovered token from the residual (p - q)+ at the first rejected step;
    // sentinel (-1) drafts carry log q = -inf.
    torch::Tensor recovered_tokens;
    {
      torch::Tensor target_log_probs =
          row_log_probs(grouped_logits, target_lse);
      torch::Tensor draft_log_probs =
          draft_logits.has_value() ? row_log_probs(*draft_logits, draft_lse)
                                   : torch::Tensor();
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
    return {commit_prefix(accepted, recovered_token_ids), first_reject};
  };

  auto greedy_sample_dispatch =
      [&]() -> std::pair<torch::Tensor, torch::Tensor> {
    CHECK_EQ(greedy_target_token_ids.sizes(), draft_token_ids.sizes())
        << "target and draft token shapes must match";
#if defined(USE_NPU)
    if (greedy_target_token_ids.device().type() == torch::kPrivateUse1) {
      const torch::Tensor committed_tokens =
          kernel::npu::tilelang::greedy_prefix_verify(
              draft_token_ids, greedy_target_token_ids, bonus_token_ids);
      // The kernel -1-fills the rejected tail, so the non-sentinel draft-column
      // count is the accepted count. Match first_rejected_indices' [B, 1]
      // shape; token id 0 is a valid accepted draft, so count non-sentinel
      // entries, not non-zero ones.
      const torch::Tensor num_accepted_tokens =
          committed_tokens.narrow(/*dim=*/1,
                                  /*start=*/1,
                                  /*length=*/num_draft_tokens);
      return {committed_tokens,
              num_accepted_tokens.ge(0)
                  .sum(/*dim=*/1, /*keepdim=*/true)
                  .to(torch::kInt32)};
    }
#endif
    const torch::Tensor accepted = greedy_target_token_ids == draft_token_ids;
    const torch::Tensor first_reject = first_rejected_indices(accepted);
    const torch::Tensor gather_reject =
        first_reject.clamp_max(num_draft_tokens - 1);
    const torch::Tensor correction = greedy_target_token_ids.gather(
        /*dim=*/1, gather_reject);
    return {commit_prefix(accepted, correction), first_reject};
  };

  if (params.sampling_mode == SamplingMode::GREEDY) {
    std::tie(output.next_tokens, output.num_accepted_tokens) =
        greedy_sample_dispatch();
  } else {
    std::tie(output.next_tokens, output.num_accepted_tokens) =
        random_sample_dispatch();
    if (params.sampling_mode == SamplingMode::MIXED) {
      const auto [greedy_tokens, greedy_counts] = greedy_sample_dispatch();
      output.next_tokens =
          torch::where(greedy_rows, greedy_tokens, output.next_tokens);
      output.num_accepted_tokens =
          torch::where(greedy_rows, greedy_counts, output.num_accepted_tokens);
    }
  }

  compute_logprobs(
      output, raw_logits.defined() ? raw_logits : grouped_logits, params);
  return output;
}

}  // namespace xllm
