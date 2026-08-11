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

#include "core/framework/speculative/adaptive_pruning_helpers.h"

#include <glog/logging.h>

namespace xllm {
namespace adaptive_pruning {

namespace {

torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

void sync_pruned_boundary_logprobs(SampleOutput& sample_output,
                                   const ForwardOutput& target_output,
                                   int32_t batch_size,
                                   int32_t num_val_tokens,
                                   const PrunedPrefixMasks& masks) {
  if (!sample_output.logprobs.defined()) {
    return;
  }

  CHECK(target_output.sample_output.logprobs.defined())
      << "target output logprobs are required for adaptive pruning";
  CHECK_EQ(
      target_output.sample_output.logprobs.numel(),
      static_cast<int64_t>(batch_size) * static_cast<int64_t>(num_val_tokens))
      << "target logprob count mismatch";
  torch::Tensor cut_mask = masks.cut_mask.to(sample_output.logprobs.device());
  torch::Tensor target_logprobs = safe_to(
      target_output.sample_output.logprobs.view({batch_size, num_val_tokens}),
      sample_output.logprobs.options(),
      /*non_blocking=*/true);
  sample_output.logprobs =
      torch::where(cut_mask, target_logprobs, sample_output.logprobs);
}

void sync_pruned_boundary_top_logprobs(SampleOutput& sample_output,
                                       const ForwardOutput& target_output,
                                       int32_t batch_size,
                                       int32_t num_val_tokens,
                                       const PrunedPrefixMasks& masks) {
  if (!sample_output.top_tokens.defined()) {
    return;
  }

  CHECK(sample_output.top_logprobs.defined())
      << "top_logprobs must be defined when top_tokens are defined";
  CHECK(target_output.sample_output.top_tokens.defined())
      << "target top_tokens are required for adaptive pruning";
  CHECK(target_output.sample_output.top_logprobs.defined())
      << "target top_logprobs are required for adaptive pruning";
  CHECK_EQ(
      target_output.sample_output.top_tokens.size(0),
      static_cast<int64_t>(batch_size) * static_cast<int64_t>(num_val_tokens))
      << "target top_tokens count mismatch";
  CHECK_EQ(
      target_output.sample_output.top_logprobs.size(0),
      static_cast<int64_t>(batch_size) * static_cast<int64_t>(num_val_tokens))
      << "target top_logprobs count mismatch";

  torch::Tensor cut_mask_broadcast =
      masks.cut_mask.to(sample_output.top_tokens.device())
          .unsqueeze(/*dim=*/-1);
  const int64_t top_k = sample_output.top_tokens.size(2);
  CHECK_EQ(target_output.sample_output.top_tokens.size(1), top_k)
      << "target top_tokens top_k mismatch";
  CHECK_EQ(target_output.sample_output.top_logprobs.size(1), top_k)
      << "target top_logprobs top_k mismatch";
  torch::Tensor target_top_tokens = target_output.sample_output.top_tokens.view(
      {batch_size, num_val_tokens, top_k});
  torch::Tensor target_top_logprobs =
      target_output.sample_output.top_logprobs.view(
          {batch_size, num_val_tokens, top_k});
  sample_output.top_tokens =
      torch::where(cut_mask_broadcast,
                   safe_to(target_top_tokens,
                           sample_output.top_tokens.options(),
                           /*non_blocking=*/true),
                   sample_output.top_tokens);
  sample_output.top_logprobs =
      torch::where(cut_mask_broadcast.to(sample_output.top_logprobs.device()),
                   safe_to(target_top_logprobs,
                           sample_output.top_logprobs.options(),
                           /*non_blocking=*/true),
                   sample_output.top_logprobs);
}

}  // namespace

PrunedPrefixMasks build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device) {
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  torch::Tensor prefix_lengths =
      safe_to(make_cpu_int_tensor(pruned_prefix_lengths),
              torch::TensorOptions().dtype(torch::kLong).device(device),
              /*non_blocking=*/true)
          .clamp(0, num_speculative_tokens);
  torch::Tensor positions =
      torch::arange(num_val_tokens, prefix_lengths.options());
  PrunedPrefixMasks masks;
  masks.keep_mask =
      positions.unsqueeze(/*dim=*/0).le(prefix_lengths.unsqueeze(/*dim=*/1));
  masks.cut_mask = positions.unsqueeze(/*dim=*/0)
                       .eq(prefix_lengths.unsqueeze(/*dim=*/1))
                       .logical_and(prefix_lengths.unsqueeze(/*dim=*/1).lt(
                           num_speculative_tokens));
  return masks;
}

torch::Tensor selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs) {
  CHECK(!draft_outputs.empty()) << "draft outputs must not be empty";
  std::vector<torch::Tensor> probs_steps;
  probs_steps.reserve(draft_outputs.size());
  int64_t batch_size = -1;
  for (const ForwardOutput& draft_output : draft_outputs) {
    torch::Tensor probs = draft_output.sample_output.probs;
    if (!probs.defined()) {
      // Fallback: compute selected probs from logits+next_tokens on the fly.
      // This bypasses the sampler's greedy fast-path which skips probs
      // assignment. Avoids triggering model-side kernel paths (e.g. Qwen3.5
      // CausalConv1d) that a sampler-side return_probs=true would touch.
      const torch::Tensor& logits = draft_output.logits;
      const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
      CHECK(logits.defined() && next_tokens.defined())
          << "adaptive pruning fallback needs draft logits and next_tokens";
      CHECK_EQ(logits.dim(), 2)
          << "adaptive pruning expects draft logits [batch,vocab], got "
          << logits.sizes();
      // Compute p_selected = exp(logit_selected - logsumexp(logits)) without
      // materializing the dense [batch, vocab] softmax. logsumexp is a
      // reduction to [batch]; the gather picks one column of logits. This
      // saves batch*vocab fp32 (≈39 MiB at batch=64, vocab=152k) per draft
      // step, executed on every adaptive decode.
      const torch::Tensor logits_f32 = logits.to(torch::kFloat32);
      const torch::Tensor logsumexp = torch::logsumexp(logits_f32, /*dim=*/-1);
      const torch::Tensor selected_logits =
          logits_f32
              .gather(/*dim=*/-1, next_tokens.view({-1, 1}).to(torch::kLong))
              .squeeze(/*dim=*/-1);
      probs = torch::exp(selected_logits - logsumexp).to(logits.dtype());
    }
    CHECK(probs.dim() == 1 || probs.dim() == 2)
        << "adaptive pruning expects draft probs [batch] or [batch,1], got "
        << probs.sizes();
    if (probs.dim() == 2) {
      CHECK_EQ(probs.size(1), 1)
          << "adaptive pruning expects draft probs [batch,1], got "
          << probs.sizes();
      probs = probs.squeeze(/*dim=*/1);
    }
    if (batch_size < 0) {
      batch_size = probs.size(0);
    }
    CHECK_EQ(probs.size(0), batch_size)
        << "adaptive pruning draft prob batch mismatch";
    probs_steps.emplace_back(probs.view({-1, 1}));
  }
  return torch::cat(probs_steps, /*dim=*/1);
}

bool has_selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs) {
  if (draft_outputs.empty()) {
    return false;
  }
  for (const ForwardOutput& draft_output : draft_outputs) {
    const torch::Tensor& probs = draft_output.sample_output.probs;
    if (probs.defined()) {
      continue;  // ok
    }
    // Fallback: selected_probs_by_step will compute from logits+next_tokens.
    const torch::Tensor& logits = draft_output.logits;
    const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
    if (!logits.defined() || !next_tokens.defined()) {
      return false;
    }
  }
  return true;
}

void clamp_prefix_lengths(std::vector<int32_t>& prefix_lengths,
                          int32_t batch_size,
                          int32_t num_speculative_tokens) {
  CHECK_EQ(prefix_lengths.size(), static_cast<size_t>(batch_size))
      << "adaptive pruning prefix length batch mismatch";
  for (int32_t& prefix_len : prefix_lengths) {
    prefix_len = std::clamp(prefix_len, 0, num_speculative_tokens);
  }
}

int32_t max_pruned_prefix_length(const std::vector<int32_t>& prefix_lengths,
                                 int32_t num_speculative_tokens) {
  if (prefix_lengths.empty()) {
    return num_speculative_tokens;
  }
  const int32_t max_prefix_len =
      *std::max_element(prefix_lengths.begin(), prefix_lengths.end());
  return std::clamp(max_prefix_len, 0, num_speculative_tokens);
}

std::vector<ForwardOutput> truncate_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    int32_t num_speculative_tokens) {
  CHECK_GE(num_speculative_tokens, 0)
      << "num_speculative_tokens must be non-negative";
  CHECK_GE(draft_outputs.size(), static_cast<size_t>(num_speculative_tokens))
      << "draft outputs are fewer than the requested validation width";
  std::vector<ForwardOutput> truncated_outputs;
  truncated_outputs.reserve(static_cast<size_t>(num_speculative_tokens));
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    truncated_outputs.emplace_back(draft_outputs[static_cast<size_t>(i)]);
  }
  return truncated_outputs;
}

void apply_pruned_prefix_lengths(SampleOutput& sample_output,
                                 const torch::Tensor& target_next_tokens,
                                 int32_t num_speculative_tokens,
                                 const PrunedPrefixMasks& masks) {
  CHECK(sample_output.next_tokens.defined())
      << "validate output tokens are undefined";
  CHECK_EQ(sample_output.next_tokens.dim(), 2)
      << "validate output tokens should be [batch, width]";
  const int32_t batch_size =
      static_cast<int32_t>(sample_output.next_tokens.size(0));
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(sample_output.next_tokens.size(1), num_val_tokens)
      << "validate output width mismatch";
  CHECK(target_next_tokens.defined())
      << "target output tokens are required for adaptive pruning";
  torch::Tensor target_next_tokens_view =
      target_next_tokens.view({batch_size, num_val_tokens});
  torch::Tensor keep_mask =
      masks.keep_mask.to(sample_output.next_tokens.device());
  torch::Tensor cut_mask =
      masks.cut_mask.to(sample_output.next_tokens.device());
  torch::Tensor target_tokens = safe_to(target_next_tokens_view,
                                        sample_output.next_tokens.options(),
                                        /*non_blocking=*/true);
  // Boundary token: replace at cut positions with the target's resample.
  // `torch::where(cond, src, dst)` is unavoidable here because the fill is a
  // tensor, not a scalar. Downstream keep-mask fills use masked_fill_ to
  // avoid the zeros_like/-ones_like scratch tensors.
  torch::Tensor next_tokens =
      torch::where(cut_mask, target_tokens, sample_output.next_tokens);
  const torch::Tensor drop_mask = keep_mask.logical_not();
  next_tokens.masked_fill_(drop_mask, -1);
  sample_output.next_tokens = next_tokens;

  if (sample_output.logprobs.defined()) {
    CHECK_EQ(sample_output.logprobs.dim(), 2)
        << "validate output logprobs should be [batch, width]";
    CHECK_EQ(sample_output.logprobs.size(0), batch_size)
        << "validate logprob batch mismatch";
    CHECK_EQ(sample_output.logprobs.size(1), num_val_tokens)
        << "validate logprob width mismatch";
    sample_output.logprobs.masked_fill_(
        drop_mask.to(sample_output.logprobs.device()), 0);
  }

  if (sample_output.top_tokens.defined()) {
    CHECK(sample_output.top_logprobs.defined())
        << "top_logprobs must be defined when top_tokens are defined";
    CHECK_EQ(sample_output.top_tokens.dim(), 3)
        << "validate top_tokens should be [batch, width, top_k]";
    CHECK_EQ(sample_output.top_logprobs.dim(), 3)
        << "validate top_logprobs should be [batch, width, top_k]";
    CHECK_EQ(sample_output.top_tokens.size(0), batch_size)
        << "validate top_tokens batch mismatch";
    CHECK_EQ(sample_output.top_tokens.size(1), num_val_tokens)
        << "validate top_tokens width mismatch";
    CHECK_EQ(sample_output.top_logprobs.size(0), batch_size)
        << "validate top_logprobs batch mismatch";
    CHECK_EQ(sample_output.top_logprobs.size(1), num_val_tokens)
        << "validate top_logprobs width mismatch";
    const torch::Tensor top_drop_mask =
        drop_mask.to(sample_output.top_tokens.device()).unsqueeze(/*dim=*/-1);
    sample_output.top_tokens.masked_fill_(top_drop_mask, 0);
    sample_output.top_logprobs.masked_fill_(
        top_drop_mask.to(sample_output.top_logprobs.device()), 0);
  }

  if (!sample_output.embeddings.defined()) {
    return;
  }
  CHECK_EQ(sample_output.embeddings.dim(), 3)
      << "validate output embeddings should be [batch, width, hidden]";
  CHECK_EQ(sample_output.embeddings.size(0), batch_size)
      << "validate embedding batch mismatch";
  CHECK_EQ(sample_output.embeddings.size(1), num_val_tokens)
      << "validate embedding width mismatch";

  const torch::Tensor embedding_drop_mask =
      drop_mask.to(sample_output.embeddings.device()).unsqueeze(/*dim=*/-1);
  sample_output.embeddings.masked_fill_(embedding_drop_mask, 0);
}

void sync_pruned_boundary_outputs(SampleOutput& sample_output,
                                  const ForwardOutput& target_output,
                                  int32_t batch_size,
                                  int32_t num_val_tokens,
                                  const PrunedPrefixMasks& masks) {
  sync_pruned_boundary_logprobs(
      sample_output, target_output, batch_size, num_val_tokens, masks);
  sync_pruned_boundary_top_logprobs(
      sample_output, target_output, batch_size, num_val_tokens, masks);
}

}  // namespace adaptive_pruning
}  // namespace xllm
