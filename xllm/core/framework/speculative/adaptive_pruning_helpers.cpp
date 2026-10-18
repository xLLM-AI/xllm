/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/speculative/adaptive_pruning_helpers.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>

#include "core/framework/sampling/logits_utils.h"

namespace xllm {
namespace adaptive_pruning {

torch::Tensor build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device) {
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  torch::Tensor prefix_lengths =
      async_h2d_tensor(
          pruned_prefix_lengths,
          torch::TensorOptions().dtype(torch::kLong).device(device))
          .clamp(0, num_speculative_tokens);
  torch::Tensor positions =
      torch::arange(num_val_tokens, prefix_lengths.options());
  return positions.unsqueeze(/*dim=*/0).gt(prefix_lengths.unsqueeze(/*dim=*/1));
}

std::optional<torch::Tensor> selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs) {
  if (draft_outputs.empty()) {
    return std::nullopt;
  }
  std::vector<torch::Tensor> probs_steps;
  probs_steps.reserve(draft_outputs.size());
  int64_t batch_size = -1;
  for (const ForwardOutput& draft_output : draft_outputs) {
    if (!draft_output.sample_output.next_tokens.defined()) {
      return std::nullopt;
    }
    // Prefer the sampler's own export: Eagle3's d2t remap rebinds
    // draft_logits (and next_tokens) to the target vocabulary, while
    // ForwardOutput::logits stays in the draft's reduced vocabulary.
    const torch::Tensor& logits =
        draft_output.sample_output.draft_logits.defined()
            ? draft_output.sample_output.draft_logits
            : draft_output.logits;
    if (!logits.defined()) {
      return std::nullopt;
    }
    torch::Tensor probs = gather_probs_from_logits(
        logits, draft_output.sample_output.next_tokens);
    if (batch_size < 0) {
      batch_size = probs.size(0);
    }
    CHECK_EQ(probs.size(0), batch_size)
        << "adaptive pruning draft prob batch mismatch";
    probs_steps.emplace_back(probs.view({-1, 1}));
  }
  return torch::cat(probs_steps, /*dim=*/1);
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
                                 int32_t num_speculative_tokens,
                                 const torch::Tensor& drop_mask) {
  CHECK(sample_output.next_tokens.defined())
      << "validate output tokens are undefined";
  CHECK_EQ(sample_output.next_tokens.dim(), 2)
      << "validate output tokens should be [batch, width]";
  const int32_t batch_size =
      static_cast<int32_t>(sample_output.next_tokens.size(0));
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(sample_output.next_tokens.size(1), num_val_tokens)
      << "validate output width mismatch";
  sample_output.next_tokens.masked_fill_(drop_mask, -1);

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

void scatter_varlen_target_output_to_dense(
    ForwardOutput& target_output,
    const std::vector<int32_t>& per_seq_val_tokens,
    int32_t batch_size,
    int32_t max_val_tokens) {
  CHECK(target_output.logits.defined())
      << "target logits must be defined for varlen->dense scatter";
  const int32_t total_tokens =
      static_cast<int32_t>(target_output.logits.size(0));
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;
  if (total_tokens == static_cast<int32_t>(padded_total)) {
    // Already uniform max width; no scatter needed.
    return;
  }
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(-1));

  // Pure order-preserving scatter: seq i's cu-packed rows [cu_offset,
  // cu_offset + seq_tokens) map 1:1 onto dense cols [0, seq_tokens) of row i.
  // The source rows are exactly [0, total_tokens) in order, so index_copy_
  // reads target_output.logits directly with no source gather. Row j is the
  // target's prediction after seeing anchor + draft[0..j-1]; col 0 predicts
  // draft[0] (anchor row) and col (seq_tokens - 1) is the per-seq bonus.
  // Trailing cols [seq_tokens, max_val_tokens) stay padded, so the bonus lands
  // at col (seq_tokens - 1), NOT the fixed last column; validate() and
  // apply_pruned_prefix_lengths therefore read the bonus/cut position per-seq.
  std::vector<int64_t> dst_indices_vec;
  dst_indices_vec.reserve(static_cast<size_t>(total_tokens));
  for (int32_t i = 0; i < batch_size; ++i) {
    const int32_t seq_tokens = per_seq_val_tokens[static_cast<size_t>(i)];
    for (int32_t j = 0; j < seq_tokens; ++j) {
      dst_indices_vec.push_back(static_cast<int64_t>(i) * max_val_tokens + j);
    }
  }
  torch::Tensor dst_indices =
      async_h2d_tensor(dst_indices_vec, target_output.logits.device());

  // Logits pad to [B*max_val_tokens, V]. Only the padding rows need the -inf
  // sentinel (strictly-rejecting distribution); using empty + a targeted
  // index_fill_ over the complement of dst_indices avoids paying vocab_size ×
  // padded_total writes when most rows get overwritten by index_copy_.
  torch::Tensor padded_logits =
      torch::empty({padded_total, vocab_size}, target_output.logits.options());
  if (dst_indices_vec.size() < static_cast<size_t>(padded_total)) {
    std::vector<bool> valid(static_cast<size_t>(padded_total), false);
    for (int64_t idx : dst_indices_vec) {
      valid[static_cast<size_t>(idx)] = true;
    }
    std::vector<int64_t> pad_indices_vec;
    pad_indices_vec.reserve(static_cast<size_t>(padded_total) -
                            dst_indices_vec.size());
    for (int64_t i = 0; i < padded_total; ++i) {
      if (!valid[static_cast<size_t>(i)]) {
        pad_indices_vec.push_back(i);
      }
    }
    torch::Tensor pad_indices =
        async_h2d_tensor(pad_indices_vec, target_output.logits.device());
    padded_logits.index_fill_(/*dim=*/0, pad_indices, -1e9);
  }
  padded_logits.index_copy_(/*dim=*/0, dst_indices, target_output.logits);
  target_output.logits = padded_logits;

  if (target_output.sample_output.embeddings.defined()) {
    const int32_t hidden_size =
        static_cast<int32_t>(target_output.sample_output.embeddings.size(-1));
    torch::Tensor padded_embeddings =
        torch::zeros({padded_total, hidden_size},
                     target_output.sample_output.embeddings.options());
    padded_embeddings.index_copy_(
        /*dim=*/0, dst_indices, target_output.sample_output.embeddings);
    target_output.sample_output.embeddings = padded_embeddings;
  }
}

}  // namespace adaptive_pruning
}  // namespace xllm
