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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>

namespace xllm::runtime::detail {

class MluDecodeRoundState {
 public:
  void require_prev_beam(int32_t round) const {
    CHECK_GT(round, 0) << "decode rounds start at 1";
    CHECK_EQ(last_beam_round_, round - 1)
        << "MLU decode round " << round
        << " requires the previous beam search to complete";
  }

  void mark_beam_complete(int32_t round) {
    CHECK_EQ(round, last_beam_round_ + 1)
        << "MLU beam search rounds must be consecutive";
    last_beam_round_ = round;
  }

  int32_t last_beam_round() const { return last_beam_round_; }

 private:
  int32_t last_beam_round_ = -1;
};

inline torch::Tensor get_mlu_decode_tokens(const torch::Tensor& prev_tokens,
                                           int64_t rows) {
  CHECK(prev_tokens.defined()) << "MLU decode requires selected beam tokens";
  CHECK_EQ(prev_tokens.scalar_type(), torch::kInt32)
      << "MLU selected beam tokens must be int32";
  CHECK(prev_tokens.is_contiguous())
      << "MLU selected beam tokens must be contiguous";
  CHECK_EQ(prev_tokens.numel(), rows)
      << "MLU selected token count mismatch: actual " << prev_tokens.numel()
      << ", expected " << rows;
  return prev_tokens.reshape({rows});
}

inline bool should_reparent_mlu_cache(int32_t round, int32_t total_rounds) {
  CHECK_GT(total_rounds, 0) << "total_rounds must be positive";
  return round > 0 && round < total_rounds - 1;
}

inline bool has_mlu_fixed_result_width(int32_t beam_width,
                                       int32_t num_return_sequences) {
  const int32_t result_width =
      num_return_sequences > 0 ? num_return_sequences : beam_width;
  return result_width == beam_width;
}

inline void update_mlu_two_stage(const torch::Tensor& block_table,
                                 int32_t decode_step,
                                 int32_t max_decode_steps,
                                 torch::Tensor& slot_mapping,
                                 torch::Tensor& unshared_seq_lens) {
  CHECK_GT(max_decode_steps, 0) << "max_decode_steps must be positive";
  CHECK_GE(decode_step, 0) << "decode_step must be non-negative";
  CHECK_LT(decode_step, max_decode_steps)
      << "decode_step " << decode_step << " exceeds capacity "
      << max_decode_steps;
  CHECK_EQ(block_table.dim(), 2)
      << "MLU block_table must have shape [total_beam, 1], got "
      << block_table.sizes();
  CHECK_EQ(block_table.size(1), 1)
      << "MLU block_table must have shape [total_beam, 1], got "
      << block_table.sizes();
  CHECK_EQ(slot_mapping.dim(), 1) << "slot_mapping must be rank 1";
  CHECK_EQ(unshared_seq_lens.dim(), 1) << "unshared_seq_lens must be rank 1";
  CHECK_EQ(slot_mapping.numel(), block_table.size(0))
      << "slot_mapping total_beam mismatch";
  CHECK_EQ(unshared_seq_lens.numel(), block_table.size(0))
      << "unshared_seq_lens total_beam mismatch";
  CHECK_EQ(block_table.scalar_type(), torch::kInt32)
      << "MLU block_table must be int32";
  CHECK_EQ(slot_mapping.scalar_type(), torch::kInt32)
      << "MLU slot_mapping must be int32";
  CHECK_EQ(unshared_seq_lens.scalar_type(), torch::kInt32)
      << "MLU unshared_seq_lens must be int32";
  CHECK_EQ(slot_mapping.device(), block_table.device())
      << "slot_mapping and block_table must be on the same device";
  CHECK_EQ(unshared_seq_lens.device(), block_table.device())
      << "unshared_seq_lens and block_table must be on the same device";

  slot_mapping.copy_(block_table.view({-1}));
  slot_mapping.mul_(max_decode_steps).add_(decode_step);
  unshared_seq_lens.fill_(decode_step + 1);
}

inline void write_first_round_beam_outputs(
    const torch::Tensor& flat_top_tokens,
    const torch::Tensor& flat_top_logprobs,
    int32_t batch_size,
    torch::Tensor& out_token_ids,
    torch::Tensor& out_log_probs,
    torch::Tensor& out_seqgroup) {
  CHECK_EQ(flat_top_tokens.dim(), 2)
      << "flat_top_tokens must be [batch * beam, 1], got "
      << flat_top_tokens.sizes();
  CHECK_EQ(flat_top_logprobs.dim(), 2)
      << "flat_top_logprobs must be [batch * beam, 1], got "
      << flat_top_logprobs.sizes();
  CHECK_EQ(flat_top_tokens.sizes(), flat_top_logprobs.sizes())
      << "flat_top_tokens/top_logprobs shape mismatch";
  CHECK_EQ(flat_top_tokens.size(0), out_token_ids.size(0))
      << "top_tokens/out_token_ids rows mismatch";
  CHECK_EQ(flat_top_logprobs.size(0), out_log_probs.size(0))
      << "top_logprobs/out_log_probs rows mismatch";
  CHECK_EQ(flat_top_tokens.size(1), 1)
      << "first-round top_tokens must have exactly one column";
  CHECK_EQ(flat_top_logprobs.size(1), 1)
      << "first-round top_logprobs must have exactly one column";
  CHECK_GT(batch_size, 0) << "batch_size must be positive";
  CHECK_EQ(flat_top_tokens.size(0) % batch_size, 0)
      << "top_tokens rows must be divisible by batch_size";
  CHECK_EQ(out_seqgroup.dim(), 3)
      << "out_seqgroup must be [batch, beam, rounds], got "
      << out_seqgroup.sizes();
  CHECK_EQ(out_seqgroup.size(0), batch_size) << "out_seqgroup batch mismatch";

  const int64_t beam_width = flat_top_tokens.size(0) / batch_size;
  CHECK_EQ(out_seqgroup.size(1), beam_width) << "out_seqgroup beam mismatch";

  out_token_ids.copy_(flat_top_tokens);
  out_log_probs.copy_(flat_top_logprobs);
  out_seqgroup.select(/*dim=*/2, /*index=*/0)
      .copy_(flat_top_tokens.view({batch_size, beam_width}));
}

// Records the cumulative beam score snapshot for each output slot into
// token_step_logprobs[batch, beam, round].
inline void write_beam_round_step_logprobs(const torch::Tensor& out_log_probs,
                                           int32_t batch_size,
                                           int32_t round,
                                           torch::Tensor& token_step_logprobs) {
  if (!out_log_probs.defined() || out_log_probs.numel() == 0 ||
      !token_step_logprobs.defined()) {
    return;
  }
  CHECK_GT(batch_size, 0);
  CHECK_EQ(out_log_probs.dim(), 2);
  const int64_t step_width = out_log_probs.size(0) / batch_size;
  CHECK_EQ(out_log_probs.size(0), batch_size * step_width);
  CHECK_GE(round, 0);
  CHECK_LT(round, token_step_logprobs.size(2));

  if (token_step_logprobs.size(1) != step_width) {
    const int64_t total_rounds = token_step_logprobs.size(2);
    auto resized = torch::zeros({batch_size, step_width, total_rounds},
                                token_step_logprobs.options());
    const int64_t copy_beams =
        std::min(token_step_logprobs.size(1), step_width);
    if (copy_beams > 0 && round > 0) {
      resized.index({torch::indexing::Slice(),
                     torch::indexing::Slice(0, copy_beams),
                     torch::indexing::Slice(0, round)}) =
          token_step_logprobs.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(0, copy_beams),
                                     torch::indexing::Slice(0, round)});
    }
    token_step_logprobs = std::move(resized);
  }

  token_step_logprobs.index(
      {torch::indexing::Slice(), torch::indexing::Slice(), round}) =
      out_log_probs.view({batch_size, step_width});
}

}  // namespace xllm::runtime::detail
