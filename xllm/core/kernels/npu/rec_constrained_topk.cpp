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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr double kInvalidLogit = -1.0e20;

bool enable_constrained_topk_trace() {
  static const bool enabled = []() {
    const char* value = std::getenv("XLLM_DEBUG_ONEREC_CONSTRAINED_TOPK_TRACE");
    if (value == nullptr) {
      return false;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
  }();
  return enabled;
}

bool env_enabled(const char* name, bool default_value = false) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
         value[0] == 'y' || value[0] == 'Y';
}

bool enable_fused_constrained_topk() {
  static const bool enabled = env_enabled(
      "XLLM_ENABLE_ONEREC_FUSED_CONSTRAINED_TOPK", /*default_value=*/true);
  return enabled;
}

bool enable_fused_constrained_topk_compare() {
  static const bool enabled =
      env_enabled("XLLM_DEBUG_ONEREC_FUSED_CONSTRAINED_TOPK_COMPARE");
  return enabled;
}

torch::Tensor index_table_by_positions(const torch::Tensor& values,
                                       const torch::Tensor& positions) {
  if (values.size(0) == 0) {
    return torch::zeros(positions.sizes(), values.options());
  }
  torch::Tensor flat_positions = positions.reshape({-1}).to(torch::kLong);
  return values.index_select(/*dim=*/0, flat_positions).view(positions.sizes());
}

int64_t max_degree_with_top_k(const torch::Tensor& degrees, int64_t top_k) {
  CHECK(degrees.defined()) << "degrees is required";
  CHECK_GT(top_k, 0);
  if (degrees.numel() == 0) {
    return top_k;
  }
  const int64_t max_degree = degrees.max().to(torch::kCPU).item<int64_t>();
  return std::max<int64_t>(top_k, max_degree);
}

std::tuple<torch::Tensor, torch::Tensor> build_range_candidates(
    const torch::Tensor& values,
    const torch::Tensor& begins,
    const torch::Tensor& degrees,
    int64_t candidate_width) {
  CHECK_GT(candidate_width, 0);
  const int64_t num_rows = begins.size(0);
  auto index_options =
      torch::TensorOptions().dtype(torch::kLong).device(begins.device());
  torch::Tensor offsets =
      torch::arange(candidate_width, index_options).unsqueeze(/*dim=*/0);
  torch::Tensor positions = begins.to(torch::kLong).unsqueeze(/*dim=*/1) +
                            offsets.expand({num_rows, candidate_width});
  torch::Tensor valid =
      offsets.lt(degrees.to(torch::kLong).unsqueeze(/*dim=*/1));
  const int64_t max_position = std::max<int64_t>(values.size(0) - 1, 0);
  torch::Tensor safe_positions =
      torch::where(valid, positions, torch::zeros_like(positions))
          .clamp(/*min=*/0, /*max=*/max_position);
  torch::Tensor candidate_tokens =
      index_table_by_positions(values, safe_positions);
  return {candidate_tokens, valid};
}

std::tuple<torch::Tensor, torch::Tensor> finish_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& candidate_tokens,
    const torch::Tensor& valid_candidates,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k) {
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D, got " << logits.sizes();
  CHECK_EQ(candidate_tokens.dim(), 2)
      << "candidate_tokens must be 2-D, got " << candidate_tokens.sizes();
  CHECK_EQ(candidate_tokens.sizes(), valid_candidates.sizes())
      << "candidate token/mask shape mismatch, tokens="
      << candidate_tokens.sizes() << ", valid=" << valid_candidates.sizes();
  CHECK_EQ(logits.size(0), candidate_tokens.size(0))
      << "logits/candidate row mismatch, logits=" << logits.sizes()
      << ", candidates=" << candidate_tokens.sizes();
  CHECK_LE(top_k, candidate_tokens.size(1))
      << "top_k exceeds candidate width, top_k=" << top_k
      << ", candidate_width=" << candidate_tokens.size(1);

  torch::Tensor token_indices = candidate_tokens.to(torch::kLong);
  torch::Tensor candidate_logits =
      logits.gather(/*dim=*/1, /*index=*/token_indices).to(torch::kFloat32);
  if (temperatures.defined()) {
    torch::Tensor temps =
        temperatures.to(torch::kFloat32).to(logits.device()).unsqueeze(1);
    temps = torch::where(temps == 0, torch::ones_like(temps), temps);
    candidate_logits = candidate_logits / temps;
  }
  candidate_logits = candidate_logits.masked_fill(
      valid_candidates.logical_not(), kInvalidLogit);

  const std::vector<int64_t> reduce_dims{1};
  torch::Tensor log_denom =
      torch::logsumexp(candidate_logits, reduce_dims, /*keepdim=*/true);
  torch::Tensor top_values;
  torch::Tensor top_indices;
  std::tie(top_values, top_indices) = candidate_logits.topk(
      top_k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  torch::Tensor top_logprobs = top_values - log_denom;
  top_logprobs = torch::where(top_values.le(kInvalidLogit / 2.0),
                              torch::full_like(top_logprobs, kInvalidLogit),
                              top_logprobs);
  torch::Tensor top_tokens =
      candidate_tokens.gather(/*dim=*/1, /*index=*/top_indices)
          .to(torch::kLong);
  if (enable_constrained_topk_trace()) {
    LOG(INFO) << "OneRec constrained topk trace, step=" << current_step
              << ", rows=" << candidate_tokens.size(0)
              << ", candidate_width=" << candidate_tokens.size(1)
              << ", top_k=" << top_k;
  }
  return {top_tokens, top_logprobs};
}

std::tuple<torch::Tensor, torch::Tensor> build_first_token_candidates(
    const torch::Tensor& logits,
    const torch::Tensor& first_token_ids,
    int64_t top_k) {
  const int64_t allowed_count = first_token_ids.size(0);
  if (allowed_count >= top_k) {
    const int64_t num_rows = logits.size(0);
    torch::Tensor candidate_tokens = first_token_ids.unsqueeze(/*dim=*/0)
                                         .expand({num_rows, allowed_count})
                                         .contiguous();
    torch::Tensor valid_candidates =
        torch::ones({num_rows, allowed_count},
                    torch::TensorOptions()
                        .dtype(torch::kBool)
                        .device(first_token_ids.device()));
    return {candidate_tokens, valid_candidates};
  }
  const int64_t candidate_width = std::max<int64_t>(allowed_count, top_k);
  auto begin_options =
      torch::TensorOptions().dtype(torch::kLong).device(logits.device());
  torch::Tensor begins = torch::zeros({logits.size(0)}, begin_options);
  torch::Tensor degrees =
      torch::full({logits.size(0)}, allowed_count, begin_options);
  return build_range_candidates(
      first_token_ids, begins, degrees, candidate_width);
}

void fill_range_bucket_topk(const torch::Tensor& logits,
                            const torch::Tensor& values,
                            const torch::Tensor& begins,
                            const torch::Tensor& degrees,
                            const torch::Tensor& temperatures,
                            const torch::Tensor& row_indices,
                            int64_t current_step,
                            int64_t top_k,
                            int64_t candidate_width,
                            torch::Tensor& top_tokens,
                            torch::Tensor& top_logprobs) {
  if (row_indices.numel() == 0) {
    return;
  }

  torch::Tensor row_begins = begins.index_select(/*dim=*/0, row_indices);
  torch::Tensor row_degrees = degrees.index_select(/*dim=*/0, row_indices);
  torch::Tensor candidate_tokens;
  torch::Tensor valid_candidates;
  std::tie(candidate_tokens, valid_candidates) =
      build_range_candidates(values, row_begins, row_degrees, candidate_width);

  torch::Tensor row_logits = logits.index_select(/*dim=*/0, row_indices);
  torch::Tensor row_temperatures;
  if (temperatures.defined()) {
    if (temperatures.dim() > 0 && temperatures.size(0) == logits.size(0)) {
      row_temperatures = temperatures.index_select(/*dim=*/0, row_indices);
    } else {
      row_temperatures = temperatures;
    }
  }
  torch::Tensor bucket_tokens;
  torch::Tensor bucket_logprobs;
  std::tie(bucket_tokens, bucket_logprobs) =
      finish_constrained_topk(row_logits,
                              candidate_tokens,
                              valid_candidates,
                              row_temperatures,
                              current_step,
                              top_k);
  top_tokens.index_copy_(/*dim=*/0, row_indices, bucket_tokens);
  top_logprobs.index_copy_(/*dim=*/0, row_indices, bucket_logprobs);
}

std::tuple<torch::Tensor, torch::Tensor> finish_bucketed_range_topk(
    const torch::Tensor& logits,
    const torch::Tensor& values,
    const torch::Tensor& begins,
    const torch::Tensor& degrees,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t candidate_width) {
  CHECK_GT(candidate_width, 0);
  const int64_t num_rows = begins.size(0);
  torch::Tensor top_tokens = torch::empty(
      {num_rows, top_k},
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  torch::Tensor top_logprobs = torch::empty(
      {num_rows, top_k},
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

  torch::Tensor remaining = torch::ones(
      {num_rows},
      torch::TensorOptions().dtype(torch::kBool).device(logits.device()));
  std::vector<int64_t> bucket_widths;
  for (int64_t width = top_k; width < candidate_width; width *= 2) {
    bucket_widths.emplace_back(width);
    if (width > candidate_width / 2) {
      break;
    }
  }

  for (const int64_t bucket_width : bucket_widths) {
    torch::Tensor bucket_mask = remaining.logical_and(degrees.le(bucket_width));
    torch::Tensor row_indices =
        torch::nonzero(bucket_mask).reshape({-1}).to(torch::kLong);
    fill_range_bucket_topk(logits,
                           values,
                           begins,
                           degrees,
                           temperatures,
                           row_indices,
                           current_step,
                           top_k,
                           bucket_width,
                           top_tokens,
                           top_logprobs);
    remaining = remaining.logical_and(bucket_mask.logical_not());
  }

  torch::Tensor row_indices =
      torch::nonzero(remaining).reshape({-1}).to(torch::kLong);
  fill_range_bucket_topk(logits,
                         values,
                         begins,
                         degrees,
                         temperatures,
                         row_indices,
                         current_step,
                         top_k,
                         candidate_width,
                         top_tokens,
                         top_logprobs);
  return {top_tokens, top_logprobs};
}

std::tuple<torch::Tensor, torch::Tensor> finish_prefix1_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix1_degree) {
  torch::Tensor sequence_group_flat =
      sequence_group.reshape({-1, sequence_group.size(-1)});
  torch::Tensor t0 =
      sequence_group_flat.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor begins = prefix1_offsets.index_select(/*dim=*/0, t0);
  torch::Tensor ends = prefix1_offsets.index_select(/*dim=*/0, t0 + 1);
  torch::Tensor degrees = ends - begins;
  const int64_t candidate_width =
      max_prefix1_degree > 0 ? std::max<int64_t>(top_k, max_prefix1_degree)
                             : max_degree_with_top_k(degrees, top_k);
  if (enable_constrained_topk_trace()) {
    torch::Tensor degrees_cpu = degrees.to(torch::kCPU);
    const int64_t min_degree = degrees_cpu.min().item<int64_t>();
    const int64_t max_degree = degrees_cpu.max().item<int64_t>();
    const int64_t le_top_k = degrees_cpu.le(top_k).sum().item<int64_t>();
    const int64_t le_512 = degrees_cpu.le(512).sum().item<int64_t>();
    const int64_t le_1024 = degrees_cpu.le(1024).sum().item<int64_t>();
    const int64_t le_2048 = degrees_cpu.le(2048).sum().item<int64_t>();
    LOG(INFO) << "OneRec constrained prefix1 degree trace, rows="
              << degrees_cpu.numel() << ", min=" << min_degree
              << ", max=" << max_degree << ", le_top_k=" << le_top_k
              << ", le_512=" << le_512 << ", le_1024=" << le_1024
              << ", le_2048=" << le_2048
              << ", candidate_width=" << candidate_width;
  }
  if (candidate_width <= top_k * 2) {
    torch::Tensor candidate_tokens;
    torch::Tensor valid_candidates;
    std::tie(candidate_tokens, valid_candidates) = build_range_candidates(
        prefix1_values, begins, degrees, candidate_width);
    return finish_constrained_topk(logits,
                                   candidate_tokens,
                                   valid_candidates,
                                   temperatures,
                                   current_step,
                                   top_k);
  }
  return finish_bucketed_range_topk(logits,
                                    prefix1_values,
                                    begins,
                                    degrees,
                                    temperatures,
                                    current_step,
                                    top_k,
                                    candidate_width);
}

std::tuple<torch::Tensor, torch::Tensor> find_prefix1_pair_indices(
    const torch::Tensor& prefix1_pair_keys,
    const torch::Tensor& t0,
    const torch::Tensor& t1,
    int64_t vocab_size) {
  CHECK(prefix1_pair_keys.defined()) << "prefix1_pair_keys is required";
  CHECK_GT(prefix1_pair_keys.size(0), 0);
  torch::Tensor query_keys =
      t0.to(torch::kLong) * vocab_size + t1.to(torch::kLong);
  torch::Tensor pair_indices =
      torch::searchsorted(prefix1_pair_keys, query_keys, /*out_int32=*/false);
  const int64_t max_pair_index =
      std::max<int64_t>(prefix1_pair_keys.size(0) - 1, 0);
  torch::Tensor safe_pair_indices =
      pair_indices.clamp(/*min=*/0, /*max=*/max_pair_index);
  torch::Tensor found_keys =
      prefix1_pair_keys.index_select(/*dim=*/0, safe_pair_indices);
  torch::Tensor pair_valid = pair_indices.lt(prefix1_pair_keys.size(0))
                                 .logical_and(found_keys.eq(query_keys));
  return {pair_valid, pair_indices};
}

std::tuple<torch::Tensor, torch::Tensor> finish_prefix2_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& prefix1_pair_keys,
    int64_t vocab_size,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix2_degree) {
  torch::Tensor sequence_group_flat =
      sequence_group.reshape({-1, sequence_group.size(-1)});
  const int64_t num_rows = sequence_group_flat.size(0);
  CHECK_EQ(sequence_group_flat.dim(), 2)
      << "sequence_group must flatten to 2-D, got "
      << sequence_group_flat.sizes();
  if (prefix2_value_offsets.size(0) < 2 || prefix2_values.size(0) == 0) {
    const int64_t prefix2_width =
        max_prefix2_degree > 0 ? std::max<int64_t>(top_k, max_prefix2_degree)
                               : top_k;
    torch::Tensor candidate_tokens =
        torch::zeros({num_rows, prefix2_width}, prefix2_values.options());
    torch::Tensor valid_candidates =
        torch::zeros({num_rows, prefix2_width},
                     torch::TensorOptions()
                         .dtype(torch::kBool)
                         .device(sequence_group.device()));
    return finish_constrained_topk(logits,
                                   candidate_tokens,
                                   valid_candidates,
                                   temperatures,
                                   current_step,
                                   top_k);
  }

  torch::Tensor t0 =
      sequence_group_flat.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor t1 = sequence_group_flat.select(/*dim=*/1, /*index=*/1);

  torch::Tensor pair_valid;
  torch::Tensor pair_indices;
  std::tie(pair_valid, pair_indices) =
      find_prefix1_pair_indices(prefix1_pair_keys, t0, t1, vocab_size);

  const int64_t max_pair_index =
      std::max<int64_t>(prefix2_value_offsets.size(0) - 2, 0);
  torch::Tensor safe_pair_index_rows =
      torch::where(pair_valid, pair_indices, torch::zeros_like(pair_indices))
          .clamp(/*min=*/0, /*max=*/max_pair_index);

  torch::Tensor prefix2_begins =
      prefix2_value_offsets.index_select(/*dim=*/0, safe_pair_index_rows);
  torch::Tensor prefix2_ends =
      prefix2_value_offsets.index_select(/*dim=*/0, safe_pair_index_rows + 1);
  torch::Tensor prefix2_degrees =
      torch::where(pair_valid,
                   prefix2_ends - prefix2_begins,
                   torch::zeros_like(prefix2_begins));

  const int64_t prefix2_width =
      max_prefix2_degree > 0 ? std::max<int64_t>(top_k, max_prefix2_degree)
                             : max_degree_with_top_k(prefix2_degrees, top_k);
  if (prefix2_width <= top_k * 2) {
    torch::Tensor candidate_tokens;
    torch::Tensor valid_candidates;
    std::tie(candidate_tokens, valid_candidates) = build_range_candidates(
        prefix2_values, prefix2_begins, prefix2_degrees, prefix2_width);
    return finish_constrained_topk(logits,
                                   candidate_tokens,
                                   valid_candidates,
                                   temperatures,
                                   current_step,
                                   top_k);
  }
  return finish_bucketed_range_topk(logits,
                                    prefix2_values,
                                    prefix2_begins,
                                    prefix2_degrees,
                                    temperatures,
                                    current_step,
                                    top_k,
                                    prefix2_width);
}

std::tuple<torch::Tensor, torch::Tensor> rec_constrained_topk_composite(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& first_token_ids,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& prefix1_pair_keys,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix1_degree,
    int64_t max_prefix2_degree) {
  CHECK(logits.defined()) << "logits is required";
  CHECK_EQ(logits.dim(), 2) << "logits must be 2-D, got " << logits.sizes();
  CHECK_GT(top_k, 0);
  CHECK(first_token_ids.defined()) << "first_token_ids is required";
  CHECK(prefix1_offsets.defined()) << "prefix1_offsets is required";
  CHECK(prefix1_values.defined()) << "prefix1_values is required";
  CHECK(prefix1_pair_keys.defined()) << "prefix1_pair_keys is required";
  CHECK_EQ(prefix1_pair_keys.size(0), prefix1_values.size(0));
  CHECK(prefix2_value_offsets.defined()) << "prefix2_value_offsets is required";
  CHECK(prefix2_values.defined()) << "prefix2_values is required";

  torch::Tensor candidate_tokens;
  torch::Tensor valid_candidates;
  if (current_step == 0) {
    std::tie(candidate_tokens, valid_candidates) =
        build_first_token_candidates(logits, first_token_ids, top_k);
  } else if (current_step == 1) {
    CHECK(sequence_group.defined()) << "sequence_group is required for step 1";
    return finish_prefix1_constrained_topk(logits,
                                           sequence_group,
                                           prefix1_offsets,
                                           prefix1_values,
                                           temperatures,
                                           current_step,
                                           top_k,
                                           max_prefix1_degree);
  } else if (current_step == 2) {
    CHECK(sequence_group.defined()) << "sequence_group is required for step 2";
    const int64_t vocab_size = prefix1_offsets.size(0) - 1;
    CHECK_GT(vocab_size, 0);
    return finish_prefix2_constrained_topk(logits,
                                           sequence_group,
                                           prefix1_pair_keys,
                                           vocab_size,
                                           prefix2_value_offsets,
                                           prefix2_values,
                                           temperatures,
                                           current_step,
                                           top_k,
                                           max_prefix2_degree);
  } else {
    LOG(FATAL) << "Unsupported OneRec constrained step: " << current_step;
  }

  return finish_constrained_topk(logits,
                                 candidate_tokens,
                                 valid_candidates,
                                 temperatures,
                                 current_step,
                                 top_k);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rec_constrained_topk(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& first_token_ids,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& prefix1_pair_keys,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k,
    int64_t max_prefix1_degree,
    int64_t max_prefix2_degree) {
  if (!enable_fused_constrained_topk()) {
    return rec_constrained_topk_composite(logits,
                                          sequence_group,
                                          first_token_ids,
                                          prefix1_offsets,
                                          prefix1_values,
                                          prefix1_pair_keys,
                                          prefix2_value_offsets,
                                          prefix2_values,
                                          temperatures,
                                          current_step,
                                          top_k,
                                          max_prefix1_degree,
                                          max_prefix2_degree);
  }

  std::optional<std::tuple<torch::Tensor, torch::Tensor>> fused_output =
      rec_constrained_topk_fused(logits,
                                 sequence_group,
                                 first_token_ids,
                                 prefix1_offsets,
                                 prefix1_values,
                                 prefix1_pair_keys,
                                 prefix2_value_offsets,
                                 prefix2_values,
                                 temperatures,
                                 current_step,
                                 top_k,
                                 max_prefix1_degree,
                                 max_prefix2_degree);
  if (!fused_output.has_value()) {
    LOG_FIRST_N(WARNING, 1)
        << "OneRec fused constrained topk is enabled but unavailable or "
           "failed; falling back to composite rec_constrained_topk.";
    return rec_constrained_topk_composite(logits,
                                          sequence_group,
                                          first_token_ids,
                                          prefix1_offsets,
                                          prefix1_values,
                                          prefix1_pair_keys,
                                          prefix2_value_offsets,
                                          prefix2_values,
                                          temperatures,
                                          current_step,
                                          top_k,
                                          max_prefix1_degree,
                                          max_prefix2_degree);
  }

  torch::Tensor fused_tokens;
  torch::Tensor fused_logprobs;
  std::tie(fused_tokens, fused_logprobs) = fused_output.value();
  fused_tokens = fused_tokens.to(torch::kLong);
  if (enable_fused_constrained_topk_compare()) {
    torch::Tensor composite_tokens;
    torch::Tensor composite_logprobs;
    std::tie(composite_tokens, composite_logprobs) =
        rec_constrained_topk_composite(logits,
                                       sequence_group,
                                       first_token_ids,
                                       prefix1_offsets,
                                       prefix1_values,
                                       prefix1_pair_keys,
                                       prefix2_value_offsets,
                                       prefix2_values,
                                       temperatures,
                                       current_step,
                                       top_k,
                                       max_prefix1_degree,
                                       max_prefix2_degree);
    const bool tokens_match = torch::equal(fused_tokens, composite_tokens);
    const bool logprobs_match = torch::allclose(fused_logprobs,
                                                composite_logprobs,
                                                /*rtol=*/1.0e-4,
                                                /*atol=*/1.0e-4);
    if (!tokens_match || !logprobs_match) {
      torch::Tensor token_diff = fused_tokens.ne(composite_tokens);
      const int64_t token_mismatch_count =
          token_diff.sum().to(torch::kCPU).item<int64_t>();
      const double max_logprob_diff = (fused_logprobs - composite_logprobs)
                                          .abs()
                                          .max()
                                          .to(torch::kCPU)
                                          .item<double>();
      torch::Tensor mismatch_rows =
          torch::nonzero(token_diff.any(/*dim=*/1)).reshape({-1});
      int64_t first_mismatch_row = -1;
      if (mismatch_rows.numel() > 0) {
        first_mismatch_row = mismatch_rows.select(/*dim=*/0, /*index=*/0)
                                 .to(torch::kCPU)
                                 .item<int64_t>();
      }
      const int64_t debug_width = std::min<int64_t>(top_k, 16);
      std::ostringstream fused_tokens_head;
      std::ostringstream composite_tokens_head;
      std::ostringstream fused_logprobs_head;
      std::ostringstream composite_logprobs_head;
      if (first_mismatch_row >= 0) {
        fused_tokens_head << fused_tokens.select(/*dim=*/0, first_mismatch_row)
                                 .slice(/*dim=*/0,
                                        /*start=*/0,
                                        /*end=*/debug_width)
                                 .to(torch::kCPU);
        composite_tokens_head
            << composite_tokens.select(/*dim=*/0, first_mismatch_row)
                   .slice(/*dim=*/0, /*start=*/0, /*end=*/debug_width)
                   .to(torch::kCPU);
        fused_logprobs_head
            << fused_logprobs.select(/*dim=*/0, first_mismatch_row)
                   .slice(/*dim=*/0, /*start=*/0, /*end=*/debug_width)
                   .to(torch::kCPU);
        composite_logprobs_head
            << composite_logprobs.select(/*dim=*/0, first_mismatch_row)
                   .slice(/*dim=*/0, /*start=*/0, /*end=*/debug_width)
                   .to(torch::kCPU);
      }
      const char* severity = logprobs_match ? "tie-order" : "value";
      std::ostringstream message;
      message << "OneRec fused constrained topk compare mismatch, kind="
              << severity << ", step=" << current_step << ", top_k=" << top_k
              << ", tokens_match=" << tokens_match
              << ", logprobs_match=" << logprobs_match
              << ", token_mismatch_count=" << token_mismatch_count
              << ", max_logprob_diff=" << max_logprob_diff
              << ", first_mismatch_row=" << first_mismatch_row
              << ", fused_tokens_head=" << fused_tokens_head.str()
              << ", composite_tokens_head=" << composite_tokens_head.str()
              << ", fused_logprobs_head=" << fused_logprobs_head.str()
              << ", composite_logprobs_head=" << composite_logprobs_head.str();
      if (logprobs_match) {
        LOG_FIRST_N(WARNING, 8) << message.str();
      } else {
        LOG(ERROR) << message.str();
      }
    }
  }
  return {fused_tokens, fused_logprobs};
}

}  // namespace xllm::kernel::npu
