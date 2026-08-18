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

#include "scheduler/profile/decode_graph_warmup_plan.h"

#include <array>
#include <cstddef>
#include <utility>

#include "platform/platform.h"

namespace xllm {
namespace {

constexpr int32_t kCompatibilityBatchStep = 16;
constexpr std::array<int32_t, 5> kCompatibilitySmallBatchSizes = {1,
                                                                  2,
                                                                  4,
                                                                  8,
                                                                  16};

}  // namespace

DecodeGraphWarmupPlan get_compatibility_decode_graph_warmup_plan(
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan;
  if (max_global_batch_size <= 0 || dp_size <= 0) {
    return plan;
  }

  const size_t max_bucket_count =
      kCompatibilitySmallBatchSizes.size() +
      static_cast<size_t>(max_global_batch_size) /
          static_cast<size_t>(kCompatibilityBatchStep) +
      1;
  plan.batch_sizes.reserve(max_bucket_count);
  for (int32_t batch_size : kCompatibilitySmallBatchSizes) {
    if (batch_size >= dp_size && batch_size <= max_global_batch_size) {
      plan.batch_sizes.emplace_back(batch_size);
    }
  }

  for (int32_t batch_size = kCompatibilityBatchStep * 2;
       batch_size <= max_global_batch_size;
       batch_size += kCompatibilityBatchStep) {
    if (batch_size >= dp_size) {
      plan.batch_sizes.emplace_back(batch_size);
    }
  }

  if (max_global_batch_size >= dp_size &&
      (plan.batch_sizes.empty() ||
       plan.batch_sizes.back() != max_global_batch_size)) {
    plan.batch_sizes.emplace_back(max_global_batch_size);
  }

  return plan;
}

DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const runtime::DecodeGraphExecutionShape& execution_shape,
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      max_global_batch_size, dp_size);
  plan.execution_shape = execution_shape;

  // MTP emits num_decoding_tokens rows per sequence. On supporting backends,
  // the graph cache is keyed by the padded number of rows rather than the
  // sequence count. Therefore the compatibility schedule can miss a graph
  // bucket (for example, batch size 9 with four decode tokens starts the
  // 48-token bucket). Platform owns this capability decision so this generic
  // plan does not depend on a device build macro. A backend must not opt in
  // until its graph keying and MTP replay behavior are covered by tests.
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    return plan;
  }

  const bool use_mtp_batches =
      !execution_shape.enable_graph_mode_decode_no_padding &&
      execution_shape.num_decoding_tokens > 1 &&
      max_global_batch_size >= dp_size && dp_size > 0;
  if (!use_mtp_batches) {
    return plan;
  }

  const int32_t max_local_batch_size = max_global_batch_size / dp_size;
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(static_cast<size_t>(max_local_batch_size) + 1);
  int64_t last_token_bucket = 0;
  for (int32_t local_batch_size = 1; local_batch_size <= max_local_batch_size;
       ++local_batch_size) {
    const int64_t num_tokens = static_cast<int64_t>(local_batch_size) *
                               execution_shape.num_decoding_tokens;
    const int64_t token_bucket = runtime::get_decode_graph_token_bucket(
        num_tokens, execution_shape.enable_graph_mode_decode_no_padding);
    if (batch_sizes.empty() || token_bucket != last_token_bucket) {
      batch_sizes.emplace_back(local_batch_size * dp_size);
      last_token_bucket = token_bucket;
    }
  }

  const int32_t max_full_global_batch_size = max_local_batch_size * dp_size;
  if (max_full_global_batch_size < max_global_batch_size) {
    batch_sizes.emplace_back(max_global_batch_size);
  }
  plan.batch_sizes = std::move(batch_sizes);

  return plan;
}

}  // namespace xllm
