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

#include "scheduler/profile/decode_graph_warmup_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "core/framework/config/speculative_config.h"
#include "platform/platform.h"
#include "runtime/acl_graph_bucket_policy.h"

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

namespace {

DecodeGraphWarmupPlan build_target_decode_graph_warmup_plan(
    const DecodeGraphWarmupConfig& warmup_config,
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      max_global_batch_size, dp_size);
  plan.warmup_config = warmup_config;

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
      warmup_config.num_speculative_tokens > 0 &&
      max_global_batch_size >= dp_size && dp_size > 0 &&
      (!warmup_config.enable_graph_mode_decode_no_padding ||
       warmup_config.max_graph_batch_size > 0);
  if (!use_mtp_batches) {
    return plan;
  }

  if (warmup_config.max_graph_batch_size > 0) {
    // The graph batch limit caps the DP-local decode batch: the executor
    // compares the largest DP-local batch against it before entering graph
    // mode, so the warmup sweep must apply the limit in local terms as well.
    const int32_t max_graph_global_batch_size =
        static_cast<int32_t>(npu::acl_graph_max_global_batch_size(
            static_cast<uint32_t>(max_global_batch_size),
            static_cast<uint32_t>(warmup_config.max_graph_batch_size),
            static_cast<uint32_t>(dp_size)));
    std::vector<int32_t> graph_batch_sizes =
        npu::get_acl_graph_mtp_warmup_batch_sizes(
            static_cast<uint32_t>(max_global_batch_size),
            static_cast<uint32_t>(dp_size),
            static_cast<uint32_t>(warmup_config.num_speculative_tokens),
            warmup_config.enable_graph_mode_decode_no_padding,
            static_cast<uint32_t>(warmup_config.max_graph_batch_size));

    std::vector<int32_t> batch_sizes;
    batch_sizes.reserve(plan.batch_sizes.size() + graph_batch_sizes.size());
    for (int32_t batch_size : plan.batch_sizes) {
      if (warmup_config.enable_graph_mode_decode_no_padding ||
          batch_size >= max_graph_global_batch_size) {
        batch_sizes.emplace_back(batch_size);
      }
    }
    batch_sizes.insert(
        batch_sizes.end(), graph_batch_sizes.begin(), graph_batch_sizes.end());
    std::sort(batch_sizes.begin(), batch_sizes.end());
    batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()),
                      batch_sizes.end());
    plan.batch_sizes = std::move(batch_sizes);
    return plan;
  }

  const int32_t max_local_batch_size =
      static_cast<int32_t>(npu::acl_graph_max_full_local_batch_size(
          static_cast<uint32_t>(max_global_batch_size),
          static_cast<uint32_t>(dp_size)));
  std::vector<int32_t> batch_sizes = npu::get_acl_graph_mtp_warmup_batch_sizes(
      static_cast<uint32_t>(max_global_batch_size),
      static_cast<uint32_t>(dp_size),
      static_cast<uint32_t>(warmup_config.num_speculative_tokens),
      warmup_config.enable_graph_mode_decode_no_padding,
      0);

  if (max_local_batch_size * dp_size < max_global_batch_size) {
    batch_sizes.emplace_back(max_global_batch_size);
  }
  plan.batch_sizes = std::move(batch_sizes);

  return plan;
}

}  // namespace

DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const DecodeGraphWarmupConfig& warmup_config,
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan = build_target_decode_graph_warmup_plan(
      warmup_config, max_global_batch_size, dp_size);
  if (!Platform::supports_mtp_decode_graph_warmup() ||
      warmup_config.num_speculative_tokens == 0 ||
      !SpeculativeConfig::is_mtp_algorithm(
          warmup_config.speculative_algorithm) ||
      warmup_config.enable_graph_mode_decode_no_padding || dp_size <= 0 ||
      max_global_batch_size < dp_size) {
    return plan;
  }
  const int32_t max_local_batch_size =
      warmup_config.max_graph_batch_size > 0
          ? std::min(max_global_batch_size / dp_size,
                     warmup_config.max_graph_batch_size)
          : max_global_batch_size / dp_size;
  std::vector<int32_t> joint_batches;
  joint_batches.reserve(plan.batch_sizes.size() +
                        kCompatibilitySmallBatchSizes.size());
  int64_t previous_bucket = 0;
  for (int32_t local = 1; local <= max_local_batch_size; ++local) {
    const int64_t bucket =
        std::min(runtime::get_decode_graph_token_bucket(local, false),
                 static_cast<int64_t>(max_local_batch_size));
    if (bucket != previous_bucket) {
      joint_batches.emplace_back(local * dp_size);
      previous_bucket = bucket;
    } else {
      joint_batches.back() = local * dp_size;
    }
  }

  // Seed the schedule with every independently required draft bucket. Add a
  // target representative only if its token bucket is still uncovered. In
  // particular, target rows 24 and 30 both pad to 32, but Draft5 requests 4
  // and 5 belong to different request buckets (4 and 8).
  std::vector<int64_t> covered_target_buckets;
  covered_target_buckets.reserve(joint_batches.size() +
                                 plan.batch_sizes.size());
  const auto target_bucket = [&warmup_config, dp_size](int32_t batch_size) {
    return runtime::get_decode_graph_token_bucket(
        static_cast<int64_t>(batch_size / dp_size) *
            (warmup_config.num_speculative_tokens + 1),
        /*enable_no_padding=*/false);
  };
  for (const int32_t batch_size : joint_batches) {
    covered_target_buckets.emplace_back(target_bucket(batch_size));
  }
  for (const int32_t batch_size : plan.batch_sizes) {
    const bool independent_batch = batch_size % dp_size != 0 ||
                                   batch_size > max_local_batch_size * dp_size;
    const int64_t bucket = target_bucket(batch_size);
    if (independent_batch ||
        std::find(covered_target_buckets.begin(),
                  covered_target_buckets.end(),
                  bucket) == covered_target_buckets.end()) {
      joint_batches.emplace_back(batch_size);
      covered_target_buckets.emplace_back(bucket);
    }
  }
  std::sort(joint_batches.begin(), joint_batches.end());
  joint_batches.erase(std::unique(joint_batches.begin(), joint_batches.end()),
                      joint_batches.end());
  plan.batch_sizes = std::move(joint_batches);
  return plan;
}

}  // namespace xllm
