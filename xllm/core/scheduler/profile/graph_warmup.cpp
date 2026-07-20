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

#include "scheduler/profile/graph_warmup.h"

#include <absl/strings/str_cat.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>
#include <iomanip>
#include <sstream>

#include "framework/request/sequence.h"

namespace xllm {
namespace {

constexpr int32_t kGraphWarmupBarWidth = 20;

}  // namespace

GraphWarmupPlan graph_warmup_plan(InstanceRole role) {
  if (role == InstanceRole::PREFILL) {
    return GraphWarmupPlan::PREFILL_ONLY;
  }
  if (role == InstanceRole::DECODE) {
    return GraphWarmupPlan::DECODE_ONLY;
  }

  return GraphWarmupPlan::UNIFIED;
}

std::vector<int32_t> graph_warmup_buckets(int32_t max_seqs_per_batch) {
  CHECK_GT(max_seqs_per_batch, 0);

  std::vector<int32_t> buckets;
  const std::vector<int32_t> small_buckets = {1, 2, 4, 8, 16};
  for (int32_t bucket : small_buckets) {
    if (bucket <= max_seqs_per_batch) {
      buckets.emplace_back(bucket);
    }
  }

  for (int32_t bucket = 32; bucket <= max_seqs_per_batch; bucket += 16) {
    buckets.emplace_back(bucket);
  }

  if (buckets.back() != max_seqs_per_batch) {
    buckets.emplace_back(max_seqs_per_batch);
  }

  return buckets;
}

bool skip_graph_bucket(int32_t bucket, int32_t dp_size) {
  CHECK_GT(bucket, 0);
  CHECK_GT(dp_size, 0);
  return bucket < dp_size;
}

std::vector<int32_t> graph_decode_buckets(int32_t max_seqs_per_batch,
                                          int32_t dp_size) {
  std::vector<int32_t> buckets = graph_warmup_buckets(max_seqs_per_batch);
  std::vector<int32_t> decode_buckets;
  decode_buckets.reserve(buckets.size());
  for (int32_t bucket : buckets) {
    if (!skip_graph_bucket(bucket, dp_size)) {
      decode_buckets.emplace_back(bucket);
    }
  }

  return decode_buckets;
}

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t bucket,
                                  double latency_ms) {
  CHECK_GT(total, 0);
  CHECK_GE(completed, 0);
  CHECK_LE(completed, total);
  CHECK_GT(bucket, 0);
  CHECK_GE(latency_ms, 0.0);

  const int32_t filled = static_cast<int32_t>(
      (static_cast<int64_t>(completed) * kGraphWarmupBarWidth + total / 2) /
      total);

  std::string bar;
  bar.reserve(kGraphWarmupBarWidth);
  bar.append(static_cast<size_t>(filled), '#');
  bar.append(static_cast<size_t>(kGraphWarmupBarWidth - filled), '-');

  const double percent =
      static_cast<double>(completed) * 100.0 / static_cast<double>(total);

  std::ostringstream oss;
  oss << "Graph warmup progress: [" << bar << "] " << completed << "/" << total
      << " " << std::fixed << std::setprecision(1) << percent
      << "%, bucket=" << bucket << ", latency=" << std::setprecision(2)
      << latency_ms << " ms";
  return oss.str();
}

std::string next_warmup_request_id() {
  static std::atomic<int64_t> counter{0};
  const int64_t id = counter.fetch_add(1, std::memory_order_relaxed);
  return absl::StrCat("warmup_", id);
}

void prepare_warmup_decode_sequence(Sequence* sequence,
                                    int64_t embedding_width,
                                    int32_t num_speculative_tokens) {
  CHECK(sequence != nullptr);
  if (num_speculative_tokens <= 0) {
    return;
  }

  CHECK_GT(embedding_width, 0);
  // Placeholder bootstrap hidden states; the worker converts dtype/device and
  // only the [1, embedding_width] shape matters for the batch input builder.
  sequence->update_mtp_bootstrap_embedding(
      torch::zeros({1, embedding_width}, /*options=*/torch::kFloat));
}

}  // namespace xllm
