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

#include "dit_cache_impl.h"

#include "dit_non_cache.h"
#include "fbcache.h"
#include "fbcache_taylorseer.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "residual_cache.h"
#include "taylorseer.h"

namespace xllm {

torch::Tensor DitCacheImpl::get_tensor_or_empty(const TensorMap& m,
                                                const std::string& k) {
  auto it = m.find(k);
  if (it != m.end()) return it->second;
  return torch::Tensor();
}

bool DitCacheImpl::is_similar(const torch::Tensor& lhs,
                              const torch::Tensor& rhs,
                              float threshold) const {
  if (!lhs.defined() || !rhs.defined()) return false;
  if (lhs.sizes() != rhs.sizes()) return false;

  if (threshold <= 0.0f) {
    return torch::allclose(lhs, rhs);
  }

  // SP shards the sequence, so a local mean would let ranks disagree on cache
  // reuse. All-reduce the raw sums + count for an identical global decision on
  // every rank. Shards are equal-length, so sum/count is the exact global mean.
  //
  // The all-reduce is a collective: every early-return above must fire
  // identically on all SP ranks, or HCCL hangs. Keep new early-outs symmetric.
  auto sum_abs_diff = (lhs - rhs).abs().to(torch::kFloat32).sum();
  auto sum_abs_lhs = lhs.abs().to(torch::kFloat32).sum();
  auto count = torch::full({},
                           static_cast<float>(lhs.numel()),
                           torch::dtype(torch::kFloat32).device(lhs.device()));

  ProcessGroup* sp_group =
      parallel_args_ != nullptr ? parallel_args_->dit_sp_group_ : nullptr;
  if (sp_group != nullptr && sp_group->world_size() > 1) {
    // Pack the three scalars into one tensor to do a single all-reduce.
    auto stats = torch::stack({sum_abs_diff, sum_abs_lhs, count});
    stats = parallel_state::reduce(stats, sp_group);
    sum_abs_diff = stats[0];
    sum_abs_lhs = stats[1];
    count = stats[2];
  }

  auto mean_diff = sum_abs_diff / count;
  auto mean_lhs = sum_abs_lhs / count;

  auto rel = mean_diff / (mean_lhs + 1e-6);
  return (rel < threshold).item<bool>();
}

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg) {
  switch (cfg.selected_policy) {
    case PolicyType::FBCache:
      return std::make_unique<FBCache>();
    case PolicyType::TaylorSeer:
      return std::make_unique<TaylorSeer>();
    case PolicyType::FBCacheTaylorSeer:
      return std::make_unique<FBCacheTaylorSeer>();
    case PolicyType::ResidualCache:
      return std::make_unique<ResidualCache>();
    default:
      return std::make_unique<DiTNonCache>();
  }
}
}  // namespace xllm
