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

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "dit_non_cache.h"
#include "fbcache.h"
#include "fbcache_taylorseer.h"
#include "framework/parallel_state/parallel_state.h"
#include "racfgcache.h"
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

  torch::Device dev = lhs.device();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

  auto sum_abs_diff = (lhs - rhs).abs().sum().to(torch::kFloat32);
  auto sum_abs_lhs = lhs.abs().sum().to(torch::kFloat32);
  auto count = torch::tensor({static_cast<float>(lhs.numel())}, opts);

  if (runtime_ctx_.sp_enabled && runtime_ctx_.sp_world_size > 1 &&
      runtime_ctx_.sp_group != nullptr) {
    auto* sp_group = static_cast<ProcessGroup*>(runtime_ctx_.sp_group);
    CHECK(sp_group != nullptr)
        << "sp_group is null in DitCacheImpl::is_similar";

    sum_abs_diff = xllm::parallel_state::reduce(sum_abs_diff, sp_group);
    sum_abs_lhs = xllm::parallel_state::reduce(sum_abs_lhs, sp_group);
    count = xllm::parallel_state::reduce(count, sp_group);
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
    case PolicyType::RACFGCache:
      return std::make_unique<RACFGCache>();
    default:
      return std::make_unique<DiTNonCache>();
  }
}
}  // namespace xllm
