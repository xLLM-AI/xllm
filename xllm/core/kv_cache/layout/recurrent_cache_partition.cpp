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

#include "core/kv_cache/layout/recurrent_cache_partition.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>

namespace xllm {

RecurrentCachePartition::RecurrentCachePartition(
    CachePartitionId id,
    int32_t checkpoint_stride,
    int32_t ssm_snapshots_per_slot,
    bool enable_prefix_cache,
    std::vector<CacheComponent> components,
    int32_t tail_tokens_per_slot)
    : id_(std::move(id)),
      checkpoint_stride_(checkpoint_stride),
      ssm_snapshots_per_slot_(ssm_snapshots_per_slot),
      enable_prefix_cache_(enable_prefix_cache),
      components_(std::move(components)),
      tail_tokens_per_slot_(tail_tokens_per_slot) {
  CHECK_GT(checkpoint_stride_, 0)
      << "Recurrent checkpoint stride must be positive";
  CHECK(!components_.empty())
      << "Recurrent cache partition must contain components";
  std::unordered_set<CacheComponentRole> roles;
  for (const auto& component : components_) {
    CHECK(component.role == CacheComponentRole::CONV ||
          component.role == CacheComponentRole::SSM ||
          component.role == CacheComponentRole::KPOOL_TAIL)
        << "Unsupported recurrent cache component";
    CHECK(roles.insert(component.role).second) << "Duplicate cache component";
    CHECK(!component.layers.empty())
        << "Recurrent cache component must own layers";
    CHECK_GE(component.layers.front(), 0);
    CHECK(std::adjacent_find(component.layers.begin(),
                             component.layers.end(),
                             std::greater_equal<int32_t>()) ==
          component.layers.end())
        << "Recurrent cache layers must be strictly increasing";
  }
  const auto* conv = find_component(CacheComponentRole::CONV);
  const auto* ssm = find_component(CacheComponentRole::SSM);
  CHECK((conv == nullptr) == (ssm == nullptr))
      << "Recurrent cache requires both Conv and SSM components";
  if (ssm != nullptr) {
    CHECK(conv->layers == ssm->layers)
        << "Conv and SSM must own the same layers";
    CHECK_GT(ssm_snapshots_per_slot_, 0)
        << "Recurrent slot must contain SSM snapshots";
  } else {
    CHECK_EQ(ssm_snapshots_per_slot_, 0)
        << "Tail-only slots must not reserve SSM snapshots";
  }
  if (find_component(CacheComponentRole::KPOOL_TAIL) != nullptr) {
    CHECK_GT(tail_tokens_per_slot_, 0) << "KPool tail must contain tokens";
  } else {
    CHECK_EQ(tail_tokens_per_slot_, 0)
        << "Tail tokens require a KPool tail component";
  }
}

const CacheComponent* RecurrentCachePartition::find_component(
    CacheComponentRole role) const {
  const auto it = std::find_if(
      components_.begin(), components_.end(), [role](const auto& component) {
        return component.role == role;
      });
  return it == components_.end() ? nullptr : &*it;
}

}  // namespace xllm
