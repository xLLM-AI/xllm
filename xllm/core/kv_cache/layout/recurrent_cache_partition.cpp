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
#include <utility>

namespace xllm {

RecurrentCachePartition::RecurrentCachePartition(CachePartitionId id,
                                                 int32_t checkpoint_stride,
                                                 int32_t ssm_snapshots_per_slot,
                                                 bool enable_prefix_cache,
                                                 std::vector<int32_t> layers,
                                                 CacheScalarType conv_dtype,
                                                 CacheScalarType ssm_dtype)
    : id_(std::move(id)),
      checkpoint_stride_(checkpoint_stride),
      ssm_snapshots_per_slot_(ssm_snapshots_per_slot),
      enable_prefix_cache_(enable_prefix_cache),
      conv_{CacheComponentRole::CONV, conv_dtype, layers},
      ssm_{CacheComponentRole::SSM, ssm_dtype, std::move(layers)} {
  CHECK_GT(checkpoint_stride_, 0)
      << "Recurrent checkpoint stride must be positive";
  CHECK_GT(ssm_snapshots_per_slot_, 0)
      << "Recurrent slot must contain SSM snapshots";
  CHECK(!conv_.layers.empty()) << "Recurrent cache partition must own layers";
  CHECK_GE(conv_.layers.front(), 0);
  CHECK(std::adjacent_find(conv_.layers.begin(),
                           conv_.layers.end(),
                           std::greater_equal<int32_t>()) == conv_.layers.end())
      << "Recurrent cache layers must be strictly increasing";
}

}  // namespace xllm
