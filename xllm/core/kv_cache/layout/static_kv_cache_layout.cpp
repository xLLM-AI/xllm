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

#include "core/kv_cache/layout/static_kv_cache_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace {

size_t checked_add(size_t left, size_t right) {
  CHECK_LE(right, std::numeric_limits<size_t>::max() - left);
  return left + right;
}

size_t checked_multiply(size_t left, size_t right) {
  CHECK(right == 0 || left <= std::numeric_limits<size_t>::max() / right);
  return left * right;
}

size_t checked_lcm(size_t left, size_t right) {
  return checked_multiply(left / std::gcd(left, right), right);
}

}  // namespace

StaticKVCacheLayout::StaticKVCacheLayout(StaticKVCacheLayoutSpec spec)
    : spec_(std::move(spec)) {
  CHECK_GT(spec_.full_row_bytes, 0u);
  CHECK_GT(spec_.conv_row_bytes, 0u);
  CHECK_GT(spec_.ssm_row_bytes, 0u);
  CHECK_GT(spec_.checkpoint_stride, 0u);
  CHECK(!spec_.full_layer_ids.empty());
  CHECK(!spec_.linear_layer_ids.empty());
  CHECK_EQ(spec_.full_row_bytes % torch::elementSize(spec_.dtype), 0u);
  CHECK_EQ(spec_.conv_row_bytes % torch::elementSize(spec_.dtype), 0u);
  CHECK_EQ(spec_.ssm_row_bytes % torch::elementSize(spec_.ssm_dtype), 0u);

  full_pair_bytes_ = checked_multiply(2, spec_.full_row_bytes);
  ssm_bundle_bytes_ =
      checked_multiply(spec_.checkpoint_stride, spec_.ssm_row_bytes);
  slab_bytes_ = checked_lcm(checked_lcm(full_pair_bytes_, ssm_bundle_bytes_),
                            spec_.conv_row_bytes);
  bytes_used_ = checked_multiply(spec_.num_slabs, slab_bytes_);
  CHECK_LE(bytes_used_,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const size_t num_layers =
      checked_add(spec_.full_layer_ids.size(), spec_.linear_layer_ids.size());
  tensors_.reserve(checked_multiply(num_layers, 2));
  std::unordered_set<int64_t> layer_ids;
  layer_ids.reserve(num_layers);

  for (size_t layer_index = 0; layer_index < spec_.full_layer_ids.size();
       ++layer_index) {
    const int64_t layer_id = spec_.full_layer_ids[layer_index];
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::KEY,
        .row_bytes = spec_.full_row_bytes,
        .lane = layer_index,
    });
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::VALUE,
        .offset_bytes = spec_.full_row_bytes,
        .row_bytes = spec_.full_row_bytes,
        .lane = layer_index,
    });
  }

  for (size_t layer_index = 0; layer_index < spec_.linear_layer_ids.size();
       ++layer_index) {
    const int64_t layer_id = spec_.linear_layer_ids[layer_index];
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    const size_t conv_group = checked_multiply(layer_index, 2);
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::CONV,
        .row_bytes = spec_.conv_row_bytes,
        .lane = layer_index,
        .linear_group = conv_group,
    });
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::SSM,
        .row_bytes = spec_.ssm_row_bytes,
        .rows_per_slot = spec_.checkpoint_stride,
        .lane = layer_index,
        .linear_group = checked_add(conv_group, 1),
    });
  }
  if (spec_.num_slabs == 0) {
    return;
  }
  for (auto& tensor : tensors_) {
    tensor.view_rows = (bytes_used_ - tensor.offset_bytes) / tensor.row_bytes;
  }
}

bool StaticKVCacheLayout::can_fit(size_t full_blocks,
                                  size_t linear_states) const {
  if (spec_.num_slabs == 0 || full_blocks >= spec_.num_slots ||
      linear_states >= spec_.num_slots) {
    return false;
  }
  const auto slabs_for = [this](size_t objects, size_t object_bytes) {
    const size_t per_slab = slab_bytes_ / object_bytes;
    return objects / per_slab + static_cast<size_t>(objects % per_slab != 0);
  };
  const size_t full_objects = checked_multiply(full_blocks, full_layer_count());
  const size_t linear_objects =
      checked_multiply(linear_states, linear_layer_count());
  const size_t required_slabs =
      checked_add(slabs_for(full_objects, full_pair_bytes_),
                  checked_add(slabs_for(linear_objects, spec_.conv_row_bytes),
                              slabs_for(linear_objects, ssm_bundle_bytes_)));
  return required_slabs <= spec_.num_slabs - 1;
}

size_t StaticKVCacheLayout::row_bytes(KVCacheTensorRole role) const {
  switch (role) {
    case KVCacheTensorRole::KEY:
    case KVCacheTensorRole::VALUE:
      return spec_.full_row_bytes;
    case KVCacheTensorRole::CONV:
      return spec_.conv_row_bytes;
    case KVCacheTensorRole::SSM:
      return spec_.ssm_row_bytes;
    default:
      LOG(FATAL) << "Unsupported static cache tensor role: "
                 << role.to_string();
      return 0;
  }
}

size_t StaticKVCacheLayout::rows_per_slab(KVCacheTensorRole role) const {
  return slab_bytes_ / row_bytes(role);
}

size_t StaticKVCacheLayout::row_index(size_t slab,
                                      KVCacheTensorRole role) const {
  return checked_multiply(slab, rows_per_slab(role));
}

size_t StaticKVCacheLayout::role_row_stride(KVCacheTensorRole role) const {
  return rows_per_slab(role);
}

size_t StaticKVCacheLayout::full_layer_index(int64_t layer_id) const {
  return tensor_layout(layer_id, KVCacheTensorRole::KEY).lane;
}

torch::ScalarType StaticKVCacheLayout::dtype(KVCacheTensorRole role) const {
  CHECK(role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
        role == KVCacheTensorRole::CONV || role == KVCacheTensorRole::SSM);
  return role == KVCacheTensorRole::SSM ? spec_.ssm_dtype : spec_.dtype;
}

bool StaticKVCacheLayout::has_layer(int64_t layer_id) const {
  return std::any_of(tensors_.begin(),
                     tensors_.end(),
                     [layer_id](const StaticKVCacheTensorLayout& tensor) {
                       return tensor.layer_id == layer_id;
                     });
}

const StaticKVCacheTensorLayout& StaticKVCacheLayout::tensor_layout(
    int64_t layer_id,
    KVCacheTensorRole role) const {
  const auto it =
      std::find_if(tensors_.begin(),
                   tensors_.end(),
                   [layer_id, role](const StaticKVCacheTensorLayout& tensor) {
                     return tensor.layer_id == layer_id && tensor.role == role;
                   });
  CHECK(it != tensors_.end())
      << "Static cache layout has no " << role.to_string()
      << " tensor for layer " << layer_id;
  return *it;
}

size_t StaticKVCacheLayout::linear_group(int64_t layer_id,
                                         KVCacheTensorRole role) const {
  CHECK(role == KVCacheTensorRole::CONV || role == KVCacheTensorRole::SSM);
  const auto it =
      std::find_if(tensors_.begin(),
                   tensors_.end(),
                   [layer_id, role](const StaticKVCacheTensorLayout& tensor) {
                     return tensor.layer_id == layer_id && tensor.role == role;
                   });
  return it == tensors_.end() ? 0 : it->linear_group;
}

}  // namespace xllm
