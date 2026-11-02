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

size_t align_up(size_t bytes, size_t alignment) {
  const size_t remainder = bytes % alignment;
  return remainder == 0 ? bytes : checked_add(bytes, alignment - remainder);
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

  constexpr size_t kFullRoles = 2;
  full_page_payload_bytes_ = checked_multiply(
      checked_multiply(kFullLayersPerPage, kFullRoles), spec_.full_row_bytes);
  const size_t ssm_payload_bytes =
      checked_multiply(spec_.checkpoint_stride, spec_.ssm_row_bytes);
  linear_page_payload_bytes_ =
      checked_add(spec_.conv_row_bytes, ssm_payload_bytes);
  const size_t ssm_offset = align_up(spec_.conv_row_bytes, spec_.ssm_row_bytes);
  const size_t linear_page_end = checked_add(ssm_offset, ssm_payload_bytes);
  const size_t row_alignment =
      checked_lcm(checked_lcm(spec_.full_row_bytes, spec_.conv_row_bytes),
                  spec_.ssm_row_bytes);
  unit_bytes_ = align_up(std::max(full_page_payload_bytes_, linear_page_end),
                         row_alignment);
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
    const size_t page_offset =
        checked_multiply(layer_index / kFullLayersPerPage, unit_bytes_);
    const size_t layer_offset = checked_multiply(
        checked_multiply(layer_index % kFullLayersPerPage, kFullRoles),
        spec_.full_row_bytes);
    const size_t key_offset = checked_add(page_offset, layer_offset);
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        layer_id, KVCacheTensorRole::KEY, key_offset, spec_.full_row_bytes});
    tensors_.back().page_group_index = layer_index / kFullLayersPerPage;
    tensors_.emplace_back(
        StaticKVCacheTensorLayout{layer_id,
                                  KVCacheTensorRole::VALUE,
                                  checked_add(key_offset, spec_.full_row_bytes),
                                  spec_.full_row_bytes});
    tensors_.back().page_group_index = layer_index / kFullLayersPerPage;
  }
  full_block_units_ =
      align_up(spec_.full_layer_ids.size(), kFullLayersPerPage) /
      kFullLayersPerPage;

  size_t linear_page_start = 0;
  for (const int64_t layer_id : spec_.linear_layer_ids) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    tensors_.emplace_back(StaticKVCacheTensorLayout{layer_id,
                                                    KVCacheTensorRole::CONV,
                                                    linear_page_start,
                                                    spec_.conv_row_bytes});
    tensors_.back().page_group_index = linear_page_start / unit_bytes_;
    tensors_.emplace_back(
        StaticKVCacheTensorLayout{layer_id,
                                  KVCacheTensorRole::SSM,
                                  checked_add(linear_page_start, ssm_offset),
                                  spec_.ssm_row_bytes,
                                  spec_.checkpoint_stride});
    tensors_.back().page_group_index = linear_page_start / unit_bytes_;
    linear_page_start = checked_add(linear_page_start, unit_bytes_);
  }
  linear_state_units_ = spec_.linear_layer_ids.size();
  CHECK(spec_.num_units == 0 ||
        spec_.num_units >= std::max(full_block_units_, linear_state_units_))
      << "Static arena cannot fit one resource extent";
  bytes_used_ = checked_multiply(spec_.num_units, unit_bytes_);
  CHECK_LE(bytes_used_,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));

  for (auto& tensor : tensors_) {
    const bool full = tensor.role == KVCacheTensorRole::KEY ||
                      tensor.role == KVCacheTensorRole::VALUE;
    tensor.extent_units = full ? full_block_units_ : linear_state_units_;
    if (spec_.num_units == 0) {
      continue;
    }
    const size_t last_start_bytes =
        checked_multiply(spec_.num_units - tensor.extent_units, unit_bytes_);
    tensor.view_rows =
        checked_add(last_start_bytes / tensor.row_bytes, tensor.rows_per_slot);
    const size_t view_bytes =
        checked_multiply(tensor.view_rows, tensor.row_bytes);
    CHECK_LE(checked_add(tensor.offset_bytes, view_bytes), bytes_used_);
  }
}

size_t StaticKVCacheLayout::role_row_stride(KVCacheTensorRole role) const {
  switch (role) {
    case KVCacheTensorRole::KEY:
    case KVCacheTensorRole::VALUE:
      return unit_bytes_ / spec_.full_row_bytes;
    case KVCacheTensorRole::CONV:
      return unit_bytes_ / spec_.conv_row_bytes;
    case KVCacheTensorRole::SSM:
      return unit_bytes_ / spec_.ssm_row_bytes;
    default:
      LOG(FATAL) << "Unsupported static cache tensor role: "
                 << role.to_string();
      return 0;
  }
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

}  // namespace xllm
