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
  const size_t payload_bytes =
      std::max(full_page_payload_bytes_, linear_page_end);
  const size_t row_alignment =
      checked_lcm(spec_.full_row_bytes, spec_.ssm_row_bytes);
  const size_t candidate = align_up(payload_bytes, row_alignment);
  const size_t conv_gap =
      spec_.conv_row_bytes - std::gcd(candidate, spec_.conv_row_bytes);
  // A rounded Conv start must still end before the page's SSM payload. Slot
  // starts are page multiples, so their Conv gap never exceeds this bound.
  // Fall back to integral Conv strides when that existing gap is too small.
  page_bytes_ =
      ssm_offset >= checked_add(spec_.conv_row_bytes, conv_gap)
          ? candidate
          : align_up(payload_bytes,
                     checked_lcm(row_alignment, spec_.conv_row_bytes));
  pages_per_slot_ = align_up(spec_.full_layer_ids.size(), kFullLayersPerPage) /
                    kFullLayersPerPage;
  slot_bytes_ = checked_multiply(pages_per_slot_, page_bytes_);
  num_linear_groups_ =
      align_up(spec_.linear_layer_ids.size(), pages_per_slot_) /
      pages_per_slot_;
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
    const size_t lane = layer_index / kFullLayersPerPage;
    const size_t layer_offset = checked_multiply(
        checked_multiply(layer_index % kFullLayersPerPage, kFullRoles),
        spec_.full_row_bytes);
    const size_t key_offset =
        checked_add(checked_multiply(lane, page_bytes_), layer_offset);
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::KEY,
        .offset_bytes = key_offset,
        .row_bytes = spec_.full_row_bytes,
        .lane = lane,
    });
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::VALUE,
        .offset_bytes = checked_add(key_offset, spec_.full_row_bytes),
        .row_bytes = spec_.full_row_bytes,
        .lane = lane,
    });
  }

  for (size_t layer_index = 0; layer_index < spec_.linear_layer_ids.size();
       ++layer_index) {
    const int64_t layer_id = spec_.linear_layer_ids[layer_index];
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    const size_t lane = layer_index % pages_per_slot_;
    const size_t page_offset = checked_multiply(lane, page_bytes_);
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::CONV,
        .offset_bytes = page_offset,
        .row_bytes = spec_.conv_row_bytes,
        .lane = lane,
        .linear_group = layer_index / pages_per_slot_,
    });
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::SSM,
        .offset_bytes = checked_add(page_offset, ssm_offset),
        .row_bytes = spec_.ssm_row_bytes,
        .rows_per_slot = spec_.checkpoint_stride,
        .lane = lane,
        .linear_group = layer_index / pages_per_slot_,
    });
  }
  bytes_used_ = checked_multiply(spec_.num_slots, slot_bytes_);
  CHECK_LE(bytes_used_,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  if (spec_.num_slots == 0) {
    return;
  }

  for (auto& tensor : tensors_) {
    tensor.view_rows = checked_add(row_index(spec_.num_slots - 1, tensor.role),
                                   tensor.rows_per_slot);
    const size_t view_bytes =
        checked_multiply(tensor.view_rows, tensor.row_bytes);
    CHECK_LE(checked_add(tensor.offset_bytes, view_bytes), bytes_used_);
  }
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

size_t StaticKVCacheLayout::row_index(size_t slot,
                                      KVCacheTensorRole role) const {
  const size_t bytes = checked_multiply(slot, slot_bytes_);
  const size_t row_size = row_bytes(role);
  return bytes / row_size + static_cast<size_t>(bytes % row_size != 0);
}

size_t StaticKVCacheLayout::role_row_stride(KVCacheTensorRole role) const {
  const size_t row_size = row_bytes(role);
  CHECK_EQ(slot_bytes_ % row_size, 0u)
      << "Cache role requires ceiling physical row mapping.";
  return slot_bytes_ / row_size;
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

size_t StaticKVCacheLayout::linear_group(int64_t layer_id) const {
  const auto it =
      std::find_if(tensors_.begin(),
                   tensors_.end(),
                   [layer_id](const StaticKVCacheTensorLayout& tensor) {
                     return tensor.layer_id == layer_id &&
                            tensor.role == KVCacheTensorRole::CONV;
                   });
  return it == tensors_.end() ? 0 : it->linear_group;
}

}  // namespace xllm
