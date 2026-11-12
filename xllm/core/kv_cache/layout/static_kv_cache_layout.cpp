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

  const size_t target_bank_count =
      std::gcd(full_layer_count(), linear_layer_count());
  full_layers_per_bank_ = full_layer_count() / target_bank_count;
  linear_layers_per_bank_ = linear_layer_count() / target_bank_count;
  CHECK_EQ(draft_layer_count() % full_layers_per_bank_, 0u)
      << "Draft Full layers must fill whole companion banks";
  bank_count_ = checked_add(target_bank_count,
                            draft_layer_count() / full_layers_per_bank_);
  full_pair_bytes_ = checked_multiply(2, spec_.full_row_bytes);
  full_bundle_bytes_ =
      checked_multiply(full_layers_per_bank_, full_pair_bytes_);
  ssm_bundle_bytes_ =
      checked_multiply(spec_.checkpoint_stride, spec_.ssm_row_bytes);
  slice_bytes_ = checked_lcm(full_bundle_bytes_, ssm_bundle_bytes_);
  // Conv rows align inside a slice instead of inflating every role to its LCM.
  // Leave room for the leading alignment gap when the sizes do not divide.
  const size_t conv_minimum = slice_bytes_ % spec_.conv_row_bytes == 0
                                  ? spec_.conv_row_bytes
                                  : checked_multiply(2, spec_.conv_row_bytes);
  if (slice_bytes_ < conv_minimum) {
    const size_t slices = conv_minimum / slice_bytes_ +
                          static_cast<size_t>(conv_minimum % slice_bytes_ != 0);
    slice_bytes_ = checked_multiply(slices, slice_bytes_);
  }
  slab_bytes_ = checked_multiply(bank_count_, slice_bytes_);
  // FP32 scratch may alias BF16 NaNs. Every bank permanently separates the
  // writable scratch bundle from its immutable Full read-padding bundle.
  const size_t scratch_bytes =
      std::max({full_bundle_bytes_, spec_.conv_row_bytes, ssm_bundle_bytes_});
  const size_t padding_offset = checked_multiply(
      scratch_bytes / full_bundle_bytes_ +
          static_cast<size_t>(scratch_bytes % full_bundle_bytes_ != 0),
      full_bundle_bytes_);
  full_read_padding_row_ = padding_offset / spec_.full_row_bytes;
  const size_t reserved_bytes = checked_add(padding_offset, full_bundle_bytes_);
  num_reserved_slabs_ = reserved_bytes / slice_bytes_ +
                        static_cast<size_t>(reserved_bytes % slice_bytes_ != 0);
  CHECK_LE(spec_.num_slabs, max_slab_count())
      << "Shared Full bank exceeds the paged attention address range";
  bank_bytes_ = checked_multiply(spec_.num_slabs, slice_bytes_);
  bytes_used_ = checked_multiply(bank_count_, bank_bytes_);
  CHECK_LE(bytes_used_,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const size_t num_layers =
      checked_add(spec_.full_layer_ids.size(), spec_.linear_layer_ids.size());
  tensors_.reserve(checked_multiply(num_layers, 2));
  draft_tensors_.reserve(checked_multiply(draft_layer_count(), 2));
  CHECK_LE(draft_layer_count(),
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  std::unordered_set<int64_t> layer_ids;
  layer_ids.reserve(num_layers);

  const auto append_full_layer =
      [this](std::vector<StaticKVCacheTensorLayout>& tensors,
             int64_t layer_id,
             size_t layer_index,
             size_t bank) {
        const size_t bank_lane = layer_index % full_layers_per_bank_;
        const size_t key_offset =
            checked_add(checked_multiply(bank, bank_bytes_),
                        checked_multiply(bank_lane, full_pair_bytes_));
        tensors.emplace_back(StaticKVCacheTensorLayout{
            .layer_id = layer_id,
            .role = KVCacheTensorRole::KEY,
            .offset_bytes = key_offset,
            .row_bytes = spec_.full_row_bytes,
            .lane = layer_index,
            .bank = bank,
            .bank_lane = bank_lane,
        });
        tensors.emplace_back(StaticKVCacheTensorLayout{
            .layer_id = layer_id,
            .role = KVCacheTensorRole::VALUE,
            .offset_bytes = checked_add(key_offset, spec_.full_row_bytes),
            .row_bytes = spec_.full_row_bytes,
            .lane = layer_index,
            .bank = bank,
            .bank_lane = bank_lane,
        });
      };
  for (size_t layer_index = 0; layer_index < spec_.full_layer_ids.size();
       ++layer_index) {
    const int64_t layer_id = spec_.full_layer_ids[layer_index];
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    append_full_layer(
        tensors_, layer_id, layer_index, layer_index / full_layers_per_bank_);
  }
  for (size_t layer_index = 0; layer_index < draft_layer_count();
       ++layer_index) {
    append_full_layer(draft_tensors_,
                      static_cast<int64_t>(layer_index),
                      layer_index,
                      target_bank_count + layer_index / full_layers_per_bank_);
  }

  for (size_t layer_index = 0; layer_index < spec_.linear_layer_ids.size();
       ++layer_index) {
    const int64_t layer_id = spec_.linear_layer_ids[layer_index];
    CHECK_GE(layer_id, 0);
    CHECK(layer_ids.emplace(layer_id).second) << "Duplicate cache layer id";
    const size_t bank = layer_index / linear_layers_per_bank_;
    const size_t bank_lane = layer_index % linear_layers_per_bank_;
    const size_t conv_group = checked_multiply(layer_index, 2);
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::CONV,
        .row_bytes = spec_.conv_row_bytes,
        .lane = layer_index,
        .bank = bank,
        .bank_lane = bank_lane,
        .linear_group = conv_group,
    });
    tensors_.emplace_back(StaticKVCacheTensorLayout{
        .layer_id = layer_id,
        .role = KVCacheTensorRole::SSM,
        .row_bytes = spec_.ssm_row_bytes,
        .rows_per_slot = spec_.checkpoint_stride,
        .lane = layer_index,
        .bank = bank,
        .bank_lane = bank_lane,
        .linear_group = checked_add(conv_group, 1),
    });
  }
  if (spec_.num_slabs == 0) {
    return;
  }
  for (auto* layouts : {&tensors_, &draft_tensors_}) {
    for (auto& tensor : *layouts) {
      const bool full = tensor.role == KVCacheTensorRole::KEY ||
                        tensor.role == KVCacheTensorRole::VALUE;
      const size_t end =
          full ? checked_multiply(tensor.bank + 1, bank_bytes_) : bytes_used_;
      CHECK_LE(tensor.offset_bytes, end);
      tensor.view_rows = full ? bank_bytes_ / spec_.full_row_bytes -
                                    (2 * full_layers_per_bank_ - 1)
                              : (end - tensor.offset_bytes) / tensor.row_bytes;
    }
  }
}

bool StaticKVCacheLayout::can_fit(size_t full_blocks,
                                  size_t linear_states) const {
  if (spec_.num_slabs < num_reserved_slabs_ || full_blocks >= spec_.num_slots ||
      linear_states >= spec_.num_slots) {
    return false;
  }
  const auto slabs_for = [this](size_t objects, KVCacheTensorRole role) {
    const size_t per_slab = objects_per_slab(role);
    return objects / per_slab + static_cast<size_t>(objects % per_slab != 0);
  };
  const size_t linear_objects =
      checked_multiply(linear_states, linear_objects_per_state());
  const size_t required_slabs = checked_add(
      slabs_for(full_blocks, KVCacheTensorRole::KEY),
      checked_add(slabs_for(linear_objects, KVCacheTensorRole::CONV),
                  slabs_for(linear_objects, KVCacheTensorRole::SSM)));
  return required_slabs <= spec_.num_slabs - num_reserved_slabs_;
}

size_t StaticKVCacheLayout::max_slab_count() const {
  // The deployed paged FIA address chain uses 32-bit relative element offsets.
  // Bound each compact bank, independently of the logical-ID or arena sizes.
  constexpr uint64_t kFiaElementSpan = uint64_t{1} << 32;
  const size_t row_elements =
      full_row_bytes() / torch::elementSize(spec_.dtype);
  const size_t full_limit =
      static_cast<size_t>(kFiaElementSpan / row_elements) /
      full_pages_per_slab();
  size_t limit = std::min(
      full_limit,
      static_cast<size_t>(std::numeric_limits<int64_t>::max()) / slab_bytes_);
  const size_t max_index = std::numeric_limits<int32_t>::max();
  limit = std::min(limit, max_index / full_pages_per_slab());
  for (const KVCacheTensorRole role :
       {KVCacheTensorRole::CONV, KVCacheTensorRole::SSM}) {
    limit = std::min(limit, max_index / rows_per_slab(role));
  }
  return limit;
}

size_t StaticKVCacheLayout::objects_per_slab(KVCacheTensorRole role) const {
  switch (role) {
    case KVCacheTensorRole::KEY:
    case KVCacheTensorRole::VALUE:
      return slice_bytes_ / full_bundle_bytes_;
    case KVCacheTensorRole::CONV:
      return checked_multiply(
          draft_layer_count() == 0 ? 1 : bank_count_,
          slice_bytes_ / spec_.conv_row_bytes -
              static_cast<size_t>(slice_bytes_ % spec_.conv_row_bytes != 0));
    case KVCacheTensorRole::SSM:
      return checked_multiply(draft_layer_count() == 0 ? 1 : bank_count_,
                              slice_bytes_ / ssm_bundle_bytes_);
    default:
      LOG(FATAL) << "Unsupported static cache object role: "
                 << role.to_string();
      return 0;
  }
}

size_t StaticKVCacheLayout::layer_bank_index(int64_t layer_id) const {
  const auto it = std::find_if(
      tensors_.begin(), tensors_.end(), [layer_id](const auto& tensor) {
        return tensor.layer_id == layer_id;
      });
  CHECK(it != tensors_.end()) << "Unknown static cache layer";
  return it->bank;
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
  const size_t bytes =
      role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE
          ? slice_bytes_
          : slab_bytes_;
  const size_t row = row_bytes(role);
  return bytes / row + static_cast<size_t>(bytes % row != 0);
}

size_t StaticKVCacheLayout::row_index(size_t slab,
                                      KVCacheTensorRole role) const {
  const size_t bytes = checked_multiply(slab, slice_bytes_);
  const size_t row = row_bytes(role);
  return bytes / row + static_cast<size_t>(bytes % row != 0);
}

size_t StaticKVCacheLayout::role_row_stride(KVCacheTensorRole role) const {
  CHECK_EQ(slice_bytes_ % row_bytes(role), 0u)
      << "Aligned Conv objects do not have a constant native row stride";
  return slice_bytes_ / row_bytes(role);
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

const StaticKVCacheTensorLayout& StaticKVCacheLayout::draft_tensor_layout(
    int64_t draft_layer_id,
    KVCacheTensorRole role) const {
  CHECK_GE(draft_layer_id, 0);
  CHECK_LT(static_cast<size_t>(draft_layer_id), draft_layer_count());
  CHECK(role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE);
  return draft_tensors_[2 * static_cast<size_t>(draft_layer_id) +
                        static_cast<size_t>(role == KVCacheTensorRole::VALUE)];
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
