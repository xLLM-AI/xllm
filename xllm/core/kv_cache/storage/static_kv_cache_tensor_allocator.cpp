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

#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"

namespace xllm {

StaticKVCacheTensorAllocator::StaticKVCacheTensorAllocator(
    StaticKVCacheLayout layout,
    const torch::Device& device)
    : layout_(std::move(layout)),
      initialized_full_block_epochs_(layout_.num_slots(), 0) {
  CHECK_GT(layout_.bytes_used(), 0u)
      << "Static tensor storage requires an arena capacity";
  arena_ =
      torch::zeros({static_cast<int64_t>(layout_.bytes_used())},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  LOG(INFO) << "Static shared KV arena: logical_slots=" << layout_.num_slots()
            << ", physical_slabs=" << layout_.num_slabs()
            << ", slab_bytes=" << layout_.slab_bytes()
            << ", banks=" << layout_.bank_count()
            << ", bank_bytes=" << layout_.bank_bytes()
            << ", slice_bytes=" << layout_.slice_bytes()
            << ", arena_bytes=" << layout_.bytes_used()
            << ", Full_pair_bytes=" << layout_.full_pair_bytes()
            << ", Conv_bytes=" << layout_.conv_row_bytes()
            << ", SSM_bundle_bytes=" << layout_.ssm_bundle_bytes()
            << ", checkpoint_stride=" << layout_.checkpoint_stride();
}

bool StaticKVCacheTensorAllocator::needs_full_block_initialization(
    const std::vector<int32_t>& rows,
    const std::vector<uint64_t>& lease_epochs) const {
  const size_t row_width = layout_.full_layer_count() + 1;
  CHECK_EQ(rows.size() % row_width, 0u);
  CHECK_EQ(lease_epochs.size(), rows.size() / row_width);
  bool needs_initialization = false;
  for (size_t i = 0; i < lease_epochs.size(); ++i) {
    const int32_t id = rows[i * row_width];
    CHECK_GE(id, 0);
    CHECK_LT(static_cast<size_t>(id), layout_.num_slots());
    if (id != 0) {
      CHECK_GT(lease_epochs[i], 0u);
      needs_initialization |=
          initialized_full_block_epochs_[id] != lease_epochs[i];
    }
  }
  return needs_initialization;
}

void StaticKVCacheTensorAllocator::initialize_full_blocks(
    const std::vector<int32_t>& rows,
    const std::vector<uint64_t>& lease_epochs) {
  const size_t row_width = layout_.full_layer_count() + 1;
  std::vector<int64_t> bundle_indices;
  bundle_indices.reserve(lease_epochs.size());
  for (size_t i = 0; i < lease_epochs.size(); ++i) {
    const int32_t* entry = rows.data() + i * row_width;
    const int32_t id = entry[0];
    if (id == 0 || initialized_full_block_epochs_[id] == lease_epochs[i]) {
      continue;
    }
    bundle_indices.emplace_back(
        static_cast<int64_t>(entry[1]) /
        static_cast<int64_t>(2 * layout_.full_layers_per_bank()));
    initialized_full_block_epochs_[id] = lease_epochs[i];
  }
  if (bundle_indices.empty()) {
    return;
  }
  std::sort(bundle_indices.begin(), bundle_indices.end());
  const int64_t bundle_bytes =
      static_cast<int64_t>(layout_.full_bundle_bytes());
  const torch::Tensor banks =
      arena_.view({static_cast<int64_t>(layout_.bank_count()),
                   static_cast<int64_t>(layout_.bank_bytes())});
  size_t range_begin = 0;
  while (range_begin < bundle_indices.size()) {
    size_t range_end = range_begin + 1;
    while (range_end < bundle_indices.size() &&
           bundle_indices[range_end] <= bundle_indices[range_end - 1] + 1) {
      ++range_end;
    }
    const int64_t offset_bytes = bundle_indices[range_begin] * bundle_bytes;
    const int64_t length_bytes =
        (bundle_indices[range_end - 1] - bundle_indices[range_begin] + 1) *
        bundle_bytes;
    // Narrow before dispatch so initialization touches only newly leased pairs.
    // Zero bits also clear pairs retyped from FP32 SSM or Conv storage.
    banks.narrow(/*dim=*/1, offset_bytes, length_bytes).zero_();
    range_begin = range_end;
  }
}

void StaticKVCacheTensorAllocator::begin_layer(int64_t layer_id) {
  CHECK(layout_.has_layer(layer_id)) << "Unknown static cache layer";
  CHECK(started_layers_.emplace(layer_id).second)
      << "Static cache layer cannot be allocated twice";
  current_layer_id_ = layer_id;
  allocated_roles_ = 0;
}

torch::Tensor StaticKVCacheTensorAllocator::allocate(
    KVCacheTensorRole role,
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    const torch::Device& device) {
  CHECK_GE(current_layer_id_, 0)
      << "Static cache allocation requires an explicit layer";
  const StaticKVCacheTensorLayout& tensor =
      layout_.tensor_layout(current_layer_id_, role);
  CHECK_EQ(device, arena_.device());
  CHECK_EQ(dtype, layout_.dtype(role));
  CHECK(!shape.empty());
  CHECK_LE(layout_.num_slots(),
           static_cast<size_t>(std::numeric_limits<int64_t>::max()) /
               tensor.rows_per_slot);
  const size_t expected_slots = layout_.num_slots() * tensor.rows_per_slot;
  CHECK_EQ(shape.front(), static_cast<int64_t>(expected_slots))
      << "Static cache logical capacity does not match the layout";
  CHECK_EQ(VirtualKVCachePageLayout::tensor_block_bytes(
               shape, torch::elementSize(dtype)),
           tensor.row_bytes)
      << "Static cache inner row geometry does not match the layout";

  const uint32_t role_mask =
      1u << static_cast<uint32_t>(static_cast<KVCacheTensorRole::Value>(role));
  CHECK_EQ(allocated_roles_ & role_mask, 0u)
      << "Static cache role cannot be allocated twice in one layer";
  std::vector<int64_t> view_shape = shape;
  view_shape.front() = static_cast<int64_t>(tensor.view_rows);
  const int64_t view_bytes =
      static_cast<int64_t>(tensor.view_rows * tensor.row_bytes);
  torch::Tensor view =
      arena_
          .narrow(
              /*dim=*/0, static_cast<int64_t>(tensor.offset_bytes), view_bytes)
          .view(dtype)
          .view(view_shape);
  CHECK(view.is_contiguous());
  allocated_roles_ |= role_mask;
  return view;
}

}  // namespace xllm
