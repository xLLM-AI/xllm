/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/host_transfer/layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <set>

namespace xllm {
namespace {

size_t get_tensor_bytes(const torch::Tensor& tensor) {
  const int64_t elements = tensor.numel();
  CHECK_GT(elements, 0) << "tensor must contain elements.";
  const size_t element_size = tensor.element_size();
  CHECK_GT(element_size, static_cast<size_t>(0)) << "tensor dtype is invalid.";
  CHECK_LE(static_cast<uint64_t>(elements),
           std::numeric_limits<size_t>::max() / element_size)
      << "tensor byte count overflows.";
  return static_cast<size_t>(elements) * element_size;
}

size_t get_block_bytes(const torch::Tensor& tensor) {
  CHECK_GT(tensor.dim(), 1) << "device tensor must contain block dimensions.";
  return get_tensor_bytes(tensor[0]);
}

void validate_host_tensor(const torch::Tensor& tensor) {
  CHECK(tensor.defined()) << "host tensor must be defined.";
  CHECK(tensor.is_cpu()) << "host tensor must be on CPU.";
  CHECK(tensor.is_contiguous()) << "host tensor must be contiguous.";
  CHECK_GT(tensor.dim(), 2)
      << "host tensor shape must be [blocks, layers, ...block_dims].";
  CHECK_GT(tensor.size(0), 0) << "host tensor must have blocks.";
  CHECK_GT(tensor.size(1), 0) << "host tensor must have layer slots.";
}

void validate_role_id(KVCacheTensorRole::Value role) {
  CHECK_GE(static_cast<int32_t>(role),
           static_cast<int32_t>(KVCacheTensorRole::KEY))
      << "host KV role is invalid.";
  CHECK_LE(static_cast<int32_t>(role),
           static_cast<int32_t>(KVCacheTensorRole::COMPRESS_INDEX_STATE))
      << "host KV role is invalid.";
}

void validate_role(const torch::Tensor& host_tensor,
                   const torch::Tensor& ref_device_tensor,
                   const std::vector<HostKVLayerLayout>& layers,
                   KVCacheTensorRole::Value role,
                   const torch::Device& device) {
  CHECK(ref_device_tensor.defined()) << "device tensor must be defined.";
  CHECK(ref_device_tensor.is_contiguous())
      << "device tensor must be contiguous.";
  CHECK_EQ(ref_device_tensor.device(), device)
      << "device tensor is on the wrong device.";
  CHECK_GT(ref_device_tensor.dim(), 1)
      << "device tensor shape must be [blocks, ...block_dims].";
  CHECK_GT(ref_device_tensor.size(0), 0) << "device tensor must have blocks.";
  CHECK_EQ(host_tensor.scalar_type(), ref_device_tensor.scalar_type())
      << "host and device role dtypes must match.";
  CHECK_EQ(host_tensor.dim(), ref_device_tensor.dim() + 1)
      << "host tensor must add one group-layer dimension.";
  for (int64_t dim = 1; dim < ref_device_tensor.dim(); ++dim) {
    CHECK_GT(ref_device_tensor.size(dim), 0)
        << "device block dimensions must be positive.";
    CHECK_EQ(host_tensor.size(dim + 1), ref_device_tensor.size(dim))
        << "host and device block shapes must match.";
  }
  CHECK_EQ(get_block_bytes(ref_device_tensor),
           get_tensor_bytes(host_tensor[0][0]))
      << "host and device block bytes must match.";

  for (const HostKVLayerLayout& layer : layers) {
    CHECK_LT(layer.group_layer_slot, host_tensor.size(1))
        << "group layer slot is out of range.";
    auto device_it = layer.device_roles.find(role);
    if (device_it == layer.device_roles.end()) {
      continue;
    }
    const torch::Tensor& tensor = device_it->second;
    CHECK(tensor.defined()) << "device tensor must be defined.";
    CHECK(tensor.is_contiguous()) << "device tensor must be contiguous.";
    CHECK_EQ(tensor.device(), device)
        << "device tensor is on the wrong device.";
    CHECK_EQ(tensor.scalar_type(), ref_device_tensor.scalar_type())
        << "device role dtypes must match across active layers.";
    CHECK_EQ(tensor.dim(), ref_device_tensor.dim())
        << "device role block dimensions must match across active layers.";
    CHECK_EQ(tensor.size(0), ref_device_tensor.size(0))
        << "device role block capacities must match across active layers.";
    for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
      CHECK_EQ(tensor.size(dim), ref_device_tensor.size(dim))
          << "device role block shapes must match across active layers.";
    }
  }
}

}  // namespace

HostKVLayout::HostKVLayout(int64_t num_layers,
                           std::vector<HostKVGroupLayout> groups,
                           const torch::Device& device)
    : num_layers_(num_layers), groups_(std::move(groups)) {
  CHECK_GT(num_layers_, 0) << "host KV layer count must be positive.";
  CHECK_LE(num_layers_,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "host KV layer count exceeds synchronizer capacity.";
  CHECK(!groups_.empty()) << "host KV groups must not be empty.";

  std::sort(groups_.begin(),
            groups_.end(),
            [](const HostKVGroupLayout& lhs, const HostKVGroupLayout& rhs) {
              return lhs.group_id < rhs.group_id;
            });
  for (size_t group_offset = 0; group_offset < groups_.size(); ++group_offset) {
    HostKVGroupLayout& group = groups_[group_offset];
    CHECK_GE(group.group_id, 0) << "host KV group id must be non-negative.";
    CHECK(!group.host_roles.empty()) << "host KV host roles must not be empty.";
    CHECK(!group.layers.empty()) << "host KV group layers must not be empty.";
    CHECK(
        groups_by_id_.emplace(group.group_id, GroupIndex{group_offset}).second)
        << "duplicate host KV group id.";

    for (const auto& role_tensor : group.host_roles) {
      validate_role_id(role_tensor.first);
      validate_host_tensor(role_tensor.second);
    }
    std::set<int64_t> absolute_layer_ids;
    std::set<int64_t> group_layer_slots;
    std::set<KVCacheTensorRole::Value> active_roles;
    for (const HostKVLayerLayout& layer : group.layers) {
      CHECK_GE(layer.absolute_layer_id, 0)
          << "absolute layer id must be non-negative.";
      CHECK_LT(layer.absolute_layer_id, num_layers_)
          << "absolute layer id is out of range.";
      CHECK(absolute_layer_ids.insert(layer.absolute_layer_id).second)
          << "duplicate absolute layer id in host KV group.";
      CHECK_GE(layer.group_layer_slot, 0)
          << "group layer slot must be non-negative.";
      CHECK(group_layer_slots.insert(layer.group_layer_slot).second)
          << "duplicate group layer slot in host KV group.";
      CHECK(!layer.device_roles.empty())
          << "host KV layer must have an active role.";
      for (const auto& role_tensor : layer.device_roles) {
        validate_role_id(role_tensor.first);
        active_roles.insert(role_tensor.first);
      }
    }
    std::sort(group.layers.begin(),
              group.layers.end(),
              [](const HostKVLayerLayout& lhs, const HostKVLayerLayout& rhs) {
                return lhs.absolute_layer_id < rhs.absolute_layer_id;
              });

    GroupIndex& index = groups_by_id_.at(group.group_id);
    index.host_block_count = group.host_roles.begin()->second.size(0);
    index.device_block_count =
        group.layers.front().device_roles.begin()->second.size(0);
    for (const auto& [role, host_tensor] : group.host_roles) {
      (void)role;
      CHECK_EQ(host_tensor.size(0), index.host_block_count)
          << "host role block capacities must match.";
    }
    for (const HostKVLayerLayout& layer : group.layers) {
      for (const auto& [role, device_tensor] : layer.device_roles) {
        (void)role;
        CHECK_EQ(device_tensor.size(0), index.device_block_count)
            << "device role block capacities must match.";
      }
    }
    index.active_roles.assign(active_roles.begin(), active_roles.end());
    for (KVCacheTensorRole::Value role : index.active_roles) {
      auto host_it = group.host_roles.find(role);
      CHECK(host_it != group.host_roles.end())
          << "active device role is missing its host tensor.";
      const torch::Tensor* ref_tensor = nullptr;
      for (const HostKVLayerLayout& layer : group.layers) {
        auto device_it = layer.device_roles.find(role);
        if (device_it != layer.device_roles.end()) {
          ref_tensor = &device_it->second;
          break;
        }
      }
      CHECK(ref_tensor != nullptr)
          << "active role is missing its device tensor.";
      validate_role(host_it->second, *ref_tensor, group.layers, role, device);
      index.role_block_bytes.emplace(role, get_block_bytes(*ref_tensor));
    }
  }
}

HostKVLayout::HostKVLayout(const HostKVLayoutInput& input,
                           const torch::Device& device)
    : HostKVLayout(input.num_layers, input.groups, device) {}

const HostKVGroupLayout* HostKVLayout::find_group(int32_t group_id) const {
  auto it = groups_by_id_.find(group_id);
  if (it == groups_by_id_.end()) {
    return nullptr;
  }
  return &groups_[it->second.group_offset];
}

const HostKVGroupLayout& HostKVLayout::group(int32_t group_id) const {
  auto it = groups_by_id_.find(group_id);
  CHECK(it != groups_by_id_.end()) << "unknown host KV group.";
  return groups_[it->second.group_offset];
}

int64_t HostKVLayout::host_block_count(int32_t group_id) const {
  return groups_by_id_.at(group_id).host_block_count;
}

int64_t HostKVLayout::device_block_count(int32_t group_id) const {
  return groups_by_id_.at(group_id).device_block_count;
}

const std::vector<KVCacheTensorRole::Value>& HostKVLayout::active_roles(
    int32_t group_id) const {
  auto it = groups_by_id_.find(group_id);
  CHECK(it != groups_by_id_.end()) << "unknown host KV group.";
  return it->second.active_roles;
}

size_t HostKVLayout::block_bytes(int32_t group_id,
                                 KVCacheTensorRole::Value role) const {
  const GroupIndex& index = groups_by_id_.at(group_id);
  auto role_it = index.role_block_bytes.find(role);
  CHECK(role_it != index.role_block_bytes.end()) << "inactive host KV role.";
  return role_it->second;
}

std::vector<const HostKVLayerLayout*> HostKVLayout::active_layers(
    int32_t group_id,
    KVCacheTensorRole::Value role,
    int64_t begin_layer,
    int64_t end_layer) const {
  CHECK_GE(begin_layer, 0) << "layer range begin must be non-negative.";
  CHECK_LE(begin_layer, end_layer) << "layer range must not be inverted.";
  CHECK_LE(end_layer, num_layers_) << "layer range end is out of range.";
  const HostKVGroupLayout& group_layout = group(group_id);
  std::vector<const HostKVLayerLayout*> layers;
  for (const HostKVLayerLayout& layer : group_layout.layers) {
    if (layer.absolute_layer_id < begin_layer ||
        layer.absolute_layer_id >= end_layer ||
        layer.device_roles.find(role) == layer.device_roles.end()) {
      continue;
    }
    layers.emplace_back(&layer);
  }
  return layers;
}

std::vector<const torch::Tensor*> HostKVLayout::active_tensors(
    int32_t group_id,
    KVCacheTensorRole::Value role,
    int64_t begin_layer,
    int64_t end_layer) const {
  const std::vector<const HostKVLayerLayout*> layers =
      active_layers(group_id, role, begin_layer, end_layer);
  std::vector<const torch::Tensor*> tensors;
  tensors.reserve(layers.size());
  for (const HostKVLayerLayout* layer : layers) {
    tensors.emplace_back(&layer->device_roles.at(role));
  }
  return tensors;
}

}  // namespace xllm
