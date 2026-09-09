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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

struct HostKVLayerLayout {
  int64_t absolute_layer_id = -1;
  int64_t group_layer_slot = -1;
  std::map<KVCacheTensorRole::Value, torch::Tensor> device_roles;
};

struct HostKVGroupLayout {
  int32_t group_id = -1;
  std::map<KVCacheTensorRole::Value, torch::Tensor> host_roles;
  std::vector<HostKVLayerLayout> layers;
};

struct HostKVMapping {
  int32_t group_id = -1;
  int64_t host_block_id = -1;
  int64_t device_block_id = -1;
};

struct HostKVRequest {
  std::vector<HostKVMapping> target_mappings;
  std::vector<HostKVMapping> draft_mappings;
};

struct HostKVLayoutInput {
  int64_t num_layers = 0;
  std::vector<HostKVGroupLayout> groups;
};

class HostKVLayout final {
 public:
  HostKVLayout(int64_t num_layers,
               std::vector<HostKVGroupLayout> groups,
               const torch::Device& device);
  HostKVLayout(const HostKVLayoutInput& input, const torch::Device& device);

  int64_t num_layers() const { return num_layers_; }
  const std::vector<HostKVGroupLayout>& groups() const { return groups_; }
  const HostKVGroupLayout* find_group(int32_t group_id) const;
  const HostKVGroupLayout& group(int32_t group_id) const;
  int64_t host_block_count(int32_t group_id) const;
  int64_t device_block_count(int32_t group_id) const;
  const std::vector<KVCacheTensorRole::Value>& active_roles(
      int32_t group_id) const;
  size_t block_bytes(int32_t group_id, KVCacheTensorRole::Value role) const;
  std::vector<const HostKVLayerLayout*> active_layers(
      int32_t group_id,
      KVCacheTensorRole::Value role,
      int64_t begin_layer,
      int64_t end_layer) const;
  std::vector<const torch::Tensor*> active_tensors(
      int32_t group_id,
      KVCacheTensorRole::Value role,
      int64_t begin_layer,
      int64_t end_layer) const;

 private:
  struct GroupIndex {
    size_t group_offset = 0;
    int64_t host_block_count = 0;
    int64_t device_block_count = 0;
    std::vector<KVCacheTensorRole::Value> active_roles;
    std::map<KVCacheTensorRole::Value, size_t> role_block_bytes;
  };

  int64_t num_layers_ = 0;
  std::vector<HostKVGroupLayout> groups_;
  std::map<int32_t, GroupIndex> groups_by_id_;
};

}  // namespace xllm
