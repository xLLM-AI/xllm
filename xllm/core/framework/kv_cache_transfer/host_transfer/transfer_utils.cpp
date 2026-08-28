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

#include "framework/kv_cache_transfer/host_transfer/transfer_utils.h"

#include <glog/logging.h>

#include <algorithm>

namespace xllm {

uint32_t get_layers_per_event(int64_t num_layers, uint32_t requested_batches) {
  CHECK_GT(num_layers, 0) << "host KV layer count must be positive.";
  uint32_t layers_per_event =
      requested_batches == 0
          ? static_cast<uint32_t>(num_layers)
          : static_cast<uint32_t>(num_layers) / requested_batches;
  return std::max<uint32_t>(layers_per_event, 1);
}

std::vector<LayerRange> build_layer_ranges(int64_t num_layers,
                                           uint32_t layers_per_event) {
  CHECK_GT(num_layers, 0) << "host KV layer count must be positive.";
  CHECK_GT(layers_per_event, 0U) << "layers per event must be positive.";
  std::vector<LayerRange> ranges;
  ranges.reserve((static_cast<uint32_t>(num_layers) + layers_per_event - 1) /
                 layers_per_event);
  for (int64_t begin = 0; begin < num_layers; begin += layers_per_event) {
    ranges.emplace_back(
        LayerRange{begin, std::min(begin + layers_per_event, num_layers)});
  }
  return ranges;
}

GroupedHostKVMappings group_mappings(const HostKVRequest& request) {
  GroupedHostKVMappings grouped;
  for (const HostKVMapping& mapping : request.mappings) {
    grouped[mapping.group_id].emplace_back(mapping);
  }
  return grouped;
}

}  // namespace xllm
