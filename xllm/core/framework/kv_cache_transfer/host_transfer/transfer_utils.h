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

#include <cstdint>
#include <map>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/layout.h"

namespace xllm {

struct LayerRange {
  int64_t begin = 0;
  int64_t end = 0;
};

using GroupedHostKVMappings = std::map<int32_t, std::vector<HostKVMapping>>;

uint32_t get_layers_per_event(int64_t num_layers, uint32_t requested_batches);

std::vector<LayerRange> build_layer_ranges(int64_t num_layers,
                                           uint32_t layers_per_event);

GroupedHostKVMappings group_mappings(const HostKVRequest& request);

}  // namespace xllm
