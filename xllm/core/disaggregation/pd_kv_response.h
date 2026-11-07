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

#pragma once

#include <cstdint>
#include <vector>

#include "disagg_pd.pb.h"

namespace xllm {

class KVCacheState;

// Flat KV exports only the unshared suffix. Grouped caches retain logical
// positions, including shared prefixes and expired SWA placeholders.
// When provided, virtual_memory_block_ids receives the unshared flat KV
// IDs appended during serialization; grouped and linear IDs are excluded.
void append_pd_kv_transfer_groups(
    const KVCacheState& kv_state,
    bool include_linear_state,
    proto::DisaggResponse* response,
    std::vector<int32_t>* virtual_memory_block_ids = nullptr);

}  // namespace xllm
