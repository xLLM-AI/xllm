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
#include <string>
#include <vector>

namespace xllm {

// Chained content hash for prompt blocks.
//
// hash(block_k) mixes block_k's tokens with hash(block_{k-1}), so a block key
// identifies the whole token prefix that produced its KV content. Prefill and
// decode compute the same sequence from the same prompt, which lets a
// content-addressed backend (DataSystem) publish on P and fetch on D without
// exchanging keys.
//
// Returns one lowercase-hex entry per prompt block, i.e.
// ceil(token_ids.size() / block_size) entries; empty when token_ids is empty.
std::vector<std::string> compute_kv_block_hashes(
    const std::vector<int32_t>& token_ids,
    uint32_t block_size,
    uint32_t seed);

}  // namespace xllm
