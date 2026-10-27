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
#include <utility>

#include "layers/common/kv_shard_batch_metadata.h"

namespace xllm::layer {

// Sort actual prefill slots and retain their K row indices. Duplicate valid
// slots fail through a bounds-checked gather on the device; padding may repeat.
std::pair<torch::Tensor, torch::Tensor> build_prefill_slot_order(
    const torch::Tensor& source_slots);

}  // namespace xllm::layer
