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

#include <cstddef>

namespace xllm {

// Physical bytes held by the execution caching allocator, including its idle
// cache. VMM pool handles are allocated directly and are excluded.
struct AllocatorMemoryStats {
  size_t reserved_bytes = 0;
  size_t peak_reserved_bytes = 0;
  size_t allocated_bytes = 0;
};

}  // namespace xllm
