/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstddef>

namespace xllm::mlu {

constexpr size_t kMinRdmaRegisterBytes = 2 * 1024 * 1024;

struct RdmaMemoryPlan {
  size_t logical_bytes = 0;
  size_t block_bytes = 0;
  size_t registered_bytes = 0;
};

RdmaMemoryPlan make_rdma_memory_plan(size_t logical_bytes, size_t block_count);

}  // namespace xllm::mlu
